"""V7: V5 (DeTPP base + diffusion residual) with a DeTR-style residual decoder.

Combines the strengths of V5 and V6:
- V5 gave 0.2253 T-mAP (close to DeTPP 0.2302) by using ConditionalHead as
  the main predictor plus a small per-slot diffusion residual. Residual
  contribution was ~8.5% of base magnitude and couldn't be pushed higher
  (alpha=0.2 reached 0.2177 because model self-shrinks the residual).
- V6 replaces per-slot Head(D, P_base) with a DeTR-style decoder (K
  learnable queries + cross-attention to denoised latents). This decoder
  is much more expressive per slot and can model cross-slot dependencies.

V7 keeps V5's guaranteed ≥ DeTPP floor (ConditionalHead base with
zero-initialised residual at start) but gives diffusion a real chance to
add value by using the DeTR decoder for the residual path.
"""
import random
import time

import torch

from hotpp.data import PaddedBatch
from ..fields import PRESENCE, PRESENCE_PROB
from .common import ScaleGradient
from .detection import DetectionLoss
from .horizon_diffusion_v5 import HorizonDiffusionLossV5
from .horizon_diffusion_v6 import DeTRDecoder


class HorizonDiffusionLossV7(HorizonDiffusionLossV5):
    def __init__(
        self,
        *args,
        detr_d_model=64,
        detr_n_heads=4,
        detr_n_layers=2,
        detr_ffn_dim=128,
        detr_dropout=0.1,
        zero_init_residual=True,  # forwarded to V5; also applied to DeTR output below
        **kwargs,
    ):
        # Parent V5 builds self._decoder (per-slot Head). We replace it with
        # DeTRDecoder after super().__init__ runs.
        super().__init__(*args, zero_init_residual=zero_init_residual, **kwargs)

        self._detr_decoder = None
        if isinstance(self._next_item, DetectionLoss):
            inner_input_size = self._next_item._next_item.input_size
            self._detr_decoder = DeTRDecoder(
                latent_dim=self._embedder.output_size,
                n_queries=self._k,
                output_per_query=inner_input_size,
                d_model=detr_d_model,
                n_heads=detr_n_heads,
                n_layers=detr_n_layers,
                ffn_dim=detr_ffn_dim,
                dropout=detr_dropout,
            )
            if zero_init_residual:
                # Ensure residual starts at exactly zero so V7 == DeTPP at init.
                torch.nn.init.zeros_(self._detr_decoder.output.weight)
                if self._detr_decoder.output.bias is not None:
                    torch.nn.init.zeros_(self._detr_decoder.output.bias)

    def _diffusion_loss(self, conditions, targets, det_inputs=None, det_mask=None, det_lengths=None):
        if not isinstance(self._next_item, DetectionLoss):
            return super()._diffusion_loss(
                conditions=conditions, targets=targets,
                det_inputs=det_inputs, det_mask=det_mask, det_lengths=det_lengths,
            )

        losses = {}
        metrics = {}
        batch_size = len(targets)
        t0 = time.perf_counter()

        embeddings = self._embed_targets(targets)

        steps = torch.randint(1, self._generation_steps + 1, [batch_size], device=embeddings.device)
        steps[random.randint(0, batch_size - 1)] = 1
        corrupted = self._corrupt(embeddings, steps)
        reconstructed = self._denoiser(corrupted, conditions, steps)
        t_denoise = time.perf_counter()

        if self._detach_embeddings_from_step is False:
            reconstruction_target = embeddings.payload
        else:
            reconstruction_target = torch.where(
                steps[:, None, None] > self._detach_embeddings_from_step,
                embeddings.payload.detach(),
                embeddings.payload,
            )
        losses["diffusion"] = ScaleGradient.apply(
            (reconstructed.payload - reconstruction_target).square().mean(),
            self._diffusion_loss_weight,
        )

        # Base DeTPP path (unchanged from V5).
        base_flat = self._compute_base_predictions(conditions)  # (V, K*P_base)
        if self._base_loss_weight != 1.0:
            base_flat = ScaleGradient.apply(base_flat, self._base_loss_weight)

        # Residual diffusion path via DeTR decoder.
        decoder_input_payload = (
            reconstructed.payload.detach() if self._detach_decoder else reconstructed.payload
        )
        decoded = self._detr_decoder(
            PaddedBatch(decoder_input_payload, reconstructed.seq_lens)
        )  # (V, K, P_base)
        residual_flat = decoded.payload.flatten(1)  # (V, K*P_base)

        final_flat = base_flat + self._residual_alpha * residual_flat
        b_det, l_det = det_mask.shape
        det_payload = torch.zeros(
            b_det, l_det, final_flat.shape[-1],
            dtype=final_flat.dtype, device=final_flat.device,
        )
        det_payload.masked_scatter_(det_mask.unsqueeze(-1), final_flat)
        det_outputs = PaddedBatch(det_payload, det_lengths)
        decoder_losses, det_metrics = self._next_item(det_inputs, det_outputs, None)

        t_decoder = time.perf_counter()
        metrics.update({"decoder_" + k: v for k, v in det_metrics.items()})
        losses.update(
            {"decoder_" + k: ScaleGradient.apply(v, self._decoder_loss_weight) for k, v in decoder_losses.items()}
        )
        losses["embedder_regularizer"] = ScaleGradient.apply(
            self._alpha_prods[self._generation_steps] * embeddings.payload.square().mean(),
            self._embedder_regularizer,
        )

        with torch.no_grad():
            metrics["v7_residual_alpha"] = float(
                self._residual_alpha.detach().mean().item()
                if self._residual_alpha.numel() > 0
                else self._residual_alpha.item()
            )
            metrics["v7_base_logit_std"] = float(base_flat.detach().std().item())
            metrics["v7_residual_logit_std"] = float(residual_flat.detach().std().item())
            metrics["v7_residual_to_base_ratio"] = float(
                (self._residual_alpha.detach() * residual_flat.detach()).std().item()
                / (base_flat.detach().std().item() + 1e-8)
            )

        metrics["perf_embed_denoise_s"] = t_denoise - t0
        metrics["perf_decoder_loss_s"] = t_decoder - t_denoise
        metrics["perf_total_loss_s"] = t_decoder - t0
        return losses, metrics

    def predict_next_k(self, outputs, states, fields=None, logits_fields_mapping=None):
        if not isinstance(self._next_item, DetectionLoss):
            return super().predict_next_k(outputs, states, fields=fields, logits_fields_mapping=logits_fields_mapping)

        mask = outputs.seq_len_mask.bool()
        conditions = outputs.payload[mask]

        # Residual diffusion path via DeTR decoder.
        batch_size = len(conditions)
        x = self._noise(batch_size, device=conditions.device, dtype=outputs.payload.dtype)
        for step in range(self._generation_steps, 0, -1):
            x = self._denoising_step(x, conditions, step)
        decoded = self._detr_decoder(x)  # (V, K, P_base)
        residual_flat = decoded.payload.flatten(1)

        # Base DeTPP path.
        base_flat = self._compute_base_predictions(conditions)  # (V, K*P_base)

        final_flat = base_flat + self._residual_alpha * residual_flat

        bsz, seq_len = outputs.shape
        det_payload = torch.zeros(
            bsz, seq_len, final_flat.shape[-1],
            dtype=final_flat.dtype, device=final_flat.device,
        )
        det_payload.masked_scatter_(mask.unsqueeze(-1), final_flat)
        det_outputs = PaddedBatch(det_payload, outputs.seq_lens)

        if fields is None:
            fields = set(self._next_item.data_fields)
        sequences = self._next_item.predict_next_k(
            det_outputs, states, fields=fields, logits_fields_mapping=logits_fields_mapping,
        )

        # Identical presence_selection logic to V5.
        presence_logits_field = PRESENCE + "_logits"
        if presence_logits_field in sequences.payload:
            presence_logits = sequences.payload[presence_logits_field].squeeze(-1)
            calibrated_logits = presence_logits - self._next_item._matching_thresholds - self._presence_threshold_bias
            if self._presence_selection == "topk":
                target_k = self._topk_target_length
                if target_k is None:
                    matching_priors = getattr(self._next_item, "_matching_priors", None)
                    if matching_priors is not None:
                        target_k = max(int(round(float(matching_priors.sum().item()))), 1)
                    else:
                        target_k = max(int(round(self._k / 4)), 1)
                target_k = min(max(int(target_k), 1), presence_logits.shape[-1])
                topk_indices = calibrated_logits.topk(target_k, dim=-1).indices
                presence_mask = torch.zeros_like(presence_logits, dtype=torch.bool)
                presence_mask.scatter_(dim=-1, index=topk_indices, value=True)
                sequences.payload[PRESENCE] = presence_mask
                sequences.payload[PRESENCE_PROB] = torch.sigmoid(calibrated_logits)
            elif self._presence_selection == "calibrated_floor":
                if self._min_presence_count is not None:
                    target_k = int(self._min_presence_count)
                else:
                    target_k = max(int(round(float(self._next_item._matching_priors.sum().item()))), 1)
                target_k = min(target_k, presence_logits.shape[-1])
                presence_mask = calibrated_logits > 0
                current_counts = presence_mask.sum(-1)
                need_more = current_counts < target_k
                if need_more.any():
                    topk_indices = calibrated_logits.topk(target_k, dim=-1).indices
                    floor_mask = torch.zeros_like(presence_mask)
                    floor_mask.scatter_(dim=-1, index=topk_indices, value=True)
                    presence_mask = torch.where(need_more.unsqueeze(-1), floor_mask, presence_mask)
                sequences.payload[PRESENCE] = presence_mask
                sequences.payload[PRESENCE_PROB] = torch.sigmoid(calibrated_logits)
            elif self._presence_selection == "zero":
                sequences.payload[PRESENCE] = presence_logits > 0
                sequences.payload[PRESENCE_PROB] = torch.sigmoid(presence_logits)
            elif self._use_detection_calibration_threshold:
                sequences.payload[PRESENCE] = calibrated_logits > 0
                sequences.payload[PRESENCE_PROB] = torch.sigmoid(calibrated_logits)
            else:
                sequences.payload[PRESENCE] = presence_logits > 0
                sequences.payload[PRESENCE_PROB] = torch.sigmoid(presence_logits)
        return sequences
