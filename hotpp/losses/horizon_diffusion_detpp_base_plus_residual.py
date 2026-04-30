"""V5: DeTPP base head + diffusion residual.

Architecture:
    backbone GRU --> context (V, D_context)
                     |
        ┌────────────┴───────────────┐
        |                            |
        ▼                            ▼
   ConditionalHead              GRU diffusion denoiser → per-slot Head
   (DeTPP-style queries)         (residual refinement)
        |                            |
        ▼                            ▼
   base_logits (V, K*P_base)   residual (V, K*P_base)
        |                            |
        └────────►  base + α·residual ─────► DetectionLoss
                  (final K*P_base predictions)

With α=0 (or with zero-initialised residual decoder), the model is
behaviourally equivalent to pure DeTPP — guaranteeing ≥ DeTPP performance
as a floor. Training learns to use the diffusion residual where it helps.

Args (in addition to V3):
    base_head_partial: Constructor for the base DeTPP head, accepting
        (input_size, output_size). Typically a ConditionalHead.
    residual_alpha: Initial mixing coefficient for the diffusion residual.
    learnable_alpha: If True, ``alpha`` is a learnable parameter; else fixed.
    zero_init_residual: If True, zero-initialize the LAST linear layer of
        the per-slot decoder so the residual starts at exactly 0 → equivalent
        to pure DeTPP at step 0.
    base_loss_weight: Scale gradient through the base head separately
        (useful if you want to slow down DeTPP base updates).
"""
import random
import time

import torch

from hotpp.data import PaddedBatch
from ..fields import PRESENCE, PRESENCE_PROB
from .common import ScaleGradient
from .detection import DetectionLoss
from .horizon_diffusion_fixed_perslot_decoder import HorizonDiffusionLossFixedPerslotDecoder


class HorizonDiffusionLossDetppBasePlusResidual(HorizonDiffusionLossFixedPerslotDecoder):
    def __init__(
        self,
        *args,
        base_head_partial=None,
        residual_alpha=0.1,
        learnable_alpha=False,
        zero_init_residual=True,
        base_loss_weight=1.0,
        detach_backbone_from_diffusion=False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self._base_head = None
        self._base_loss_weight = float(base_loss_weight)
        self._detach_backbone_from_diffusion = bool(detach_backbone_from_diffusion)
        if isinstance(self._next_item, DetectionLoss):
            if base_head_partial is None:
                raise ValueError("Need `base_head_partial` for V5 (e.g. ConditionalHead).")
            condition_size = self._denoiser.condition_size
            self._base_head = base_head_partial(condition_size, self._next_item.input_size)

            alpha_tensor = torch.tensor(float(residual_alpha))
            if learnable_alpha:
                self._residual_alpha = torch.nn.Parameter(alpha_tensor)
            else:
                self.register_buffer("_residual_alpha", alpha_tensor)

            if zero_init_residual:
                # Zero the last Linear in self._decoder so residual = 0 at init.
                last_linear = None
                for m in self._decoder.modules():
                    if isinstance(m, torch.nn.Linear):
                        last_linear = m
                if last_linear is not None:
                    torch.nn.init.zeros_(last_linear.weight)
                    if last_linear.bias is not None:
                        torch.nn.init.zeros_(last_linear.bias)

    def _compute_base_predictions(self, conditions):
        """ConditionalHead applied to flat (V, D) -> (V, K*P_base)."""
        v = conditions.shape[0]
        device = conditions.device
        wrapped = PaddedBatch(
            conditions.unsqueeze(1),
            torch.ones(v, dtype=torch.long, device=device),
        )
        out = self._base_head(wrapped)  # (V, 1, K*P_base)
        return out.payload.squeeze(1)  # (V, K*P_base)

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
        # Optionally cut the diffusion gradient flow back to the shared backbone;
        # this keeps the backbone representation driven only by the base path
        # (ConditionalHead + DetectionLoss), matching pure DeTPP training dynamics.
        diff_conditions = conditions.detach() if self._detach_backbone_from_diffusion else conditions
        reconstructed = self._denoiser(corrupted, diff_conditions, steps)
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

        # ---------------- Base DeTPP path -----------------
        base_flat = self._compute_base_predictions(conditions)  # (V, K*P_base)
        if self._base_loss_weight != 1.0:
            base_flat = ScaleGradient.apply(base_flat, self._base_loss_weight)

        # ---------------- Residual diffusion path -----------------
        decoder_input_payload = (
            reconstructed.payload.detach() if self._detach_decoder else reconstructed.payload
        )
        decoded = self._decoder(PaddedBatch(decoder_input_payload, reconstructed.seq_lens))  # (V, K, P_base)
        residual_flat = decoded.payload.flatten(1)  # (V, K*P_base)

        # ---------------- Combine and feed DetectionLoss -----------------
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

        # V5 telemetry.
        with torch.no_grad():
            metrics["v5_residual_alpha"] = float(
                self._residual_alpha.detach().mean().item()
                if self._residual_alpha.numel() > 0
                else self._residual_alpha.item()
            )
            metrics["v5_base_logit_std"] = float(base_flat.detach().std().item())
            metrics["v5_residual_logit_std"] = float(residual_flat.detach().std().item())
            metrics["v5_residual_to_base_ratio"] = float(
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
        conditions = outputs.payload[mask]  # (V, D_context)

        # Residual diffusion path.
        batch_size = len(conditions)
        x = self._noise(batch_size, device=conditions.device, dtype=outputs.payload.dtype)
        for step in range(self._generation_steps, 0, -1):
            x = self._denoising_step(x, conditions, step)
        decoded = self._decoder(x)  # (V, K, P_base)
        residual_flat = decoded.payload.flatten(1)  # (V, K*P_base)

        # Base DeTPP path.
        base_flat = self._compute_base_predictions(conditions)  # (V, K*P_base)

        # Combine.
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

        # Same presence_selection logic as V3.
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
