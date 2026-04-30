"""V6: DeTR-style cross-attention decoder over denoised diffusion latents.

Why V3/V4/V5 per-slot paths underperform:
- V3 (per-slot Head D->P_base): no cross-slot mixing, too narrow.
- V4 (self-attention over K slots): starts from scratch, overfits/destabilizes.
- V5 (DeTPP base + per-slot residual): residual too limited; ratio plateaus.

V6 applies the DeTR decoder idea: K learnable queries (== K detection heads)
cross-attend to the K denoised diffusion latents. Each query produces a
slot-specific (P_base) prediction. This preserves:
- Per-slot output structure (K -> K), as required by the advisor.
- DeTPP-style slot identity via learnable queries.
- Cross-slot information access via cross-attention (what monolithic V1
  had via the flat Linear, and what V3 per-slot lacks).

Everything else (horizon window construction, mask embedding, presence
calibration, presence_selection logic) is identical to V3.
"""
import random
import time

import torch

from hotpp.data import PaddedBatch
from ..fields import PRESENCE, PRESENCE_PROB
from .common import ScaleGradient
from .detection import DetectionLoss
from .horizon_diffusion_v3 import HorizonDiffusionLossV3


class DeTRDecoder(torch.nn.Module):
    """DeTR-style decoder: K learnable queries cross-attend to denoised latents.

    Shapes: input (V, K_mem, D_latent); output (V, K_queries, output_per_query).
    """

    def __init__(
        self,
        latent_dim,
        n_queries,
        output_per_query,
        d_model=64,
        n_heads=4,
        n_layers=2,
        ffn_dim=128,
        dropout=0.1,
    ):
        super().__init__()
        self._n_queries = n_queries
        self._output_per_query = output_per_query
        self._d_model = d_model

        if latent_dim != d_model:
            self.input_proj = torch.nn.Linear(latent_dim, d_model)
        else:
            self.input_proj = torch.nn.Identity()

        # Learnable per-slot queries (like DeTR object queries / DeTPP queries).
        self.queries = torch.nn.Parameter(torch.randn(n_queries, d_model) * 0.02)

        decoder_layer = torch.nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
            activation="relu",
            norm_first=True,  # pre-LN is more stable for training from scratch.
        )
        self.decoder = torch.nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.output = torch.nn.Linear(d_model, output_per_query)

    def forward(self, latents_batch):
        # latents_batch: PaddedBatch with payload (V, K_mem, latent_dim).
        latents = latents_batch.payload
        v = latents.shape[0]

        memory = self.input_proj(latents)  # (V, K_mem, d_model)
        queries = self.queries.unsqueeze(0).expand(v, -1, -1)  # (V, K_queries, d_model)

        out = self.decoder(tgt=queries, memory=memory)  # (V, K_queries, d_model)
        predictions = self.output(out)  # (V, K_queries, output_per_query)

        seq_lens = torch.full(
            [v], self._n_queries, dtype=torch.long, device=predictions.device,
        )
        return PaddedBatch(predictions, seq_lens)


class HorizonDiffusionLossV6(HorizonDiffusionLossV3):
    """V3 with DeTR-style decoder for the DetectionLoss branch."""

    def __init__(
        self,
        *args,
        detr_d_model=64,
        detr_n_heads=4,
        detr_n_layers=2,
        detr_ffn_dim=128,
        detr_dropout=0.1,
        zero_init_output=False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

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
            if zero_init_output:
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

        # DeTR-style decoder.
        decoder_input_payload = (
            reconstructed.payload.detach() if self._detach_decoder else reconstructed.payload
        )
        decoded = self._detr_decoder(
            PaddedBatch(decoder_input_payload, reconstructed.seq_lens)
        )  # (V, K, P_base)
        decoded_flat = decoded.payload.flatten(1)  # (V, K*P_base)

        b_det, l_det = det_mask.shape
        det_payload = torch.zeros(
            b_det, l_det, decoded_flat.shape[-1],
            dtype=decoded_flat.dtype, device=decoded_flat.device,
        )
        det_payload.masked_scatter_(det_mask.unsqueeze(-1), decoded_flat)
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
        metrics["perf_embed_denoise_s"] = t_denoise - t0
        metrics["perf_decoder_loss_s"] = t_decoder - t_denoise
        metrics["perf_total_loss_s"] = t_decoder - t0
        return losses, metrics

    def predict_next_k(self, outputs, states, fields=None, logits_fields_mapping=None):
        if not isinstance(self._next_item, DetectionLoss):
            return super().predict_next_k(outputs, states, fields=fields, logits_fields_mapping=logits_fields_mapping)

        mask = outputs.seq_len_mask.bool()
        conditions = outputs.payload[mask]

        batch_size = len(conditions)
        x = self._noise(batch_size, device=conditions.device, dtype=outputs.payload.dtype)
        for step in range(self._generation_steps, 0, -1):
            x = self._denoising_step(x, conditions, step)

        decoded = self._detr_decoder(x)  # (V, K, P_base)
        decoded_flat = decoded.payload.flatten(1)

        bsz, seq_len = outputs.shape
        det_payload = torch.zeros(
            bsz, seq_len, decoded_flat.shape[-1],
            dtype=decoded_flat.dtype, device=decoded_flat.device,
        )
        det_payload.masked_scatter_(mask.unsqueeze(-1), decoded_flat)
        det_outputs = PaddedBatch(det_payload, outputs.seq_lens)

        if fields is None:
            fields = set(self._next_item.data_fields)
        sequences = self._next_item.predict_next_k(
            det_outputs, states, fields=fields, logits_fields_mapping=logits_fields_mapping,
        )

        # Identical presence_selection logic to V3.
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
