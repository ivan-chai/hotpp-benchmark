"""V41: V3 architecture but mask the diffusion MSE loss on absent slots.

Rationale (from V40 result):
- V3 forces ALL 37/48 absent slots to reconstruct the SAME shared mask
  embedding -> 37 identical latents -> 37 identical decoded predictions ->
  Hungarian effectively has K~=11 candidates instead of 48.
- V40 fixed it with K per-slot learnable masks. Diffusion loss dropped
  (0.66 vs 1.39) so denoiser DOES learn diverse latents. But calibration
  broke: 37 diverse absent latents produce VARIED presence_logits, some
  passing the threshold -> mean-predicted-length jumped to 21.4 (target
  15.6) -> many false positives, next-item-accuracy collapsed to 0.07.
- V41 removes the conflict: do NOT supervise absent slots in MSE at all.
  Denoiser is only pulled toward real event embeddings for present slots.
  Absent slots' latents are shaped purely by the DetectionLoss BCE-negative
  signal (unmatched -> low presence). No fake target, no calibration breakage.
"""
import random
import time

import torch

from hotpp.data import PaddedBatch
from ..fields import PRESENCE
from .common import ScaleGradient
from .detection import DetectionLoss
from .horizon_diffusion_v3 import HorizonDiffusionLossV3


class HorizonDiffusionLossV41(HorizonDiffusionLossV3):
    """V3 with presence-masked diffusion MSE: only present slots are supervised."""

    def _diffusion_loss(self, conditions, targets, det_inputs=None, det_mask=None, det_lengths=None):
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

        # Mask MSE by per-slot presence: only supervise slots that have a real
        # horizon event. Absent slots are left to be shaped by DetectionLoss.
        present = targets.payload[PRESENCE][:, 1:].float().unsqueeze(-1)  # (V, K, 1)
        squared_err = (reconstructed.payload - reconstruction_target).square()  # (V, K, D)
        # Average over D, then weight by per-slot presence and renormalise.
        d = reconstructed.payload.shape[-1]
        per_slot_mse = squared_err.mean(dim=-1)  # (V, K)
        present_2d = present.squeeze(-1)  # (V, K)
        denom = present_2d.sum().clamp(min=1.0)
        diff_loss = (per_slot_mse * present_2d).sum() / denom

        losses["diffusion"] = ScaleGradient.apply(diff_loss, self._diffusion_loss_weight)

        if isinstance(self._next_item, DetectionLoss):
            if (det_inputs is None) or (det_mask is None) or (det_lengths is None):
                raise RuntimeError("Detection branch requires det_inputs, det_mask, det_lengths.")

            decoder_input_payload = (
                reconstructed.payload.detach() if self._detach_decoder else reconstructed.payload
            )
            decoded = self._decoder(
                PaddedBatch(decoder_input_payload, reconstructed.seq_lens)
            )
            decoded_flat = decoded.payload.flatten(1)
            b_det, l_det = det_mask.shape
            det_payload = torch.zeros(
                b_det, l_det, decoded_flat.shape[-1],
                dtype=decoded_flat.dtype, device=decoded_flat.device,
            )
            det_payload.masked_scatter_(det_mask.unsqueeze(-1), decoded_flat)
            det_outputs = PaddedBatch(det_payload, det_lengths)
            decoder_losses, det_metrics = self._next_item(det_inputs, det_outputs, None)
        else:
            decoded = self._decoder(
                PaddedBatch(decoder_input_payload, reconstructed.seq_lens)
            )
            decoded_payload = torch.cat([decoded.payload, torch.empty_like(decoded.payload[:, :1])], dim=1)
            decoded_batch = PaddedBatch(decoded_payload, targets.seq_lens)
            decoder_losses, det_metrics = self._next_item(targets, decoded_batch, None)

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
        metrics["v41_presence_ratio_for_mse"] = float(present_2d.mean().item())
        return losses, metrics
