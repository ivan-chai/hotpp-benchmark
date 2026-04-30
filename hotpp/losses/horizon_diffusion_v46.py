"""V46: V44 + diffusion conditioned on multi-scale raw event history.

Идея: дать денойзеру информацию, которой нет у DeTPP ConditionalHead.
DeTPP видит RNN hidden state (recurrent summary). Диффузия в V46 видит
multi-scale pooled raw event embeddings (commutative, bag-of-events
summary at multiple scales). Это другой view на ту же последовательность —
RNN кодирует трактовку, multi-scale pool — просто статистика.

Декодер по-прежнему использует RNN context (как в DeTPP) для предсказаний.
"""
import torch

from hotpp.data import PaddedBatch
from .horizon_diffusion_v44 import HorizonDiffusionLossV44


class MultiScaleHistoryPooler(torch.nn.Module):
    """Causal multi-scale sliding-mean of raw event embeddings.

    For each position p, computes mean of last `n` events at multiple
    scales `n=1,5,32,128`, projects each scale, concatenates, then a final
    Linear maps to `output_dim`.

    Input:  (B, L, D_embed) raw embeddings
    Output: (B, L, D_output) per-position multi-scale history vector
    """

    def __init__(self, embed_dim, output_dim, scales=(1, 5, 32, 128)):
        super().__init__()
        self.scales = tuple(scales)
        per_scale = max(8, output_dim // len(self.scales))
        self.scale_projs = torch.nn.ModuleList([
            torch.nn.Linear(embed_dim, per_scale) for _ in self.scales
        ])
        self.act = torch.nn.GELU()
        self.combine = torch.nn.Linear(per_scale * len(self.scales), output_dim)

    @staticmethod
    def causal_window_mean(x, n):
        """Mean of last `n` elements along seq dim 1, causal.

        Uses 1D causal convolution with a uniform kernel — deterministic-friendly
        (cumsum on CUDA isn't deterministic, F.conv1d is).
        """
        b, l, d = x.shape
        device = x.device
        if n <= 1:
            return x  # window of size 1 = identity
        x_t = x.transpose(1, 2).contiguous()  # (B, D, L)
        kernel = torch.ones(d, 1, n, device=device, dtype=x.dtype)
        x_padded = torch.nn.functional.pad(x_t, (n - 1, 0))  # left-pad with zeros
        window_sum = torch.nn.functional.conv1d(x_padded, kernel, groups=d)  # (B, D, L)
        idx = torch.arange(1, l + 1, device=device, dtype=x.dtype).clamp(max=n)
        return (window_sum / idx.view(1, 1, l)).transpose(1, 2).contiguous()

    def forward(self, embeddings):
        scaled = []
        for proj, n in zip(self.scale_projs, self.scales):
            pooled = self.causal_window_mean(embeddings, n)  # (B, L, D)
            scaled.append(self.act(proj(pooled)))
        cat = torch.cat(scaled, dim=-1)
        return self.combine(cat)


class HorizonDiffusionLossV46(HorizonDiffusionLossV44):
    def __init__(self, *args, history_scales=(1, 5, 32, 128), **kwargs):
        super().__init__(*args, **kwargs)
        cond_size = self._denoiser.condition_size  # 64
        embed_dim = self._embedder.output_size  # 33
        self._history_pooler = MultiScaleHistoryPooler(
            embed_dim=embed_dim,
            output_dim=cond_size,
            scales=history_scales,
        )
        # Stash for cross-method access during one forward pass.
        self._cached_history_cond_flat = None

    # ---- helpers ----
    def _compute_history_cond(self, inputs):
        """Embed all input events, run multi-scale pooler, return (B, L, cond)."""
        embed_inputs = PaddedBatch(
            {k: inputs.payload[k] for k in self._diffusion_fields},
            inputs.seq_lens,
            set(self._diffusion_fields),
        )
        raw = self._embedder(self._compute_time_deltas(embed_inputs))  # PaddedBatch (B, L, D)
        return self._history_pooler(raw.payload)  # (B, L, cond_size)

    def forward(self, inputs, outputs, states):
        # Compute history conditioning BEFORE parent forward (which flattens).
        history_full = self._compute_history_cond(inputs)  # (B, L, cond_size)
        # Stash flat version for use inside _diffusion_loss.
        # Need to slice and flatten in the same way parent does.
        # Parent forward will: lengths = (target seq_lens), outputs trimmed to targets.shape[1],
        # then mask = targets.seq_len_mask, and conditions = outputs.payload[mask].
        # We mirror that here: compute mask consistently.
        # For simplicity, stash full history; slicing happens in _diffusion_loss override.
        self._cached_history_full = history_full
        return super().forward(inputs, outputs, states)

    def _diffusion_loss(self, conditions, targets, det_inputs=None, det_mask=None, det_lengths=None):
        if det_mask is None or self._cached_history_full is None:
            return super()._diffusion_loss(
                conditions=conditions, targets=targets,
                det_inputs=det_inputs, det_mask=det_mask, det_lengths=det_lengths,
            )

        # Slice cached history with the same mask used for `conditions`.
        b_det, l_det = det_mask.shape
        history_trim = self._cached_history_full[:, :l_det]  # match seq dim to det layout
        history_flat = history_trim[det_mask]  # (V, cond_size)

        # Replace `conditions` for the denoiser ONLY; decoder still uses RNN context.
        # Do the diffusion math here mirroring V44, but pass history_flat to denoiser
        # and original `conditions` (RNN context) to decoder.
        import random
        import time
        from .common import ScaleGradient
        from .detection import DetectionLoss

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

        # Use history_flat as denoiser conditioning (DIFFERENT from RNN context).
        # Match dtype to corrupted (fp16 under mixed precision) — pooler may produce fp32.
        history_flat = history_flat.to(corrupted.payload.dtype)
        reconstructed = self._denoiser(corrupted, history_flat, steps)
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

        # Decoder uses RNN context (parent V44 ContextQueryAugmentedHead expects context kwarg).
        decoder_input_payload = (
            reconstructed.payload.detach() if self._detach_decoder else reconstructed.payload
        )
        decoded = self._decoder(
            PaddedBatch(decoder_input_payload, reconstructed.seq_lens),
            context=conditions,
        )
        decoded_flat = decoded.payload.flatten(1)

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
        # Use cached history if forward was just called (validation/test in same pass).
        if self._cached_history_full is None:
            # No cache: fall back to parent behaviour (shouldn't happen in normal flow).
            return super().predict_next_k(outputs, states, fields=fields, logits_fields_mapping=logits_fields_mapping)
        from .detection import DetectionLoss
        from ..fields import PRESENCE, PRESENCE_PROB

        if not isinstance(self._next_item, DetectionLoss):
            return super().predict_next_k(outputs, states, fields=fields, logits_fields_mapping=logits_fields_mapping)

        mask = outputs.seq_len_mask.bool()
        conditions = outputs.payload[mask]  # RNN context for decoder
        history_flat = self._cached_history_full[:, :outputs.shape[1]][mask]  # raw history for denoiser

        batch_size = len(conditions)
        x = self._noise(batch_size, device=conditions.device, dtype=outputs.payload.dtype)
        # Match denoiser hidden dtype to noise dtype (avoid fp16/fp32 mismatch).
        history_flat = history_flat.to(x.payload.dtype)
        for step in range(self._generation_steps, 0, -1):
            x = self._denoising_step(x, history_flat, step)

        decoded = self._decoder(x, context=conditions)
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

        # Same presence_selection logic as V3/V44.
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
