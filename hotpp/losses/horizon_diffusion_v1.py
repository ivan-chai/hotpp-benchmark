import random
import time

import torch

from hotpp.data import PaddedBatch
from ..fields import PRESENCE, PRESENCE_PROB
from .common import ScaleGradient
from .detection import DetectionLoss
from .diffusion import DiffusionLoss


class HorizonDiffusionLossV1(DiffusionLoss):
    """Diffusion in latent space with DeTPP-style decoder supervision on the horizon.

    Design:
    1. Build a fixed-length horizon target chain of length ``K + 1``.
    2. Replace padded future events with a learned mask embedding before diffusion.
    3. Keep denoising unchanged: diffusion operates only in latent space.
    4. Decode denoised latents into DeTPP outputs and apply the usual DetectionLoss.

    This keeps the responsibilities separate:
    - diffusion learns latent horizon trajectories;
    - detection/matching lives entirely at the decoder level.
    """

    def __init__(
        self,
        *args,
        horizon=None,
        use_detection_calibration_threshold=True,
        presence_threshold_bias=0.0,
        presence_selection="calibrated",
        topk_target_length=None,
        min_presence_count=None,
        detection_sequence_decoder_partial=None,
        **kwargs
    ):
        decoder_partial = kwargs.get("decoder_partial", None)
        super().__init__(*args, **kwargs)

        self._horizon = getattr(self._next_item, "_horizon", None) if horizon is None else horizon
        if self._horizon is None:
            raise ValueError("Need `horizon` or a next_item_loss with `_horizon`.")

        self._use_detection_calibration_threshold = use_detection_calibration_threshold
        self._presence_threshold_bias = float(presence_threshold_bias)
        self._presence_selection = presence_selection
        self._topk_target_length = topk_target_length
        self._min_presence_count = min_presence_count
        if self._presence_selection not in {"calibrated", "zero", "topk", "calibrated_floor"}:
            raise ValueError(f"Unknown presence_selection: {self._presence_selection}.")

        # Used only for diffusion target embeddings.
        self._mask_embedding = torch.nn.Parameter(torch.zeros(self._embedder.output_size))

        self._diffusion_fields = self.fields
        if hasattr(self._next_item, "data_fields"):
            self._diffusion_fields = self._next_item.data_fields

        self._detection_sequence_decoder = None
        if isinstance(self._next_item, DetectionLoss):
            decoder_ctor = detection_sequence_decoder_partial or decoder_partial
            if decoder_ctor is None:
                raise ValueError("Need `detection_sequence_decoder_partial` or `decoder_partial`.")
            self._detection_sequence_decoder = decoder_ctor(
                self._k * self._embedder.output_size,
                self._next_item.input_size,
            )

    def _build_horizon_windows(self, inputs):
        """Build horizon targets of shape (B, L, K + 1) for diffusion embedding.

        The first position in each window is the current event.
        Future positions contain the next events within the horizon; all remaining
        slots are padded and marked with presence = 0.
        """
        b, l = inputs.shape
        k1 = self._k + 1
        device = inputs.device

        base = torch.arange(l, device=device)[None, :, None]
        offs = torch.arange(k1, device=device)[None, None, :]
        valid_index = base + offs < inputs.seq_lens[:, None, None]

        current_timestamps = inputs.payload[self._timestamps_field]
        rolled_timestamps = torch.stack(
            [current_timestamps.roll(-i, 1) for i in range(k1)],
            dim=2,
        )
        horizon_valid = (rolled_timestamps - current_timestamps.unsqueeze(2)) < self._horizon
        valid = valid_index & horizon_valid

        windows = {}
        for name in self._diffusion_fields:
            values = inputs.payload[name]
            window = torch.stack([values.roll(-i, 1) for i in range(k1)], dim=2)
            if name == self._timestamps_field:
                pad_value = current_timestamps.unsqueeze(2) + self._horizon + 1
            else:
                pad_value = torch.zeros_like(window)
            window = torch.where(valid, window, pad_value)
            windows[name] = window

        windows[PRESENCE] = valid.long()
        return PaddedBatch(windows, inputs.seq_lens, set(self._diffusion_fields) | {PRESENCE})

    def _embed_targets(self, targets):
        embed_inputs = PaddedBatch(
            {k: targets.payload[k] for k in self._diffusion_fields},
            targets.seq_lens,
            set(self._diffusion_fields),
        )
        embeddings = self._embedder(self._compute_time_deltas(embed_inputs))
        embeddings = PaddedBatch(
            embeddings.payload[:, 1:],
            (embeddings.seq_lens - 1).clip(min=0),
        )

        presence = targets.payload[PRESENCE][:, 1:].bool()
        embeddings.payload = torch.where(
            presence.unsqueeze(-1),
            embeddings.payload,
            self._mask_embedding.view(1, 1, -1),
        )
        return embeddings

    def _decode_detection_sequence(self, sequence):
        flattened = sequence.payload.flatten(1)
        lengths = torch.ones(len(flattened), dtype=torch.long, device=sequence.device)
        decoded = self._detection_sequence_decoder(PaddedBatch(flattened.unsqueeze(1), lengths))
        return decoded.payload[:, 0]

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

        losses["diffusion"] = ScaleGradient.apply(
            (reconstructed.payload - reconstruction_target).square().mean(),
            self._diffusion_loss_weight,
        )

        if isinstance(self._next_item, DetectionLoss):
            if (det_inputs is None) or (det_mask is None) or (det_lengths is None):
                raise RuntimeError("Detection branch requires det_inputs, det_mask, det_lengths.")

            decoded_values = self._decode_detection_sequence(
                PaddedBatch(
                    reconstructed.payload.detach() if self._detach_decoder else reconstructed.payload,
                    reconstructed.seq_lens,
                )
            )
            decoded_dim = decoded_values.shape[1]
            b_det, l_det = det_mask.shape
            det_payload = torch.zeros(b_det, l_det, decoded_dim, dtype=decoded_values.dtype, device=decoded_values.device)
            det_payload.masked_scatter_(det_mask.unsqueeze(-1), decoded_values)
            det_outputs = PaddedBatch(det_payload, det_lengths)
            decoder_losses, metrics = self._next_item(det_inputs, det_outputs, None)
        else:
            decoded = self._decoder(
                PaddedBatch(
                    reconstructed.payload.detach() if self._detach_decoder else reconstructed.payload,
                    reconstructed.seq_lens,
                )
            )
            decoded_payload = torch.cat([decoded.payload, torch.empty_like(decoded.payload[:, :1])], dim=1)
            decoded_batch = PaddedBatch(decoded_payload, targets.seq_lens)
            decoder_losses, metrics = self._next_item(targets, decoded_batch, None)

        t_decoder = time.perf_counter()
        metrics.update({"decoder_" + k: v for k, v in metrics.items()})
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

    def forward(self, inputs, outputs, states):
        t0 = time.perf_counter()

        targets = self._build_horizon_windows(inputs)
        lengths = targets.seq_lens.clone()

        outputs = PaddedBatch(outputs.payload[:, :targets.shape[1]], lengths)
        det_inputs = PaddedBatch(
            {
                k: (
                    v[:, :targets.shape[1]].clone()
                    if (k in inputs.seq_names) and isinstance(v, torch.Tensor)
                    else v
                )
                for k, v in inputs.payload.items()
            },
            lengths,
            inputs.seq_names,
        )

        if self._loss_step > 1:
            lengths = (lengths - self._loss_step - 1).div(self._loss_step, rounding_mode="floor").clip(min=-1) + 1
            targets = PaddedBatch(
                {k: v[:, self._loss_step::self._loss_step] for k, v in targets.payload.items()},
                lengths,
                targets.seq_names,
            )
            outputs = PaddedBatch(outputs.payload[:, self._loss_step::self._loss_step], lengths)
            det_inputs = PaddedBatch(
                {
                    k: (
                        v[:, self._loss_step::self._loss_step].clone()
                        if (k in det_inputs.seq_names) and isinstance(v, torch.Tensor)
                        else v
                    )
                    for k, v in det_inputs.payload.items()
                },
                lengths,
                det_inputs.seq_names,
            )

        mask = targets.seq_len_mask.bool()
        flat_lengths = torch.full([mask.sum().item()], self._k + 1, device=mask.device, dtype=torch.long)
        targets = PaddedBatch(
            {k: v[mask] for k, v in targets.payload.items()},
            flat_lengths,
            targets.seq_names,
        )
        flat_outputs = outputs.payload[:, :mask.shape[1]][mask]

        losses, metrics = self._diffusion_loss(
            conditions=flat_outputs,
            targets=targets,
            det_inputs=det_inputs,
            det_mask=mask,
            det_lengths=lengths,
        )
        metrics["perf_build_windows_s"] = time.perf_counter() - t0
        metrics["perf_num_windows"] = float(len(targets))
        metrics["perf_presence_ratio"] = float(targets.payload[PRESENCE][:, 1:].float().mean().item())
        return losses, metrics

    def predict_next_k(self, outputs, states, fields=None, logits_fields_mapping=None):
        if isinstance(self._next_item, DetectionLoss):
            mask = outputs.seq_len_mask.bool()
            conditions = outputs.payload[mask]

            batch_size = len(conditions)
            x = self._noise(batch_size, device=conditions.device, dtype=outputs.payload.dtype)
            for step in range(self._generation_steps, 0, -1):
                x = self._denoising_step(x, conditions, step)

            decoded_values = self._decode_detection_sequence(x)
            bsz, seq_len = outputs.shape
            decoded_dim = decoded_values.shape[1]
            det_payload = torch.zeros(bsz, seq_len, decoded_dim, dtype=decoded_values.dtype, device=decoded_values.device)
            det_payload.masked_scatter_(mask.unsqueeze(-1), decoded_values)
            det_outputs = PaddedBatch(det_payload, outputs.seq_lens)

            if fields is None:
                fields = set(self._next_item.data_fields)
            sequences = self._next_item.predict_next_k(
                det_outputs,
                states,
                fields=fields,
                logits_fields_mapping=logits_fields_mapping,
            )

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
                    current_counts = presence_mask.sum(-1)  # (B, L).
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

        return super().predict_next_k(outputs, states, fields=fields, logits_fields_mapping=logits_fields_mapping)
