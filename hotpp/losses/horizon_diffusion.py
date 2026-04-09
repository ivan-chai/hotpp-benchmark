import random
import time
import torch

from hotpp.data import PaddedBatch
from ..fields import PRESENCE, PRESENCE_PROB
from .common import ScaleGradient
from .detection import DetectionLoss
from .diffusion import DiffusionLoss


class HorizonDiffusionLoss(DiffusionLoss):
    def __init__(
        self,
        *args,
        padding_prob=0.0,
        use_detection_calibration_threshold=True,
        presence_threshold_bias=0.0,
        **kwargs
    ):
        decoder_partial = kwargs.get("decoder_partial", None)
        super().__init__(*args, **kwargs)
        self._padding_prob = padding_prob
        self._use_detection_calibration_threshold = use_detection_calibration_threshold
        self._presence_threshold_bias = float(presence_threshold_bias)
        self._mask_embedding = torch.nn.Parameter(torch.zeros(self._embedder.output_size))
        self._diffusion_fields = self.fields
        if hasattr(self._next_item, "data_fields"):
            self._diffusion_fields = self._next_item.data_fields
        if isinstance(self._next_item, DetectionLoss):
            if decoder_partial is None:
                raise ValueError("Need decoder_partial for DetectionLoss branch.")
            # For DetectionLoss we decode the whole denoised sequence jointly:
            # (K, D) -> (K * P), which is closer to DeTPP semantics than per-token decoding.
            self._decoder = decoder_partial(self._k * self._embedder.output_size, self._next_item.input_size)

    def _build_horizon_windows(self, inputs):
        b, l = inputs.shape
        k1 = self._k + 1
        device = inputs.device

        base = torch.arange(l, device=device)[None, :, None]  # (1, L, 1)
        offs = torch.arange(k1, device=device)[None, None, :]  # (1, 1, K + 1)
        valid = base + offs < inputs.seq_lens[:, None, None]  # (B, L, K + 1)

        windows = {}
        for name in self._diffusion_fields:
            x = inputs.payload[name]  # (B, L)
            parts = [x.roll(-i, 1) for i in range(k1)]
            w = torch.stack(parts, 2)  # (B, L, K + 1)
            if name == self._timestamps_field:
                max_ts = x[inputs.seq_len_mask].max().item() if inputs.seq_len_mask.any() else 0
                pad_value = max_ts + (self._max_time_delta or 1) + 1
            else:
                pad_value = 0
            w = w.masked_fill(~valid, pad_value)
            windows[name] = w

        windows[PRESENCE] = valid.long()
        lengths = inputs.seq_lens.clone()
        seq_names = set(inputs.seq_names) | {PRESENCE}
        return PaddedBatch(windows, lengths, seq_names)

    def _embed_targets(self, targets):
        embed_inputs = PaddedBatch(
            {k: targets.payload[k] for k in self._diffusion_fields},
            targets.seq_lens,
            set(self._diffusion_fields),
        )
        embeddings = self._embedder(self._compute_time_deltas(embed_inputs))  # (B, K + 1, D).
        embeddings = PaddedBatch(embeddings.payload[:, 1:], (embeddings.seq_lens - 1).clip(min=0))  # (B, K, D).
        if PRESENCE in targets.payload:
            presence = targets.payload[PRESENCE][:, 1:].bool()  # (B, K).
            embeddings.payload = torch.where(
                presence.unsqueeze(-1),
                embeddings.payload,
                self._mask_embedding.view(1, 1, -1),
            )
        return embeddings

    def _decode_detection_sequence(self, sequence):
        flattened = sequence.payload.flatten(1)  # (V, K * D).
        lengths = torch.ones(len(flattened), dtype=torch.long, device=sequence.device)
        decoded = self._decoder(PaddedBatch(flattened.unsqueeze(1), lengths))  # (V, 1, K * P).
        return decoded.payload[:, 0]

    def _diffusion_loss(self, conditions, targets, det_inputs=None, det_mask=None, det_lengths=None):
        losses = {}
        metrics = {}
        b = len(targets)
        t0 = time.perf_counter()

        embeddings = self._embed_targets(targets)

        steps = torch.randint(1, self._generation_steps + 1, [b], device=embeddings.device)
        steps[random.randint(0, b - 1)] = 1
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

        # Run decoder loss
        if isinstance(self._next_item, DetectionLoss):
            if (det_inputs is None) or (det_mask is None) or (det_lengths is None):
                raise RuntimeError("DetectionLoss branch requires det_inputs, det_mask, and det_lengths.")
            b_det, l_det = det_mask.shape
            det_values = self._decode_detection_sequence(
                PaddedBatch(
                    reconstructed.payload.detach() if self._detach_decoder else reconstructed.payload,
                    reconstructed.seq_lens,
                )
            )
            p_det = det_values.shape[1]
            det_payload = torch.zeros(b_det, l_det, p_det, dtype=det_values.dtype, device=det_values.device)
            det_payload.masked_scatter_(det_mask.unsqueeze(-1), det_values)
            det_outputs = PaddedBatch(det_payload, det_lengths)
            decoder_losses, metrics = self._next_item(det_inputs, det_outputs, None)
        else:
            decoded = self._decoder(
                PaddedBatch(
                    reconstructed.payload.detach() if self._detach_decoder else reconstructed.payload,
                    reconstructed.seq_lens,
                )
            )
            decoded_payload = torch.cat([decoded.payload, torch.empty_like(decoded.payload[:, :1])], 1)
            decoded_batch = PaddedBatch(decoded_payload, targets.seq_lens)
            decoder_losses, metrics = self._next_item(targets, decoded_batch, None)

        t_det = time.perf_counter()
        metrics.update({"decoder_" + k: v for k, v in metrics.items()})
        losses.update(
            {"decoder_" + k: ScaleGradient.apply(v, self._decoder_loss_weight) for k, v in decoder_losses.items()}
        )

        losses["embedder_regularizer"] = ScaleGradient.apply(
            self._alpha_prods[self._generation_steps] * embeddings.payload.square().mean(),
            self._embedder_regularizer,
        )
        metrics["perf_embed_denoise_s"] = t_denoise - t0
        metrics["perf_detection_s"] = t_det - t_denoise
        metrics["perf_total_loss_s"] = t_det - t0
        return losses, metrics

    def forward(self, inputs, outputs, states):
        t0 = time.perf_counter()
        targets = self._build_horizon_windows(inputs)  # (B, L, K + 1) with presence-aware padding.
        lengths = targets.seq_lens

        outputs = PaddedBatch(outputs.payload[:, :targets.shape[1]], lengths)
        det_inputs = PaddedBatch(
            {
                k: (
                    v[:, :targets.shape[1]].clone() if k in inputs.seq_names and isinstance(v, torch.Tensor)
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
                        if k in det_inputs.seq_names and isinstance(v, torch.Tensor)
                        else v
                    )
                    for k, v in det_inputs.payload.items()
                },
                lengths,
                det_inputs.seq_names,
            )

        mask = targets.seq_len_mask.bool()
        v_lengths = torch.full([mask.sum().item()], self._k + 1, device=mask.device)
        targets = PaddedBatch({k: v[mask] for k, v in targets.payload.items()}, v_lengths, targets.seq_names)
        outputs = outputs.payload[:, :mask.shape[1]][mask]

        losses, metrics = self._diffusion_loss(
            conditions=outputs,
            targets=targets,
            det_inputs=det_inputs,
            det_mask=mask,
            det_lengths=lengths,
        )
        metrics["perf_build_windows_s"] = time.perf_counter() - t0
        metrics["perf_num_windows"] = float(len(targets))
        if PRESENCE in targets.payload:
            metrics["perf_presence_ratio"] = float(targets.payload[PRESENCE][:, 1:].float().mean().item())
        return losses, metrics

    def predict_next_k(self, outputs, states, fields=None, logits_fields_mapping=None):
        if isinstance(self._next_item, DetectionLoss):
            mask = outputs.seq_len_mask.bool()  # (B, L).
            conditions = outputs.payload[mask]  # (V, P).

            # Generate and denoise latent trajectories.
            b = len(conditions)
            x = self._noise(b, device=conditions.device, dtype=outputs.payload.dtype)  # (V, K, D).
            for i in range(self._generation_steps, 0, -1):
                x = self._denoising_step(x, conditions, i)

            det_values = self._decode_detection_sequence(x)  # (V, K * P_base).

            # Scatter back to sequence layout expected by DetectionLoss.
            bsz, seq_len = outputs.shape
            p_det = det_values.shape[1]
            det_payload = torch.zeros(bsz, seq_len, p_det, dtype=det_values.dtype, device=det_values.device)
            det_payload.masked_scatter_(mask.unsqueeze(-1), det_values)
            det_outputs = PaddedBatch(det_payload, outputs.seq_lens)  # (B, L, K * P_base).

            if fields is None:
                fields = set(self._next_item.data_fields)
            sequences = self._next_item.predict_next_k(
                det_outputs,
                states,
                fields=fields,
                logits_fields_mapping=logits_fields_mapping
            )
            presence_logits_field = PRESENCE + "_logits"
            if self._use_detection_calibration_threshold and (self._presence_threshold_bias != 0):
                if presence_logits_field in sequences.payload:
                    presence_logits = sequences.payload[presence_logits_field].squeeze(-1)
                    thresholds = self._next_item._matching_thresholds + self._presence_threshold_bias
                    sequences.payload[PRESENCE] = presence_logits > thresholds
                    sequences.payload[PRESENCE_PROB] = torch.sigmoid(presence_logits - thresholds)
            if not self._use_detection_calibration_threshold:
                if presence_logits_field in sequences.payload:
                    presence_logits = sequences.payload[presence_logits_field].squeeze(-1)
                    sequences.payload[PRESENCE] = presence_logits > 0
                    sequences.payload[PRESENCE_PROB] = torch.sigmoid(presence_logits)
            # Keep metric evaluation positions identical to other losses.
            # Mismatched sequence lengths here can silently change target statistics.
            if sequences.shape[1] != outputs.shape[1]:
                sequences = PaddedBatch(
                    {
                        k: (v[:, :outputs.shape[1]] if k in sequences.seq_names else v)
                        for k, v in sequences.payload.items()
                    },
                    torch.minimum(sequences.seq_lens, outputs.seq_lens),
                    sequences.seq_names
                )
            assert sequences.shape[1] == outputs.shape[1], "predict_next_k length mismatch"
            assert (sequences.seq_lens == outputs.seq_lens).all(), "predict_next_k seq_lens mismatch"
            return sequences

        if fields is None:
            if hasattr(self._next_item, "data_fields"):
                fields = set(self._next_item.data_fields)
            else:
                fields = set(self.fields)
        return super().predict_next_k(outputs, states, fields=fields, logits_fields_mapping=logits_fields_mapping)
