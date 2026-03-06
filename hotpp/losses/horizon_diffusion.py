import random
import time
import torch

from hotpp.data import PaddedBatch
from ..fields import PRESENCE
from .common import ScaleGradient
from .detection import DetectionLoss
from .diffusion import DiffusionLoss


class HorizonDiffusionLoss(DiffusionLoss):
    def __init__(self, *args, padding_prob=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self._padding_prob = padding_prob
        self._diffusion_fields = self.fields
        if hasattr(self._next_item, "data_fields"):
            self._diffusion_fields = self._next_item.data_fields

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

    def _diffusion_loss(self, conditions, targets, det_inputs=None, det_mask=None, det_lengths=None):
        losses = {}
        metrics = {}
        b = len(targets)
        t0 = time.perf_counter()

        embed_inputs = PaddedBatch(
            {k: targets.payload[k] for k in self._diffusion_fields},
            targets.seq_lens,
            set(self._diffusion_fields),
        )
        embeddings = self._embedder(self._compute_time_deltas(embed_inputs))  # (B, K + 1, D)
        embeddings = PaddedBatch(embeddings.payload[:, 1:], (embeddings.seq_lens - 1).clip(min=0))  # (B, K, D)

        presence_full = targets.payload[PRESENCE].clone()  # (B, K + 1)

        targets = targets.clone()
        targets.payload[PRESENCE] = presence_full

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

        decoded = self._decoder(
            PaddedBatch(
                reconstructed.payload.detach() if self._detach_decoder else reconstructed.payload,
                reconstructed.seq_lens,
            )
        )

        # Run decoder loss
        if isinstance(self._next_item, DetectionLoss):
            if (det_inputs is None) or (det_mask is None) or (det_lengths is None):
                raise RuntimeError("DetectionLoss branch requires det_inputs, det_mask, and det_lengths.")
            b_det, l_det = det_mask.shape
            det_values = decoded.payload[:, 0]  # (V, P_det).
            p_det = det_values.shape[1]
            det_payload = torch.zeros(b_det, l_det, p_det, dtype=decoded.payload.dtype, device=decoded.device)
            det_payload.masked_scatter_(det_mask.unsqueeze(-1), det_values)
            det_outputs = PaddedBatch(det_payload, det_lengths)
            decoder_losses, metrics = self._next_item(det_inputs, det_outputs, None)
        else:
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
        targets = self._build_horizon_windows(inputs)  # (B, L, K + 1)
        lengths = targets.seq_lens
        l = targets.shape[1]
        outputs = PaddedBatch(outputs.payload[:, :l], lengths)
        det_inputs = PaddedBatch(
            {k: (v[:, :l] if k in inputs.seq_names else v) for k, v in inputs.payload.items()},
            lengths,
            inputs.seq_names,
        )

        if self._loss_step > 1:
            lengths = (lengths - self._loss_step - 1).div(self._loss_step, rounding_mode="floor").clip(min=-1) + 1
            payload = {k: v[:, self._loss_step::self._loss_step] for k, v in targets.payload.items()}
            targets = PaddedBatch(payload, lengths, targets.seq_names)
            outputs = PaddedBatch(outputs.payload[:, self._loss_step::self._loss_step], lengths)
            det_inputs = PaddedBatch(
                {
                    k: (v[:, self._loss_step::self._loss_step] if k in det_inputs.seq_names else v)
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
        metrics["perf_presence_ratio"] = float(targets.payload[PRESENCE][:, 1:].float().mean().item())
        return losses, metrics

    def predict_next_k(self, outputs, states, fields=None, logits_fields_mapping=None):
        if fields is None:
            if hasattr(self._next_item, "data_fields"):
                fields = set(self._next_item.data_fields)
            else:
                fields = set(self.fields)
        return super().predict_next_k(outputs, states, fields=fields, logits_fields_mapping=logits_fields_mapping)
