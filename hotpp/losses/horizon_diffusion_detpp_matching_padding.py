"""V44 (DeTPP-all-comps-in-decoder) + Hungarian-aligned padding insertion.

Same idea as `HorizonDiffusionLossMatchingPadding` (which builds on V3),
but on top of V44 — where the per-slot decoder uses
``(latent_k, query_k, context)`` and so slots have **per-slot learnable
identity** via the K queries. Hungarian-matching the diffusion target
should have a stronger effect here because slots can actually
specialise.

Algorithm per training step (after warm-up):
    1. predict via diffusion (full reverse process, no_grad), using V44's
       ContextQueryAugmentedHead which needs `context=conditions`;
    2. classical Hungarian matching between K predictions and T GT events;
    3. realign the diffusion target windows: matched slots get real events,
       unmatched slots get padding (mask_embedding via presence=0);
    4. standard noise/denoise/update on the realigned targets, using
       V44's _diffusion_loss (which calls decoder with context).
"""
import time
import torch

from hotpp.data import PaddedBatch
from ..fields import PRESENCE
from .detection import DetectionLoss
from .horizon_diffusion_detpp_all_comps_in_decoder import (
    HorizonDiffusionLossDetppAllCompsInDecoder,
)


class HorizonDiffusionLossDetppMatchingPadding(HorizonDiffusionLossDetppAllCompsInDecoder):
    def __init__(self, *args, warmup_steps=0, **kwargs):
        super().__init__(*args, **kwargs)
        self._warmup_steps = int(warmup_steps)
        self.register_buffer("_align_step_count", torch.zeros(1, dtype=torch.long))

    # --- Step 1 + 2: full diffusion inference + Hungarian matching ---
    def _predict_and_match_for_alignment(self, inputs, outputs):
        """Returns matches: tensor (B, L, K) with values in {-1, 0..T-1}."""
        targets_for_lengths = self._build_horizon_windows(inputs)
        lengths = targets_for_lengths.seq_lens.clone()
        det_inputs = PaddedBatch(
            {
                k: (
                    v[:, :targets_for_lengths.shape[1]].clone()
                    if (k in inputs.seq_names) and isinstance(v, torch.Tensor)
                    else v
                )
                for k, v in inputs.payload.items()
            },
            lengths,
            inputs.seq_names,
        )

        mask = outputs.seq_len_mask.bool()
        conditions = outputs.payload[mask]                              # (V, D)
        batch_size = len(conditions)
        x = self._noise(
            batch_size,
            device=conditions.device,
            dtype=outputs.payload.dtype,
        )
        for step in range(self._generation_steps, 0, -1):
            x = self._denoising_step(x, conditions, step)
        # NB: V44 decoder requires `context=` arg.
        decoded = self._decoder(x, context=conditions)                  # (V, K, P_base)
        decoded_flat = decoded.payload.flatten(1)

        bsz, seq_len = outputs.shape
        det_payload = torch.zeros(
            bsz, seq_len, decoded_flat.shape[-1],
            dtype=decoded_flat.dtype, device=decoded_flat.device,
        )
        det_payload.masked_scatter_(mask.unsqueeze(-1), decoded_flat)

        P_base = decoded.payload.shape[-1]
        det_outputs_4d = PaddedBatch(
            det_payload.reshape(bsz, seq_len, self._k, P_base),
            outputs.seq_lens,
        )
        det_targets = self._next_item.extract_structured_windows(det_inputs)
        matches_pb, _, _ = self._next_item.match_targets(
            det_outputs_4d, det_targets,
        )
        return matches_pb.payload                                       # (B, L, K)

    # --- Step 3: realign target windows ---
    def _realign_targets_by_matches(self, targets, matches):
        """Same as V3 matching-padding: gather windows according to matches+1
        (anchor index 0 stays untouched). Unmatched slots get presence=0.
        """
        gather_idx = (matches.clamp(min=0) + 1).long()                  # (B, L, K)
        unmatched = (matches < 0)

        new_payload = {}
        for name, raw in targets.payload.items():
            gathered = torch.gather(raw, dim=2, index=gather_idx)       # (B, L, K)
            if name == PRESENCE:
                gathered = torch.where(
                    unmatched, torch.zeros_like(gathered), gathered,
                )
            new_window = torch.cat([raw[:, :, :1], gathered], dim=2)    # (B, L, K+1)
            new_payload[name] = new_window
        return PaddedBatch(new_payload, targets.seq_lens, targets.seq_names)

    # --- forward ---
    def forward(self, inputs, outputs, states):
        if not isinstance(self._next_item, DetectionLoss):
            return super().forward(inputs, outputs, states)

        t0 = time.perf_counter()
        targets = self._build_horizon_windows(inputs)

        do_realign = self.training and (self._align_step_count.item() > self._warmup_steps)
        if self.training:
            self._align_step_count += 1

        if do_realign:
            with torch.no_grad():
                matches = self._predict_and_match_for_alignment(inputs, outputs)
            targets = self._realign_targets_by_matches(targets, matches)

        # The rest mirrors V3's forward; V44._diffusion_loss handles
        # context-aware decoder internally.
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
        metrics["perf_presence_ratio"] = float(
            targets.payload[PRESENCE][:, 1:].float().mean().item()
        )
        metrics["matching_padding_active"] = float(do_realign)
        return losses, metrics
