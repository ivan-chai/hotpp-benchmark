"""V3 + Hungarian-aligned padding insertion (научник's idea).

Standard V3 (`HorizonDiffusionLossFixedPerslotDecoder`) builds diffusion target
windows POSITIONALLY:

    windows[k=0]    = current event   (anchor)
    windows[k=1]    = 1-st future event by time
    windows[k=2]    = 2-nd future event
    ...
    windows[k=T]    = T-th future event
    windows[k=T+1..K] = mask_embedding   (padding always at the tail)

This is the "mask collapse" we observed in debug:
- Slots 0..T are supervised on real events (positionally ordered);
- Slots T+1..K are ALWAYS supervised on the same shared `mask_embedding`.

Result: the denoiser learns to reproduce one constant vector for ~80% of the
slots. K-slot diversity collapses; per-slot decoder produces near-identical
predictions on the tail; Hungarian (in DetectionLoss) effectively chooses from
~T unique candidates instead of K.

This module implements the научник's proposal:

    Step 1: predict via diffusion (full reverse process, no_grad)
    Step 2: classical Hungarian matching between K predictions and T GT events
    Step 3: rearrange the diffusion targets so that:
            target[k] = real_event[matches[k]]   if matches[k] >= 0
                     = mask_embedding (presence=0)  if matches[k] == -1
    Step 4: standard noise/denoise/update on these matching-aligned targets

So padding now lands on slots that the model is currently uncertain about
(matches==-1), and slots the model "wants to fire" get supervised on real
events. The supervision becomes self-consistent (DETR-style).

Cost: each training step now does 1 extra full diffusion inference
(`generation_steps` denoising passes) under no_grad, plus a Hungarian match.

Cold start: at the very beginning of training, the denoiser is random, so
predictions are noise and matching is essentially random. To stabilise, a
`warmup_steps` parameter falls back to V3 positional padding for the first N
training steps before switching to matching-based padding.
"""
import time
import torch

from hotpp.data import PaddedBatch
from ..fields import PRESENCE
from .detection import DetectionLoss
from .horizon_diffusion_fixed_perslot_decoder import HorizonDiffusionLossFixedPerslotDecoder


class HorizonDiffusionLossMatchingPadding(HorizonDiffusionLossFixedPerslotDecoder):
    def __init__(self, *args, warmup_steps=0, **kwargs):
        super().__init__(*args, **kwargs)
        self._warmup_steps = int(warmup_steps)
        # Persistent step counter (saved with state_dict so resumed runs keep timing).
        self.register_buffer("_align_step_count", torch.zeros(1, dtype=torch.long))

    # ------------------------------------------------------------------
    # Step 1 + Step 2: full diffusion inference + Hungarian matching
    # ------------------------------------------------------------------
    def _predict_and_match_for_alignment(self, inputs, outputs):
        """Full diffusion inference (no_grad) + Hungarian matching.

        Returns:
            matches: tensor of shape (B, L, K), values in {-1, 0..T-1}.
                matches[b, l, k] = t means decoder slot k was assigned to GT
                event index t at position (b, l). t = -1 means no match.
        """
        # Build det_inputs exactly the same way as V3's forward does, so that
        # extract_structured_windows produces identical target windows.
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

        # Full diffusion inference: from random noise through `generation_steps`
        # reverse steps, conditioned on the RNN context (== `outputs`).
        mask = outputs.seq_len_mask.bool()
        conditions = outputs.payload[mask]                        # (V, D)
        batch_size = len(conditions)
        x = self._noise(
            batch_size,
            device=conditions.device,
            dtype=outputs.payload.dtype,
        )
        for step in range(self._generation_steps, 0, -1):
            x = self._denoising_step(x, conditions, step)
        decoded = self._decoder(x)                                # (V, K, P_base)
        decoded_flat = decoded.payload.flatten(1)                 # (V, K * P_base)

        # Scatter back to (B, L, K * P_base) for DetectionLoss API.
        bsz, seq_len = outputs.shape
        det_payload = torch.zeros(
            bsz, seq_len, decoded_flat.shape[-1],
            dtype=decoded_flat.dtype, device=decoded_flat.device,
        )
        det_payload.masked_scatter_(mask.unsqueeze(-1), decoded_flat)

        # match_targets() expects (B, L, K, P_base).
        P_base = decoded.payload.shape[-1]
        det_outputs_4d = PaddedBatch(
            det_payload.reshape(bsz, seq_len, self._k, P_base),
            outputs.seq_lens,
        )

        # GT target windows from DetectionLoss.
        det_targets = self._next_item.extract_structured_windows(det_inputs)

        # Run Hungarian.
        matches_pb, _, _ = self._next_item.match_targets(
            det_outputs_4d, det_targets,
        )
        return matches_pb.payload                                 # (B, L, K)

    # ------------------------------------------------------------------
    # Step 3: realign target windows according to matches
    # ------------------------------------------------------------------
    def _realign_targets_by_matches(self, targets, matches):
        """Rearrange diffusion target windows according to Hungarian matching.

        Args:
            targets: PaddedBatch with payload {field: (B, L, K+1)}; the K+1
                slots are positionally ordered (slot 0 = current/anchor,
                slots 1..K = K future events; out-of-horizon slots already
                have presence=0).
            matches: tensor (B, L, K) with values in {-1, 0..T-1}.
                NB: matches[b, l, k] = t corresponds to the (t+1)-th index in
                the K+1 windows (because index 0 is the anchor). The match_targets
                API treats GT-event index 0 as the FIRST FUTURE event.

        Returns:
            PaddedBatch with realigned slots:
              new_window[:, :, 0]  = original anchor (unchanged);
              new_window[:, :, k]  = original window[matches[k-1] + 1] if matched;
                                   = padding (presence=0) if matches[k-1] == -1.
        """
        # gather_idx[b, l, k] = which raw slot to place at position k+1.
        # Use matches+1, and clamp -1 → 0 (we'll override unmatched via PRESENCE).
        gather_idx = (matches.clamp(min=0) + 1).long()             # (B, L, K)
        unmatched = (matches < 0)                                  # (B, L, K)

        new_payload = {}
        for name, raw in targets.payload.items():
            # raw: (B, L, K+1)
            gathered = torch.gather(raw, dim=2, index=gather_idx)  # (B, L, K)
            if name == PRESENCE:
                # Force presence=0 for unmatched slots → mask_embedding in
                # _embed_targets() of the parent V3.
                gathered = torch.where(
                    unmatched, torch.zeros_like(gathered), gathered,
                )
            # Anchor stays untouched (raw[:, :, 0:1]).
            new_window = torch.cat([raw[:, :, :1], gathered], dim=2)
            new_payload[name] = new_window

        return PaddedBatch(new_payload, targets.seq_lens, targets.seq_names)

    # ------------------------------------------------------------------
    # forward()
    # ------------------------------------------------------------------
    def forward(self, inputs, outputs, states):
        # Fallback for non-DetectionLoss branches (e.g., if NextItemLoss is used).
        if not isinstance(self._next_item, DetectionLoss):
            return super().forward(inputs, outputs, states)

        t0 = time.perf_counter()

        # Build positional targets (V3 style).
        targets = self._build_horizon_windows(inputs)

        # During training, after warm-up: predict + match + realign.
        do_realign = self.training and (self._align_step_count.item() > self._warmup_steps)
        if self.training:
            self._align_step_count += 1

        if do_realign:
            with torch.no_grad():
                matches = self._predict_and_match_for_alignment(inputs, outputs)
            targets = self._realign_targets_by_matches(targets, matches)

        # The rest of the function mirrors V3's forward, with the realigned
        # `targets` flowing through the standard noise/denoise/update step.
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
        # Diagnostic: 1.0 if matching-padding active this step, 0.0 if warmup.
        metrics["matching_padding_active"] = float(do_realign)
        return losses, metrics
