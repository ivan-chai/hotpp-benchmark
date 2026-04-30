"""V42: pure diffusion supervision + DetectionLoss matching.

Difference from V3:
- V3 builds a horizon-aware target window: events past horizon are replaced
  with a fixed mask_embedding. Denoiser is trained to reconstruct masks.
- V42 does NOT mask. The denoiser is trained EXACTLY like pure
  diffusion_gru (positional reconstruction of next K events). Horizon
  filtering happens later, at DetectionLoss matching, via cost penalty
  on predictions whose timestamp is past horizon.

This is the clean way to combine the two ideas: take pure diffusion's
working supervision as-is, and bolt DetectionLoss with presence head
and Hungarian matching on top, exactly like DeTPP.
"""
import torch

from hotpp.data import PaddedBatch
from ..fields import PRESENCE
from .horizon_diffusion_v3 import HorizonDiffusionLossV3


class HorizonDiffusionLossV42(HorizonDiffusionLossV3):
    def _build_horizon_windows(self, inputs):
        """Build K+1 target window per position WITHOUT horizon mask.

        ``valid`` only enforces the sequence-length boundary (slot i is
        valid if position p+i is inside the original sequence). Events
        past the horizon stay marked as present, so the denoiser sees
        their real embeddings — same supervision as pure diffusion_gru.
        """
        b, l = inputs.shape
        k1 = self._k + 1
        device = inputs.device

        base = torch.arange(l, device=device)[None, :, None]
        offs = torch.arange(k1, device=device)[None, None, :]
        valid = base + offs < inputs.seq_lens[:, None, None]  # (B, L, K+1)
        # NOTE: no horizon_valid filter here — events past horizon stay valid.

        current_timestamps = inputs.payload[self._timestamps_field]

        windows = {}
        for name in self._diffusion_fields:
            values = inputs.payload[name]
            window = torch.stack([values.roll(-i, 1) for i in range(k1)], dim=2)
            if name == self._timestamps_field:
                # For slots past seq_lens (no real event), use a far-future
                # placeholder so DetectionLoss sees them as out-of-horizon.
                pad_value = current_timestamps.unsqueeze(2) + self._horizon + 1
            else:
                pad_value = torch.zeros_like(window)
            window = torch.where(valid, window, pad_value)
            windows[name] = window

        # Presence indicates "real event" (within sequence) vs "padding".
        # Out-of-horizon events stay presence=1 so denoiser learns them.
        windows[PRESENCE] = valid.long()
        return PaddedBatch(windows, inputs.seq_lens, set(self._diffusion_fields) | {PRESENCE})
