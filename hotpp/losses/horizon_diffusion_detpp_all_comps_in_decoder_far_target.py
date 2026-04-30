"""V48: V44 + diffusion target = events at LONG-RANGE future (delta in [h_min, h_max]).

Идея: денойзер учится представлять события за пределами near-horizon
[0, 10] — например, в [10, 30]. Decoder ВСЁ ЕЩЁ предсказывает события
near-horizon (DetectionLoss matches against [0, 10]). Latent_k несёт
ортогональную информацию (long-range pattern) которая может помочь
near-future prediction через decoder.

Если far-future events коррелируют с near-future (например, частые типы
событий продолжают появляться) — диффузия даст decoder'у дополнительный
сигнал о тренде.
"""
import torch

from hotpp.data import PaddedBatch
from ..fields import PRESENCE
from .horizon_diffusion_detpp_all_comps_in_decoder import HorizonDiffusionLossDetppAllCompsInDecoder


class HorizonDiffusionLossDetppAllCompsInDecoderFarTarget(HorizonDiffusionLossDetppAllCompsInDecoder):
    def __init__(
        self,
        *args,
        diffusion_target_min_delta=10.0,
        diffusion_target_max_delta=30.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._diff_target_min = float(diffusion_target_min_delta)
        self._diff_target_max = float(diffusion_target_max_delta)

    def _build_horizon_windows(self, inputs):
        """Build K+1 target window: target events at delta in [min, max] (long-range)."""
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
        delta = rolled_timestamps - current_timestamps.unsqueeze(2)
        # Long-range filter: keep events whose delta falls in [min, max].
        # i=0 is current event itself (delta=0); we keep it as anchor.
        long_range_valid = (delta >= self._diff_target_min) & (delta < self._diff_target_max)
        long_range_valid[:, :, 0] = True  # keep current as i=0 anchor
        valid = valid_index & long_range_valid

        windows = {}
        for name in self._diffusion_fields:
            values = inputs.payload[name]
            window = torch.stack([values.roll(-i, 1) for i in range(k1)], dim=2)
            if name == self._timestamps_field:
                pad_value = current_timestamps.unsqueeze(2) + self._diff_target_max + 1
            else:
                pad_value = torch.zeros_like(window)
            window = torch.where(valid, window, pad_value)
            windows[name] = window

        windows[PRESENCE] = valid.long()
        return PaddedBatch(windows, inputs.seq_lens, set(self._diffusion_fields) | {PRESENCE})
