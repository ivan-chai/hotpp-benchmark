"""V50: V48 + three typed mask embeddings (near / far / pad).

V48 supervises diffusion on events at long range (delta in
[diffusion_target_min_delta, diffusion_target_max_delta)). Slots that do
not have such an event fall into THREE distinct semantic categories,
which V48 collapses into a single shared mask vector:

  * near : valid_index=True, delta in [0, diffusion_target_min_delta)
           — there IS an event in the window but it is too soon to be a
           long-range target.
  * far  : valid_index=True, delta >= diffusion_target_max_delta
           — there IS an event but it is past the target horizon.
  * pad  : valid_index=False
           — out of the input sequence (true padding).

V50 gives each category its own learnable embedding of shape (D,).
The slot type is carried as an extra ``_slot_type`` field through
``_build_horizon_windows`` and used in ``_embed_targets`` to look up
the right mask. This lets the denoiser distinguish "no event yet" from
"event already happened" from "out of sequence", and gives diffusion
real semantic information instead of one ambiguous mask.

Slot 0 is the anchor (current event), set to type=real and is later
sliced away by ``[:, 1:]``, so its type does not matter.
"""
import torch

from hotpp.data import PaddedBatch
from ..fields import PRESENCE
from .horizon_diffusion_v48 import HorizonDiffusionLossV48


SLOT_TYPE_FIELD = "_slot_type"
SLOT_REAL = 0
SLOT_NEAR = 1
SLOT_FAR = 2
SLOT_PAD = 3


class HorizonDiffusionLossV50(HorizonDiffusionLossV48):
    def __init__(self, *args, mask_embedding_init_scale=0.02, **kwargs):
        super().__init__(*args, **kwargs)
        d = self._embedder.output_size
        scale = float(mask_embedding_init_scale)
        # Three typed masks; type=real reuses the actual event embedding,
        # so we only need 3 entries (indices 1..3).
        self._mask_near = torch.nn.Parameter(torch.randn(d) * scale)
        self._mask_far = torch.nn.Parameter(torch.randn(d) * scale)
        self._mask_pad = torch.nn.Parameter(torch.randn(d) * scale)

    def _build_horizon_windows(self, inputs):
        """V48 windows + an extra ``_slot_type`` payload field of the same
        shape as PRESENCE, encoding {real, near, far, pad} per slot."""
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
        long_range_valid = (delta >= self._diff_target_min) & (delta < self._diff_target_max)
        long_range_valid[:, :, 0] = True  # anchor
        valid = valid_index & long_range_valid

        # Slot type:
        #   real if valid_index AND in long-range window
        #   near if valid_index AND delta < target_min
        #   far  if valid_index AND delta >= target_max
        #   pad  otherwise (valid_index = False)
        slot_type = torch.full_like(valid_index, SLOT_PAD, dtype=torch.long)
        slot_type = torch.where(
            valid_index & (delta < self._diff_target_min),
            torch.full_like(slot_type, SLOT_NEAR),
            slot_type,
        )
        slot_type = torch.where(
            valid_index & (delta >= self._diff_target_max),
            torch.full_like(slot_type, SLOT_FAR),
            slot_type,
        )
        slot_type = torch.where(
            valid,  # real long-range event
            torch.full_like(slot_type, SLOT_REAL),
            slot_type,
        )
        # Anchor (slot 0) is always real (will be sliced away by [:, 1:]).
        slot_type[:, :, 0] = SLOT_REAL

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
        windows[SLOT_TYPE_FIELD] = slot_type
        return PaddedBatch(
            windows,
            inputs.seq_lens,
            set(self._diffusion_fields) | {PRESENCE, SLOT_TYPE_FIELD},
        )

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

        slot_type = targets.payload[SLOT_TYPE_FIELD][:, 1:]  # (V, K).

        # Stack masks at type indices: (4, D) — index 0 (real) unused for replacement
        # since presence-real slots keep the actual event embedding.
        d = self._mask_near.shape[0]
        # Use real embedding placeholder = zeros (won't be selected when type==real).
        mask_real_placeholder = torch.zeros(
            d, dtype=self._mask_near.dtype, device=self._mask_near.device
        )
        mask_table = torch.stack(
            [mask_real_placeholder, self._mask_near, self._mask_far, self._mask_pad],
            dim=0,
        )  # (4, D).
        # Look up per-slot mask: (V, K, D).
        per_slot_mask = mask_table[slot_type]

        is_real = slot_type == SLOT_REAL  # (V, K).
        embeddings.payload = torch.where(
            is_real.unsqueeze(-1),
            embeddings.payload,
            per_slot_mask,
        )
        return embeddings
