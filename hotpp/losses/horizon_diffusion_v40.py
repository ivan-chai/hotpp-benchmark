"""V40: V3 architecture with per-slot learnable mask embeddings.

Forensic finding (forensic_compare.py + V3 trace on stackoverflow):
- For each position, a horizon-K+1 window has on average ~11 valid future
  events; the remaining ~37 slots are "absent" (past horizon).
- V3 replaces all absent embeddings with a SINGLE shared learnable
  ``_mask_embedding`` vector.
- Consequence: denoiser sees ONE identical reconstruction target across
  37 of the 48 K-slots. After denoising, those 37 latents collapse to
  near-identical values, and the per-slot decoder produces ~37 identical
  decoded predictions per position. Hungarian matching effectively has
  only ~11 unique candidates instead of K=48.

In contrast, pure DeTPP learns K=48 distinct query vectors that produce
K diverse predictions out of the box, so Hungarian gets full K-way choice.

V40 fix: replace the single shared mask vector with K learnable per-slot
mask embeddings. Each absent slot is supervised to reconstruct its own
slot-specific mask, so K denoised latents stay diverse even when most are
"absent". The decoder then has K distinct inputs to project from, mirroring
DeTPP's K-query diversity.
"""
import random
import time

import torch

from hotpp.data import PaddedBatch
from ..fields import PRESENCE, PRESENCE_PROB
from .common import ScaleGradient
from .detection import DetectionLoss
from .horizon_diffusion_v3 import HorizonDiffusionLossV3


class HorizonDiffusionLossV40(HorizonDiffusionLossV3):
    def __init__(self, *args, mask_embedding_init_scale=0.02, **kwargs):
        super().__init__(*args, **kwargs)
        # Replace the single _mask_embedding (D,) with per-slot (K, D).
        d = self._embedder.output_size
        # Init with small random values, not zeros, so each slot starts distinct.
        self._mask_embedding = torch.nn.Parameter(
            torch.randn(self._k, d) * float(mask_embedding_init_scale)
        )

    def _embed_targets(self, targets):
        """Same as V3 but per-slot mask: ``_mask_embedding`` has shape (K, D)."""
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

        presence = targets.payload[PRESENCE][:, 1:].bool()  # (V, K)
        # _mask_embedding shape: (K, D). Broadcast to (1, K, D) and pick per slot.
        mask = self._mask_embedding.unsqueeze(0).expand(embeddings.payload.shape[0], -1, -1)
        embeddings.payload = torch.where(
            presence.unsqueeze(-1),
            embeddings.payload,
            mask,
        )
        return embeddings
