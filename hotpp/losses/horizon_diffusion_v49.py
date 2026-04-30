"""V49: V48 + per-slot learnable mask embedding (K, D).

Audit of V48 showed that with diffusion target ``delta in [10, 30)``
~94% of K=48 slots in any window get ``presence=0`` and are therefore
mapped to a SINGLE shared mask vector. The denoiser then has to
reconstruct the same vector for ~45 slots, which collapses those
latents and erases per-slot identity. Diffusion still helps V48 only
through its gradient flow into the RNN backbone (multi-task
regularization), not because the K latents themselves are diverse.

V49 keeps the V48 supervision (long-range target) but replaces the
shared mask vector with K learnable per-slot mask embeddings. Each
absent slot ``k`` is supervised against its own ``mask_k``, so the
denoiser is free to keep K diverse latents even when most of them
are "absent" — mirroring V40's idea on top of V48's better target.
"""
import torch

from .horizon_diffusion_v48 import HorizonDiffusionLossV48


class HorizonDiffusionLossV49(HorizonDiffusionLossV48):
    def __init__(self, *args, mask_embedding_init_scale=0.02, **kwargs):
        super().__init__(*args, **kwargs)
        d = self._embedder.output_size
        self._mask_embedding = torch.nn.Parameter(
            torch.randn(self._k, d) * float(mask_embedding_init_scale)
        )

    def _embed_targets(self, targets):
        from hotpp.data import PaddedBatch
        from ..fields import PRESENCE

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

        presence = targets.payload[PRESENCE][:, 1:].bool()  # (V, K).
        # (K, D) -> (V, K, D) by broadcast.
        mask = self._mask_embedding.unsqueeze(0).expand(
            embeddings.payload.shape[0], -1, -1
        )
        embeddings.payload = torch.where(
            presence.unsqueeze(-1),
            embeddings.payload,
            mask,
        )
        return embeddings
