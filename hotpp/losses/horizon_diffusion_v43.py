"""V43: V42 + per-slot learnable queries (DeTPP-style ConditionalHead on top).

Architecture:
- Diffusion supervision: positional, no horizon mask (inherited from V42).
- Decoder: K learnable queries are concatenated with K denoised latents,
  then a shared MLP maps each (latent_k, query_k) -> (presence, time, label).
  This mirrors DeTPP's ConditionalHead but the per-slot context comes from
  diffusion latents instead of a single RNN context.
- Hungarian matching with presence head (inherited from DetectionLoss).

Combines:
- Diffusion's K-way diversity (slot k is denoised toward k-th future event)
- DeTPP's K-way query specialization (slot k has its own learnable identity)

If queries are useless -> MLP ignores them -> V43 == V42 (~pure diffusion).
If latents are weak -> MLP relies on queries -> V43 ~ pure DeTPP.
If both contribute -> V43 better than either.
"""
import torch

from hotpp.data import PaddedBatch
from .detection import DetectionLoss
from .horizon_diffusion_v42 import HorizonDiffusionLossV42


class QueryAugmentedHead(torch.nn.Module):
    """Per-slot decoder that concatenates each latent slot with a learnable
    query embedding (DeTPP-style) and applies a shared MLP.

    Input:  PaddedBatch payload (V, K, D_latent)
    Output: PaddedBatch payload (V, K, P_base)
    """

    def __init__(
        self,
        latent_dim,
        output_size,
        k,
        query_size=64,
        hidden_dims=(128, 256),
        use_batch_norm=True,
        query_init_scale=0.02,
    ):
        super().__init__()
        self.queries = torch.nn.Parameter(
            torch.randn(k, query_size) * float(query_init_scale)
        )
        self.k = k
        self.output_size = output_size

        layers = []
        last_dim = latent_dim + query_size
        if use_batch_norm:
            layers.append(torch.nn.BatchNorm1d(last_dim))
        for dim in hidden_dims or []:
            layers.append(torch.nn.Linear(last_dim, dim, bias=not use_batch_norm))
            if use_batch_norm:
                layers.append(torch.nn.BatchNorm1d(dim))
            layers.append(torch.nn.ReLU())
            last_dim = dim
        layers.append(torch.nn.Linear(last_dim, output_size))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        latents = x.payload  # (V, K, D_latent)
        v = latents.shape[0]
        if latents.shape[1] != self.k:
            raise ValueError(
                f"QueryAugmentedHead expects K={self.k} slots, got {latents.shape[1]}"
            )

        queries = self.queries.unsqueeze(0).expand(v, -1, -1)  # (V, K, D_query)
        combined = torch.cat([latents, queries], dim=-1)  # (V, K, D_latent + D_query)
        flat = combined.flatten(0, 1)  # (V*K, D_latent + D_query)
        out = self.mlp(flat)  # (V*K, P_base)
        out = out.view(v, self.k, self.output_size)

        return PaddedBatch(out, x.seq_lens)


class HorizonDiffusionLossV43(HorizonDiffusionLossV42):
    def __init__(
        self,
        *args,
        query_size=64,
        query_hidden_dims=(128, 256),
        query_use_batch_norm=True,
        query_init_scale=0.02,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if isinstance(self._next_item, DetectionLoss):
            inner_input_size = self._next_item._next_item.input_size  # P_base
            self._decoder = QueryAugmentedHead(
                latent_dim=self._embedder.output_size,
                output_size=inner_input_size,
                k=self._k,
                query_size=query_size,
                hidden_dims=tuple(query_hidden_dims) if query_hidden_dims else None,
                use_batch_norm=query_use_batch_norm,
                query_init_scale=query_init_scale,
            )
