import torch
from hotpp.data import PaddedBatch
from .state import TransformerState


class PositionalEncoder(torch.nn.Module):
    def __init__(self, size, m=1, M=2000):
        super().__init__()
        if size % 2 != 0:
            raise ValueError("Need even size")
        self.size = size
        self.m = m
        self.M = M

    def forward(self, times):
        """Convert times with shape (B, L) to positional embeddings with shape (B, L, D)."""
        d = torch.linspace(0, 1, self.size // 2, device=times.device)
        denum = self.m * (5 * self.M / self.m) ** d
        args = times.unsqueeze(-1) / denum
        embeddings = torch.cat([torch.sin(args), torch.cos(args)], -1)
        return embeddings


class AttNHPTransformerLayer(torch.nn.TransformerEncoderLayer):
    def __init__(self, hidden_size, n_heads, ff_size=None, dropout=0):
        super().__init__(hidden_size, n_heads,
                         batch_first=True,
                         dim_feedforward=ff_size or hidden_size,
                         dropout=dropout)

    def forward(self, embeddings, mask, attn_mask):
        """Apply causal inference.

        Args:
            embeddings: Input embeddings with shape (B, L, D).
            mask: Input mask with ones at padding positions with shape (B, L).
            attn_mask: Attention mask with ones at disabled positions with shape (L, L).
        """
        result = super().forward(embeddings,
                                 src_mask=attn_mask,
                                 src_key_padding_mask=mask)
        return result


class AttNHPTransformer(torch.nn.Module):
    def __init__(self, input_size, hidden_size, n_heads, n_layers, pos_m=1, pos_M=2000):
        # TODO: adjust m / M.
        super().__init__()
        self.pos_encoder = PositionalEncoder(hidden_size, m=pos_m, M=pos_M)
        self.inter_token = torch.nn.Parameter(torch.randn(hidden_size))
        self.proj = torch.nn.Linear(input_size, hidden_size)
        layers = []
        for i in range(n_layers):
            layers.append(AttNHPTransformerLayer(hidden_size, n_heads))
        self.layers = torch.nn.ModuleList(layers)

    @property
    def num_layers(self):
        return len(self.layers)

    def forward(self, embeddings: PaddedBatch, times: PaddedBatch):
        """Encode input sequences with a causal mask.

        Args:
            embeddings: Input embeddings with shape (B, L, D).
            times: Input times with shape (B, L), absolute.

        Returns:
            Outputs with shape (B, L, D) and activations with shape (N, B, L, D).
        """
        x, mask = embeddings.payload, ~embeddings.seq_len_mask.bool()
        x = self.proj(x)  # (B, L, D).
        x = x + self.pos_encoder(times.payload)
        rng = torch.arange(x.shape[1], device=x.device)  # (L).
        attn_mask = rng[:, None] < rng[None, :]  # (L, L), exclude self-reference.
        results = [x]
        for layer in self.layers:
            results.append(layer(results[-1], mask, attn_mask))  # (B, L, D).
        outputs = PaddedBatch(results[-1], embeddings.seq_lens)
        states = torch.stack(results[:-1])  # (N, B, L, D).
        states = TransformerState(times.payload, states, embeddings.seq_lens)
        return outputs, states

    def decode(self, embeddings: PaddedBatch, times: PaddedBatch, history_states: TransformerState):
        """Compute activations for queries with full mask.

        Args:
            embeddings: Input embeddings with shape (B, L', D), right-aligned.
            times: Times to predict with shape (B, L').
            history_states: Historical activations with shape (N, B, L, D).
                Decoder will ignore the index.

        Returns:
            Outputs with shape (B, L', D), states with shape (N, B, L', D).
        """
        b, l = embeddings.shape
        lh = history_states.payload.shape[2]
        x = self.proj(embeddings.payload) + self.pos_encoder(times.payload)  # (B, L', D).

        mask = torch.cat([~history_states.seq_len_mask.bool(), ~embeddings.seq_len_mask.bool()], 1)  # (B, L + L').
        attn_mask = ~torch.eye(lh + l, dtype=torch.bool, device=x.device)  # (L + L', L + L').
        attn_mask[lh:, :lh] = False
        results = [x]  # (B, L', D).
        assert len(history_states) == len(self.layers)
        for layer, states in zip(self.layers, history_states.payload):
            z = torch.cat([states, results[-1]], 1)  # (B, L + L', D).
            results.append(layer(z, mask, attn_mask)[:, lh:])  # (B, L', D).
        outputs = PaddedBatch(results[-1], embeddings.seq_lens)
        states = torch.stack(results[:-1])  # (N, B, L', D).
        states = TransformerState(times.payload, states, embeddings.seq_lens)
        return outputs, states

    def interpolate(self, times: PaddedBatch, history_states: TransformerState, last_history_index=None):
        """Compute activations for queries with full mask.

        Args:
            times: Times to predict with shape (B, L', S).
            history_states: Historical activations with shape (N, B, L, D).
                Index is used to align states with attn_mask.
            last_history_index: The last history state index used for prediction with shape (L').

        Returns:
            Outputs with shape (B, L', S, D).
        """
        b, l, s = times.payload.shape
        lh = history_states.payload.shape[2]
        if last_history_index is not None:
            if not (history_states.index[1:] == history_states.index[:1]).all():
                raise NotImplementedError("Need uniform index for attention mask during interpolation.")
            last_history_index = history_states.index[0].take_along_dim(last_history_index, 0)  # (L').

        x = self.inter_token[None, None] + self.pos_encoder(times.payload.reshape(b, l * s))  # (B, L'S, D).

        mask = torch.cat([~history_states.seq_len_mask.bool(),
                          torch.zeros(b, l * s, dtype=torch.bool, device=history_states.device)],
                         1)  # (B, L + L'S).
        attn_mask = ~torch.eye(lh + l * s, dtype=torch.bool, device=x.device)  # (L + L'S, L + L'S).
        if last_history_index is not None:
            history_attn_inv_mask = last_history_index[:, None] < torch.arange(lh, device=history_states.device)  # (L', L).
            attn_mask[lh:, :lh].copy_(history_attn_inv_mask.unsqueeze(1).repeat(1, s, 1).reshape(l * s, lh))  # (L'S, L).
        else:
            attn_mask[lh:, :lh] = False
        assert len(history_states) == len(self.layers)
        for layer, states in zip(self.layers, history_states.payload):
            z = torch.cat([states, x], 1)  # (B, L + L'S, D).
            x = layer(z, mask, attn_mask)[:, lh:]  # (B, L'S, D).
        outputs = PaddedBatch(x.reshape(b, l, s, -1), times.seq_lens)  # (B, L', S, D).
        return outputs
