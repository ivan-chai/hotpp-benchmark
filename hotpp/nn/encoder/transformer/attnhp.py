"""An implementation of the fast continuous transformer for TPP.

1. Self attention depends only on history (excluding current token), to make thinning more efficient.
2. The transformer works in two modes: simple forward and attention to history.
2.a. Simple forward mode employs causal self-attention (excluding reference to the current token).
2.b. Attention to history computes keys and values only for historical events.
"""
import math
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


class AttNHPTransformerLayer(torch.nn.Module):
    """
    Args:
        dim_value: The size of the value
        dim_feedforward: Internal FF size. Use 0 to disable FF block.
    """
    def __init__(self, d_model, nhead, dim_feedforward=None, dim_value=None,
                 dropout=0.1, ninf=-1e6):
        if dim_feedforward is None:
            dim_feedforward = d_model
        if dim_value is None:
            dim_value = d_model
        super().__init__()
        self.nhead = nhead
        self.dim_value = dim_value
        self.ninf = ninf

        self.in_proj = torch.nn.Linear(d_model, 2 * nhead * d_model + nhead * dim_value)
        if nhead * dim_value != d_model:
            self.out_proj = torch.nn.Linear(nhead * dim_value, d_model)
        else:
            self.out_proj = torch.nn.Identity()
        self.dropout1 = torch.nn.Dropout(dropout)
        self.norm1 = torch.nn.LayerNorm(d_model)

        self.use_ff = dim_feedforward > 0
        if self.use_ff:
            self.linear1 = torch.nn.Linear(d_model, dim_feedforward)
            self.activation = torch.nn.ReLU()
            self.dropout = torch.nn.Dropout(dropout)
            self.linear2 = torch.nn.Linear(dim_feedforward, d_model)
            self.dropout2 = torch.nn.Dropout(dropout)
            self.norm2 = torch.nn.LayerNorm(d_model)

    def forward(self, src, mask=None, attn_mask=None, history=None):
        """Apply self-attention layer.

        Args:
            src: Input embeddings with shape (B, L, D).
            mask: Input mask with ones at padding positions with shape (B, L)
                or history mask with shape (B, H) if history is provided.
            attn_mask: Attention mask with ones at disabled positions with shape (L, L)
                or history cross-attention mask with shape (L, H) if history is provided.
            history: Historical embeddings with shape (B, H, D).
        """
        x = src
        x = self.norm1(x + self._sa_block(x, mask, attn_mask, history))
        if self.use_ff:
            x = self.norm2(x + self._ff_block(x))
        return x

    def _sa_block(self, embeddings, mask=None, attn_mask=None, history=None):
        b, l, d = embeddings.shape
        if history is None:
            lh = l
            proj = self.in_proj(embeddings)  # (B, L, 2 * ND + NV).
            q = proj[..., :self.nhead * d].reshape(b, l, self.nhead, -1)  # (B, L, N, D).
            kv = proj[..., self.nhead * d:]  # (B, H, ND + NV).
        else:
            bh, lh, dh = history.shape
            if (bh != b) or (dh != d):
                raise ValueError("Embeddings and history shape mismatch.")
            q = torch.nn.functional.linear(embeddings,
                                           self.in_proj.weight[:self.nhead * d],
                                           self.in_proj.bias[:self.nhead * d]).reshape(b, l, self.nhead, -1)  # (B, L, N, D).
            kv = torch.nn.functional.linear(history,
                                            self.in_proj.weight[self.nhead * d:],
                                            self.in_proj.bias[self.nhead * d:])  # (B, H, ND + NV).
        k = kv[..., :self.nhead * d].reshape(b, lh, self.nhead, -1)  # (B, H, N, D).
        v = kv[..., self.nhead * d:].reshape(b, lh, self.nhead, -1)  # (B, H, N, V).
        outputs = []
        for i in range(self.nhead):
            weights = torch.bmm(
                q[:, :, i],  # (B, L, D).
                k[:, :, i].transpose(1, 2) / math.sqrt(d)  # (B, D, H).
            )  # (B, L, H).
            if mask is not None:
                weights.masked_fill_(mask.unsqueeze(1), self.ninf)
            if attn_mask is not None:
                weights.masked_fill_(attn_mask[None], self.ninf)
            weights = torch.nn.functional.softmax(weights, -1)  # (B, L, H).
            sa = torch.bmm(weights, v[:, :, i])  # (B, L, D).
            outputs.append(sa)
        result = self.out_proj(torch.cat(outputs, -1))  # (B, L, D).

        # Replace output with zero if history is empty.
        invalid = mask[:, None, :] if mask is not None else None  # (B, L, H).
        if attn_mask is not None:
            invalid = attn_mask[None] if invalid is None else torch.logical_or(invalid, attn_mask[None])
        if invalid is not None:
            invalid = invalid.all(2, keepdim=True)  # (B, L, 1).
            result = result.masked_fill(invalid, 0)  # (B, L, D).

        return self.dropout1(result)

    def _ff_block(self, embeddings):
        x = self.linear2(self.dropout(self.activation(self.linear1(embeddings))))
        return self.dropout2(x)


class AttNHPTransformer(torch.nn.Module):
    """Att-NHP transformer model.

    Args:
        sample_to_batch: Whether to duplicate batch for each sample or append them as independent tokens.
    """
    def __init__(self, input_size, hidden_size, n_heads, n_layers,
                 dim_feedforward=None, dim_value=None, dropout=0.1,
                 pos_m=1, pos_M=2000, sample_to_batch=False):
        super().__init__()
        self.pos_encoder = PositionalEncoder(hidden_size, m=pos_m, M=pos_M)
        self.inter_token = torch.nn.Parameter(torch.randn(hidden_size))
        self.proj = torch.nn.Linear(input_size, hidden_size)
        layers = []
        for i in range(n_layers):
            layers.append(AttNHPTransformerLayer(hidden_size, n_heads,
                                                 dim_feedforward=dim_feedforward,
                                                 dim_value=dim_value,
                                                 dropout=dropout))
        self.layers = torch.nn.ModuleList(layers)
        self.hidden_size = hidden_size
        self.sample_to_batch = sample_to_batch

    @property
    def delta_time(self):
        """Whether to take delta time or raw timestamps at input."""
        # Need raw time for positional encoding.
        return False

    @property
    def output_size(self):
        return self.hidden_size

    @property
    def num_layers(self):
        return len(self.layers)

    def forward(self, embeddings: PaddedBatch, times: PaddedBatch,
                return_states=False):
        """Encode input sequences with a causal mask.

        Args:
            embeddings: Input embeddings with shape (B, L, D).
            times: Input times with shape (B, L), absolute.
            return_states: Whether to return final states with shape (B, D), full states with shape (B, T, D)
                or no states (either False, "last" or "full").

        Returns:
            Outputs with shape (B, L, D) and activations with shape (N, B, L, D).
        """
        x, mask = embeddings.payload, ~embeddings.seq_len_mask.bool()
        x = self.proj(x)  # (B, L, D).
        x = x + self.pos_encoder(times.payload)
        rng = torch.arange(x.shape[1], device=x.device)  # (L).
        attn_mask = rng[:, None] <= rng[None, :]  # (L, L), exclude self-reference.
        results = [x]
        for layer in self.layers:
            results.append(layer(results[-1], mask, attn_mask))  # (B, L, D).
        outputs = PaddedBatch(results[-1], embeddings.seq_lens)
        if not return_states:
            states = None
        elif return_states == "last":
            raise NotImplementedError("Only full or no states can be returned.")
        elif return_states == "full":
            states = torch.stack(results[:-1])  # (N, B, L, D).
            states = TransformerState(times.payload, states, embeddings.seq_lens)
        else:
            raise ValueError(f"Unknown states flag: {return_states}")
        return outputs, states

    def decode(self, embeddings: PaddedBatch, times: PaddedBatch, history_states: TransformerState):
        """Compute activations for queries with full mask.

        Args:
            embeddings: Input embeddings with shape (B, L, D), right-aligned.
            times: Times to predict with shape (B, L).
            history_states: Historical activations with shape (N, B, H, D).
                Decoder will ignore the index.

        Returns:
            Outputs with shape (B, L, D), states with shape (N, B, L, D).
        """
        b, l = embeddings.shape
        lh = history_states.payload.shape[2]
        x = self.proj(embeddings.payload) + self.pos_encoder(times.payload)  # (B, L, D).

        mask = ~history_states.seq_len_mask.bool()  # (B, H).
        results = [x]  # (B, L, D).
        assert len(history_states) == len(self.layers)
        for layer, states in zip(self.layers, history_states.payload):
            results.append(layer(results[-1], mask, history=states))  # (B, L, D).
        outputs = PaddedBatch(results[-1], embeddings.seq_lens)
        states = torch.stack(results[:-1])  # (N, B, L, D).
        states = TransformerState(times.payload, states, embeddings.seq_lens)
        return outputs, states

    def interpolate(self, times: PaddedBatch, history_states: TransformerState, last_history_index=None):
        """Compute activations for queries with full mask.

        Args:
            times: Times to predict with shape (B, L, S).
            history_states: Historical activations with shape (N, B, H, D).
                Index is used to align states with attn_mask.
            last_history_index: The last history state index used for prediction with shape (L).

        Returns:
            Outputs with shape (B, L, S, D).
        """
        b, l, s = times.payload.shape
        if self.sample_to_batch and (s > 1):
            sb_times = PaddedBatch(times.payload.transpose(1, 2).reshape(b * s, l, 1), times.seq_lens[:, None].repeat(1, s).flatten())  # (BS, L', 1).
            sb_history_states = TransformerState(
                times=history_states.times[:, None, :].repeat(1, s, 1).flatten(0, 1),  # (BS, L).
                states=history_states.payload[:, :, None, :, :].repeat(1, 1, s, 1, 1).flatten(1, 2),  # (N, BS, L, D).
                seq_lens=history_states.seq_lens[:, None].repeat(1, s).flatten(),
                index=history_states.index[:, None].repeat(1, s, *([1] * (history_states.index.ndim - 1))).flatten(0, 1),  # (BS, *).
                index_lens=history_states.index_lens[:, None].repeat(1, s).flatten())
            result = self.interpolate(sb_times, sb_history_states, last_history_index)  # (BS, L', 1, D).
            return PaddedBatch(result.payload.reshape(b, s, l, -1).transpose(1, 2), times.seq_lens)  # (B, L', S, D).

        lh = history_states.payload.shape[2]
        if last_history_index is not None:
            if not (history_states.index[1:] == history_states.index[:1]).all():
                raise NotImplementedError("Need uniform index for attention mask during interpolation.")
            last_history_index = history_states.index[0].take_along_dim(last_history_index, 0)  # (L).

        x = self.inter_token[None, None] + self.pos_encoder(times.payload.reshape(b, l * s))  # (B, LS, D).

        mask = ~history_states.seq_len_mask.bool()  # (B, H).
        if last_history_index is not None:
            history_attn_inv_mask = last_history_index[:, None] < torch.arange(lh, device=history_states.device)  # (L, H).
            attn_mask = history_attn_inv_mask.unsqueeze(1).repeat(1, s, 1).reshape(l * s, lh)  # (LS, H).
        else:
            attn_mask = None
        assert len(history_states) == len(self.layers)
        for layer, states in zip(self.layers, history_states.payload):
            x = layer(x, mask, attn_mask, history=states)  # (B, LS, D).
        outputs = PaddedBatch(x.reshape(b, l, s, -1), times.seq_lens)  # (B, L, S, D).
        return outputs
