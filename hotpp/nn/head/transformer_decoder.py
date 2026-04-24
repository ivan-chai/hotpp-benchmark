import torch
import torch.nn as nn
from hotpp.data import PaddedBatch


class _CrossOnlyDecoderLayer(nn.Module):
    """Transformer decoder layer with cross-attention only (no self-attention between queries)."""

    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, tgt, memory, memory_key_padding_mask=None, **kwargs):
        tgt2, _ = self.cross_attn(
            tgt, memory, memory,
            key_padding_mask=memory_key_padding_mask,
        )
        tgt = self.norm1(tgt + self.dropout(tgt2))
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = self.norm2(tgt + self.dropout(tgt2))
        return tgt


class _CrossOnlyDecoder(nn.Module):
    """Stack of cross-attention-only decoder layers."""

    def __init__(self, n_layers, d_model, nhead, dim_feedforward, dropout):
        super().__init__()
        self.layers = nn.ModuleList([
            _CrossOnlyDecoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, tgt, memory, memory_key_padding_mask=None):
        for layer in self.layers:
            tgt = layer(tgt, memory, memory_key_padding_mask=memory_key_padding_mask)
        return tgt


class TransformerDecoderHead(nn.Module):
    """Prediction head using transformer cross-attention over encoder history.

    Instead of attending to a single aggregated context vector (as in ConditionalHead),
    each of the K query vectors attends to the full history of encoder hidden states
    up to the current timestep via cross-attention. This allows queries to selectively
    focus on different parts of the history.

    Architecture (per timestep t):
        memory  = encoder_hiddens[:, :t+1, :]   (B, t+1, D)
        queries = learnable (K, D_q)
        output  = TransformerDecoder(tgt=queries, memory=memory)  (K, D_q)
        preds   = output_proj(output).flatten()                    (K*P,)

    Args:
        input_size: Encoder hidden size (backbone output dimension).
        output_size: Total output dimension (K * P, where P = params per event).
        k: Number of output event candidates.
        query_size: Decoder model dimension. Defaults to input_size.
        n_heads: Number of attention heads. Must evenly divide query_size.
        n_layers: Number of decoder layers.
        dim_feedforward: FFN hidden size inside each decoder layer.
            Defaults to 4 * query_size.
        dropout: Dropout probability.
        use_self_attention: If True, uses a full TransformerDecoder layer
            (self-attention + cross-attention + FFN). If False, uses only
            cross-attention + FFN (no communication between queries before
            cross-attention).
    """

    def __init__(self, input_size, output_size, k,
                 query_size=None, n_heads=4, n_layers=2,
                 dim_feedforward=None, dropout=0.0,
                 use_self_attention=True):
        super().__init__()
        if output_size % k != 0:
            raise ValueError("output_size must be divisible by k.")

        self.k = k
        self.output_size = output_size
        query_size = query_size or input_size
        self.query_size = query_size
        dim_feedforward = dim_feedforward or 4 * query_size

        # K learnable query vectors — one slot per predicted future event.
        self.queries = nn.Parameter(torch.randn(k, query_size))

        # Project encoder hidden dim → query_size when they differ.
        self.input_proj = (
            nn.Linear(input_size, query_size)
            if input_size != query_size
            else None
        )

        if use_self_attention:
            # Full decoder: self-attn(queries) → cross-attn(queries, memory) → FFN
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=query_size,
                nhead=n_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
            )
            self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        else:
            # Lightweight decoder: cross-attn(queries, memory) → FFN only
            self.decoder = _CrossOnlyDecoder(
                n_layers, query_size, n_heads, dim_feedforward, dropout
            )

        # Per-query projection to event parameter space.
        self.output_proj = nn.Linear(query_size, output_size // k)

    def forward_impl(self, memory, key_padding_mask=None):
        """Run the decoder for a batch of memory sequences.

        Args:
            memory: Encoder hidden states, shape (N, S, D).
            key_padding_mask: Boolean mask (N, S). True where memory is padding
                and should be ignored by attention.

        Returns:
            Predictions of shape (N, K*P).
        """
        N = memory.shape[0]

        if self.input_proj is not None:
            memory = self.input_proj(memory)  # (N, S, D_q)

        queries = self.queries.unsqueeze(0).expand(N, -1, -1)  # (N, K, D_q)

        output = self.decoder(
            tgt=queries,
            memory=memory,
            memory_key_padding_mask=key_padding_mask,
        )  # (N, K, D_q)

        output = self.output_proj(output)  # (N, K, P)
        return output.flatten(1)           # (N, K*P)

    def forward(self, x, indices=None):
        """Predict K future events for each sequence position.

        Uses the "fake-batch" trick: every (batch_item, position) pair is
        treated as an independent sample, enabling full parallelism.

        Args:
            x: PaddedBatch with payload (B, L, D) — encoder hidden states.
            indices: Optional PaddedBatch with payload["index"] (B, I) —
                positions selected for loss computation during training.

        Returns:
            PaddedBatch with payload (B, L, K*P) or (B, I, K*P).
        """
        payload = x.payload    # (B, L, D)
        seq_lens = x.seq_lens  # (B,)
        B, L, D = payload.shape
        pos_range = torch.arange(L, device=payload.device)

        if indices is None:
            # Val / test: process all L positions in parallel.
            # For position t, valid memory positions are j ∈ [0, t] ∩ [0, seq_len).

            # causal_mask[t, j] = True  →  j > t  (future, invisible)
            causal_mask = pos_range.unsqueeze(0) > pos_range.unsqueeze(1)  # (L, L)
            causal_mask = causal_mask.unsqueeze(0).expand(B, -1, -1)       # (B, L, L)

            # seq_mask[b, *, j] = True  →  j >= seq_lens[b]  (padding)
            seq_mask = pos_range.view(1, 1, L) >= seq_lens.view(B, 1, 1)   # (B, 1, L)

            key_padding_mask = (causal_mask | seq_mask).reshape(B * L, L)  # (B*L, L)

            # memory: same payload repeated for each of the L positions.
            memory = payload.unsqueeze(1).expand(-1, L, -1, -1).reshape(B * L, L, D)

            output = self.forward_impl(memory, key_padding_mask)  # (B*L, K*P)
            output = output.reshape(B, L, self.output_size)        # (B, L, K*P)
            return PaddedBatch(output, seq_lens)

        else:
            # Training: process only the I selected positions (loss subset).
            idx = indices.payload["index"]  # (B, I)
            out_lengths = indices.seq_lens   # (B,)
            I = idx.shape[1]

            # causal_mask[b, i, j] = True  →  j > idx[b, i]
            causal_mask = pos_range.view(1, 1, L) > idx.unsqueeze(2)       # (B, I, L)

            # seq_mask[b, *, j] = True  →  j >= seq_lens[b]
            seq_mask = pos_range.view(1, 1, L) >= seq_lens.view(B, 1, 1)   # (B, 1, L)

            key_padding_mask = (causal_mask | seq_mask).reshape(B * I, L)  # (B*I, L)

            memory = payload.unsqueeze(1).expand(-1, I, -1, -1).reshape(B * I, L, D)

            output = self.forward_impl(memory, key_padding_mask)  # (B*I, K*P)
            output = output.reshape(B, I, self.output_size)        # (B, I, K*P)
            return PaddedBatch(output, out_lengths)
