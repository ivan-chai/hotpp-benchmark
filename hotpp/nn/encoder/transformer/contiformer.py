"""An adapter of ContiFormer."""
import math
import torch
from hotpp.data import PaddedBatch
from physiopro.network.contiformer import ContiFormer
from .state import TransformerState

# samples: last value is timestamp
# input: B x L x D
# pad_input: повторить последний эмбеддинг с t_max временем.


class ContiformerTransformer(torch.nn.Module):
    """Contiformer model."""
    def __init__(self, input_size, hidden_size, n_heads, n_layers,
                 dim_feedforward=None, dim_kv=None, dropout=0.1):
        if dim_feedforward is None:
            dim_feedforward = hidden_size * 4
        if dim_kv is None:
            if hidden_size % n_heads != 0:
                raise ValueError("Can't infer KV dimensions")
            dim_kv = hidden_size // n_heads
        super().__init__()
        self.input_size = input_size
        self.model = ContiFormer(
            input_size=input_size,
            d_model=hidden_size,
            d_inner=dim_feedforward,
            n_layers=n_layers,
            n_head=n_heads,
            d_k=dim_kv,
            d_v=dim_kv,
            dropout=dropout
        )

    @property
    def output_size(self):
        return self.model.encoder.d_model

    @property
    def num_layers(self):
        return 1  # Mimic multiple layers for consistency with the state's shape.

    def forward(self, embeddings: PaddedBatch, times: PaddedBatch):
        """Encode input sequences with a causal mask.

        Args:
            embeddings: Input embeddings with shape (B, L, D).
            times: Input times with shape (B, L), absolute.

        Returns:
            Outputs with shape (B, L, D) and activations with shape (N, B, L, D).
        """
        x, mask = embeddings.payload, embeddings.seq_len_mask.bool().unsqueeze(2)  # (B, L, D), (B, L, 1).
        x, _ = self.model(x, times.payload, mask)  # (B, L, D).
        outputs = PaddedBatch(x, embeddings.seq_lens)  # (B, L, D).
        states = torch.cat([embeddings.payload, x], -1)  # (B, L, DI + DO).
        states = TransformerState(times.payload, states[None], embeddings.seq_lens)  # (1, B, L, DI + DO).
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
        history_embeddings = history_states.payload[0, ..., :self.input_size]  # (B, H, DI).
        all_embeddings = torch.cat([history_embeddings, embeddings.payload], 1)  # (B, H + L, D).
        all_times = torch.cat([history_states.times, times.payload], 1)  # (B, H + L).
        mask = torch.zeros(b, lh + l, lh + l, dtype=torch.bool, device=embeddings.device)
        rng = torch.arange(lh, device=embeddings.device)
        # Mask: B x target x source.
        # Target can access all source tokens to the left.
        mask[:, :lh, :lh] = rng[None, :, None] >= rng[None, None, :]  # Causal mask for historical events.
        # New tokens can see all valid historical tokens.
        mask[:, lh:, :lh] = history_states.seq_len_mask  # Allow accces to history.
        x, _ = self.model(all_embeddings, all_times, mask)[:, lh:]  # (B, L, D).
        outputs = PaddedBatch(x, embeddings.seq_lens)  # (B, L, D).
        states = TransformerState(times.payload, embeddings.payload[None], embeddings.seq_lens)  # (1, B, L, D).
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
        # Discrete model, don't interpolate.
        outputs = history_states.payload[0, ..., self.input_size:]  # (B, H, D).
        if last_history_index is not None:
            outputs = outputs.take_along_dim(last_history_index[None, :, None], 1)  # (B, L, D).
        elif times.shape[1] != history_states.payload.shape[2]:
            raise ValueError("Incompatible times and states lengths.")
        return outputs
