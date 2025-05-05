import math
import torch

from hotpp.data import PaddedBatch


class PositionalEncoding(torch.nn.Module):
    """Positional encoder.

    Time embeddings are motivated by the following paper:
    Mei H., Yang C., Eisner J. "Transformer embeddings of irregularly spaced events and their participants", ICLR 2021.

    Args:
        pos_type: Either `pos-embedding`, `pos-angular`, `time-angular-abs`, `time-angular-rel`, or `none`.
        max_duration: Must be provided if time encodings are used.
        min_time_step: The minimum time step (> 0). By default it is max_duration / n_positions.
    """
    def __init__(self, n_embd, n_positions, pos_type="pos-angular", max_duration=None, min_time_step=None, dropout=0.1):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        self.pos_type = pos_type
        if pos_type == "pos-angular":
            position = torch.arange(n_positions).unsqueeze(1)  # (L, 1).
            div_term = torch.exp(torch.arange(0, n_embd, 2) * (-math.log(10000.0) / n_embd))  # (D // 2).
            pe = torch.zeros(n_positions, n_embd)  # (L, D).
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer("pe", pe, persistent=False)
        elif pos_type == "pos-embedding":
            pe = torch.arange(n_positions)
            self.register_buffer("pe", pe, persistent=False)
            self.embeddings = torch.nn.Embedding(n_positions, n_embd)
        elif pos_type in ["time-angular-abs", "time-angular-rel"]:
            if max_duration is None:
                raise ValueError("Need max_duration for time embeddings.")
            if n_embd % 2 != 0:
                raise NotImplementedError("Need an even embedding dimension.")
            if min_time_step is None:
                min_time_step = max_duration / n_positions
            pe = min_time_step * (5 * max_duration / min_time_step) ** torch.linspace(0, 1, n_embd // 2)
            self.register_buffer("pe", pe, persistent=False)
        elif pos_type != "none":
            raise ValueError(f"Unknown positional embedding type: {pos_type}")

    def forward(self, x, timestamps=None):
        # x: (B, L, D).
        # timestamps: (B, L).
        if self.pos_type == "pos-angular":
            embeddings = self.pe[:x.shape[1]]
        elif self.pos_type == "pos-embedding":
            embeddings = self.embeddings(self.pe[:x.shape[1]])
        elif self.pos_type in ["time-angular-abs", "time-angular-rel"]:
            if timestamps is None:
                raise ValueError("Need timestamps for the selected positional encoding scheme.")
            if self.pos_type == "time-angular-rel":
                timestamps = timestamps - timestamps[:, :1]
            args = timestamps.unsqueeze(-1) / self.pe  # (B, L, D).
            embeddings = torch.cat([torch.sin(args), torch.cos(args)], -1)
        else:
            assert self.pos_type == "none"
        return self.dropout(x + embeddings)


class SimpleTransformer(torch.nn.Module):
    """Simple transformer mimicing HuggingFace interface.

    Args:
        pos_type: Either `pos-embedding`, `pos-angular`, `time-angular-abs`, `time-angular-rel`, or `none`.
        max_duration: Must be provided if time encodings are used.
        min_time_step: The minimum time step (> 0). By default it is max_duration / n_positions.
    """
    def __init__(self, input_size, n_positions=1024, n_embd=768, n_layer=12, n_head=12,
                 n_inner=None, dropout=0.1, causal=False,
                 pos_type="pos-angular", max_duration=None, min_time_step=None):
        super().__init__()
        n_inner = n_inner if n_inner is not None else 4 * n_embd
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_inner = n_inner
        self.dropout = dropout
        self.causal = causal

        self.input_projection = torch.nn.Linear(input_size, n_embd)

        # We use norm_first by default.
        # See the original paper: Xiong R. et al. "On layer normalization in the transformer architecture" ICML 2020.
        layer = torch.nn.TransformerEncoderLayer(d_model=n_embd,
                                                 nhead=n_head,
                                                 dim_feedforward=n_inner,
                                                 dropout=dropout,
                                                 norm_first=True,
                                                 batch_first=True)
        self.encoder = torch.nn.TransformerEncoder(layer, n_layer)
        self.positional = PositionalEncoding(n_embd=n_embd,
                                             n_positions=n_positions,
                                             pos_type=pos_type,
                                             max_duration=max_duration,
                                             min_time_step=min_time_step,
                                             dropout=dropout)
        if causal:
            sa_mask = torch.triu(torch.ones((n_positions, n_positions), dtype=torch.bool), diagonal=1)
            self.register_buffer("sa_mask", sa_mask)
        else:
            self.sa_mask = None

    @property
    def output_size(self):
        return self.n_embd

    @property
    def delta_time(self):
        return False

    def forward(self, x, timestamps, states=None, return_states=False):
        """Apply Transformer.

        Args:
            x: Batch with shape (B, L, D).
            timestamps: Inputs timestamps.
            states (unused): Initial states with shape (N, B, D), where N is the number of layers.
            return_states: Whether to return final states with shape (B, D), full states with shape (B, T, D)
                or no states (either False, "last" or "full"). Must be False. Added only for interface compatibility.

        Returns:
            Outputs with shape (B, L, D) and None (states are not supported).
        """
        if return_states:
            raise ValueError("Transformers encoder doesn't support states return")
        embeddings = self.input_projection(x.payload)  # (B, L, D).
        embeddings = self.positional(embeddings, timestamps.payload)  # (B, L, D).

        b, l, d = embeddings.shape
        outputs = self.encoder(embeddings,
                               mask=self.sa_mask[:l, :l] if self.sa_mask is not None else None,
                               src_key_padding_mask=~x.seq_len_mask.bool(),
                               is_causal=self.causal)  # (B, L, D).
        state = None
        return PaddedBatch(outputs, x.seq_lens), state
