import math
import torch

from hotpp.data import PaddedBatch


class PositionalAngularEmbedding(torch.nn.Module):
    def __init__(self, n_embd, n_positions):
        super().__init__()
        position = torch.arange(n_positions).unsqueeze(1)  # (L, 1).
        div_term = torch.exp(torch.arange(0, n_embd, 2) * (-math.log(10000.0) / n_embd))  # (D // 2).
        pe = torch.zeros(n_positions, n_embd)  # (L, D).
        # Use interleave instead of concatenation to achieve correct processing with multiple attention heads.
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x, timestamps=None):
        b, l, _ = x.shape
        return self.pe[:x.shape[1]][None].expand(b, l, self.pe.shape[-1])


class PositionalEmbedding(torch.nn.Embedding):
    def __init__(self, n_embd, n_positions):
        super().__init__(n_positions, n_embd)
        pe = torch.arange(n_positions)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x, timestamps=None):
        b, l, _ = x.shape
        return super().forward(self.pe[:x.shape[1]])[None].expand(b, l, self.pe.shape[-1])


class TimeAngularEmbedding(torch.nn.Module):
    def __init__(self, n_embd, n_positions, max_duration, min_time_step=None, relative=False):
        super().__init__()
        if min_time_step is None:
            min_time_step = max_duration / n_positions
        pe = torch.exp(torch.arange(0, n_embd, 2) * (-math.log(5 * max_duration / min_time_step) / n_embd)) / min_time_step  # (D // 2).
        self.register_buffer("pe", pe, persistent=False)
        self.relative = relative

    def forward(self, x, timestamps=None):
        if timestamps is None:
            raise ValueError("Need timestamps for the selected positional encoding scheme.")
        if self.relative:
            timestamps = timestamps - timestamps[:, :1]
        args = timestamps.unsqueeze(-1) * self.pe  # (B, L, D).
        # Use interleave instead of concatenation to achieve correct processing with multiple attention heads.
        return torch.stack([torch.sin(args), torch.cos(args)], -1).flatten(2, 3)  # (B, L, D).


class PositionalEncoding(torch.nn.Module):
    """Positional encoder.

    Time embeddings are motivated by the following paper:
    Mei H., Yang C., Eisner J. "Transformer embeddings of irregularly spaced events and their participants", ICLR 2021.

    Args:
        pos_type: Either `pos-embedding`, `pos-angular`, `time-angular-abs`, `time-angular-rel`, or a list of values (probably, empty).
        max_duration: Must be provided if time encodings are used.
        min_time_step: The minimum time step (> 0). By default it is max_duration / n_positions.
    """
    def __init__(self, n_embd, n_positions, pos_type="pos-angular", max_duration=None, min_time_step=None, dropout=0.1):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        self.pos_type = pos_type
        embedders = []
        if isinstance(pos_type, str):
            pos_type = [pos_type]
        if n_embd % len(pos_type) != 0:
            raise ValueError("The embedding size must be divisible by the number of positional embedders")
        embedder_size = n_embd // len(pos_type)
        for name in pos_type:
            if name == "pos-angular":
                embedders.append(PositionalAngularEmbedding(embedder_size, n_positions))
            elif name == "pos-embedding":
                embedders.append(PositionalEmbedding(embedder_size, n_positions))
            elif name == "time-angular-abs":
                embedders.append(TimeAngularEmbedding(embedder_size, n_positions, max_duration, min_time_step=min_time_step))
            elif name == "time-angular-rel":
                embedders.append(TimeAngularEmbedding(embedder_size, n_positions, max_duration, min_time_step=min_time_step,
                                                      relative=True))
            else:
                raise ValueError(f"Unknown positional embedding type: {name}")
        self.embedders = torch.nn.ModuleList(embedders)

    def forward(self, x, timestamps=None):
        # x: (B, L, D).
        # timestamps: (B, L).
        embeddings = [embedder(x, timestamps) for embedder in self.embedders]  # N x (B, L, D / N).
        # Use interleave instead of concatenation to achieve correct processing with multiple attention heads.
        embeddings = torch.stack(embeddings, -1).flatten(2, 3)  # (B, L, D).
        return self.dropout(x + embeddings)


class SimpleTransformer(torch.nn.Module):
    """Simple transformer mimicing HuggingFace interface.

    Args:
        pos_type: Either `pos-embedding`, `pos-angular`, `time-angular-abs`, `time-angular-rel`, or a list of values (probably, empty).
        max_duration: Must be provided if time encodings are used.
        min_time_step: The minimum time step (> 0). By default it is max_duration / n_positions.
    """
    def __init__(self, input_size, n_positions=1024, n_embd=768, n_layer=12, n_head=12,
                 n_inner=None, dropout=0.1, causal=False,
                 activation=torch.nn.functional.relu,
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
                                                 activation=activation,
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
