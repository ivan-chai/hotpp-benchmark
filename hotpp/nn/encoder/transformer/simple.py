import math
import torch
from contextlib import contextmanager

from hotpp.data import PaddedBatch


class PositionalAngularEmbedding(torch.nn.Module):
    def __init__(self, n_embd, n_positions, trainable=False):
        super().__init__()
        position = torch.arange(n_positions).unsqueeze(1)  # (L, 1).
        div_term = torch.exp(torch.arange(0, n_embd, 2) * (-math.log(10000.0) / n_embd))  # (D // 2).
        pe = torch.zeros(n_positions, n_embd)  # (L, D).
        # Use interleave instead of concatenation to achieve correct processing with multiple attention heads.
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        if trainable:
            self.pe = torch.nn.Parameter(pe)
        else:
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
        return super().forward(self.pe[:x.shape[1]])[None].expand(b, l, self.weight.shape[1])


class TimeAngularEmbedding(torch.nn.Module):
    def __init__(self, n_embd, n_positions, max_duration, min_time_step=None, relative=False, trainable=False):
        super().__init__()
        if min_time_step is None:
            min_time_step = max_duration / n_positions
        pe = torch.exp(torch.arange(0, n_embd, 2) * (-math.log(5 * max_duration / min_time_step) / n_embd)) / min_time_step  # (D // 2).
        if trainable:
            self.pe = torch.nn.Parameter(pe)
        else:
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
        pos_type: Either `pos-embedding`, `pos-angular[-train]`, `time-angular[-train]-abs`, `time-angular[-train]-rel`, or a list of values (probably, empty).
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
            if name in ["pos-angular", "pos-angular-train"]:
                embedders.append(PositionalAngularEmbedding(embedder_size, n_positions,
                                                            trainable="-train" in name))
            elif name == "pos-embedding":
                embedders.append(PositionalEmbedding(embedder_size, n_positions))
            elif name in {"time-angular-abs", "time-angular-train-abs", "time-angular-rel", "time-angular-train-rel"}:
                embedders.append(TimeAngularEmbedding(embedder_size, n_positions, max_duration, min_time_step=min_time_step,
                                                      relative="-rel" in name,
                                                      trainable="-train" in name))
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


class ExtendedLayer:
    def __init__(self, layer):
        self.layer = layer

    def __call__(self, *args, **kwargs):
        self.activation = self.layer(*args, **kwargs)
        return self.activation


class ExtendedTransformer:
    def __init__(self, model, cache_hiddens=False):
        self.model = model

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    @property
    def activations(self):
        return [layer.activation for layer in self.model.layers]


@contextmanager
def extended_transformer(transformer, cache_hiddens=False):
    if not cache_hiddens:
        yield transformer
        return
    layers = [ExtendedLayer(layer) for layer in transformer.layers]
    backup_values = transformer.layers._modules.values
    transformer.layers._modules.values = lambda: layers
    try:
        yield ExtendedTransformer(transformer)
    finally:
        transformer.layers._modules.values = backup_values


class SimpleTransformer(torch.nn.Module):
    """Simple transformer mimicing HuggingFace interface.

    Args:
        pos_type: Either `pos-embedding`, `pos-angular`, `time-angular[-train]-abs`, `time-angular[-train]-rel`, or a list of values (probably, empty).
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
            self.register_buffer("sa_mask", sa_mask, persistent=False)
        else:
            self.sa_mask = None

    @property
    def output_size(self):
        return self.n_embd

    @property
    def delta_time(self):
        return False

    def transform(self, embeddings, return_states=False, attention_mask=None):
        """Apply encoder after input projection and positional encoding.

        Args:
            attention_mask: Additional attention mask with shape (L, L) or (B, L, L) which contains True for masked connections.
                The mask will be merged with causal mask if causal transformer is applied.

        Returns:
            Outputs and activations, if return_states is "full".
        """
        b, l = embeddings.shape
        causal_hint = self.causal if attention_mask is None else False
        if attention_mask is None:
            attention_mask = self.sa_mask[:l, :l] if self.sa_mask is not None else None
        elif self.sa_mask is not None:
            sa_mask = self.sa_mask[:l, :l]
            if attention_mask.ndim == 3:
                sa_mask = sa_mask[None]
                if attention_mask.shape[0] == b:
                    attention_mask = attention_mask.repeat_interleave(self.n_head, dim=0)
            attention_mask = torch.logical_or(attention_mask, sa_mask)
        else:
            attention_mask = None

        with extended_transformer(self.encoder, cache_hiddens=(return_states == "full")) as encoder:
            outputs = encoder(embeddings.payload,
                              mask=attention_mask,
                              src_key_padding_mask=~embeddings.seq_len_mask.bool() if not self.causal else None,
                              is_causal=causal_hint)  # (B, L, D).
            if return_states == "full":
                states = encoder.activations
            else:
                states = None
        return PaddedBatch(outputs, embeddings.seq_lens), states

    def forward(self, x, timestamps, states=None, return_states=False, attention_mask=None):
        """Apply Transformer.

        Args:
            x: Batch with shape (B, L, D).
            timestamps: Inputs timestamps.
            states (unused): Initial states with shape (N, B, D), where N is the number of layers.
            return_states: Whether to return states with shape (B, T, D) or not (either False or "full").
            attention_mask: Additional attention mask with shape (L, L) which contains True for masked connections.
                The mask will be merged with causal mask if causal transformer is applied.

        Returns:
            Outputs with shape (B, L, D) and None (states are not supported).
        """
        if return_states not in {False, "full"}:
            raise ValueError(f"Unknown states mode: {return_states}")

        embeddings = self.input_projection(x.payload)  # (B, L, D).
        embeddings = self.positional(embeddings, timestamps.payload)  # (B, L, D).
        embeddings = PaddedBatch(embeddings, x.seq_lens)
        outputs, states = self.transform(embeddings, attention_mask=attention_mask)
        return outputs, states
