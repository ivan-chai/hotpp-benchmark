import math
import torch
from torch.nn import functional as F
from contextlib import contextmanager

from hotpp.data import PaddedBatch
from .rope import MultiheadAttentionRoPE, TimeRoPEEncoding


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
        pos_type: Either `none`, `pos-embedding`, `pos-angular[-train]`, `time-angular[-train]-abs`, `time-angular[-train]-rel`, or a list of values (probably, empty).
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
            if name in "none":
                continue
            elif name in ["pos-angular", "pos-angular-train"]:
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
        if len(self.embedders) == 0:
            embeddings = x
        else:
            embeddings = [embedder(x, timestamps) for embedder in self.embedders]  # N x (B, L, D / N).
            # Use interleave instead of concatenation to achieve correct processing with multiple attention heads.
            embeddings = torch.stack(embeddings, -1).flatten(2, 3)  # (B, L, D).
            embeddings = x + embeddings
        return self.dropout(embeddings)


class HoTPPTransformerEncoderLayer(torch.nn.TransformerEncoderLayer):
    """TransformerEncoderLayer with RoPE support.

    Args:
        normalization: Normalization class.
        mlp: Either "default" or "gated".
    """
    def __init__(self, d_model, nhead, dim_feedforward,
                 dropout=0.1,
                 activation=F.relu,
                 normalization=torch.nn.LayerNorm,
                 layer_norm_eps=1e-5,
                 mlp="default",
                 group_size=1,
                 batch_first=False,
                 norm_first=False,
                 bias=True,
                 device=None,
                 dtype=None):
        super().__init__(d_model, nhead,
                         dim_feedforward=dim_feedforward,
                         dropout=dropout,
                         activation=activation,
                         layer_norm_eps=layer_norm_eps,
                         batch_first=batch_first,
                         norm_first=norm_first,
                         bias=bias,
                         device=device,
                         dtype=dtype)

        factory_kwargs = {"device": device, "dtype": dtype}

        # Update normalization.
        if normalization is not torch.nn.LayerNorm:
            norm_kwargs = dict(factory_kwargs)
            if (normalization is torch.nn.LayerNorm) or (normalization is torch.nn.RMSNorm):
                norm_kwargs["eps"] = layer_norm_eps
            assert hasattr(self, "norm1")
            self.norm1 = normalization(d_model, **norm_kwargs)
            assert hasattr(self, "norm2")
            self.norm2 = normalization(d_model, **norm_kwargs)

        # Update MLP.
        if mlp == "gated":
            self.gate = torch.nn.Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
        elif mlp != "default":
            raise ValueError(f"Unknown MLP type: {mlp}.")
        self.mlp = mlp

        # Update attention block.
        self.self_attn = MultiheadAttentionRoPE(
            d_model,
            nhead,
            dropout=dropout,
            bias=bias,
            batch_first=batch_first,
            group_size=group_size,
            **factory_kwargs,
        )

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False, rope=None):
        rope = rope or getattr(self, "_rope", [None])[0]
        self.self_attn._rope = [rope]
        try:
            return super().forward(src, src_mask, src_key_padding_mask, is_causal)
        finally:
            del self.self_attn._rope

    def _ff_block(self, x):
        if self.mlp == "gated":
            x = self.linear2(self.dropout(self.activation(self.gate(x)) * self.linear1(x)))
        else:
            assert self.mlp == "default"
            x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class HoTPPTransformerEncoder(torch.nn.TransformerEncoder):
    """TransformerEncoder with RoPE support."""
    def __init__(self,
                 layers,
                 norm=None,
                 mask_check=True):
        super(torch.nn.TransformerEncoder, self).__init__()
        self.layers = torch.nn.ModuleList(layers)
        self.num_layers = len(layers)
        self.norm = norm
        # this attribute saves the value providedat object construction
        self.enable_nested_tensor = False
        # this attribute controls whether nested tensors are used
        self.use_nested_tensor = False
        self.mask_check = mask_check

    def forward(self, src, mask=None, src_key_padding_mask=None, is_causal=False, rope=None):
        if rope is None:
            return super().forward(src, mask, src_key_padding_mask, is_causal)
        for layer in self.layers:
            layer._rope = [rope]
        try:
            return super().forward(src, mask, src_key_padding_mask, is_causal)
        finally:
            for layer in self.layers:
                del layer._rope

class ExtendedLayer:
    def __init__(self, layer):
        self.layer = layer

    @property
    def self_attn(self):
        return self.layer.self_attn

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
    backup_values = dict(transformer.layers._modules)
    transformer.layers._modules.update({name: ExtendedLayer(layer) for name, layer in transformer.layers._modules.items()})
    try:
        yield ExtendedTransformer(transformer)
    finally:
        transformer.layers._modules.update(backup_values)


@contextmanager
def no_mha_fast_path():
    revert = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        yield None
    finally:
        torch.backends.mha.set_fastpath_enabled(revert)


class SimpleTransformer(torch.nn.Module):
    """Simple transformer mimicing HuggingFace interface.

    Args:
        pos_type: Either `pos-embedding`, `pos-angular`, `time-angular[-train]-abs`, `time-angular[-train]-rel`, or a list of values (probably, empty).
        max_duration: Must be provided if time encodings are used.
        min_time_step: The minimum time step (> 0). By default it is max_duration / n_positions.
        rope: Either "time[-train]", "none" or None.
    """
    def __init__(self, input_size, n_positions=1024, n_embd=768, n_layer=12, n_head=12,
                 n_inner=None, dropout=0.1, causal=False,
                 activation=torch.nn.functional.relu,
                 normalization=torch.nn.LayerNorm,
                 mlp="default", pos_type="pos-angular", rope=None, group_size=1,
                 max_duration=None, min_time_step=None):
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
        layers = [HoTPPTransformerEncoderLayer(d_model=n_embd,
                                               nhead=n_head,
                                               dim_feedforward=n_inner,
                                               activation=activation,
                                               normalization=normalization,
                                               mlp=mlp,
                                               dropout=dropout,
                                               group_size=group_size,
                                               norm_first=True,
                                               batch_first=True)
                  for _ in range(n_layer)]
        self.encoder = HoTPPTransformerEncoder(layers)
        self.positional = PositionalEncoding(n_embd=n_embd,
                                             n_positions=n_positions,
                                             pos_type=pos_type,
                                             max_duration=max_duration,
                                             min_time_step=min_time_step,
                                             dropout=dropout)
        if rope in {"time", "time-train"}:
            self.rope = TimeRoPEEncoding(
                head_dim=n_embd // n_head,
                n_positions=n_positions,
                max_duration=max_duration,
                min_time_step=min_time_step,
                trainable="train" in rope
            )
        elif (rope is not None) and (rope != "none"):
            raise ValueError(f"Wrong rope value: {rope}")
        else:
            self.rope = None
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
            attention_mask = self.sa_mask[:l, :l] if self.causal else None
        else:
            if attention_mask.ndim == 3:
                if attention_mask.shape[0] == b:
                    attention_mask = attention_mask.repeat_interleave(self.n_head, dim=0)
            if self.causal:
                sa_mask = self.sa_mask[:l, :l]
                if attention_mask.ndim == 3:
                    sa_mask = sa_mask[None]
                attention_mask = torch.logical_or(attention_mask, sa_mask)

        with no_mha_fast_path():
            with extended_transformer(self.encoder, cache_hiddens=(return_states == "full")) as encoder:
                outputs = encoder(embeddings.payload,
                                mask=attention_mask,
                                src_key_padding_mask=~embeddings.seq_len_mask.bool() if not self.causal else None,
                                is_causal=causal_hint,
                                rope=self.rope)  # (B, L, D).
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
        if self.rope is not None:
            with self.rope.cache(timestamps.payload):
                outputs, states = self.transform(embeddings, attention_mask=attention_mask)
        else:
            outputs, states = self.transform(embeddings, attention_mask=attention_mask)
        return outputs, states
