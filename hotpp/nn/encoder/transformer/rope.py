import torch
from torch import Tensor
from torch.nn import functional as F
from torch.nn.functional import *
from torch.nn.functional import _canonical_mask, _none_or_dtype
from torch.nn.modules.activation import _arg_requires_grad, _is_make_fx_tracing
from typing import Optional


class TimeRoPEEncoding(torch.nn.Module):
    """RoPE computer.

    Usage:
    ```
    sa = MultiheadAttentionRoPE(...)
    rope = TimeRoPEEncoding(...)
    with rope.cache(timestamps):
        x = sa(q, k, v, rope=rope)
    ```
    """
    def __init__(self, head_dim, n_positions, max_duration, min_time_step=None, trainable=False):
        if head_dim % 2 != 0:
            raise ValueError("Head dim must be divisible by 2")
        super().__init__()
        if min_time_step is None:
            min_time_step = max_duration / n_positions
        pe = torch.exp(torch.arange(0, head_dim, 2) * (-math.log(5 * max_duration / min_time_step) / head_dim)) / min_time_step  # (D // 2).
        if trainable:
            self.pe = torch.nn.Parameter(pe)
        else:
            self.register_buffer("pe", pe, persistent=False)

    def __enter__(self):
        if not hasattr(self, "_cache"):
            raise RuntimeError("Context must be created after computing the cache.")

    def __exit__(self, exc_type, exc_val, exc_tb):
        del self._cache

    def get_sin_cos(self, timestamps):
        args = timestamps.unsqueeze(-1) * self.pe  # (B, L, D // 2).
        # Use interleave instead of concatenation to achieve correct processing with multiple attention heads.
        sin, cos = torch.sin(args), torch.cos(args)  # (B, L, D // 2).
        return sin, cos

    def cache(self, timestamps):
        """Precompute sines and cosines."""
        if hasattr(self, "_cache"):
            raise RuntimeError("Nested caching.")
        self._cache = self.get_sin_cos(timestamps)
        return self

    def forward(self, q, k, timestamps=None):
        """Apply RoPE.

        Args:
            q: Queries with shape (B, H, L, D).
            k: Keys with shape (B, H, L, D).
            timestamps: Timestamps with shape (B, L). Use cached values, if not provided.

        Returns:
            New query and key values.
        """
        if timestamps is None:
            if hasattr(self, "_cache"):
                sin, cos = self._cache
            else:
                raise RuntimeError("Forward outside context")
        else:
            sin, cos = self.get_sin_cos(timestamps)
        sin = sin.unsqueeze(1)  # (B, 1, L, D // 2).
        cos = cos.unsqueeze(1)  # (B, 1, L, D // 2).
        result = []
        for v in [q, k]:
            b, h, l, d = v.shape
            v = v.reshape(b, h, l, d // 2, 2)
            v0 = v[..., 0]
            v1 = v[..., 1]
            v = torch.stack([
                cos * v0 - sin * v1,
                sin * v0 + cos * v1
            ], -1)  # (B, H, L, D // 2, 2).
            result.append(v.reshape(b, h, l, d))
        return tuple(result)


def _in_projection(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    w_q: Tensor,
    w_k: Tensor,
    w_v: Tensor,
    b_q: Optional[Tensor] = None,
    b_k: Optional[Tensor] = None,
    b_v: Optional[Tensor] = None,
):
    return linear(q, w_q, b_q), linear(k, w_k, b_k), linear(v, w_v, b_v)


def _in_projection_packed(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    w: Tensor,
    b: Optional[Tensor] = None,
):
    E = q.size(-1)
    Ekv = (w.shape[0] - E) // 2
    if k is v:
        if q is k:
            # self-attention
            proj = linear(q, w, b)  # (L, B, E + 2 * Ekv).
            proj_q, proj_kv = proj.split([E, 2 * Ekv], -1)
            proj_kv = (
                proj_kv.unflatten(-1, (2, Ekv))
                .unsqueeze(0)
                .transpose(0, -2)
                .squeeze(-2)
                .contiguous()
            )
            return proj_q.contiguous(), proj_kv[0], proj_kv[1]
        else:
            # encoder-decoder attention
            w_q, w_kv = w.split([E, Ekv * 2])
            if b is None:
                b_q = b_kv = None
            else:
                b_q, b_kv = b.split([E, Ekv * 2])
            q_proj = linear(q, w_q, b_q)
            kv_proj = linear(k, w_kv, b_kv)
            # reshape to 2, E and not E, 2 is deliberate for better memory coalescing and keeping same order as chunk()
            kv_proj = (
                kv_proj.unflatten(-1, (2, Ekv))
                .unsqueeze(0)
                .transpose(0, -2)
                .squeeze(-2)
                .contiguous()
            )
            return (q_proj, kv_proj[0], kv_proj[1])
    else:
        w_q, w_k, w_v = w.split([E, Ekv, Ekv])
        if b is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = b.split([E, Ekv, Ekv])
        return linear(q, w_q, b_q), linear(k, w_k, b_k), linear(v, w_v, b_v)


def multi_head_attention_rope_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    embed_dim_to_check: int,
    num_heads: int,
    in_proj_weight: Optional[Tensor],
    in_proj_bias: Optional[Tensor],
    bias_k: Optional[Tensor],
    bias_v: Optional[Tensor],
    add_zero_attn: bool,
    dropout_p: float,
    out_proj_weight: Tensor,
    out_proj_bias: Optional[Tensor],
    training: bool = True,
    key_padding_mask: Optional[Tensor] = None,
    need_weights: bool = True,
    attn_mask: Optional[Tensor] = None,
    use_separate_proj_weight: bool = False,
    q_proj_weight: Optional[Tensor] = None,
    k_proj_weight: Optional[Tensor] = None,
    v_proj_weight: Optional[Tensor] = None,
    average_attn_weights: bool = True,
    is_causal: bool = False,
    rope: Optional[TimeRoPEEncoding] = None,
    enable_gqa: bool = False,
) -> tuple[Tensor, Optional[Tensor]]:
    if use_separate_proj_weight:
        kvdim = k_proj_weight.shape[0]
    else:
        assert (in_proj_weight.shape[0] - embed_dim_to_check) % 2 == 0
        kvdim = (in_proj_weight.shape[0] - embed_dim_to_check) // 2
    head_dim = embed_dim_to_check // num_heads
    assert kvdim % head_dim == 0
    num_kv_heads = kvdim // head_dim
    assert num_heads % num_kv_heads == 0
    group_size = num_heads // num_kv_heads
    torch_version = list(map(int, torch.__version__.split(".")[:2]))
    if (rope is None) and (group_size == 1) and ((not enable_gqa) or (torch_version >= (2, 5))):
        kwargs = {}
        if enable_gqa:
            kwargs["enable_gqa"] = enable_gqa
        return multi_head_attention_forward(
            query,
            key,
            value,
            embed_dim_to_check,
            num_heads,
            in_proj_weight,
            in_proj_bias,
            bias_k,
            bias_v,
            add_zero_attn,
            dropout_p,
            out_proj_weight,
            out_proj_bias,
            training=training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            use_separate_proj_weight=use_separate_proj_weight,
            q_proj_weight=q_proj_weight,
            k_proj_weight=k_proj_weight,
            v_proj_weight=v_proj_weight,
            average_attn_weights=average_attn_weights,
            is_causal=is_causal,
            **kwargs
        )

    if query.ndim != 3:
        raise NotImplementedError("Need batched input")

    # set up shape vars
    tgt_len, bsz, embed_dim = query.shape  # (L, B, D).
    src_len, _, _ = key.shape

    key_padding_mask = _canonical_mask(
        mask=key_padding_mask,
        mask_name="key_padding_mask",
        other_type=_none_or_dtype(attn_mask),
        other_name="attn_mask",
        target_type=query.dtype,
    )

    if is_causal and attn_mask is None:
        raise RuntimeError(
            "Need attn_mask if specifying the is_causal hint. "
            "You may use the Transformer module method "
            "`generate_square_subsequent_mask` to create this mask."
        )

    if is_causal and key_padding_mask is None and not need_weights:
        # when we have a kpm or need weights, we need attn_mask
        # Otherwise, we use the is_causal hint go as is_causal
        # indicator to SDPA.
        attn_mask = None
    else:
        attn_mask = _canonical_mask(
            mask=attn_mask,
            mask_name="attn_mask",
            other_type=None,
            other_name="",
            target_type=query.dtype,
            check_other=False,
        )

        if key_padding_mask is not None:
            # We have the attn_mask, and use that to merge kpm into it.
            # Turn off use of is_causal hint, as the merged mask is no
            # longer causal.
            is_causal = False

    assert (
        embed_dim == embed_dim_to_check
    ), f"was expecting embedding dimension of {embed_dim_to_check}, but got {embed_dim}"
    if isinstance(embed_dim, torch.Tensor):
        # embed_dim can be a tensor when JIT tracing
        head_dim = embed_dim.div(num_heads, rounding_mode="trunc")
    else:
        head_dim = embed_dim // num_heads
    assert (
        head_dim * num_heads == embed_dim
    ), f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"
    if use_separate_proj_weight:
        # allow MHA to have different embedding dimensions when separate projection weights are used
        assert (
            key.shape[:2] == value.shape[:2]
        ), f"key's sequence and batch dims {key.shape[:2]} do not match value's {value.shape[:2]}"
    else:
        assert (
            key.shape == value.shape
        ), f"key shape {key.shape} does not match value shape {value.shape}"

    #
    # compute in-projection
    #
    if not use_separate_proj_weight:
        assert (
            in_proj_weight is not None
        ), "use_separate_proj_weight is False but in_proj_weight is None"
        q, k, v = _in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)
    else:
        assert (
            q_proj_weight is not None
        ), "use_separate_proj_weight is True but q_proj_weight is None"
        assert (
            k_proj_weight is not None
        ), "use_separate_proj_weight is True but k_proj_weight is None"
        assert (
            v_proj_weight is not None
        ), "use_separate_proj_weight is True but v_proj_weight is None"
        if in_proj_bias is None:
            b_q = b_k = b_v = None
        else:
            b_q = in_proj_bias[:embed_dim]
            b_k, b_v = in_proj_bias[embed_dim:].chunk(2)

        q, k, v = _in_projection(
            query,
            key,
            value,
            q_proj_weight,
            k_proj_weight,
            v_proj_weight,
            b_q,
            b_k,
            b_v,
        )

    # prep attention mask

    if attn_mask is not None:
        # ensure attn_mask's dim is 3
        if attn_mask.dim() == 2:
            correct_2d_size = (tgt_len, src_len)
            if attn_mask.shape != correct_2d_size:
                raise RuntimeError(
                    f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}."
                )
            attn_mask = attn_mask.unsqueeze(0)
        elif attn_mask.dim() == 3:
            correct_3d_size = (bsz * num_heads, tgt_len, src_len)
            if attn_mask.shape != correct_3d_size:
                raise RuntimeError(
                    f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}."
                )
        else:
            raise RuntimeError(
                f"attn_mask's dimension {attn_mask.dim()} is not supported"
            )

    # add bias along batch dimension (currently second)
    if bias_k is not None and bias_v is not None:
        k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
        v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
        if attn_mask is not None:
            attn_mask = pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = pad(key_padding_mask, (0, 1))
    else:
        assert bias_k is None
        assert bias_v is None

    #
    # reshape q, k, v for multihead attention and make them batch first
    #
    q = q.view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)  # (BH, L, D).
    k = k.view(k.shape[0], bsz * num_kv_heads, head_dim).transpose(0, 1)  # (BH, L, D).
    v = v.view(v.shape[0], bsz * num_kv_heads, head_dim).transpose(0, 1)  # (BH, L, D).

    # add zero attention along batch dimension (now first)
    if add_zero_attn:
        zero_attn_shape = (bsz * num_kv_heads, 1, head_dim)
        k = torch.cat(
            [k, torch.zeros(zero_attn_shape, dtype=k.dtype, device=k.device)], dim=1
        )
        v = torch.cat(
            [v, torch.zeros(zero_attn_shape, dtype=v.dtype, device=v.device)], dim=1
        )
        if attn_mask is not None:
            attn_mask = pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = pad(key_padding_mask, (0, 1))

    # update source sequence length after adjustments
    src_len = k.size(1)

    # merge key padding and attention masks
    if key_padding_mask is not None:
        key_padding_mask = (
            key_padding_mask.view(bsz, 1, 1, src_len)
            .expand(-1, num_heads, -1, -1)
            .reshape(bsz * num_heads, 1, src_len)
        )
        if attn_mask is None:
            attn_mask = key_padding_mask
        else:
            attn_mask = attn_mask + key_padding_mask

    # adjust dropout probability
    if not training:
        dropout_p = 0.0

    # Apply RoPE.
    if rope is not None:
        # q, k: (B * H, L, D).
        q = q.view(bsz, num_heads, tgt_len, head_dim)
        k = k.view(bsz, num_kv_heads, tgt_len, head_dim)
        q, k = rope(q, k)
        q = q.view(bsz * num_heads, tgt_len, head_dim)
        k = k.view(bsz * num_kv_heads, tgt_len, head_dim)

    #
    # (deep breath) calculate attention and out projection
    #

    if need_weights:
        _B, _Nt, E = q.shape
        q_scaled = q * math.sqrt(1.0 / float(E))
        if group_size > 1:
            k = k.repeat_interleave(group_size, -3)
            v = v.repeat_interleave(group_size, -3)

        assert not (
            is_causal and attn_mask is None
        ), "FIXME: is_causal not implemented for need_weights"

        if attn_mask is not None:
            attn_output_weights = torch.baddbmm(
                attn_mask, q_scaled, k.transpose(-2, -1)
            )
        else:
            attn_output_weights = torch.bmm(q_scaled, k.transpose(-2, -1))
        attn_output_weights = softmax(attn_output_weights, dim=-1)
        if dropout_p > 0.0:
            attn_output_weights = dropout(attn_output_weights, p=dropout_p)

        attn_output = torch.bmm(attn_output_weights, v)

        attn_output = (
            attn_output.transpose(0, 1).contiguous().view(tgt_len * bsz, embed_dim)
        )
        attn_output = linear(attn_output, out_proj_weight, out_proj_bias)
        attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))

        # optionally average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        if average_attn_weights:
            attn_output_weights = attn_output_weights.mean(dim=1)

        return attn_output, attn_output_weights
    else:
        # attn_mask can be either (L,S) or (N*num_heads, L, S)
        # if attn_mask's shape is (1, L, S) we need to unsqueeze to (1, 1, L, S)
        # in order to match the input for SDPA of (N, num_heads, L, S)
        if attn_mask is not None:
            if attn_mask.size(0) == 1 and attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(0)
            else:
                attn_mask = attn_mask.view(bsz, num_heads, -1, src_len)

        q = q.view(bsz, num_heads, tgt_len, head_dim)
        k = k.view(bsz, num_kv_heads, src_len, head_dim)
        v = v.view(bsz, num_kv_heads, src_len, head_dim)

        kwargs = {}
        if enable_gqa:
            # Need PyTorch > 2.5.0.
            kwargs["enable_gqa"] = True

        attn_output = scaled_dot_product_attention(
            q, k, v, attn_mask, dropout_p, is_causal,
            **kwargs
        )
        attn_output = (
            attn_output.permute(2, 0, 1, 3).contiguous().view(bsz * tgt_len, embed_dim)
        )

        attn_output = linear(attn_output, out_proj_weight, out_proj_bias)
        attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))
        return attn_output, None


class MultiheadAttentionRoPE(torch.nn.MultiheadAttention):
    def __init__(self,
                 embed_dim,
                 num_heads,
                 *,
                 group_size=1,
                 device=None,
                 dtype=None,
                 **kwargs
        ):
        super().__init__(embed_dim, num_heads,
                         device=device, dtype=dtype,
                         **kwargs)
        self.group_size = group_size
        if group_size > 1:
            assert embed_dim % num_heads == 0
            head_dim = embed_dim // num_heads
            assert num_heads % group_size == 0
            num_kv_heads = num_heads // group_size
            kvdim = head_dim * num_kv_heads

            factory_kwargs = {"device": device, "dtype": dtype}
            if self._qkv_same_embed_dim:
                self.in_proj_weight = torch.nn.Parameter(
                    torch.empty((embed_dim + 2 * kvdim, embed_dim), **factory_kwargs)
                )
            else:
                self.k_proj_weight = torch.nn.Parameter(
                    torch.empty((kvdim, self.kdim), **factory_kwargs)
                )
                self.v_proj_weight = torch.nn.Parameter(
                    torch.empty((kvdim, self.vdim), **factory_kwargs)
                )
            if self.in_proj_bias is not None:
                self.in_proj_bias = torch.nn.Parameter(torch.empty(embed_dim + 2 * kvdim, **factory_kwargs))
            self._reset_parameters()

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
        rope=None
    ) -> tuple[Tensor, Optional[Tensor]]:
        rope = rope if rope is not None else getattr(self, "_rope", [None])[0]
        if (rope is None) and (self.group_size == 1):
            return super().forward(query, key, value,
                                   key_padding_mask=key_padding_mask,
                                   need_weights=need_weights,
                                   attn_mask=attn_mask,
                                   average_attn_weights=average_attn_weights,
                                   is_causal=is_causal)
        if query.dim() != 3:
            if self.batch_first:
                raise ValueError("Expected batched input with shape (B, L, D).")
            else:
                raise ValueError("Expected batched input with shape (L, B, D).")

        key_padding_mask = F._canonical_mask(
            mask=key_padding_mask,
            mask_name="key_padding_mask",
            other_type=F._none_or_dtype(attn_mask),
            other_name="attn_mask",
            target_type=query.dtype,
        )

        attn_mask = F._canonical_mask(
            mask=attn_mask,
            mask_name="attn_mask",
            other_type=None,
            other_name="",
            target_type=query.dtype,
            check_other=False,
        )

        any_nested = query.is_nested or key.is_nested or value.is_nested
        assert not any_nested, (
            "MultiheadAttention does not support NestedTensor outside of its fast path. "
        )

        if self.batch_first:
            # make sure that the transpose op does not affect the "is" property
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = (x.transpose(1, 0) for x in (query, key))
                    value = key
            else:
                query, key, value = (x.transpose(1, 0) for x in (query, key, value))

        # QKV: (L, B, D).

        attn_output, attn_output_weights = multi_head_attention_rope_forward(
            query,
            key,
            value,
            self.embed_dim,
            self.num_heads,
            self.in_proj_weight,
            self.in_proj_bias,
            self.bias_k,
            self.bias_v,
            self.add_zero_attn,
            self.dropout,
            self.out_proj.weight,
            self.out_proj.bias,
            training=self.training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            use_separate_proj_weight=not self._qkv_same_embed_dim,
            q_proj_weight=self.q_proj_weight,
            k_proj_weight=self.k_proj_weight,
            v_proj_weight=self.v_proj_weight,
            average_attn_weights=average_attn_weights,
            is_causal=is_causal,
            rope=rope,
            enable_gqa=self.group_size > 1
        )
        if self.batch_first:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights
