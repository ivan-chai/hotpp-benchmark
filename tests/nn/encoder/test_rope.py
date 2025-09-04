from unittest import TestCase, main

import torch
import itertools
from torch.nn import MultiheadAttention

from hotpp.nn.encoder.transformer.rope import TimeRoPEEncoding, MultiheadAttentionRoPE


class TestRoPE(TestCase):
    def test_rope(self):
        bs = 8
        l = 14
        dim = 32
        nh = 4
        max_dur = 10
        layer = MultiheadAttention(dim, nh, batch_first=True)
        layer_rope = MultiheadAttentionRoPE(dim, nh, batch_first=True)
        layer_rope.load_state_dict(layer.state_dict())

        q = torch.randn(bs, l, dim)
        k = torch.randn(bs, l, dim)
        v = torch.randn(bs, l, dim)
        key_padding_mask = torch.rand(bs, l) > 0.5  # (B, L).
        key_padding_mask[:, 0] = False
        attn_mask = (torch.rand(l, l) + torch.eye(l)) > 0.5  # (L, L).

        # O: (B, L, D).
        # W: (B, L, L).
        o_gt, w_gt = layer(q, k, v, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        o_gt.nan_to_num_(0, 0, 0)
        w_gt.nan_to_num_(0, 0, 0)

        # Test without RoPE.
        o2, w2 = layer_rope(q, k, v, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        o2.nan_to_num_(0, 0, 0)
        w2.nan_to_num_(0, 0, 0)
        assert o_gt.allclose(o2) and w_gt.allclose(w2)

        # Test with zero timestamps.
        timestamps = torch.zeros(bs, l)
        rope = TimeRoPEEncoding(dim // nh, l, max_dur)
        with rope.cache(timestamps):
            o2, w2 = layer_rope(q, k, v, key_padding_mask=key_padding_mask, attn_mask=attn_mask, rope=rope)
        o2.nan_to_num_(0, 0, 0)
        w2.nan_to_num_(0, 0, 0)
        assert o_gt.allclose(o2) and w_gt.allclose(w2)

        # Test with random timestamps, different result.
        timestamps = torch.rand(bs, l) * max_dur / 2
        with rope.cache(timestamps):
            o2, w2 = layer_rope(q, k, v, key_padding_mask=key_padding_mask, attn_mask=attn_mask, rope=rope)
        o2.nan_to_num_(0, 0, 0)
        w2.nan_to_num_(0, 0, 0)
        assert (not o2.allclose(o_gt)) and (not w2.allclose(w_gt))

        timestamps = timestamps + max_dur / 2  # Shift time, same result.
        with rope.cache(timestamps):
            o3, w3 = layer_rope(q, k, v, key_padding_mask=key_padding_mask, attn_mask=attn_mask, rope=rope)
        o3.nan_to_num_(0, 0, 0)
        w3.nan_to_num_(0, 0, 0)
        assert o2.allclose(o3, atol=1e-6) and w2.allclose(w3, atol=1e-6)

        timestamps = timestamps * 2  # Multiply time, different result.
        with rope.cache(timestamps):
            o3, w3 = layer_rope(q, k, v, key_padding_mask=key_padding_mask, attn_mask=attn_mask, rope=rope)
        o3.nan_to_num_(0, 0, 0)
        w3.nan_to_num_(0, 0, 0)
        assert (not o2.allclose(o3, atol=1e-6)) and (not w2.allclose(w3, atol=1e-6))


class TestMHA(TestCase):
    def test_mha(self):
        dim = 16
        for batch_first, group_size in itertools.product([True, False], [1, 2, 4]):
            kvdim = dim // group_size
            base = MultiheadAttentionRoPE(
                dim,  # embed_dim.
                4,   # num_heads.
                group_size=group_size,
                batch_first=batch_first,
                dropout=0
            )
            assert base._qkv_same_embed_dim
            split = MultiheadAttentionRoPE(
                dim,  # embed_dim.
                4,   # num_heads.
                group_size=group_size,
                batch_first=batch_first,
                dropout=0
            )
            split.load_state_dict(base.state_dict())
            split._qkv_same_embed_dim = False
            split.q_proj_weight = torch.nn.Parameter(split.in_proj_weight.data[:dim].clone())
            split.k_proj_weight = torch.nn.Parameter(split.in_proj_weight.data[dim:dim + kvdim].clone())
            split.v_proj_weight = torch.nn.Parameter(split.in_proj_weight.data[dim + kvdim:].clone())

            b, l, d = 18, 7, 16
            q = torch.randn(b, l, d)
            k = torch.randn(b, l, d)
            v = torch.randn(b, l, d)
            if not batch_first:
                q = q.transpose(0, 1)
                k = k.transpose(0, 1)
                v = v.transpose(0, 1)
            y_gt, w_gt = base(q, k, v, need_weights=False)
            self.assertTrue(w_gt is None)
            y, w_gt = base(q, k, v, need_weights=True)
            self.assertTrue(y.allclose(y_gt, atol=1e-6))
            y, w = split(q, k, v, need_weights=False)
            self.assertTrue(w is None)
            self.assertTrue(y.allclose(y_gt, atol=1e-6))
            y, w = split(q, k, v, need_weights=True)
            self.assertTrue(y.allclose(y_gt, atol=1e-6))
            self.assertTrue(w.allclose(w_gt, atol=1e-6))


if __name__ == "__main__":
    main()
