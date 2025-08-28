from unittest import TestCase, main

import torch
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


if __name__ == "__main__":
    main()
