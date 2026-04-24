from unittest import TestCase, main

import torch

from hotpp.data import PaddedBatch
from hotpp.nn.head.transformer_decoder import TransformerDecoderHead


def make_batch(B=4, L=10, D=64, seq_lens=None):
    if seq_lens is None:
        seq_lens = torch.tensor([10, 8, 6, 4])
    payload = torch.randn(B, L, D)
    return PaddedBatch(payload, seq_lens)


def make_indices(B=4, L=10, positions=None, seq_lens=None):
    """Create a PaddedBatch of indices as used during training."""
    if positions is None:
        positions = [2, 4, 6]
    idx = torch.tensor([positions] * B)  # (B, I)
    if seq_lens is None:
        seq_lens = torch.full((B,), len(positions))
    full_mask = torch.ones(B, len(positions), dtype=torch.bool)
    return PaddedBatch({"index": idx, "full_mask": full_mask}, seq_lens)


class TestOutputShape(TestCase):
    """Forward pass returns correct tensor shapes."""

    def _check_shapes(self, use_self_attention, query_size=None):
        B, L, D, K, P = 4, 10, 64, 8, 18
        head = TransformerDecoderHead(
            input_size=D, output_size=K * P, k=K,
            query_size=query_size, n_heads=4, n_layers=2,
            use_self_attention=use_self_attention,
        )
        x = make_batch(B, L, D)

        # Val/test path.
        out = head(x)
        self.assertEqual(out.payload.shape, (B, L, K * P))
        self.assertTrue(out.seq_lens.equal(x.seq_lens))

        # Train path.
        indices = make_indices(B, L, positions=[2, 4, 6])
        out_train = head(x, indices=indices)
        self.assertEqual(out_train.payload.shape, (B, 3, K * P))
        self.assertTrue(out_train.seq_lens.equal(indices.seq_lens))

    def test_shape_with_self_attention(self):
        self._check_shapes(use_self_attention=True)

    def test_shape_without_self_attention(self):
        self._check_shapes(use_self_attention=False)

    def test_shape_with_input_proj(self):
        # query_size != input_size → triggers input_proj
        self._check_shapes(use_self_attention=True, query_size=32)


class TestCausality(TestCase):
    """Output at position t must not depend on positions j > t."""

    def _check_causality(self, use_self_attention):
        B, L, D, K, P = 2, 8, 16, 4, 6
        head = TransformerDecoderHead(
            input_size=D, output_size=K * P, k=K,
            n_heads=2, n_layers=1,
            use_self_attention=use_self_attention,
        )
        head.eval()

        seq_lens = torch.tensor([8, 8])
        payload = torch.randn(B, L, D)
        x = PaddedBatch(payload.clone(), seq_lens)
        out_orig = head(x).payload  # (B, L, K*P)

        # Corrupt positions t+1..L-1 for t=3 and check output at t=3 unchanged.
        t = 3
        payload_corrupted = payload.clone()
        payload_corrupted[:, t + 1:, :] = torch.randn(B, L - t - 1, D) * 100
        x_corrupted = PaddedBatch(payload_corrupted, seq_lens)
        out_corrupted = head(x_corrupted).payload

        self.assertTrue(
            out_orig[:, :t + 1, :].allclose(out_corrupted[:, :t + 1, :], atol=1e-5),
            "Output at positions ≤ t changed after corrupting future positions"
        )

    def test_causality_with_self_attention(self):
        self._check_causality(use_self_attention=True)

    def test_causality_without_self_attention(self):
        self._check_causality(use_self_attention=False)


class TestPaddingInvariance(TestCase):
    """Output at valid positions must not depend on padding values."""

    def test_padding_invariance(self):
        B, L, D, K, P = 3, 10, 16, 4, 6
        head = TransformerDecoderHead(
            input_size=D, output_size=K * P, k=K,
            n_heads=2, n_layers=1,
        )
        head.eval()

        seq_lens = torch.tensor([5, 7, 3])
        payload = torch.randn(B, L, D)

        x1 = PaddedBatch(payload.clone(), seq_lens)
        out1 = head(x1).payload  # (B, L, K*P)

        # Replace padding region with large random values.
        payload_corrupted = payload.clone()
        for b in range(B):
            payload_corrupted[b, seq_lens[b]:, :] = torch.randn(L - seq_lens[b], D) * 100

        x2 = PaddedBatch(payload_corrupted, seq_lens)
        out2 = head(x2).payload

        # Only check positions within valid sequence lengths.
        for b in range(B):
            sl = seq_lens[b].item()
            self.assertTrue(
                out1[b, :sl, :].allclose(out2[b, :sl, :], atol=1e-5),
                f"Batch item {b}: output changed at valid positions after corrupting padding"
            )


class TestTrainValConsistency(TestCase):
    """Train path (with indices) and val path (without indices) must agree."""

    def _check_consistency(self, use_self_attention):
        B, L, D, K, P = 4, 10, 16, 4, 6
        head = TransformerDecoderHead(
            input_size=D, output_size=K * P, k=K,
            n_heads=2, n_layers=1,
            use_self_attention=use_self_attention,
        )
        head.eval()

        seq_lens = torch.tensor([10, 9, 8, 7])
        payload = torch.randn(B, L, D)
        x = PaddedBatch(payload, seq_lens)

        # Val path: output for all positions.
        out_val = head(x).payload  # (B, L, K*P)

        # Train path: select a subset of positions.
        positions = [1, 3, 5]
        indices = make_indices(B, L, positions=positions)
        out_train = head(x, indices=indices).payload  # (B, 3, K*P)

        for i, t in enumerate(positions):
            self.assertTrue(
                out_val[:, t, :].allclose(out_train[:, i, :], atol=1e-5),
                f"Mismatch between val and train path at position t={t}"
            )

    def test_consistency_with_self_attention(self):
        self._check_consistency(use_self_attention=True)

    def test_consistency_without_self_attention(self):
        self._check_consistency(use_self_attention=False)


class TestGradientFlow(TestCase):
    """Gradients must flow to queries and decoder parameters in both paths."""

    def _check_gradients(self, use_indices):
        B, L, D, K, P = 2, 6, 16, 4, 6
        head = TransformerDecoderHead(
            input_size=D, output_size=K * P, k=K,
            n_heads=2, n_layers=1,
        )

        seq_lens = torch.tensor([6, 5])
        payload = torch.randn(B, L, D)
        x = PaddedBatch(payload, seq_lens)

        if use_indices:
            indices = make_indices(B, L, positions=[1, 3])
            out = head(x, indices=indices).payload
        else:
            out = head(x).payload

        loss = out.sum()
        loss.backward()

        self.assertIsNotNone(head.queries.grad, "No gradient for queries")
        self.assertFalse(
            head.queries.grad.eq(0).all(),
            "Gradient for queries is all zeros"
        )

        for name, param in head.decoder.named_parameters():
            self.assertIsNotNone(param.grad, f"No gradient for decoder.{name}")

    def test_gradients_val_path(self):
        self._check_gradients(use_indices=False)

    def test_gradients_train_path(self):
        self._check_gradients(use_indices=True)


if __name__ == "__main__":
    main()
