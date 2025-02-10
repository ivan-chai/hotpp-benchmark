import math
from unittest import TestCase, main

import torch

from hotpp.data import PaddedBatch
from hotpp.nn.encoder import Encoder
from hotpp.nn.encoder.transformer.state import TransformerState


class CumSumModel(torch.nn.Module):
    """Implement cumsum along time dimension."""
    def __init__(self):
        super().__init__()
        self.inter_token = 1

    @property
    def delta_time(self):
        """Whether to take delta time or raw timestamps at input."""
        # Need raw time for positional encoding.
        return False

    def forward(self, embeddings: PaddedBatch, times: PaddedBatch, return_states=False):
        """Encode input sequences with a causal mask.

        Args:
            embeddings: Input embeddings with shape (B, L, D).
            times: Input times with shape (B, L), absolute.

        Returns:
            Outputs with shape (B, L, D) and activations with shape (N, B, L, D).
        """
        outputs = torch.stack([times.payload, embeddings.payload[..., 1]], -1)
        states = outputs[None]  # (1, B, L, D).
        outputs = (outputs * embeddings.seq_len_mask.unsqueeze(2)).cumsum(1)  # (B, L, D).
        outputs = PaddedBatch(outputs, embeddings.seq_lens)
        if not return_states:
            states = None
        else:
            assert return_states == "full"
            states = TransformerState(times.payload, states, embeddings.seq_lens)
        return outputs, states


class SimpleEmbedder(torch.nn.Module):
    def forward(self, x):
        payload = torch.stack([x.payload["timestamps"], x.payload["labels"]], -1)
        return PaddedBatch(payload, x.seq_lens)


class SimpleSequenceEncoder(Encoder):
    def __init__(self, max_context=None, autoreg_batch_size=None):
        super(Encoder, self).__init__({})
        self.embedder = SimpleEmbedder()
        self.model = CumSumModel()
        self.max_context = max_context
        self.autoreg_batch_size = autoreg_batch_size


class TestEncoder(TestCase):
    def test_encoder(self):
        features = torch.tensor([
            [[0, 0],
             [1, 1],
             [2, 1],
             [4, 0],
             [6, 0]],
            [[2, 1],
             [3, 1],
             [5, 1],
             [0, 0],
             [0, 0]]
           ]).float()  # (B, T, D).
        b, l, d = features.shape
        s = 3
        lengths = torch.tensor([5, 3])
        batch = PaddedBatch({"timestamps": features[..., 0],
                             "labels": features[..., 1]}, lengths)
        indices = torch.tensor([
            [0, 1],
            [2, 0]  # Second is unused.
        ])
        indices = PaddedBatch(indices, torch.tensor([2, 1]))

        encoder = SimpleSequenceEncoder()

        # Test forward.
        fw_outputs, fw_states = encoder(batch, return_states="full")
        mask = batch.seq_len_mask.unsqueeze(-1)
        self.assertEqual(fw_states.shape, (1, b, l, d))
        self.assertTrue((fw_states.payload[0] * mask).allclose(features * mask))
        gt_outputs = features.cumsum(1)
        self.assertTrue((fw_outputs.payload * mask).allclose(gt_outputs * mask))

        # Test generation.
        # Deltas GT:
        #    [0, 0, 0]
        #    [1, 3, 8]
        #    [10, 25, 65]
        times_gt = torch.tensor([
            [0, 0, 0],
            [2, 5, 13],
            [15, 40, 105],
            [0, 0, 0]
        ]).reshape(2, 2, 3).float()  # (2, 2, 3).
        labels_gt = torch.tensor([
            [0, 0, 0],
            [1, 2, 4],
            [3, 6, 12],
            [0, 0, 0]
        ]).reshape(2, 2, 3)  # (2, 2, 3).
        def predict_fn(outputs, states):
            return PaddedBatch({
                "timestamps": outputs.payload[:, :, 0],
                "labels": outputs.payload[:, :, 1].long()
            }, outputs.seq_lens)
        for kwargs in [{}, {"autoreg_batch_size": 1}, {"autoreg_batch_size": indices.seq_lens.sum().item() // 2}]:
            encoder = SimpleSequenceEncoder(**kwargs)
            sequences = encoder.generate(batch, indices, predict_fn, n_steps=3)
            mask = indices.seq_len_mask  # (B, I).
            self.assertTrue(sequences.payload["timestamps"][mask].allclose(times_gt[mask]))
            self.assertTrue(sequences.payload["labels"][mask].allclose(labels_gt[mask]))

        # Test max_context. History of each prefix is truncated to this value.
        encoder = SimpleSequenceEncoder(max_context=2)
        sequences = encoder.generate(batch, indices, predict_fn, n_steps=3)
        mask = indices.seq_len_mask  # (B, I).
        # Deltas GT:
        #    [0, 0, 0]
        #    [1, 3, 8]
        #    [8, 21, 55]
        times_gt = torch.tensor([
            [0, 0, 0],
            [2, 5, 13],
            [13, 34, 89],
            [0, 0, 0]
        ]).reshape(2, 2, 3).float()  # (2, 2, 3).
        labels_gt = torch.tensor([
            [0, 0, 0],
            [1, 2, 4],
            [2, 4, 8],
            [0, 0, 0]
        ]).reshape(2, 2, 3)  # (2, 2, 3).
        self.assertTrue(sequences.payload["timestamps"][mask].allclose(times_gt[mask]))
        self.assertTrue(sequences.payload["labels"][mask].allclose(labels_gt[mask]))


if __name__ == "__main__":
    main()
