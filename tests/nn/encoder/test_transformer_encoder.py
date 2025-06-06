import math
from unittest import TestCase, main

import torch

from hotpp.data import PaddedBatch
from hotpp.nn.encoder import TransformerEncoder
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
        outputs = (embeddings.payload * embeddings.seq_len_mask.unsqueeze(2)).cumsum(1)  # (B, L, D).
        states = embeddings.payload[None]  # (1, B, L, D).
        outputs = PaddedBatch(outputs, embeddings.seq_lens)
        if not return_states:
            states = None
        else:
            assert return_states == "full"
            states = TransformerState(times.payload, states, embeddings.seq_lens)
        return outputs, states

    def decode(self, embeddings: PaddedBatch, times: PaddedBatch, history_states: TransformerState):
        """Compute activations for queries with full mask.

        Args:
            embeddings: Input embeddings with shape (B, L', D), right-aligned.
            times: Times to predict with shape (B, L').
            history_states: Historical activations with shape (N, B, L, D).

        Returns:
            Outputs with shape (B, L', D), states with shape (N, B, L', D).
        """
        outputs = (history_states.payload[0] * history_states.seq_len_mask.unsqueeze(2)).sum(1)  # (B, D).
        outputs = embeddings.payload + outputs.unsqueeze(1)  # (B, L', D).
        outputs = PaddedBatch(outputs, embeddings.seq_lens)
        states = embeddings.payload[None]  # (1, B, L, D).
        states = TransformerState(times.payload, states, embeddings.seq_lens)
        return outputs, states

    def interpolate(self, times: PaddedBatch, history_states: TransformerState, last_history_index=None):
        """Compute activations for queries with full mask.

        Args:
            times: Times to predict with shape (B, L', S).
            history_states: Historical activations with shape (N, B, L, D).
            last_history_index: The last history state index used for prediction with shape (L').

        Returns:
            Outputs with shape (B, L', S, D).
        """
        b, l, s = times.payload.shape
        lh = history_states.payload.shape[2]
        masked_states = history_states.payload[0] * history_states.seq_len_mask.unsqueeze(2)  # (B, L, D).
        masked_states = masked_states.unsqueeze(1).expand(b, l, history_states.shape[2], history_states.payload.shape[-1])  # (B, L', L, D).
        if last_history_index is not None:
            if not (history_states.index[1:] == history_states.index[:1]).all():
                raise NotImplementedError("Need uniform index for attention mask during interpolation.")
            last_history_index = history_states.index[0].take_along_dim(last_history_index, 0)  # (L').
            attn_mask = last_history_index[:, None] >= torch.arange(lh, device=history_states.device)  # (L', L).
            masked_states = masked_states * attn_mask[None, :, :, None]
        outputs = masked_states.sum(2)  # (B, L', D).
        outputs = self.inter_token + outputs.unsqueeze(2).expand(b, l, s, outputs.shape[-1])  # (B, L', S, D).
        outputs = PaddedBatch(outputs, times.seq_lens)
        return outputs


class SimpleEmbedder(torch.nn.Module):
    def forward(self, x):
        payload = torch.stack([x.payload["timestamps"], x.payload["labels"]], -1)
        return PaddedBatch(payload, x.seq_lens)


class SimpleSequenceEncoder(TransformerEncoder):
    def __init__(self, max_context=None, autoreg_batch_size=None):
        super(TransformerEncoder, self).__init__({})
        self.embedder = SimpleEmbedder()
        self.transformer = CumSumModel()
        self.max_context = max_context
        self.autoreg_batch_size = autoreg_batch_size


class TestTransformerEncoder(TestCase):
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

        # Test interpoloate.
        time_deltas = PaddedBatch(torch.zeros(b, l, s), lengths)
        int_outputs = encoder.interpolate(fw_states, time_deltas)  # (B, L, S, D).
        gt_outputs = fw_outputs.payload + encoder.transformer.inter_token  # (B, L, D).
        gt_outputs = gt_outputs.unsqueeze(2).repeat(1, 1, s, 1)  # (B, L, S, D).
        self.assertTrue((int_outputs.payload * mask.unsqueeze(2)).allclose(gt_outputs * mask.unsqueeze(2)))

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

        # Test max_context. The first token uses full history, while the remaining uses only recent history.
        encoder = SimpleSequenceEncoder(max_context=2)
        sequences = encoder.generate(batch, indices, predict_fn, n_steps=3)
        mask = indices.seq_len_mask  # (B, I).
        # Deltas GT:
        #    [0, 0, 0]
        #    [1, 3, 8]
        #    [10, 23, 61]
        times_gt = torch.tensor([
            [0, 0, 0],
            [2, 5, 13],
            [15, 38, 99],
            [0, 0, 0]
        ]).reshape(2, 2, 3).float()  # (2, 2, 3).
        labels_gt = torch.tensor([
            [0, 0, 0],
            [1, 2, 4],
            [3, 5, 10],
            [0, 0, 0]
        ]).reshape(2, 2, 3)  # (2, 2, 3).
        self.assertTrue(sequences.payload["timestamps"][mask].allclose(times_gt[mask]))
        self.assertTrue(sequences.payload["labels"][mask].allclose(labels_gt[mask]))


if __name__ == "__main__":
    main()
