import logging
from unittest import TestCase, main

import torch
from torch import Tensor

from esp_horizon.data import PaddedBatch
from esp_horizon.modules.next_item.autoreg import RNNSequencePredictor, NextItemRNNAdapter, BaseAdapter, BaseRNNAdapter


class SimpleRNNAdapter(BaseRNNAdapter):
    """Model input and output are time deltas."""
    def __init__(self):
        model = torch.nn.Identity()
        super().__init__(model, time_field="timestamps")

    def eval_states(self, x):
        # Hiddens are equal to previous inputs or zeros for the first step.
        init_hidden = torch.zeros_like(x.payload["labels"][:, :1, None])
        hiddens = torch.cat([init_hidden, x.payload["labels"][:, :-1, None]], dim=1)
        return PaddedBatch(hiddens, x.seq_lens)

    def forward(self, x, states):
        new_states = x.payload["labels"][:, -1] + 1
        new_x = {
            "timestamps": 1 - 0.5 * x.payload["timestamps"][:, -1].long(),
            "labels": new_states
        }
        return new_x, new_states


class TestSequencePredictor(TestCase):
    def test_sequence_predictor(self):
        # The first feature is time.
        features = torch.tensor([
            [[0, 1],
             [1, 2],
             [2, 3],
             [4, 4],
             [6, 5]],
            [[2, 6],
             [3, 7],
             [5, 8],
             [0, 0],
             [0, 0]]
        ]).float()  # (B, T, D).
        lengths = torch.tensor([5, 3])
        batch = {"labels": features[..., 0], "timestamps": features[..., 1]}
        batch = PaddedBatch(batch, lengths)
        indices = torch.tensor([
            [0, 1],
            [1, 0]  # Second is unused.
        ])
        indices = PaddedBatch(indices, torch.tensor([2, 1]))

        # The model predicts hidden state + 1
        features1_gt = torch.tensor([
            [[1, 2],
             [2, 2.5],
             [3, 3.5]],
            [[2, 2.5],
             [3, 3.5],
             [4, 4]]
        ])

        features2_gt = torch.tensor([
            [[4, 7.5],
             [5, 8.5],
             [6, 9]],
        ])

        rnn_predictor = RNNSequencePredictor(SimpleRNNAdapter(), max_steps=3)
        for predictor in [rnn_predictor]:
            results = predictor(batch, indices)

            self.assertEqual(results.shape, (2, 2))
            self.assertEqual(results.seq_lens.tolist(), [2, 1])
            self.assertEqual(results.payload["labels"][0, :2].tolist(), features1_gt[..., 0].tolist())
            self.assertEqual(results.payload["timestamps"][0, :2].tolist(), features1_gt[..., 1].tolist())
            self.assertEqual(results.payload["labels"][1, :1].tolist(), features2_gt[..., 0].tolist())
            self.assertEqual(results.payload["timestamps"][1, :1].tolist(), features2_gt[..., 1].tolist())


if __name__ == "__main__":
    main()
