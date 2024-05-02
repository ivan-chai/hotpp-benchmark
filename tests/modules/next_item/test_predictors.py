import logging
from unittest import TestCase, main

import torch
from torch import Tensor

from esp_horizon.data import PaddedBatch
from esp_horizon.modules.next_item.autoreg import RNNSequencePredictor, NextItemRNNAdapter, BaseAdapter, BaseRNNAdapter


class SimpleRNNAdapter(BaseRNNAdapter):
    def __init__(self):
        model = torch.nn.Identity()
        super().__init__(model, time_field="time_delta")

    def eval_states(self, x):
        # Hiddens are equal to previous inputs or zeros for the first step.
        init_hidden = torch.zeros_like(x.payload["features"][:, :1])
        hiddens = torch.cat([init_hidden, x.payload["features"][:, :-1]], dim=1)
        return PaddedBatch(hiddens, x.seq_lens)

    def forward(self, x, states):
        # Predict input + 1.
        new_states = x.payload["features"][:, -1] + 1
        new_x = {
            "features": new_states,
            "time_delta": 0.5 + 0.5 * (x.payload["features"][:, -1, 0].long() % 2)
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
        batch = {"features": features, "time_delta": torch.zeros_like(features[..., 0])}
        batch = PaddedBatch(batch, lengths)
        indices = torch.tensor([
            [0, 1],
            [1, 0]  # Second is unused.
        ])
        indices = PaddedBatch(indices, torch.tensor([2, 1]))

        # The model predicts hidden state + 1 and time delta equal to 0.5 + 0.5 * (time % 2)
        # Time deltas for item 0 index 0: 0.5, 1, 0.5, (stop) 1 ...
        # Time deltas for item 0 index 1: 1, 0.5, (stop) 1 ...
        # Time deltas for item 1 index 0: 1, 0.5, (stop) 1 ...
        r1_gt = [
            [[1, 2],
             [2, 3],
             [3, 4]],
            [[2, 3],
             [3, 4],
             [4, 5]]
        ]

        r2_gt = [
            [[4, 8],
             [5, 9],
             [6, 10]],
        ]

        rnn_predictor = RNNSequencePredictor(SimpleRNNAdapter(), max_steps=3)
        for predictor in [rnn_predictor]:
            results = predictor(batch, indices)

            self.assertEqual(len(results), 2)
            r1, r2 = results
            self.assertEqual(len(r1), 2)
            self.assertEqual(r1.seq_lens.tolist(), [3, 3])
            self.assertEqual(r1.payload["features"][0].tolist(), r1_gt[0])
            self.assertEqual(r1.payload["features"][1].tolist(), r1_gt[1])
            self.assertEqual(len(r2), 1)
            self.assertEqual(r2.seq_lens.tolist(), [3])
            self.assertEqual(r2.payload["features"][0].tolist(), r2_gt[0])


if __name__ == "__main__":
    main()
