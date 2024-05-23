import logging
from unittest import TestCase, main

import torch

from esp_horizon.data import PaddedBatch
from esp_horizon.modules import BaseModule, NextItemModule
from esp_horizon.nn.encoder import RnnEncoder
from esp_horizon.nn.encoder.autoreg import autoreg_prepare_features, autoreg_revert_features, autoreg_output_to_next_input_inplace


class SimpleEmbedder(torch.nn.Module):
    def forward(self, x):
        payload = torch.stack([x.payload["timestamps"], x.payload["labels"]], -1)
        return PaddedBatch(payload, x.seq_lens)


class SimpleRNN(torch.nn.GRU):
    def __init__(self):
        super().__init__(1, 1)

    def forward(self, x, h_0=None):
        return x * 2 + 1, None


class SimpleSequenceEncoder(RnnEncoder):
    num_layers = 1
    _max_context = None
    _context_step = None

    def __init__(self):
        super(RnnEncoder, self).__init__({})
        self.embedder = SimpleEmbedder()
        self.rnn = SimpleRNN()


class SimpleLoss:
    def predict_next(self, embeddings):
        return PaddedBatch({
            "timestamps": embeddings.payload[:, :, 0].long(),
            "labels": embeddings.payload[:, :, 1].long()
        }, embeddings.seq_lens)


class SimpleModule(NextItemModule):
    def __init__(self):
        super(BaseModule, self).__init__()
        self.seq_encoder = SimpleSequenceEncoder()
        self._head = None
        self._autoreg_max_steps = 2
        self.loss = SimpleLoss()


class TestAutoreg(TestCase):
    def assertEqualBatch(self, batch1, batch2):
        self.assertEqual(len(batch1), len(batch2))
        self.assertEqual(set(batch1.payload), set(batch2.payload))
        self.assertTrue((batch1.seq_lens == batch2.seq_lens).all())
        assert batch1.left == batch2.left
        for k, v1 in batch1.payload.items():
            v1 = v1.float()
            v2 = batch2.payload[k].float()
            self.assertEqual(v1.shape, v2.shape)
            for i, l in enumerate(batch1.seq_lens):
                if batch1.left:
                    t = v1.shape[1]
                    self.assertTrue(v1[i, t - l:].allclose(v2[i, t - l:]))
                else:
                    self.assertTrue(v1[i, :l].allclose(v2[i, :l]))

    def test_time(self):
        times = torch.tensor([
            [3, 5, 5, 7],
            [0, 2, -5, 200]
        ])
        deltas = torch.tensor([
            [0, 2, 0, 2],
            [0, 2, -1, -1]
        ])
        left_times = torch.tensor([
            [3, 5, 5, 7],
            [-8, 100, 0, 2]
        ])
        left_deltas = torch.tensor([
            [0, 2, 0, 2],
            [-1, -1, 0, 2]
        ])
        lengths = torch.tensor([4, 2])
        batch = PaddedBatch({"timestamps": times}, lengths)
        left_batch = PaddedBatch({"timestamps": left_times}, lengths, left=True)

        features = {
            "timestamps": torch.tensor([3, 0])
        }

        # Right Batch.
        x = autoreg_prepare_features(batch)
        gt = PaddedBatch({"timestamps": deltas,
                          "_times": times},
                         lengths)
        self.assertEqualBatch(x, gt)

        y = autoreg_output_to_next_input_inplace(x, features)
        self.assertEqual(y["timestamps"].tolist(), [3, 0])
        self.assertEqual(y["_times"].tolist(), [10, 2])

        x = autoreg_revert_features(x)
        self.assertEqualBatch(x, batch)

        # Left Batch.
        x = autoreg_prepare_features(left_batch)
        gt = PaddedBatch({"timestamps": left_deltas,
                          "_times": left_times},
                         lengths, left=True)
        self.assertEqualBatch(x, gt)

        y = autoreg_output_to_next_input_inplace(x, features)
        self.assertEqual(y["timestamps"].tolist(), [3, 0])
        self.assertEqual(y["_times"].tolist(), [10, 2])

        x = autoreg_revert_features(x)
        self.assertEqualBatch(x, left_batch)

    def test_inference(self):
        features = torch.tensor([
            [[0, 0],
             [1, 1],
             [2, 1],
             [4, 0],
             [6, 0]],
            [[2, 1],
             [4, 1],
             [5, 1],
             [0, 0],
             [0, 0]]
           ]).float()  # (B, T, D).
        lengths = torch.tensor([5, 3])
        batch = PaddedBatch({"timestamps": features[..., 0],
                             "labels": features[..., 1]}, lengths)
        indices = torch.tensor([
            [0, 1],
            [2, 0]  # Second is unused.
        ])
        indices = PaddedBatch(indices, torch.tensor([2, 1]))

        # After time delta:
        # 0 1 1 2 2
        # 0 2 1 -5 0

        # Initial states (time_delta, label):
        # [0, 0], [1, 1]
        # [5, 3]

        # Prediction function is x * 2 + 1
        # The model predicts labels and time delta.
        #
        # Results:
        # [1, 1], [4, 3]
        # [4, 3], [11, 7]
        # [8, 7], [15, 15]

        gt_times = [[[1, 4], [4, 11]], [[8, 15]]]
        gt_labels = [[[1, 3], [3, 7]], [[3, 7]]]

        # Initialize predictors.
        module = SimpleModule()

        for predictor in [module]:
            results = predictor.generate_sequences(batch, indices)

            self.assertEqual(results.shape, (2, 2))
            self.assertEqual(results.seq_lens.tolist(), [2, 1])
            self.assertEqual(results.payload["timestamps"][0].tolist(), gt_times[0])
            self.assertEqual(results.payload["labels"][0].tolist(), gt_labels[0])
            self.assertEqual(results.payload["timestamps"][1, :1].tolist(), gt_times[1])
            self.assertEqual(results.payload["labels"][1, :1].tolist(), gt_labels[1])


if __name__ == "__main__":
    main()
