import logging
from unittest import TestCase, main

import torch

from esp_horizon.data import PaddedBatch
from esp_horizon.modules import NextItemModule
from ptls.nn.seq_encoder import RnnEncoder
from ptls.nn.seq_encoder.containers import SeqEncoderContainer
from esp_horizon.modules.next_item.autoreg import RNNSequencePredictor, NextItemRNNAdapter, BaseAdapter


class SimpleEncoder(torch.nn.Module):
    category_names = {"timestamps", "labels"}

    def __init__(self):
        super().__init__()
        self.output_size = 2
        self.embeddings = torch.nn.ModuleDict({
            "labels": torch.nn.Embedding(2, 1)
        })

    def forward(self, x):
        payload = torch.stack([x.payload["timestamps"], x.payload["labels"]], -1)
        return PaddedBatch(payload, x.seq_lens)


class SimpleSequenceEncoder(RnnEncoder):
    rnn_type = "gru"
    num_layers = 1
    bidirectional = False
    trainable_starter = "static"

    def __init__(self, input_size, is_reduce_sequence):
        super(RnnEncoder, self).__init__(is_reduce_sequence=is_reduce_sequence)

    @property
    def starter_h(self):
        return torch.zeros(1, 1, 2)

    def forward(self, x, h_0=None):
        return PaddedBatch(x.payload * 2 + 1, x.seq_lens)


class SimpleContainer(SeqEncoderContainer):
    def __init__(self):
        super().__init__(trx_encoder=SimpleEncoder(),
                         seq_encoder_cls=SimpleSequenceEncoder,
                         input_size=2,
                         seq_encoder_params={},
                         is_reduce_sequence=False)


class SimpleModule(NextItemModule):
    def __init__(self):
        super(NextItemModule, self).__init__()
        self.seq_encoder = SimpleContainer()

    def get_modes(self, embeddings):
        return PaddedBatch({
            "timestamps": embeddings.payload[:, :, 0].long(),
            "labels": embeddings.payload[:, :, 1].long()
        }, embeddings.seq_lens)


class SimpleAdapter(BaseAdapter):
    def __init__(self, **kwargs):
        super().__init__(torch.nn.Identity(), "timestamps", **kwargs)

    def forward(self, batch):
        return {"timestamps": torch.zeros(len(batch))}


class TestAdapters(TestCase):
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

        # Input delta and output delta.
        adapter = SimpleAdapter(time_int=True)
        x = adapter.prepare_features(batch)
        gt = PaddedBatch({"timestamps": deltas,
                          "_times": times},
                         lengths)
        self.assertEqualBatch(x, gt)

        d, y = adapter.output_to_next_input(x, features)
        self.assertEqual(d.tolist(), [3, 0])
        self.assertEqual(y["timestamps"].tolist(), [3, 0])
        self.assertEqual(y["_times"].tolist(), [10, 2])

        x = adapter.revert_features(x)
        self.assertEqualBatch(x, batch)

        # Left Batch.
        x = adapter.prepare_features(left_batch)
        gt = PaddedBatch({"timestamps": left_deltas,
                          "_times": left_times},
                         lengths, left=True)
        self.assertEqualBatch(x, gt)

        d, y = adapter.output_to_next_input(x, features)
        self.assertEqual(d.tolist(), [3, 0])
        self.assertEqual(y["timestamps"].tolist(), [3, 0])
        self.assertEqual(y["_times"].tolist(), [10, 2])

        x = adapter.revert_features(x)
        self.assertEqualBatch(x, left_batch)

        # Input raw time and output delta.
        adapter = SimpleAdapter(time_int=True, time_input_delta=False)
        x = adapter.prepare_features(batch)
        gt = PaddedBatch({"timestamps": times,
                          "_times": times},
                         lengths)
        self.assertEqualBatch(x, gt)

        d, y = adapter.output_to_next_input(x, features)
        self.assertEqual(d.tolist(), [3, 0])
        self.assertEqual(y["timestamps"].tolist(), [10, 2])
        self.assertEqual(y["_times"].tolist(), [10, 2])

        x = adapter.revert_features(PaddedBatch({"timestamps": deltas, "_times": times}, x.seq_lens))
        self.assertEqualBatch(x, batch)

        # Left Batch.
        x = adapter.prepare_features(left_batch)
        gt = PaddedBatch({"timestamps": left_times,
                          "_times": left_times},
                         lengths, left=True)
        self.assertEqualBatch(x, gt)

        d, y = adapter.output_to_next_input(x, features)
        self.assertEqual(d.tolist(), [3, 0])
        self.assertEqual(y["timestamps"].tolist(), [10, 2])
        self.assertEqual(y["_times"].tolist(), [10, 2])

        x = adapter.revert_features(PaddedBatch({"timestamps": left_deltas, "_times": left_times}, x.seq_lens, left=True))
        self.assertEqualBatch(x, left_batch)

    def test_inference(self):
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
        lengths = torch.tensor([5, 3])
        batch = PaddedBatch({"timestamps": features[..., 0],
                             "labels": features[..., 1]}, lengths)
        indices = torch.tensor([
            [0, 1],
            [1, 0]  # Second is unused.
        ])
        indices = PaddedBatch(indices, torch.tensor([2, 1]))

        # Initial states:
        # [0, 0], [1, 1]
        # [3, 1]

        # Prediction function is x * 2 + 1
        # Results:
        # [1, 1], [3, 3]
        # [3, 3], [7, 7]
        # [7, 3], [15, 7]

        gt_times = [[[1, 3], [3, 7]], [[7, 15]]]
        gt_labels = [[[1, 3], [3, 7]], [[3, 7]]]

        # Initialize predictors.
        module = SimpleModule()
        adapter = NextItemRNNAdapter(module, "timestamps", time_input_delta=False, time_output_delta=False)
        rnn_predictor_modes = RNNSequencePredictor(adapter, max_steps=2)

        for predictor in [rnn_predictor_modes]:
            results = predictor(batch, indices)

            self.assertEqual(results.shape, (2, 2))
            self.assertEqual(results.seq_lens.tolist(), [2, 1])
            self.assertEqual(results.payload["timestamps"][0].tolist(), gt_times[0])
            self.assertEqual(results.payload["labels"][0].tolist(), gt_labels[0])
            self.assertEqual(results.payload["timestamps"][1, :1].tolist(), gt_times[1])
            self.assertEqual(results.payload["labels"][1, :1].tolist(), gt_labels[1])


if __name__ == "__main__":
    main()
