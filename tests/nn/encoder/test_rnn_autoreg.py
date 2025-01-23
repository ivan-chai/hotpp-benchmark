import logging
from unittest import TestCase, main

import torch

from hotpp.data import PaddedBatch
from hotpp.modules import BaseModule, NextItemModule
from hotpp.nn.encoder import RnnEncoder


class SimpleEmbedder(torch.nn.Module):
    def forward(self, x):
        payload = torch.stack([x.payload["timestamps"], x.payload["labels"]], -1)
        return PaddedBatch(payload, x.seq_lens)


class SimpleRNN(torch.nn.GRU):
    def __init__(self):
        super().__init__(2, 2)
        self.init_state = torch.zeros(1, 2)

    def forward(self, x, timestamps, states=None, return_states=False):
        if states is None:
            b, l, d = x.payload.shape
            states = torch.zeros(1, b, l, d)
        else:
            states = states.unsqueeze(2).repeat(1, 1, x.shape[1], 1)
        if not return_states:
            states = None
        elif return_states == "last":
            states = states.take_along_dim((x.seq_lens - 1).clip(min=0)[None, :, None, None], 2).squeeze(2)  # (N, B, D).
        else:
            assert return_states == "full"
        return PaddedBatch(x.payload * 2 + 1, x.seq_lens), states


class SimpleSequenceEncoder(RnnEncoder):
    num_layers = 1
    _max_context = None
    _context_step = None

    def __init__(self):
        super(RnnEncoder, self).__init__({})
        self.embedder = SimpleEmbedder()
        self.rnn = SimpleRNN()


class SimpleLoss:
    def predict_next(self, outputs, states, fields, logits_fields_mapping):
        return PaddedBatch({
            "timestamps": outputs.payload[:, :, 0],
            "labels": outputs.payload[:, :, 1].long()
        }, outputs.seq_lens)


class SimpleModule(NextItemModule):
    def __init__(self):
        super(BaseModule, self).__init__()
        self._seq_encoder = SimpleSequenceEncoder()
        self._head = torch.nn.Identity()
        self._autoreg_max_steps = 2
        self._loss = SimpleLoss()
        self._timestamps_field = "timestamps"
        self._labels_field = "labels"
        self._labels_logits_field = "logits"


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

    def compare_with_simple_rnn_forward(self):
        # One step of autoregression is equivalent to RNN forward.
        encoder = RnnEncoder(
            {"labels": {"in": 4, "out": 8}}
        )
        batch = PaddedBatch({
            "timestamps": torch.rand(1, 10).cumsum(1),
            "labels": torch.randint(0, 4, (1, 10))
        }, torch.tensor([10]))  # (1, 10).
        indices = PaddedBatch(torch.arange(10)[None], torch.tensor([10]))

        def predict_fn(outputs):
            return PaddedBatch({
                "timestamps": outputs.payload[:, :, 0],
                "labels": (outputs.payload[:, :, 1] * 100).abs().long().clip(min=0, max=3)
            }, outputs.seq_lens)
        result = encoder.generate(batch, indices, predict_fn, 1)  # (B, I, 1).
        result = PaddedBatch({k: v.squeeze(2) for k, v in result.payload.items()}, result.seq_lens)
        result_gt = predict_fn(encoder(batch)["outputs"])
        result_gt.payload["timestamps"].clip_(min=0)
        result_gt.payload["timestamps"] += batch.payload["timestamps"]
        self.assertEqualBatch(result, result_gt)

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
            [2, 0]  # Second is unused.
        ])
        indices = PaddedBatch(indices, torch.tensor([2, 1]))

        # After time delta:
        # 0 1 1 2 2
        # 0 1 2 -5 0

        # Initial states (state is the output from the previous feature, i.e. 2 * x + 1):
        # [0, 0], [1, 1]
        # [3, 3]

        # Initial fetures (time_delta, label):
        # [0, 0], [1, 1]
        # [2, 1]

        # Prediction function is x * 2 + 1, where x is input.
        # The model predicts labels and time delta.
        #
        # Results (time_delta, label):
        # [1, 1], [3, 3]
        # [3, 3], [7, 7]
        # [5, 3], [11, 7]
        #
        # Results (time_delta cumsum, label):
        # [1, 1], [4, 3]
        # [3, 3], [10, 7]
        # [5, 3], [16, 7]
        #
        # Results (time, label):
        # [1, 1], [4, 3]
        # [4, 3], [11, 7]
        # [10, 3], [21, 7]

        gt_times = [[[1, 4], [4, 11]], [[10, 21]]]
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
