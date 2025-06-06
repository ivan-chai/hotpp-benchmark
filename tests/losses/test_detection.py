#!/usr/bin/env python3
import math
from unittest import TestCase, main

import torch

from hotpp.data import PaddedBatch
from hotpp.losses import TimeMSELoss, CrossEntropyLoss, BinaryCrossEntropyLoss
from hotpp.losses import NextItemLoss, DetectionLoss


def inv_softplus(x):
    return math.log(math.exp(x) - 1)


class Model(torch.nn.Module):
    def __init__(self, length, num_events, num_labels):
        super().__init__()
        self.length = length
        self.weight = torch.nn.Parameter(torch.randn(1, length, num_events, num_labels + 2))  # (B, L, T, P).

    def forward(self):
        return PaddedBatch(self.weight.reshape(1, self.length, -1),
                           torch.tensor([self.length]))


class TestDetectionLoss(TestCase):
    def test_matching(self):
        times = torch.tensor([10, 11, 12, 12, 13, 20]).float()
        labels = torch.tensor([0, 0, 1, 2, 1, 0]).long()
        horizon = 4
        num_events = 4
        # Two similar sequences with different lengths.
        batch = PaddedBatch({
            "timestamps": torch.stack([times, times + 5]),
            "labels": torch.stack([labels, labels])
        }, torch.tensor([6, 5]))

        losses = {
            "_presence": BinaryCrossEntropyLoss(),
            "timestamps": TimeMSELoss(delta="start"),
            "labels": CrossEntropyLoss(3)
        }
        loss = DetectionLoss(NextItemLoss(losses), num_events, horizon,
                             drop_partial_windows=True, match_weights={"timestamps": 1})
        # Input length is 5, window size is 4 (= 2 / 0.5).
        # There are 2 extracted windows for the first item and 1 window for the second.
        zeros = torch.zeros(2, 6, num_events)
        anchors = torch.arange(num_events)[None, None, :].repeat(2, 6, 1)  # (B, L, T).
        outputs = {
            "_presence": zeros.unsqueeze(-1) + 1, # Logit.
            "timestamps": anchors.unsqueeze(-1),  # (B, L, T, 1).
            "labels": torch.stack([zeros, zeros + 1, zeros])  # (B, L, T, 3): labels logits.
        }
        outputs = {k: v.reshape(2, 6, 4, -1) for k, v in outputs.items()}  # (B, L, T, P).
        outputs = torch.cat([outputs[k] for k in loss.fields], -1)
        outputs = PaddedBatch(outputs, torch.tensor([6, 5]))
        windows = loss.extract_structured_windows(batch)  # (B, L - k, k + 1, D).
        indices, matching, losses, stats = loss.get_subset_matching(batch, outputs)  # (B, L - k, T).
        # Select valid matching.
        matching = PaddedBatch(matching.payload[:, :2], indices.payload["full_mask"].sum(1))
        # Relative to absolute.
        mask = matching.payload < 0
        matching.payload += torch.arange(1, matching.shape[1] + 1)[:, None]
        matching.payload[mask] = -1
        self.assertEqual(matching.payload.shape, (2, 2, 4))
        self.assertEqual(matching.seq_lens.tolist(), [2, 1])
        self.assertEqual(stats["prediction_match_rate"], (8 + 3) / (3 * 4))
        self.assertEqual(stats["target_match_rate"], (8 + 3) / (4 + 3 + 4))

        matching_gt = torch.full((2, 6 - 4, 4), -1, dtype=torch.long)
        matching_gt[:, 0, 0] = 1  # t=11.
        matching_gt[:, 0, 1] = 2  # t=12.
        matching_gt[:, 0, 2] = 3  # t=12.
        matching_gt[:, 0, 3] = 4  # t=13.
        matching_gt[0, 1, 0] = 2  # t=12.
        matching_gt[0, 1, 1] = 3  # t=12.
        matching_gt[0, 1, 2] = 4  # t=13.
        self.assertEqual(matching.payload.tolist(), matching_gt.tolist())

    def test_convergence(self):
        torch.manual_seed(0)

        losses = {
            "_presence": BinaryCrossEntropyLoss(),
            "timestamps": TimeMSELoss(delta="start"),
            "labels": CrossEntropyLoss(10)
        }

        loss = DetectionLoss(NextItemLoss(losses), k=6, horizon=3,
                             drop_partial_windows=True,
                             match_weights={"timestamps": 1, "labels": 1})

        batch = PaddedBatch({
            "timestamps": torch.tensor([5.0, 5.3, 6.0, 6.1, 7.2, 8.0, 9.0, 20, 21, 21.5]).reshape(1, 10),
            "labels":     torch.tensor([5  , 9  , 0  , 1  , 0  , 1  , 5  , 7 , 8 , 1]).reshape(1, 10),
        }, torch.tensor([10]))

        model = Model(batch.payload["timestamps"].shape[1], 6, batch.payload["labels"].max().item() + 1)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        model.train()
        loss.train()
        for step in range(100):
            prediction = model()
            optimizer.zero_grad()
            losses, _ = loss(batch, prediction, None)
            loss_value = sum(losses.values())
            if step % 10 == 0:
                print(f"Loss: {loss_value.item():.4f}")
            loss_value.backward()
            optimizer.step()
        print(f"Final loss: {loss_value.item():.4f}")
        print()

        model.eval()
        loss.eval()
        with torch.no_grad():
            outputs = model()
            states = outputs.payload[None]
            predictions = loss.predict_next_k(outputs, states).payload  # (B, L, K)
            predictions["timestamps"] += batch.payload["timestamps"].unsqueeze(2)

        n_valid = 10 - 6
        for i in range(n_valid):
            # Get horizon length.
            k = (batch.payload["timestamps"][0, i + 1:] - batch.payload["timestamps"][0, i] < 3).sum().item()
            gt_times = batch.payload["timestamps"][0, i + 1:i + 1 + k]
            gt_labels = batch.payload["labels"][0, i + 1:i + 1 + k]
            mask = predictions["_presence"][0, i].bool()
            self.assertEqual(predictions["_presence"][0, i].sum().item(), k)
            self.assertEqual(predictions["labels"][0, i, mask].tolist(), gt_labels.tolist())
            self.assertTrue(predictions["timestamps"][0, i, mask].allclose(gt_times, atol=0.1))


if __name__ == "__main__":
    main()
