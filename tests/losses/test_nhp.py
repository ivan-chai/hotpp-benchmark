import logging
import math
from unittest import TestCase, main

import torch
from torch import Tensor

from esp_horizon.data import PaddedBatch
from esp_horizon.losses import NHPLoss, TimeRMTPPLoss


class RMTPPInterpolator:
    """Output make NHP similar to RMTPP."""
    def __call__(self, states, time_deltas):
        s = time_deltas.payload.shape[2]
        states = states[-1]  # (B, L, D).
        states = states.unsqueeze(2).repeat(1, 1, s, 1)  # (B, L, S, D)
        payload = states - time_deltas.payload.unsqueeze(3)  # (B, L, S, D).
        payload = (payload.exp().exp() - 1).log()
        return PaddedBatch(payload, time_deltas.seq_lens)


def inv_softplus(x):
    return (1 - x.exp()).clip(min=1e-6).log()


class TestRMTPPLoss(TestCase):
    def test_compare_nhp_to_rmtpp(self):
        # RMTPP uses closed form solution for likelihood computation.
        # Check general NHP algorithm is equal to RMTPP, when intensity matches.
        rmtpp = TimeRMTPPLoss(max_delta=5, expectation_steps=10000)
        nhp = NHPLoss(num_classes=1, max_delta=5, likelihood_sample_size=1000, expectation_steps=10000)
        nhp.interpolator = RMTPPInterpolator()

        b, l = 5, 7
        timestamps = (torch.rand(b, l) * 2).cumsum(1)
        labels = torch.zeros(b, l, dtype=torch.long)
        lengths = torch.full([b], l)
        x = PaddedBatch({"timestamps": timestamps, "labels": labels}, lengths)
        predictions = torch.randn(b, l, 1)
        states = predictions[None]  # (N, B, L, D).

        loss_gt = rmtpp(timestamps, predictions)[0].item()
        loss = nhp(x, PaddedBatch(predictions, lengths), states)[0]["nhp"].item()
        self.assertTrue(abs(loss - loss_gt) < 0.01)

        # Compare sampling.
        sample_gt = rmtpp.predict_means(predictions).squeeze(2).flatten()
        sample = nhp.predict_next(x, states).payload["timestamps"].flatten()
        mean = torch.cat([sample_gt, sample]).mean()
        sample_gt -= mean
        sample -= mean
        norm = max(torch.linalg.norm(sample).item(), torch.linalg.norm(sample_gt).item())
        delta = torch.linalg.norm(sample - sample_gt).item() / norm
        self.assertTrue(delta < 0.2)


if __name__ == "__main__":
    main()
