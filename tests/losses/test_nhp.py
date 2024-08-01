import logging
import math
from unittest import TestCase, main

import torch
from torch import Tensor

from hotpp.data import PaddedBatch
from hotpp.losses import NHPLoss, TimeRMTPPLoss


class RMTPPInterpolator:
    """Output make NHP similar to RMTPP."""
    def __call__(self, states, time_deltas):
        s = time_deltas.payload.shape[2]
        states = states[-1]  # (B, L, D).
        states = states.unsqueeze(2).repeat(1, 1, s, 1)  # (B, L, S, D)
        payload = states - time_deltas.payload.unsqueeze(3)  # (B, L, S, D).
        payload = (payload.exp().exp() - 1).log()
        return PaddedBatch(payload, time_deltas.seq_lens)

    def modules(self):
        return []

    def train(self):
        pass

    def eval(self):
        pass


class IdentityInterpolator:
    """Output make NHP similar to RMTPP."""
    def __call__(self, states, time_deltas):
        s = time_deltas.payload.shape[2]
        states = states[-1]  # (B, L, D).
        states = states.unsqueeze(2).repeat(1, 1, s, 1)  # (B, L, S, D)
        return PaddedBatch(states, time_deltas.seq_lens)


class TestNHPLoss(TestCase):
    def test_compare_nhp_to_rmtpp(self):
        # RMTPP uses closed form solution for likelihood computation.
        # Check general NHP algorithm is equal to RMTPP, when intensity matches.
        rmtpp = TimeRMTPPLoss(thinning_params={"max_delta": 5, "max_steps": 10000})
        nhp = NHPLoss(num_classes=1, likelihood_sample_size=1000, thinning_params={"max_delta": 5, "max_steps": 10000})
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
        sample = nhp.predict_next(PaddedBatch(predictions, lengths), states).payload["timestamps"].flatten()
        mean = torch.cat([sample_gt, sample]).mean()
        sample_gt -= mean
        sample -= mean
        norm = max(torch.linalg.norm(sample).item(), torch.linalg.norm(sample_gt).item())
        delta = torch.linalg.norm(sample - sample_gt).item() / norm
        self.assertTrue(delta < 0.25)

    def test_labels_prediction(self):
        nc = 5
        nhp = NHPLoss(num_classes=nc, likelihood_sample_size=1000, thinning_params={"max_delta": 5, "max_steps": 1000})
        nhp.interpolator = IdentityInterpolator()

        b, l = 5, 7
        lengths = torch.full([b], l)
        predictions = torch.rand(b, l, nc)
        for i in range(b):
            for j in range(l):
                predictions[i, j, (i + j) % nc] = 5
        sample_gt = (torch.arange(b)[:, None] + torch.arange(l)[None]) % nc
        states = predictions[None]  # (N, B, L, D).
        sample = nhp.predict_next(PaddedBatch(predictions, lengths), states).payload["labels"]  # (b, l).
        self.assertEqual(sample.tolist(), sample_gt.tolist())



if __name__ == "__main__":
    main()
