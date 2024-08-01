import logging
import math
from unittest import TestCase, main

import torch
from torch import Tensor

from hotpp.losses import TimeRMTPPLoss


class TestRMTPPLoss(TestCase):
    def test_rmtpp_loss(self):
        loss = TimeRMTPPLoss(init_influence=3, force_negative_influence=False)

        mask = torch.tensor([
            [1, 1, 1],
            [0, 1, 1]
        ]).bool()
        biases = torch.tensor([
            [0, 1, -1],
            [2, 0,  1]
        ]).double()
        times = torch.tensor([
            [2, 2, 3],
            [5, 5, 7]
        ]).double()
        result = loss(times, biases.unsqueeze(2), mask)[0].item()
        # After delta:
        # mask:
        # [[1, 1]
        #  [0, 1]]
        # times:
        # [[0, 1],
        #  [0, 2]]
        # biases:
        # [[0, 1],
        #  [2, 0]]

        # Equation (12) from the original paper.
        log_f00 = 0 + 3 * 0 + 1 / 3 * math.exp(0) - 1 / 3 * math.exp(0 + 3 * 0)
        log_f01 = 1 + 3 * 1 + 1 / 3 * math.exp(1) - 1 / 3 * math.exp(1 + 3 * 1)
        log_f11 = 0 + 3 * 2 + 1 / 3 * math.exp(0) - 1 / 3 * math.exp(0 + 3 * 2)
        neg_loglike_gt = -(log_f00 + log_f01 + log_f11) / 3
        self.assertAlmostEqual(result, neg_loglike_gt)

        modes = loss.predict_modes(biases.unsqueeze(2)).squeeze(2)
        modes_gt = torch.tensor([
            [(math.log(3) - 0) / 3, (math.log(3) - 1) / 3, (math.log(3) + 1) / 3],
            [(math.log(3) - 2) / 3, (math.log(3) - 0) / 3, (math.log(3) - 1) / 3],
        ], dtype=torch.double).clip(min=0)
        self.assertTrue(modes.allclose(modes_gt))

    def test_rmtpp_thinning(self):
        a = -0.1
        loss = TimeRMTPPLoss(init_influence=a, thinning_params={"max_delta": 1000, "max_steps": 10000})
        biases = torch.linspace(-2, 2, 11)  # (L).
        means = loss.predict_means(biases[None, :, None].double()).squeeze(2).squeeze(0)  # (L).

        xs = torch.linspace(0, 100, 100000)  # (N).
        l = a * xs[:, None] + biases  # (N, L).
        scale = 1 / (1 - (biases.exp() / a).exp())  # (L).
        pdfs = scale * (l - 1 / a * (l.exp() - biases.exp())).exp()  # (N, L).
        means_gt = (pdfs * xs[:, None] * (xs[-1] - xs[0])).mean(0)  # (L).
        self.assertTrue((means_gt < means * 1.1).all())
        self.assertTrue((means_gt > means * 0.7).all())


if __name__ == "__main__":
    main()
