#!/usr/bin/env python3
import math
from unittest import TestCase, main

import torch

from esp_horizon.metrics import MAPMetric, NextItemMetric, HorizonMetric


class TestMetrics(TestCase):
    def setUp(self):
        self.horizon = 5  # TODO: Check 4 is invalid.
        self.mask = torch.tensor([
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
        ]).bool()
        self.times = torch.tensor([
            [-7, -5, -4, -2, -1, 2, 4, 8, 11, 12]
        ])
        self.labels = torch.tensor([
            [ 0,  0,  1,  1,  0, 1, 0, 1, 1,  0]
        ])
        self.predicted_times = torch.tensor([
            [ -5, -5, -3, 0,  1, 2, 4, 8, 12, 15]
        ])
        self.predicted_labels_logits = torch.tensor([[
            [5, 4],       # 0, GT 0.
            [0, 2],       # 1, GT 1.
            [-5, -1],     # 1, GT 1.
            [-0.2, 0.5],  # 1, GT 0.
            [-0.2, 0.7],  # 1, GT 1.
            [5, 0],       # 0, GT 0.
            [1, 0],       # 0, GT 1.
            [0.1, 0.2],   # 1, GT 1.
            [0.2, 0.1],   # 0, GT -. (masked).
            [-0.7, -0.5]  # 1, GT -. (not available).
        ]])
        self.seq_indices = torch.tensor([
            [3, 6]
        ])
        self.seq_indices_lens = torch.tensor([2])
        self.seq_target_mask = torch.tensor([
            [True, True],
            [False, True]
        ])
        self.seq_target_times = torch.tensor([
            [-1, 2],
            [ 0, 8]
        ])
        self.seq_target_labels = torch.tensor([
            [ 0, 1],
            [-1, 1]
        ])
        self.seq_predicted_mask = torch.tensor([
            [True, True, False, True],
            [True, False, True, False]
        ])
        self.seq_predicted_times = torch.tensor([
            [2, 0, 3, 1],
            [4, 9, 8, 10]
        ])
        self.seq_predicted_labels_logits = torch.tensor([
            [
                [0, 0.9],
                [0.05, 0.05],
                [0, 0],
                [0, 1],
            ],
            [
                [0, 0.8],
                [0, 0],
                [0, 0.09],
                [0, 0]
            ]
        ]).clip(min=1e-6).log()  # (1, 2, 4, 2).

    def test_map_metric(self):
        metric = MAPMetric(time_delta_thresholds=[1, 2])
        metric.update(
            target_mask=self.seq_target_mask,
            target_times=self.seq_target_times,
            target_labels=self.seq_target_labels,
            predicted_mask=self.seq_predicted_mask,
            predicted_times=self.seq_predicted_times,
            predicted_labels_logits=self.seq_predicted_labels_logits
        )
        # Matching (prediction -> target):
        # Batch 1: 0 -> 1 for horizon 1 and 3 -> 1, 1 -> 0 for horizon 2.
        # Batch 2: 2 -> 1.
        #
        # Scores horizon 1, batch 1:
        # class 0: Unmatched.
        # class 1: 0.9 (pos), 0, 1.
        #
        # Scores horizon 2, batch 1:
        # class 0: 0.0.
        # class 1: 0.9, 0, 1 (pos).
        #
        # Scores horizon 1, batch 2:
        # class 0: Empty.
        # class 1: 0.8, 0.09 (pos).
        #
        # Scores horizon 2, batch 2:
        # class 0: Empty.
        # class 1: 0.8, 0.09 (pos).
        #
        # All scores horizon 1:
        # class 0: Unmatched, recall is always 0.
        # class 1: 0, 0.09 (pos), 0.8, 0.9 (pos), 1.
        #
        # All scores horizon 2:
        # class 0: 0, 0.05 (pos), 0, 0, 0.
        # class 1: 0, 0.09 (pos), 0.8, 0.9, 1 (pos).
        ap_h1_c0 = 0
        ap_h1_c1 = 0.5
        ap_h2_c0 = 1
        ap_h2_c1 = 0.75
        map_gt = (ap_h1_c0 + ap_h1_c1 + ap_h2_c0 + ap_h2_c1) / 4
        self.assertAlmostEqual(metric.compute()["detection-mAP"], map_gt)

    def test_next_item_metric(self):
        metric = NextItemMetric()
        metric.update(
            target_mask=self.mask[:, 1:],
            target_labels=self.labels[:, 1:],
            predicted_labels_logits=self.predicted_labels_logits[:, :-1]
        )
        acc_gt = 6 / 8
        self.assertAlmostEqual(metric.compute()["next-item-accuracy"], acc_gt)

    def test_end_to_end(self):
        metric = HorizonMetric(self.horizon, horizon_evaluation_step=3,
                               map_thresholds=[1, 2],
                               max_target_length=self.seq_target_mask.shape[1])
        seq_lens = self.mask.sum(1)
        metric.update_next_item(seq_lens=seq_lens,
                                timestamps=self.times,
                                labels=self.labels,
                                predicted_timestamps=self.predicted_times,
                                predicted_labels_logits=self.predicted_labels_logits)
        indices = metric.select_horizon_indices(seq_lens)
        self.assertEqual(indices.seq_lens.tolist(), [2])
        self.assertTrue((indices.payload == self.seq_indices).all())
        metric.update_horizon(seq_lens=seq_lens,
                              timestamps=self.times,
                              labels=self.labels,
                              indices=self.seq_indices,
                              indices_lens=self.seq_indices_lens,
                              seq_predicted_timestamps=self.seq_predicted_times[None],
                              seq_predicted_labels_logits=self.seq_predicted_labels_logits[None])
        metrics = metric.compute()

        acc_gt = 6 / 8
        self.assertAlmostEqual(metrics["next-item-accuracy"], acc_gt)

        ap_h1_c0 = 0
        ap_h1_c1 = 0.5
        ap_h2_c0 = 1
        ap_h2_c1 = 0.75
        map_gt = (ap_h1_c0 + ap_h1_c1 + ap_h2_c0 + ap_h2_c1) / 4
        self.assertAlmostEqual(metrics["detection-mAP"], map_gt)


if __name__ == "__main__":
    main()
