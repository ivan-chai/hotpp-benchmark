from unittest import TestCase, main

import torch

from hotpp.data import PaddedBatch
from hotpp.fields import PRESENCE_PROB
from hotpp.modules import RecentHistoryModule, MostPopularModule, HistoryDensityModule


class TestStatisticalBaselines(TestCase):
    def test_generation(self):
        rh_module = RecentHistoryModule(2, 3)
        mp_module = MostPopularModule(2, 3)
        hd_module = HistoryDensityModule(3, [1, 2], amounts_field="amounts")  # num_classes, horizons.
        batch = PaddedBatch({
            "timestamps": torch.tensor([0.5, 1, 2.5, 3])[None],
            "labels": torch.tensor([2, 0, 1, 0])[None],
            "amounts": torch.tensor([1, 2, 3, 4])[None]
        }, torch.tensor([4]))
        indices = PaddedBatch(torch.tensor([1, 3])[None], torch.tensor([2]))

        # Test RecentHistory.
        sequences = rh_module.generate_sequences(batch, indices)
        # Input deltas:
        # 0, 0.5, 1.5, 0.5
        # Deltas:
        # 0, 0.5
        # 1.5, 0.5
        # Timestamps: cumsum deltas + initial
        # 1, 1.5
        # 4.5, 5
        timestamps_gt = [[1, 1.5], [4.5, 5]]
        labels_gt = [[2, 0], [1, 0]]
        self.assertEqual(sequences.payload["timestamps"].squeeze().tolist(), timestamps_gt)
        self.assertEqual(sequences.payload["labels"].squeeze().tolist(), labels_gt)

        # Test MostPopular.
        sequences = mp_module.generate_sequences(batch, indices)
        # Input deltas:
        # 0, 0.5, 1.5, 0.5
        # Mean deltas (ignoring the first one): 0, 0.5, 1, 2.5 / 3
        # Top labels: 2, 0, 0, 0
        # Timestamps: cumsum deltas + initial
        # 1.25 1.5
        # 3.625 4.25
        # Labels:
        # 0 0
        # 0 0
        timestamps_gt = [[1.5, 2.0], [3 + 2.5 / 3, 3 + 5 / 3]]
        labels_gt = [[0, 0], [0, 0]]
        self.assertTrue(sequences.payload["timestamps"].squeeze().allclose(torch.tensor(timestamps_gt)))
        self.assertEqual(sequences.payload["labels"].squeeze().tolist(), labels_gt)

        # Test HistoryDensity.
        sequences = hd_module.generate_sequences(batch, indices)
        # Time offsets
        #  0.5, 0.5, 2.0, 2.5
        # Time deltas:
        #  0, 0.5, 1.5, 0.5.
        # Mean time deltas:
        #  0, 0.25, 2 / 3, 2.5 / 4
        # Median time deltas:
        #  0, 0, 0.5, 0.5
        #
        # Label 0 densities:
        #  0, 2, 0.5, 0.8
        # Label 1 densities:
        #  0, 0, 0.5, 0.4
        # Label 2 densities:
        #  2, 2, 0.5, 0.4
        #
        # Amounts 0 densities:
        #  0, 4, 1, 2.4
        # Amounts 1 densities:
        #  0, 0, 1.5, 1.2
        # Amounts 2 densities:
        #  2, 2, 0.5, 0.4
        #
        # Horizon 1 (same as horizon 2 - horizon 1)
        # avg counts:
        # [0, 0, 2], [2, 0, 2], [0.5, 0.5, 0.5], [0.8, 0.4, 0.4]
        # avg amounts:
        # [0, 0, 2], [4, 0, 2], [1, 1.5, 0.5], [2.4, 1.2, 0.4]
        #
        # Sorting order (index 1):
        # 0, 2, 1
        # Sorting order (index 3):
        # 0, 1, 2
        counts_gt = [[2.0, 2.0, 0.0, 2.0, 2.0, 0.0], [0.8, 0.4, 0.4, 0.8, 0.4, 0.4]]
        probs_gt = [[1.0, 1.0, 0.0, 1.0, 1.0, 0.0], [0.8, 0.4, 0.4, 0.8, 0.4, 0.4]]
        timestamps_gt = [[1.5, 1.5, 1.5, 2.5, 2.5, 2.5], [3.5, 3.5, 3.5, 4.5, 4.5, 4.5]]
        labels_gt = [[0, 2, 1, 0, 2, 1], [0, 1, 2, 0, 1, 2]]
        amounts_gt = [[4.0, 2.0, 0.0, 4.0, 2.0, 0.0], [2.4, 1.2, 0.4, 2.4, 1.2, 0.4]]
        self.assertTrue(sequences.payload[PRESENCE_PROB].squeeze().allclose(torch.tensor(probs_gt)))
        self.assertEqual(sequences.payload["timestamps"].squeeze().tolist(), timestamps_gt)
        self.assertTrue(sequences.payload["labels"].squeeze().allclose(torch.tensor(labels_gt)))
        expected_amounts = sequences.payload[PRESENCE_PROB] * sequences.payload["amounts"]
        self.assertTrue(expected_amounts.squeeze().allclose(torch.tensor(amounts_gt)))

        next_items = hd_module.predict_next(batch, None, None)  # (1, 4).
        probs_gt = [1.0, 1, 0.5, 0.8]
        timestamps_gt = [1.0, 1.5, 3.0, 3.5]
        labels_gt = [2, 0, 0, 0]
        amounts_gt = [2, 4, 1, 2.4]
        self.assertTrue(next_items.payload[PRESENCE_PROB].squeeze().allclose(torch.tensor(probs_gt), atol=1e-6))
        self.assertEqual(next_items.payload["timestamps"].squeeze().tolist(), timestamps_gt)
        self.assertEqual(next_items.payload["labels"].squeeze().tolist(), labels_gt)
        expected_amounts = next_items.payload[PRESENCE_PROB] * next_items.payload["amounts"]
        self.assertTrue(expected_amounts.squeeze().allclose(torch.tensor(amounts_gt), atol=1e-6))


if __name__ == "__main__":
    main()
