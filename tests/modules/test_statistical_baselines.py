from unittest import TestCase, main

import torch

from esp_horizon.data import PaddedBatch
from esp_horizon.modules import RecentHistoryModule, MostPopularModule


class TestStatisticalBaselines(TestCase):
    def test_generation(self):
        rh_module = RecentHistoryModule(2, 3)
        mp_module = MostPopularModule(2, 3)
        batch = PaddedBatch({
            "timestamps": torch.tensor([0.5, 1, 2.5, 3])[None],
            "labels": torch.tensor([2, 0, 1, 0])[None]
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
        # Mean deltas: 0, 0.25, 2.166..., 0.625
        # Top labels: 2, 0, 0, 0
        # Timestamps: cumsum deltas + initial
        # 1.25 1.5
        # 3.625 4.25
        # Labels:
        # 0 0
        # 0 0
        timestamps_gt = [[1.25, 1.5], [3.625, 4.25]]
        labels_gt = [[0, 0], [0, 0]]
        self.assertEqual(sequences.payload["timestamps"].squeeze().tolist(), timestamps_gt)
        self.assertEqual(sequences.payload["labels"].squeeze().tolist(), labels_gt)


if __name__ == "__main__":
    main()
