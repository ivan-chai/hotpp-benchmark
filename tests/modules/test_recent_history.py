from unittest import TestCase, main

import torch

from esp_horizon.data import PaddedBatch
from esp_horizon.modules import RecentHistoryModule


class TestRecentHistory(TestCase):
    def test_generation(self):
        module = RecentHistoryModule(2, 3)
        batch = PaddedBatch({
            "timestamps": torch.tensor([0.5, 1, 2.5, 3])[None],
            "labels": torch.tensor([2, 0, 1, 0])[None]
        }, torch.tensor([4]))
        indices = PaddedBatch(torch.tensor([1, 3])[None], torch.tensor([2]))
        sequences = module.generate_sequences(batch, indices)
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


if __name__ == "__main__":
    main()
