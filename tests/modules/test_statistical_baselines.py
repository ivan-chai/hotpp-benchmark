from unittest import TestCase, main

import torch

from hotpp.data import PaddedBatch
from hotpp.fields import PRESENCE_PROB
from hotpp.modules import RecentHistoryModule, MostPopularModule, MergeHistoryModule


class TestStatisticalBaselines(TestCase):
    def test_generation(self):
        rh_module = RecentHistoryModule(2, 3)
        mp_module = MostPopularModule(2, 3)
        mh_module = MergeHistoryModule(3, [1, 2], amounts_field="amounts")  # num_classes, horizons.
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

        # Test MergeHistory.
        sequences = mh_module.generate_sequences(batch, indices)
        # Horizon 1
        # windows:
        # [], [0.5], [], [2.5]
        # labels:
        # [], [2], [], [1]
        # amounts:
        # [], [1], [], [3]
        # avg counts:
        # [0, 0, 0], [0, 0, 0.5], [0, 0, 0.33], [0, 0.25, 0.25]
        # avg amounts:
        # [0, 0, 0], [0, 0, 0.5], [0, 0, 0.33], [0, 0.75, 0.25]
        #
        # Horizon 2
        # windows:
        # [], [0.5], [0.5, 1], [1, 2.5]
        # labels:
        # [], [2], [2, 0], [0, 1]
        # amounts:
        # [], [1], [1, 2], [2, 3]
        # avg counts:
        # [0, 0, 0], [0, 0, 0.5], [0.33, 0, 0.66], [0.5, 0.25, 0.5]
        # avg amounts:
        # [0, 0, 0], [0, 0, 0.5], [0.66, 0, 0.66], [1.0, 0.75, 0.5]
        #
        # Horizon delta
        # avg counts:
        # [0, 0, 0], [0, 0, 0], [0.33, 0, 0.33], [0.5, 0, 0.25]
        # avg amounts:
        # [0, 0, 0], [0, 0, 0], [0.66, 0, 0.33], [1.0, 0, 0.25]
        probs_gt = [[0.0, 0.0, 0.5, 0.0, 0.0, 0.0], [0.0, 0.25, 0.25, 0.5, 0.0, 0.25]]
        timestamps_gt = [[1.5, 1.5, 1.5, 2.5, 2.5, 2.5], [3.5, 3.5, 3.5, 4.5, 4.5, 4.5]]
        labels_gt = [[0, 1, 2, 0, 1, 2], [0, 1, 2, 0, 1, 2]]
        amounts_gt = [[0.0, 0.0, 0.5, 0.0, 0.0, 0.0], [0.0, 0.75, 0.25, 1, 0.0, 0.25]]
        self.assertEqual(sequences.payload[PRESENCE_PROB].squeeze().tolist(), probs_gt)
        self.assertEqual(sequences.payload["timestamps"].squeeze().tolist(), timestamps_gt)
        self.assertEqual(sequences.payload["labels"].squeeze().tolist(), labels_gt)
        expected_amounts = sequences.payload[PRESENCE_PROB] * sequences.payload["amounts"]
        self.assertEqual(expected_amounts.squeeze().tolist(), amounts_gt)

        next_items = mh_module.predict_next(batch, None, None)  # (1, 4).
        probs_gt = [0.0, 0.5, 1 / 3, 0.25]
        timestamps_gt = [1.0, 1.5, 3.0, 3.5]
        labels_gt = [0, 2, 2, 1]
        amounts_gt = [0.0, 0.5, 1 / 3, 0.75]
        self.assertTrue(next_items.payload[PRESENCE_PROB].squeeze().allclose(torch.tensor(probs_gt), atol=1e-6))
        self.assertEqual(next_items.payload["timestamps"].squeeze().tolist(), timestamps_gt)
        self.assertEqual(next_items.payload["labels"].squeeze().tolist(), labels_gt)
        expected_amounts = next_items.payload[PRESENCE_PROB] * next_items.payload["amounts"]
        self.assertTrue(expected_amounts.squeeze().allclose(torch.tensor(amounts_gt), atol=1e-6))


if __name__ == "__main__":
    main()
