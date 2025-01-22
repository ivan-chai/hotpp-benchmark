import math
from unittest import TestCase, main

import torch

from hotpp.data import PaddedBatch
from hotpp.nn.aggregator import *


class TestAggregators(TestCase):
    def test_aggregators(self):
        embeddings = torch.tensor([
            [0, 0, 1, 0, 0],
            [1, 0, 1, 0, 100],
            [1, -1, 1, -1, 1]
        ]).unsqueeze(2)  # (B, L, 1).
        lengths = torch.tensor([5, 3, 0])
        embeddings = PaddedBatch(embeddings, lengths)

        aggregator = MeanAggregator()
        result = aggregator(embeddings)
        gt = torch.tensor([1/5, 2/3, 0]).unsqueeze(1)
        self.assertTrue(result.allclose(gt))
        
        aggregator = LastAggregator()
        result = aggregator(embeddings)
        gt = torch.tensor([0, 1, 0]).unsqueeze(1)
        self.assertTrue(result.allclose(gt))

        aggregator = MiddleAggregator()
        result = aggregator(embeddings)
        gt = torch.tensor([1, 0, 0]).unsqueeze(1)
        self.assertTrue(result.allclose(gt))

        aggregator = MeanLastAggregator(n=2)
        result = aggregator(embeddings)
        gt = torch.tensor([0, 0.5, 0]).unsqueeze(1)
        self.assertTrue(result.allclose(gt))


if __name__ == "__main__":
    main()
