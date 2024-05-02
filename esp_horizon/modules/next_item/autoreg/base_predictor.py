from abc import ABC, abstractmethod
from typing import List

import torch
from esp_horizon.data import PaddedBatch


class BaseSequencePredictor(ABC, torch.nn.Module):
    """Predict future sequences for multiple positions in input data."""
    def __init__(self, max_steps):
        super().__init__()
        self.max_steps = max_steps

    @abstractmethod
    def forward(self, batch: PaddedBatch, indices: PaddedBatch) -> List[PaddedBatch]:
        """Predict sequences as a list of PaddedBatches each with shape (I, T), where I is the number of indices."""
        pass
