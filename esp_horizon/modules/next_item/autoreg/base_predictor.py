from abc import ABC, abstractmethod
from typing import List

import torch
from esp_horizon.data import PaddedBatch


class BaseSequencePredictor(ABC, torch.nn.Module):
    """Predict future sequences for multiple positions in input data.

    Args:
        max_steps: The maximum number of generated items.
    """
    def __init__(self, max_steps):
        super().__init__()
        self.max_steps = max_steps

    @abstractmethod
    def forward(self, batch: PaddedBatch, indices: PaddedBatch) -> PaddedBatch:
        """Predict future events as a batch with the shape (B, I, N), where I is the number of indices and N is the number of steps."""
        pass
