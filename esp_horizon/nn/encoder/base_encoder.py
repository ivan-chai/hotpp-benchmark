from abc import ABC, abstractmethod, abstractproperty
import torch

from ptls.nn import TrxEncoder
from esp_horizon.data import PaddedBatch


class BaseEncoder(torch.nn.Module):
    """Combines embedder and sequential model.

    Args:
        embeddings: Dict with categorical feature names. Values must be like this `{'in': dictionary_size, 'out': embedding_size}`.
        timestamps_field: The name of the timestamps field.
    """
    def __init__(self,
                 embeddings,
                 timestamps_field="timestamps"):
        super().__init__()
        self.embedder = TrxEncoder(
            embeddings=embeddings,
            numeric_values={timestamps_field: "identity"}
        )
        self._timestamps_field = timestamps_field

    @abstractproperty
    def embedding_size(self):
        pass

    def compute_time_deltas(self, x):
        """Replace timestamps with time deltas."""
        field = self._timestamps_field
        deltas = x.payload[field].clone()
        deltas[:, 1:] -= x.payload[field][:, :-1]
        deltas[:, 0] = 0
        x = x.clone()
        x.payload[field] = deltas
        return x

    def embed(self, x, compute_time_deltas=True):
        if compute_time_deltas:
            x = self.compute_time_deltas(x)
        return self.embedder(x)

    @abstractmethod
    def forward(self, x):
        """Apply the model.

        Args:
            x: PaddedBatch with input features with shape (B, T).

        Returns:
            Dictionary with "outputs" and optional "states" keys.
            Outputs is a PaddedBatch with shape (B, T, D).
        """
        pass

    @abstractmethod
    def generate(self, x, indices, predictor_fn, n_steps):
        """Use auto-regression to generate future sequence.

        Args:
            x: Batch of inputs with shape (B, T).
            indices: Output prediction indices for each element of the batch with shape (B, I).
            predictor_fn: A mapping from embedding to input features.
            n_steps: The maximum number of generated items.

        Returns:
            Predicted sequences as a batch with the shape (B, I, N), where I is the number of indices and N is the number of steps.
        """
        pass
