from abc import ABC, abstractmethod, abstractproperty
import torch

from hotpp.data import PaddedBatch
from .embedder import Embedder


class BaseEncoder(torch.nn.Module):
    """Combines embedder and sequential model.

    Args:
        embedder: An instance of embedder Class for input events encoding.
        numeric_values: Dict with numeric feature names (including timestamps). Values must be one of "identity", "sigmoid", "log", and "year".
        timestamps_field: The name of the timestamps field.
        max_time_delta: Limit maximum time delta at the model input.
        embedder_batch_norm: Use batch normalization in embedder.
    """
    def __init__(self,
                 embedder,
                 timestamps_field="timestamps",
                 max_time_delta=None):
        super().__init__()
        self.embedder = embedder
        self._timestamps_field = timestamps_field
        self._max_time_delta = max_time_delta

    @abstractproperty
    def need_states(self):
        """Whether encoder uses states for inference optimization or not."""
        pass

    @abstractproperty
    def hidden_size(self):
        """The output dimension of the encoder."""
        pass

    def compute_time_deltas(self, x):
        """Replace timestamps with time deltas."""
        field = self._timestamps_field
        deltas = x.payload[field].clone()
        deltas[:, 1:] -= x.payload[field][:, :-1]
        deltas[:, 0] = 0
        deltas.clip_(min=0, max=self._max_time_delta)
        x = x.clone()
        x.payload[field] = deltas
        return x

    def apply_embedder(self, x, compute_time_deltas=True):
        """Transform input features into embeddings."""
        if compute_time_deltas:
            x = self.compute_time_deltas(x)
        return self.embedder(x)

    def embed(self, x):
        """Extract embeddings with shape (B, D)."""
        raise NotImplementedError("Encoder doesn't support embeddings extraction.")

    @abstractmethod
    def forward(self, x, return_states=False):
        """Apply the model.

        Args:
            x: PaddedBatch with input features with shape (B, T).
            return_states: Whether to return final states with shape (B, D), full states with shape (B, T, D)
                or no states (either False, "last" or "full").

        Returns:
            Outputs is with shape (B, T, D) and states with shape (N, B, D) or (N, B, T, D) or None, if return_states is False.
        """
        pass

    @abstractmethod
    def interpolate(self, states, time_deltas):
        """Compute model outputs in continuous time.

        Args:
            states: Model states with shape (N, B, L, H), where N is the number of layers.
            time_deltas: Relative timestamps with shape (B, L, S), where S is a sample size.

        Returns:
            Outputs with shape (B, L, S, D).
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
