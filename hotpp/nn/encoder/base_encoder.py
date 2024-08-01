from abc import ABC, abstractmethod, abstractproperty
import torch

from ptls.nn import TrxEncoder
from hotpp.data import PaddedBatch


class BaseEncoder(torch.nn.Module):
    """Combines embedder and sequential model.

    Args:
        embeddings: Dict with categorical feature names. Values must be like this `{'in': dictionary_size, 'out': embedding_size}`.
        timestamps_field: The name of the timestamps field.
        max_time_delta: Limit maximum time delta at the model input.
        embedder_batch_norm: Use batch normalization in embedder.
    """
    def __init__(self,
                 embeddings,
                 timestamps_field="timestamps",
                 max_time_delta=None,
                 embedder_batch_norm=True):
        super().__init__()
        self.embedder = TrxEncoder(
            embeddings=embeddings,
            numeric_values={timestamps_field: "identity"},
            use_batch_norm=embedder_batch_norm,
            use_batch_norm_with_lens=True
        )
        self._timestamps_field = timestamps_field
        self._max_time_delta = max_time_delta

    @abstractproperty
    def hidden_size(self):
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

    def embed(self, x, compute_time_deltas=True):
        if compute_time_deltas:
            x = self.compute_time_deltas(x)
        embeddings = self.embedder(x)
        # Convert PTLS batch to HoTPP batch.
        return PaddedBatch(embeddings.payload, embeddings.seq_lens)

    @abstractmethod
    def forward(self, x, return_full_states=False):
        """Apply the model.

        Args:
            x: PaddedBatch with input features with shape (B, T).
            return_full_states: Whether to return full states with shape (B, T, D)
                or only final states with shape (B, D).

        Returns:
            Dictionary with "outputs" and optional "states" keys.
            Outputs is a PaddedBatch with shape (B, T, D).
            States (if provided) is a PaddedBatch with shape (N, B, T, D).
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
