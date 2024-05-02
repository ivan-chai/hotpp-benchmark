from abc import ABC, abstractmethod
from typing import Tuple, Dict
import torch

from esp_horizon.data import PaddedBatch


class BaseAdapter(ABC, torch.nn.Module):
    """Base class for ESP models adapters.

    Args:
      model: Base model.
      time_field: Time feature name.
      time_int: If true, cast time to int before model inference.
      time_input_delta: Whether the model accepts time deltas or raw time features.
      time_output_delta: Whether the model return time deltas or raw time features.
      category_mapping: Mapping dictionaries for category features.
    """
    def __init__(self, model, time_field="timestamps", time_int=False,
                 time_input_delta=True, time_output_delta=True,
                 category_mapping=None):
        super().__init__()
        self.model = model
        self.time_field = time_field
        self.time_int = time_int
        self.time_input_delta = time_input_delta
        self.time_output_delta = time_output_delta
        self.categories = {name: i for i, name in enumerate(category_mapping or {})}
        for name, i in self.categories.items():
            orig, new = zip(*category_mapping[name].items())
            mapping = torch.zeros(max(orig) + 1, dtype=torch.long).scatter_(0, torch.tensor(orig), torch.tensor(new))
            inv_mapping = torch.zeros(max(new) + 1, dtype=torch.long).scatter_(0, torch.tensor(new), torch.tensor(orig))
            self.register_buffer(f"category_mapping_{i}", mapping)
            self.register_buffer(f"category_inv_mapping_{i}", inv_mapping)

    def prepare_features(self, batch: PaddedBatch) -> PaddedBatch:
        """Format features for the model.

        The method handles time in a special way:
        1. saves initial time for delta inverse,
        2. computes deltas if necessary,
        3. applies rounding if integer time is required.

        Args:
          batch: PaddedBatch with shape (B, T, D).
        """
        batch = batch.clone()
        times = batch.payload[self.time_field].to(torch.float, copy=True)

        # Save time for delta inversion.
        batch.payload["_times"] = times.clone()
        batch.seq_names.add("_times")

        # Compute time delta if necessary.
        if self.time_input_delta:
            times[:, 1:] -= batch.payload["_times"][:, :-1]
            if batch.left:
                # Set initial delta to zero.
                starts = (times.shape[1] - batch.seq_lens).clip(max=times.shape[1] - 1).unsqueeze(1)  # (B, 1).
                times.scatter_(1, starts, 0)
            else:
                times[:, 0] = 0

        if self.time_int:
            times = times.round_().long()

        batch.payload[self.time_field] = times

        for field, i in self.categories.items():
            mapping = getattr(self, f"category_mapping_{i}")
            batch.payload[field] = mapping[batch.payload[field].flatten()].reshape(*batch.payload[field].shape)
        return batch

    def revert_features(self, batch: PaddedBatch) -> PaddedBatch:
        """Inverse of prepare features.

        The method handles time in a special way:
        1. reverts deltas if necessary,
        2. applies rounding to get Unix time.

        Args:
          batch: PaddedBatch with shape (B, T, D).
        """
        batch = batch.clone()

        for field, i in self.categories.items():
            mapping = getattr(self, f"category_inv_mapping_{i}")
            batch.payload[field] = mapping[batch.payload[field].flatten()].reshape(*batch.payload[field].shape)

        # Replace model time (can be delta or rounded) with global time.
        times = batch.payload.pop("_times")
        if self.time_int:
            times = times.round_().long()
        batch.payload[self.time_field] = times

        return batch

    def output_to_next_input(self, batch: PaddedBatch, outputs: Dict) -> Tuple[torch.Tensor, Dict]:
        """Update features between generation iterations.

        The method handles time in a special way:
        1. Converts output delta to input time if necessary,
        2. Converts output time to input delta if necessary,
        3. Applies rounding if integer time is required,
        4. Computes output time deltas.

        Args:
          batch: Current input batch of features with shape (B, T).
          outputs: Model output dict with feature shapes (B) (single token without time dimension).

        Returns:
          Time offsets and next iteration features.
        """
        outputs = outputs.copy()
        if not batch.left:
            last = (batch.seq_lens - 1).unsqueeze(1)  # (B).

        # Get current time for delta inversion.
        if batch.left:
            current_times = batch.payload["_times"][:, -1]  # (B).
        else:
            current_times = batch.payload["_times"].take_along_dim(last, 1).squeeze(1)  # (B).

        times = outputs[self.time_field].to(torch.float, copy=True)  # (B).
        assert times.ndim == 1

        # Force non-decreasing time.
        if self.time_output_delta:
            times.clip_(min=0)
        else:
            times = torch.maximum(times, current_times)

        # Compute time deltas (UNIX time).
        if self.time_output_delta:
            time_deltas = times.clone()
        else:
            time_deltas = times - current_times

        # Generate new input.
        if self.time_input_delta and (not self.time_output_delta):
            times -= current_times
        if self.time_output_delta and (not self.time_input_delta):
            times += current_times
        if self.time_int:
            times = times.round_().long()
        outputs[self.time_field] = times

        # Update _times.
        outputs["_times"] = current_times + time_deltas

        return time_deltas, outputs


class BaseRNNAdapter(BaseAdapter):
    @abstractmethod
    def eval_states(self, x: PaddedBatch) -> PaddedBatch:
        """Apply encoder to the batch of features and produce batch of input hidden states for each iteration.

        Args:
          - x: Payload contains dictionary of features with shapes (B, T, D_i).

        Returns:
          PaddedBatch with payload containing hidden states with shape (B, T, D).
        """
        pass

    @abstractmethod
    def forward(self, x: PaddedBatch, states: torch.Tensor=None) -> Tuple[Dict, torch.Tensor]:
        """Predict features given inputs and hidden states.

        Args:
          - x: Payload contains dictionary of features with shapes (B, T, D_i).
          - states: Initial states with shape (B, D).

        Returns:
          Next token features and new states with shape (B, D).
        """
        pass
