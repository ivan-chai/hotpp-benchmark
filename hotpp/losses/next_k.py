import torch

from hotpp.data import PaddedBatch
from hotpp.utils.torch import deterministic
from .next_item import NextItemLoss


class NextKLoss(torch.nn.Module):
    """Hybrid loss for next item prediction.

    Args:
        next_item_loss: An instance of the next event prediction loss used for pairwise loss computation.
        k: The number of future events to predict.
        prediction: The type of prediction (either `mean` or `mode`).
        loss_step: The period of loss evaluation.
    """
    def __init__(self, next_item_loss, k, timestamps_field="timestamps", loss_step=1):
        super().__init__()
        self._next_item = next_item_loss
        self._k = k
        self._timestamps_field = timestamps_field
        self._loss_step = loss_step

    @property
    def need_interpolator(self):
        return self._next_item.need_interpolator

    @property
    def interpolator(self):
        return self._next_item.interpolator

    @interpolator.setter
    def interpolator(self, value):
        self._next_item.interpolator = value

    @property
    def num_events(self):
        return self._k

    @property
    def fields(self):
        return self._next_item.fields

    @property
    def input_size(self):
        return self._k * self._next_item.input_size

    def get_delta_type(self, field):
        """Get time delta type."""
        return self._next_item.get_delta_type(field)

    def forward(self, inputs, outputs, states):
        """Extract targets and compute loss between predictions and targets.

        Args:
            inputs: Input features with shape (B, L).
            outputs: Predicted values with shape (B, L, P).
            states: Hidden model states with shape (N, B, L, D), where N is the number of layers.

        Returns:
            Losses dict and metrics dict.
        """
        # Join targets before windowing.
        b, l = inputs.shape
        targets = torch.stack([inputs.payload[name] for name in self.fields], -1)  # (B, L, D).

        # Extract windows.
        targets = self.extract_windows(targets, self._k + 1)   # (B, L - k, k + 1, D).
        assert targets.shape[:3] == (b, max(l - self._k, 0), self._k + 1)
        lengths = (inputs.seq_lens - self._k).clip(min=0)

        # Truncate lengths to match targets.
        outputs = PaddedBatch(outputs.payload[:, :targets.shape[1]], lengths)
        states = states[:, :, :targets.shape[1]]

        # Apply step.
        if self._loss_step > 1:
            lengths = (lengths - self._loss_step - 1).div(self._loss_step, rounding_mode="floor").clip(min=-1) + 1
            targets = targets[:, self._loss_step::self._loss_step]  # (B, L', k + 1, D).
            outputs = PaddedBatch(outputs.payload[:, self._loss_step::self._loss_step], lengths)  # (B, L', P).
            states = states[:, :, self._loss_step::self._loss_step]  # (N, B, L', D).

        # Split targets.
        assert len(self.fields) == targets.shape[-1]
        windows = {name: targets[..., i] for i, name in enumerate(self.fields)}  # (B, L', k + 1).
        targets = PaddedBatch(windows, lengths, inputs.seq_names)  # (B, L', k + 1).

        # Reshape predictions.
        b, l_ = outputs.shape
        outputs = PaddedBatch(outputs.payload.reshape(b, l_, self._k, self._next_item.input_size),
                              outputs.seq_lens)  # (B, L', k, P).

        # Select by mask.
        mask = targets.seq_len_mask.bool()  # (B, L').
        lengths = torch.full([mask.sum().item()], self._k + 1)  # (V).
        targets = PaddedBatch({k: v[mask] for k, v in targets.payload.items()},
                              lengths, targets.seq_names)  # (V, k + 1).
        payload = outputs.payload[:, :mask.shape[1]][mask]  # (V, k, P).
        payload = torch.cat([payload, payload[:, -1:]], dim=1)  # (V, k + 1, P).
        outputs = PaddedBatch(payload, lengths)  # (V, k + 1, P).
        n, _, _, d = states.shape
        states = states.masked_select(mask[None, :, :, None]).reshape(n, len(lengths), 1, d)  # (N, V, 1, D).

        losses, metrics = self._next_item(targets, outputs, states)
        return losses, metrics

    def predict_next(self, outputs, states, fields=None, logits_fields_mapping=None):
        """Predict next events.

        Args:
            outputs: Model outputs.
            states: Hidden model states with shape (N, B, L, D), where N is the number of layers.
            fields: The fields to predict next values for. By default, predict all fields.
            logits_fields_mapping: A mapping from field to the output logits field to predict logits for.

        Returns:
            PaddedBatch with predictions with shape (B, L) or (B, L, C) for logits.
        """
        # Select parameters of the first predicted event.
        b, l = outputs.shape
        lengths = outputs.seq_lens
        outputs = PaddedBatch(outputs.payload.reshape(b * l, self._k, self._next_item.input_size)[:, :1, :],
                              torch.ones(b * l, device=outputs.device, dtype=torch.long))  # (BL, 1, P).
        states = states.reshape(len(states), b * l, 1, -1)  # (N, BL, , D).
        next_values = self._next_item.predict_next(outputs, states,
                                                   fields=fields,
                                                   logits_fields_mapping=logits_fields_mapping)  # (BL, 1) or (BL, 1, C).
        return PaddedBatch({k: v.reshape(b, l, *v.shape[2:]) for k, v in next_values.payload.items()},
                           lengths)  # (B, L) or (B, L, C).

    def predict_next_k(self, outputs, states, fields=None, logits_fields_mapping=None):
        """Predict K future events.

        Args:
            outputs: Model outputs.
            states: Hidden model states with shape (N, B, L, D), where N is the number of layers.
            fields: The fields to predict next values for. By default, predict all fields.
            logits_fields: A mapping from field to the output logits field to predict logits for.

        Returns:
            PaddedBatch with predictions with shape (B, L, K) or (B, L, K, C) for logits.
        """
        b, l = outputs.shape
        lengths = outputs.seq_lens
        outputs = PaddedBatch(outputs.payload.reshape(b * l, self._k, self._next_item.input_size),
                              torch.full([b * l], self._k, device=outputs.device, dtype=torch.long))  # (BL, K, P).
        states = states.reshape(len(states), b * l, 1, -1)  # (N, BL, 1, D).
        next_values = self._next_item.predict_next(outputs, states,
                                                   fields=fields,
                                                   logits_fields_mapping=logits_fields_mapping)  # (BL, K) or (BL, K, C).
        sequences = PaddedBatch({k: v.reshape(b, l, self._k, *v.shape[2:]) for k, v in next_values.payload.items()},
                                lengths)  # (B, L, K) or (B, L, K, C).
        self.revert_delta_and_sort_time_inplace(sequences)
        return sequences

    def revert_delta_and_sort_time_inplace(self, sequences):
        # Revert delta type.
        delta_type = self.get_delta_type(self._timestamps_field)
        if delta_type == "last":
            with deterministic(False):
                sequences.payload[self._timestamps_field].cumsum_(2)
        elif delta_type != "start":
            raise ValueError(f"Unknown delta type: {self.delta_type}.")
        # Sort by time.
        order = sequences.payload[self._timestamps_field].argsort(dim=2)  # (B, L, K).
        for k in sequences.seq_names:
            payload = sequences.payload[k]
            shaped_order = order.reshape(*(list(order.shape) + [1] * (payload.ndim - order.ndim)))  # (B, L, K, *).
            sequences.payload[k] = payload.take_along_dim(shaped_order, dim=2)  # (B, L, K, *).
        return sequences

    @staticmethod
    def extract_windows(x, t):
        """Convert tensor with shape (B, L, D) to a tensor with
        shape (B, L - t + 1, t, D) containing sliding windows of length t."""
        b, l, d = x.shape
        if l - t + 1 <= 0:
            return x[:, :0].reshape(b, 0, t, d)
        parts = [x.roll(-i, 1) for i in range(t)]
        return torch.stack(parts, 2)[:, :l - t + 1]
