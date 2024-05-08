import torch
from esp_horizon.data import PaddedBatch
from ..next_item import NextItemLoss


class NextKLoss(torch.nn.Module):
    """Hybrid loss for next item prediction.

    Args:
        k: The number of future events to predict.
        losses: Mapping from the feature name to the loss function.
        prediction: The type of prediction (either `mean` or `mode`).
        loss_step: The period of loss evaluation.
    """
    def __init__(self, k, losses, prediction="mean", loss_step=1):
        super().__init__()
        self._k = k
        self._loss_step = loss_step
        self._next_item = NextItemLoss(losses, prediction=prediction)

    @property
    def num_events(self):
        return self._k

    @property
    def fields(self):
        return self._next_item.fields

    @property
    def input_dim(self):
        return self._k * self._next_item.input_dim

    def forward(self, predictions, targets):
        """Compute loss between predictions and targets.

        Args:
            predictions: Predicted values with shape (B, L, C).
            targets: Inputs for predictions (without shift) with shape (B, L).

        Returns:
            Losses dict and metrics dict.
        """
        x = targets  # Inputs.

        # Join targets before windowing.
        b, l = x.shape
        targets = torch.stack([x.payload[name] for name in self.fields], -1)  # (B, L, D).

        # Extract windows.
        targets = self.extract_windows(targets, self._k + 1)   # (B, L - k, k + 1, D).
        assert targets.shape[:3] == (b, max(l - self._k, 0), self._k + 1)
        lengths = (x.seq_lens - self._k).clip(min=0)

        # Skip input.
        targets = targets[:, :, 1:]  # (B, L - k, k, D).

        # Apply step.
        if self._loss_step > 1:
            lengths = (lengths - self._loss_step - 1).div(self._loss_step, rounding_mode="floor").clip(min=-1) + 1
            targets = targets[:, self._loss_step::self._loss_step]  # (B, L', k, D).
            predictions = PaddedBatch(predictions.payload[:, self._loss_step::self._loss_step], lengths)  # (B, L', P).

        # Split targets.
        assert len(self.fields) == targets.shape[-1]
        windows = {name: targets[..., i] for i, name in enumerate(self.fields)}  # (B, L', k).
        targets = PaddedBatch(windows, lengths, x.seq_names)  # (B, L', k).

        # Reshape predictions.
        b, l_ = predictions.shape
        predictions = PaddedBatch(predictions.payload.reshape(b, l_, self._k, self._next_item.input_dim),
                                  predictions.seq_lens)  # (B, L', k, P).

        # Select by mask.
        mask = targets.seq_len_mask.bool()
        lengths = torch.full([mask.sum().item()], self._k)  # (V).
        targets = PaddedBatch({k: v[mask] for k, v in targets.payload.items()},
                              lengths, targets.seq_names)  # (V, k).
        predictions = PaddedBatch(predictions.payload[:, :mask.shape[1]][mask], lengths)  # (V, k, P).

        losses, metrics = self._next_item(predictions, targets, shifted=True)
        return losses, metrics

    def predict_next(self, predictions, fields=None):
        # Select parameters of the first predicted event.
        b, l = predictions.shape
        predictions = PaddedBatch(predictions.payload.reshape(b, l, self._k, self._next_item.input_dim)[:, :, 0, :],
                                  predictions.seq_lens)  # (B, L, P).
        return self._next_item.predict_next(predictions, fields=fields)  # (B, L).

    def predict_next_category_logits(self, predictions, fields=None):
        # Select parameters of the first predicted event.
        b, l = predictions.shape
        predictions = PaddedBatch(predictions.payload.reshape(b, l, self._k, self._next_item.input_dim)[:, :, 0, :],
                                  predictions.seq_lens)  # (B, L, P).
        return self._next_item.predict_next_category_logits(predictions, fields=fields)  # (B, L).

    def predict_next_k(self, predictions, fields=None, dump_category_logits=None):
        b, l = predictions.shape
        predictions = PaddedBatch(predictions.payload.reshape(b, l, self._k, self._next_item.input_dim),
                                  predictions.seq_lens)  # (B, L, K, P).
        results = self._next_item.predict_next(predictions, fields=fields)  # (B, L, K).
        if dump_category_logits:
            logits = self._next_item.predict_next_category_logits(predictions, fields=set(dump_category_logits))
            payload = dict(results.payload)
            payload.update({logits_field: logits.payload[field]
                            for field, logits_field in dump_category_logits.items()})
            result = PaddedBatch(payload, results.seq_lens,
                                 seq_names=set(results.seq_names) | set(dump_category_logits.values()))
        return result

    @staticmethod
    def extract_windows(x, t):
        """Convert tensor with shape (B, L, D) to a tensor with
        shape (B, L - t + 1, t, D) containing sliding windows of length t."""
        b, l, d = x.shape
        if l - t + 1 <= 0:
            return x[:, :0].reshape(b, 0, t, d)
        parts = [x.roll(-i, 1) for i in range(t)]
        return torch.stack(parts, 2)[:, :l - t + 1]
