import torch

from esp_horizon.data import PaddedBatch


class NextItemLoss(torch.nn.Module):
    """Hybrid loss for next item prediction.

    Args:
        losses: Mapping from the feature name to the loss function.
        prediction: The type of prediction (either `mean` or `mode`).
    """
    def __init__(self, losses, prediction="mean"):
        super().__init__()
        self._losses = torch.nn.ModuleDict(losses)
        self._order = list(sorted(losses))
        self._prediction = prediction

    @property
    def fields(self):
        return self._order

    @property
    def input_dim(self):
        return sum([loss.input_dim for loss in self._losses.values()])

    def get_delta_type(self, field):
        """Get time delta type."""
        return self._losses[field].delta

    def forward(self, inputs, predictions):
        """Extract targets and compute loss between predictions and targets.

        Args:
            inputs: Input features with shape (B, L).
            predictions: Predicted values with shape (B, L, P).

        Returns:
            Losses dict and metrics dict.
        """
        # Align lengths.
        l = min(predictions.shape[1], inputs.shape[1])
        lengths = torch.minimum(predictions.seq_lens, inputs.seq_lens)
        predictions = PaddedBatch(predictions.payload[:, :l], lengths)
        inputs = PaddedBatch({k: (v[:, :l] if k in inputs.seq_names else v)
                              for k, v in inputs.payload.items()},
                             lengths, inputs.seq_names)

        # Compute losses. It is assumed that predictions lengths are equal to targets lengths.
        predictions = self._split_predictions(predictions)
        mask = inputs.seq_len_mask.bool() if (inputs.seq_lens != inputs.shape[1]).any() else None
        losses = {}
        metrics = {}
        for name, output in predictions.items():
            losses[name], loss_metrics = self._losses[name](inputs.payload[name], output, mask)
            for k, v in loss_metrics.items():
                metrics[f"{name}-{k}"] = v
        return losses, metrics

    def predict_next(self, predictions, fields=None):
        seq_lens = predictions.seq_lens
        predictions = self._split_predictions(predictions)
        result = {}
        for name in (fields or self._losses):
            if self._prediction == "mode":
                result[name] = self._losses[name].predict_modes(predictions[name]).squeeze(-1)  # (B, L).
            elif self._prediction == "mean":
                result[name] = self._losses[name].predict_means(predictions[name]).squeeze(-1)  # (B, L).
            else:
                raise ValueError(f"Unknown prediction type: {self._prediction}.")
        return PaddedBatch(result, seq_lens)

    def predict_next_category_logits(self, predictions, fields=None):
        if fields is None:
            fields = [name for name, loss in self._losses.items() if hasattr(loss, "predict_logits")]
        seq_lens = predictions.seq_lens
        predictions = self._split_predictions(predictions)
        result = {}
        for name in fields:
            result[name] = self._losses[name].predict_logits(predictions[name])  # (B, L, C).
        return PaddedBatch(result, seq_lens)

    def _split_predictions(self, predictions):
        """Convert parameters tensor to the dictionary with parameters for each loss."""
        offset = 0
        result = {}
        for name in self._order:
            loss = self._losses[name]
            result[name] = predictions.payload[..., offset:offset + loss.input_dim]
            offset += loss.input_dim
        if offset != self.input_dim:
            raise ValueError("Predictions tensor has inconsistent size.")
        return result