import torch
from esp_horizon.data import PaddedBatch


class TimeMAELoss(torch.nn.Module):
    """MAE for delta T prediction."""
    input_dim = 1
    target_dim = 1

    def forward(self, predictions, targets, mask):
        """Compute MAE loss between predictions and targets.

        Args:
            predictions: Predicted values with shape (B, L, 1).
            targets: Target values with shape (B, L), shifted w.r.t. predictions.
            mask: Sequence lengths mask with shape (B, L).
        """
        if targets.ndim != 2:
            raise ValueError(f"Expected targets with shape (B, L), got {targets.shape}.")
        # An input sequence contains predictions shifted w.r.t. targets:
        # prediction: 1, 2, 3, ...
        # target: 2, 3, 4, ...
        #
        # After a time delta computation, the model have to predict an offset to the next event:
        # new_prediction: 2, 3, ...
        # delta_target: 3 - 2, 4 - 3, ...
        predictions = predictions[:, 1:].squeeze(2)  #  (B, L - 1).
        delta = targets[:, 1:] - targets[:, :-1]  # (B, L - 1).
        mask = mask[:, 1:]  # (B, L - 1).
        losses = (predictions - delta).abs()  # (B, L - 1).
        assert losses.ndim == 2
        return losses[mask].mean()

    def get_modes(self, predictions):
        return predictions  # (B, L, 1).


class CrossEntropyLoss(torch.nn.Module):
    target_dim = 1

    def __init__(self, num_classes):
        super().__init__()
        self.input_dim = num_classes

    @property
    def num_classes(self):
        return self.input_dim

    def forward(self, predictions, targets, mask):
        """Compute cross-entropy loss between predictions and targets.

        Args:
            predictions: Predicted values with shape (B, L, C).
            targets: Target values with shape (B, L), shifted w.r.t. predictions.
            mask: Sequence lengths mask with shape (B, L).
        """
        if targets.ndim != 2:
            raise ValueError(f"Expected targets with shape (B, L), got {targets.shape}.")
        losses = torch.nn.functional.cross_entropy(predictions.permute(0, 2, 1), targets.long(), reduction="none")  # (B, T).
        assert losses.ndim == 2
        return losses[mask].mean()

    def get_log_proba(self, predictions):
        return torch.nn.functional.log_softmax(predictions, -1)  # (B, L, C).

    def get_proba(self, predictions):
        return torch.nn.functional.softmax(predictions, -1)  # (B, L, C).

    def get_modes(self, predictions):
        return predictions.argmax(-1).unsqueeze(2)  # (B, L, 1).


class NextItemLoss(torch.nn.Module):
    """Hybrid loss for next item prediction.

    Args:
        losses: Mapping from the feature name to the loss function.
    """
    def __init__(self, losses):
        super().__init__()
        self._losses = losses
        self._order = list(sorted(losses))

    @property
    def loss_names(self):
        return self._order

    @property
    def input_dim(self):
        return sum([loss.input_dim for loss in self._losses.values()])

    def __getitem__(self, key):
        return self._losses[key]

    def forward(self, predictions, targets):
        """Compute loss between predictions and targets.

        Args:
            predictions: Predicted values with shape (B, L, C).
            targets: Target values with shape (B, L), shifted w.r.t. predictions.
        """
        # Align lengths.
        l = min(predictions.shape[1], targets.shape[1])
        lengths = torch.minimum(predictions.seq_lens, targets.seq_lens)
        predictions = PaddedBatch(predictions.payload[:, :l], lengths)
        targets = PaddedBatch({k: (v[:, :l] if k in targets.seq_names else v)
                               for k, v in targets.payload.items()},
                              lengths, targets.seq_names)

        # Compute losses.
        predictions = self.split_predictions(predictions)
        mask = targets.seq_len_mask.bool()
        losses = {}
        for name, output in predictions.items():
            target = targets.payload[name]
            losses[name] = self._losses[name](output, target, mask)
        return losses

    def get_modes(self, predictions):
        seq_lens = predictions.seq_lens
        predictions = self.split_predictions(predictions)
        result = {}
        for name, prediction in predictions.items():
            result[name] = self._losses[name].get_modes(prediction).squeeze(2)  # (B, L).
        return PaddedBatch(result, seq_lens)

    def get_logits(self, predictions):
        seq_lens = predictions.seq_lens
        predictions = self.split_predictions(predictions)
        result = {}
        for name, prediction in predictions.items():
            try:
                result[name] = self._losses[name].get_log_proba(prediction).squeeze(2)  # (B, L, D).
            except AttributeError:
                continue
        return PaddedBatch(result, seq_lens)

    def split_predictions(self, predictions):
        """Convert parameters tensor to the dictionary with parameters for each loss."""
        offset = 0
        result = {}
        for name in self.loss_names:
            loss = self._losses[name]
            result[name] = predictions.payload[:, :, offset:offset + loss.input_dim]
            offset += loss.input_dim
        if offset != self.input_dim:
            raise ValueError("Predictions tensor has inconsistent size.")
        return result
