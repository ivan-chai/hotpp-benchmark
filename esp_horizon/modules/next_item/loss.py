import torch


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
        delta = targets[:, 1:] - targets[:, :-1]  # (B, L - 1).
        predictions = predictions[:, 1:].squeeze(2)  #  (B, L - 1).
        mask = mask[:, 1:]  # (B, L - 1).
        losses = (predictions - delta).abs()  # (B, L - 1).
        assert losses.ndim == 2
        return losses[mask].mean()


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

    def get_proba(self, predictions):
        return torch.nn.functional.softmax(predictions, -1)  # (B, L, C).


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
        predictions = self.split_predictions(predictions)
        mask = targets.seq_len_mask.bool()
        losses = []
        for name, output in predictions.items():
            target = targets.payload[name]
            losses.append(self._losses[name](output, target, mask))
        return torch.stack(losses).sum()

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
