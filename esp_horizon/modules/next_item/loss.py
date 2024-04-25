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
    def input_dim(self):
        return sum([loss.input_dim for loss in self._losses.values()])

    def forward(self, predictions, targets):
        shifted_lengths = (targets.seq_lens - 1).clip(min=0)  # (B).
        mask = torch.arange(targets.seq_feature_shape[1] - 1, device=targets.device)[None] < shifted_lengths[:, None]  # (B, L).
        input_offset = 0
        losses = []
        for name in self._order:
            loss = self._losses[name]
            output = predictions.payload[:, :, input_offset:input_offset + loss.input_dim]
            target = targets.payload[name]
            # Shift target.
            output = output[:, :-1]
            target = target[:, 1:]
            losses.append(loss(output, target, mask))
            input_offset += loss.input_dim
        assert input_offset == self.input_dim
        return torch.stack(losses).sum()
