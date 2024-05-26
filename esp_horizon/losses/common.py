from abc import ABC, abstractmethod

import torch


class ScaleGradient(torch.autograd.Function):
    """Scale gradient during backward pass."""

    @staticmethod
    def forward(ctx, src, weight):
        ctx._weight = weight
        return src

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx._weight, None


def compute_delta(inputs, mask=None, delta="last", max_delta=None):
    if delta == "last":
        deltas = inputs[:, 1:] - inputs[:, :-1]  # (B, L - 1).
        mask = torch.logical_and(mask[:, 1:], mask[:, :-1]) if mask is not None else None  # (B, L - 1).
    elif delta == "start":
        deltas = inputs[:, 1:] - inputs[:, :1]  # (B, L - 1).
        mask = torch.logical_and(mask[:, 1:], mask[:, :1]) if mask is not None else None  # (B, L - 1).
    else:
        raise ValueError(f"Unknown delta type: {delta}.")
    if max_delta is not None:
        deltas = deltas.clip(max=max_delta)
    return deltas, mask


class BaseLoss(ABC, torch.nn.Module):
    """Base loss class.

    Each loss has some parametrization (input_size) and a number of outputs (target_size).

    Args:
        input_size: The number of parameters.
        target_size: Target value dimension.
        grad_scale: The backward pass gradient scale.
    """
    def __init__(self, input_size, target_size, grad_scale=None):
        super().__init__()
        self._input_size = input_size
        self._target_size = target_size
        self._grad_scale = grad_scale

    @property
    def input_size(self):
        return self._input_size

    @property
    def target_size(self):
        return self._target_size

    def forward(self, inputs, predictions, mask=None):
        """Compute loss between predictions and targets.

        NOTE: the model predicts next inputs.

        Args:
            inputs: Input features with shape (B, L).
            predictions: Mode outputs with shape (B, L, P).
            mask: Sequence lengths mask with shape (B, L) or None.

        Returns:
            Loss and metrics.
        """
        losses, mask, metrics = self.compute_loss(inputs, predictions, mask)
        assert losses.ndim == 2
        if mask is not None:
            losses = losses[mask]
        loss = losses.mean()
        if self._grad_scale is not None:
            loss = ScaleGradient.apply(loss, self._grad_scale)
        return loss, metrics

    @abstractmethod
    def compute_loss(self, inputs, predictions, mask=None):
        """Compute losses and metrics.

        NOTE: the model predicts next inputs.

        Args:
            inputs: Input features with shape (B, L).
            predictions: Mode outputs with shape (B, L, P).
            mask: Sequence lengths mask with shape (B, L) or None.

        Returns:
            Losses tensor with shape (B, L'), (optional) mask tensor with shape (B, L') and metrics dictionary.
        """
        pass

    @abstractmethod
    def predict_modes(self, predictions):
        """Get most probable outputs.

        Args:
            predictions: Mode outputs with shape (B, L, P).

        Returns:
            Modes with shape (B, L, D).
        """
        pass

    @abstractmethod
    def predict_means(self, predictions):
        """Get expected outputs.

        Args:
            predictions: Mode outputs with shape (B, L, P).

        Returns:
            Means with shape (B, L, D).
        """
        pass


class TimeMAELoss(BaseLoss):
    """MAE for delta T prediction.

    Args:
        delta: The type of time delta computation (`last` or `start`).
    """
    def __init__(self, delta="last", max_delta=None, grad_scale=None):
        super().__init__(input_size=1, target_size=1,
                         grad_scale=grad_scale)
        self.delta = delta
        self.max_delta = max_delta

    def compute_loss(self, inputs, predictions, mask=None):
        """Compute losses and metrics.

        NOTE: the model predicts next inputs.

        Args:
            inputs: Input features with shape (B, L).
            predictions: Mode outputs with shape (B, L, D).
            mask: Sequence lengths mask with shape (B, L) or None.

        Returns:
            Losses tensor with shape (B, L'), (optional) mask tensor with shape (B, L') and metrics dictionary.
        """
        assert predictions.shape[2] == 1
        predictions = predictions[:, :-1].squeeze(2)  # (B, L - 1).
        deltas, mask = compute_delta(inputs, mask, delta=self.delta, max_delta=self.max_delta)
        losses = (predictions - deltas).abs()  # (B, L - 1).
        return losses, mask, {}

    def predict_modes(self, predictions):
        # Delta is always positive.
        return predictions.clip(min=0)  # (B, L, 1).

    def predict_means(self, predictions):
        # Delta is always positive.
        return predictions.clip(min=0)  # (B, L, 1).


class CrossEntropyLoss(BaseLoss):
    target_size = 1

    def __init__(self, num_classes, grad_scale=None):
        super().__init__(input_size=num_classes, target_size=1,
                         grad_scale=grad_scale)

    @property
    def num_classes(self):
        return self.input_size

    def compute_loss(self, inputs, predictions, mask=None):
        """Compute cross-entropy loss between predictions and targets.

        NOTE: the model predicts next inputs.

        Args:
            inputs: Input features with shape (B, L).
            predictions: Mode outputs with shape (B, L, D).
            mask: Sequence lengths mask with shape (B, L) or None.

        Returns:
            Losses tensor with shape (B, L'), (optional) mask tensor with shape (B, L') and metrics dictionary.
        """
        if inputs.ndim != 2:
            raise ValueError(f"Expected labels with shape (B, L), got {inputs.shape}.")
        # Extract targets from features.
        predictions = predictions[:, :-1]  # (B, L - 1).
        targets = inputs[:, 1:].long()  # (B, L - 1).
        mask = torch.logical_and(mask[:, 1:], mask[:, :-1]) if mask is not None else None  # (B, L - 1).

        # Compute loss.
        losses = torch.nn.functional.cross_entropy(predictions.permute(0, 2, 1), targets, reduction="none")  # (B, T).
        return losses, mask, {}

    def predict_logits(self, predictions):
        return predictions  # (B, L, C).

    def predict_modes(self, predictions):
        return predictions.argmax(-1).unsqueeze(-1)  # (B, L, 1).

    def predict_means(self, predictions):
        # There is no mean for a categorical distribution. Return modes.
        return self.predict_modes(predictions)
