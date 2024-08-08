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


def compute_delta(inputs, mask=None,
                  delta="last", smoothing=None,
                  max_delta=None, exclude_out_of_horizon=False):
    if delta == "last":
        deltas = inputs[:, 1:] - inputs[:, :-1]  # (B, L - 1).
        mask = torch.logical_and(mask[:, 1:], mask[:, :-1]) if mask is not None else None  # (B, L - 1).
    elif delta == "start":
        deltas = inputs[:, 1:] - inputs[:, :1]  # (B, L - 1).
        mask = torch.logical_and(mask[:, 1:], mask[:, :1]) if mask is not None else None  # (B, L - 1).
    else:
        raise ValueError(f"Unknown delta type: {delta}.")
    deltas = deltas.clip(min=0, max=max_delta)
    if smoothing is not None:
        deltas = deltas.float()
        deltas = deltas + (torch.rand_like(deltas) - 0.5) * smoothing  # (-S / 2; S / 2).
        deltas = deltas.abs()  # Make positive, but don't generate a large amount of zeros, like in clipping.
    if (max_delta is not None) and exclude_out_of_horizon:
        mask[deltas >= max_delta] = False
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
        self._interpolator = None

    @property
    def interpolator(self):
        return self._interpolator

    @interpolator.setter
    def interpolator(self, value):
        self._interpolator = value

    @property
    def input_size(self):
        return self._input_size

    @property
    def target_size(self):
        return self._target_size

    def forward(self, inputs, predictions, mask=None, reduction="mean"):
        """Compute loss between predictions and targets.

        NOTE: the model predicts next inputs.

        Args:
            inputs: Input features with shape (B, L, *).
            predictions: Mode outputs with shape (B, L, *, P).
            mask: Sequence lengths mask with shape (B, L) or None.
            reduction: `mean` or `none`.

        Returns:
            Loss and metrics.
        """
        loss, mask, metrics = self.compute_loss(inputs, predictions, mask)
        if (reduction != "none") and (mask is not None):
            loss = loss[mask]
        if reduction == "mean":
            loss = loss.mean()
        elif reduction != "none":
            raise ValueError(f"Unknown reduction: {reduction}.")
        if self._grad_scale is not None:
            loss = ScaleGradient.apply(loss, self._grad_scale)
        return loss, metrics

    @abstractmethod
    def compute_loss(self, inputs, predictions, mask=None):
        """Compute losses and metrics.

        NOTE: the model predicts next inputs.

        Args:
            inputs: Input features with shape (B, L, *).
            predictions: Mode outputs with shape (B, L, *, P) or (B, 1, *, P).
            mask: Sequence lengths mask with shape (B, L) or None.

        Returns:
            Losses tensor with shape (B, L', *), (optional) mask tensor with shape (B, L') and metrics dictionary.
        """
        pass

    @abstractmethod
    def predict_modes(self, predictions):
        """Get most probable outputs.

        Args:
            predictions: Model outputs with shape (B, L, P).

        Returns:
            Modes with shape (B, L, D).
        """
        pass

    @abstractmethod
    def predict_means(self, predictions):
        """Get expected outputs.

        Args:
            predictions: Model outputs with shape (B, L, P).

        Returns:
            Means with shape (B, L, D).
        """
        pass

    @abstractmethod
    def predict_samples(self, predictions, temperature=1):
        """Sample outputs.

        Args:
            predictions: Model outputs with shape (B, L, P).

        Returns:
            Samples with shape (B, L, D).
        """
        pass


class TimeMAELoss(BaseLoss):
    """MAE for delta T prediction.

    Args:
        delta: The type of time delta computation (`last` or `start`).
        smoothing: The amount of noise to add to time deltas. Useful for discrete time to prevent spiky intensity.
    """
    def __init__(self, delta="last", max_delta=None, smoothing=None, grad_scale=None):
        super().__init__(input_size=1, target_size=1,
                         grad_scale=grad_scale)
        self.delta = delta
        self.max_delta = max_delta
        self.smoothing = smoothing

    def compute_loss(self, inputs, predictions, mask=None):
        """Compute losses and metrics.

        NOTE: the model predicts next inputs.

        Args:
            inputs: Input features with shape (B, L, *).
            predictions: Mode outputs with shape (B, L, *, P) or (B, 1, *, P).
            mask: Sequence lengths mask with shape (B, L) or None.

        Returns:
            Losses tensor with shape (B, L', *), (optional) mask tensor with shape (B, L') and metrics dictionary.
        """
        assert predictions.shape[-1] == 1
        predictions = predictions.squeeze(-1)  # (B, L, *).
        broadcast = (predictions.shape[1] != inputs.shape[1]) and (predictions.shape[1] == 1)
        predictions = predictions if broadcast else predictions[:, :-1]  # (B, L - 1, *).
        deltas, mask = compute_delta(inputs, mask, delta=self.delta,
                                     max_delta=self.max_delta, smoothing=self.smoothing)  # (B, L - 1, *).
        losses = (predictions - deltas).abs()  # (B, L - 1, *).
        return losses, mask, {}

    def predict_modes(self, predictions):
        # Delta is always positive.
        return predictions.clip(min=0)  # (B, L, 1).

    def predict_means(self, predictions):
        # Delta is always positive.
        return predictions.clip(min=0)  # (B, L, 1).

    def predict_samples(self, predictions, temperature=1):
        # Delta is always positive.
        return predictions.clip(min=0)  # (B, L, 1).


class TimeMSELoss(BaseLoss):
    """MSE for delta T prediction.

    Args:
        delta: The type of time delta computation (`last` or `start`).
        smoothing: The amount of noise to add to time deltas. Useful for discrete time to prevent spiky intensity.
    """
    def __init__(self, delta="last", max_delta=None, smoothing=None, grad_scale=None):
        super().__init__(input_size=1, target_size=1,
                         grad_scale=grad_scale)
        self.delta = delta
        self.max_delta = max_delta
        self.smoothing = smoothing

    def compute_loss(self, inputs, predictions, mask=None):
        """Compute losses and metrics.

        NOTE: the model predicts next inputs.

        Args:
            inputs: Input features with shape (B, L, *).
            predictions: Mode outputs with shape (B, L, *, P) or (B, 1, *, P).
            mask: Sequence lengths mask with shape (B, L) or None.

        Returns:
            Losses tensor with shape (B, L', *), (optional) mask tensor with shape (B, L') and metrics dictionary.
        """
        assert predictions.shape[-1] == 1
        predictions = predictions.squeeze(-1)  # (B, L, *).
        broadcast = (predictions.shape[1] != inputs.shape[1]) and (predictions.shape[1] == 1)
        predictions = predictions if broadcast else predictions[:, :-1]  # (B, L - 1, *).
        deltas, mask = compute_delta(inputs, mask, delta=self.delta,
                                     max_delta=self.max_delta, smoothing=self.smoothing)  # (B, L - 1, *).
        losses = (predictions - deltas).square()  # (B, L - 1, *).
        return losses, mask, {}

    def predict_modes(self, predictions):
        # Delta is always positive.
        return predictions.clip(min=0)  # (B, L, 1).

    def predict_means(self, predictions):
        # Delta is always positive.
        return predictions.clip(min=0)  # (B, L, 1).

    def predict_samples(self, predictions, temperature=1):
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
            inputs: Input features with shape (B, L, *).
            predictions: Mode outputs with shape (B, L, *, P) or (B, 1, *, P).
            mask: Sequence lengths mask with shape (B, L) or None.

        Returns:
            Losses tensor with shape (B, L', *), (optional) mask tensor with shape (B, L') and metrics dictionary.
        """
        if inputs.ndim < 2:
            raise ValueError(f"Expected labels with shape (B, L, *), got {inputs.shape}.")
        # Extract targets from features.
        broadcast = (predictions.shape[1] != inputs.shape[1]) and (predictions.shape[1] == 1)
        predictions = predictions if broadcast else predictions[:, :-1]  # (B, L - 1, *, C).

        targets = inputs[:, 1:].long()  # (B, L - 1, *).
        mask = torch.logical_and(mask[:, 1:], mask[:, :-1]) if mask is not None else None  # (B, L - 1).

        # Compute loss.
        lognorms = torch.logsumexp(predictions, dim=-1)  # (B, L - 1, *).
        logits = predictions.take_along_dim(targets.unsqueeze(-1), -1).squeeze(-1)  # (B, L - 1, *).
        losses = lognorms - logits  # (B, L - 1, *).
        return losses, mask, {}

    def predict_logits(self, predictions):
        return predictions  # (B, L, C).

    def predict_modes(self, predictions):
        return predictions.argmax(-1).unsqueeze(-1)  # (B, L, 1).

    def predict_means(self, predictions):
        # There is no mean for a categorical distribution. Return modes.
        return self.predict_modes(predictions)

    def predict_samples(self, predictions, temperature=1):
        probs = torch.nn.functional.softmax(predictions / temperature, dim=-1)  # (B, L, C).
        return torch.distributions.categorical.Categorical(probs).sample().unsqueeze(-1)  # (B, L, 1).


class BinaryCrossEntropyLoss(BaseLoss):
    target_size = 1

    def __init__(self, grad_scale=None):
        super().__init__(input_size=1, target_size=1,
                         grad_scale=grad_scale)

    @property
    def num_classes(self):
        return 2

    def compute_loss(self, inputs, predictions, mask=None):
        """Compute cross-entropy loss between predictions and targets.

        NOTE: the model predicts next inputs.

        Args:
            inputs: Input features with shape (B, L, *).
            predictions: Mode outputs with shape (B, L, *, P) or (B, 1, *, P).
            mask: Sequence lengths mask with shape (B, L) or None.

        Returns:
            Losses tensor with shape (B, L', *), (optional) mask tensor with shape (B, L') and metrics dictionary.
        """
        if inputs.ndim < 2:
            raise ValueError(f"Expected labels with shape (B, L, *), got {inputs.shape}.")
        # Extract targets from features.
        assert predictions.shape[-1] == 1
        predictions = predictions.squeeze(-1)  # (B, L, *).
        broadcast = (predictions.shape[1] != inputs.shape[1]) and (predictions.shape[1] == 1)
        predictions = predictions if broadcast else predictions[:, :-1]  # (B, L - 1, *).
        targets = inputs[:, 1:].long()  # (B, L - 1, *).
        mask = torch.logical_and(mask[:, 1:], mask[:, :-1]) if mask is not None else None  # (B, L - 1).

        # Compute loss.
        neg_logprobs = torch.nn.functional.logsigmoid(-predictions)
        pos_logprobs = torch.nn.functional.logsigmoid(predictions)
        losses = -torch.where(targets.bool(), pos_logprobs, neg_logprobs)  # (B, L - 1, *).
        return losses, mask, {}

    def predict_logits(self, predictions):
        return torch.nn.functional.logsigmoid(predictions)  # (B, L, 1).

    def predict_modes(self, predictions):
        return (predictions > 0).long()  # (B, L, 1).

    def predict_means(self, predictions):
        # There is no mean for a categorical distribution. Return modes.
        return self.predict_modes(predictions)

    def predict_samples(self, predictions, temperature=1):
        probs = torch.sigmoid(predictions / temperature)  # (B, L, 1).
        return (torch.rand_like(probs) < probs).long()
