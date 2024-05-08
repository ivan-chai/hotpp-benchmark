import math
from abc import ABC, abstractmethod
import torch
from esp_horizon.data import PaddedBatch


def time_to_delta(predictions, targets, mask):
    """Convert parameters to time delta prediction."""
    if targets.ndim != 2:
        raise ValueError(f"Expected targets with shape (B, L), got {targets.shape}.")
    # An input sequence contains predictions shifted w.r.t. targets:
    # prediction: 1, 2, 3, ...
    # target: 2, 3, 4, ...
    #
    # After a time delta computation, the model have to predict an offset to the next event:
    # new_prediction: 2, 3, ...
    # delta_target: 3 - 2, 4 - 3, ...
    predictions = predictions[:, 1:]  #  (B, L - 1).
    delta = targets[:, 1:] - targets[:, :-1]  # (B, L - 1).
    mask = torch.logical_and(mask[:, 1:], mask[:, :-1])  # (B, L - 1).
    return predictions, delta, mask


class ScaleGradient(torch.autograd.Function):
    """Scale gradient."""

    @staticmethod
    def forward(ctx, src, weight):
        ctx._weight = weight
        return src

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx._weight, None


class BaseLoss(ABC, torch.nn.Module):
    def __init__(self, grad_scale=None):
        super().__init__()
        self.grad_scale = grad_scale

    def forward(self, predictions, targets, mask):
        """Compute loss between predictions and targets.

        Args:
            predictions: Predicted values with shape (B, L, 1).
            targets: Target values with shape (B, L), shifted w.r.t. predictions.
            mask: Sequence lengths mask with shape (B, L).

        Returns:
            Loss and metrics.
        """
        loss, metrics = self.compute_loss(predictions, targets, mask)
        if self.grad_scale is not None:
            loss = ScaleGradient.apply(loss, self.grad_scale)
        return loss, metrics

    @abstractmethod
    def compute_loss(predictions, targets, mask):
        pass


class TimeMAELoss(BaseLoss):
    """MAE for delta T prediction."""
    input_dim = 1
    target_dim = 1

    def compute_loss(self, predictions, targets, mask):
        """Compute MAE loss between predictions and targets.

        Args:
            predictions: Predicted values with shape (B, L, 1).
            targets: Target values with shape (B, L), shifted w.r.t. predictions.
            mask: Sequence lengths mask with shape (B, L).

        Returns:
            Loss and metrics.
        """
        assert predictions.shape[2] == 1
        predictions = predictions.squeeze(2)
        predictions, deltas, mask = time_to_delta(predictions, targets, mask)
        losses = (predictions - deltas).abs()  # (B, L - 1).
        assert losses.ndim == 2
        return losses[mask].mean(), {}

    def predict_modes(self, predictions):
        return predictions  # (B, L, 1).

    def predict_means(self, predictions):
        return predictions  # (B, L, 1).


class TimeRMTPPLoss(BaseLoss):
    """Temporal Point Process loss from RMTPP.

    See the original paper for details:

    Du, Nan, et al. "Recurrent marked temporal point processes:
    Embedding event history to vector." Proceedings of the 22nd ACM
    SIGKDD international conference on knowledge discovery and data
    mining. 2016.


    Args:
        init_influence: Initial value of the influence parameter.
        max_intensity: Intensity threshold for preventing explosion.
        force_negative_influence: Force current_influence is always negative.
        max_delta: Maximum time offset in sampling.
        expectation_steps: The maximum sample size used for means prediction.
        grad_scale: Gradient scale to balance loss functions.
        eps: Small value used for influence thresholding in modes prediction.

    """
    input_dim = 1
    target_dim = 1

    def __init__(self, init_influence=1, max_intensity=None, force_negative_influence=True,
                 max_delta=None, expectation_steps=None,
                 grad_scale=None, eps=1e-6):
        super().__init__(grad_scale=grad_scale)
        self.eps = eps
        self.max_intensity = max_intensity
        self.force_negative_influence = force_negative_influence
        self.max_delta = max_delta
        self.expectation_steps = expectation_steps
        # TODO: use predicted influence.
        # TODO: per-label parameter?
        self._hidden_current_influence = torch.nn.Parameter(torch.full([], init_influence, dtype=torch.float))

    @property
    def current_influence(self):
        value = self._hidden_current_influence
        if self.force_negative_influence:
            value = -value.abs()
        return value

    def _log_intensity(self, biases, deltas):
        log_intencities = self.current_influence * deltas + biases
        if self.max_intensity is not None:
            log_intencities = log_intencities.clip(max=math.log(self.max_intensity))
        return log_intencities

    def compute_loss(self, predictions, targets, mask):
        """Compute RMTPP loss between predictions and targets.

        Args:
            predictions: Predicted values with shape (B, L, 1).
            targets: Target values with shape (B, L), shifted w.r.t. predictions.
            mask: Sequence lengths mask with shape (B, L).

        Returns:
            Loss and metrics.
        """
        assert predictions.shape[2] == 1
        predictions = predictions.squeeze(2)
        biases, deltas, mask = time_to_delta(predictions, targets, mask)  # (B, L).

        log_intencities = self._log_intensity(biases, deltas)  # (B, L).
        log_densities = log_intencities - (log_intencities.exp() - biases.exp()) / self.current_influence  # (B, L).
        losses = -log_densities  # (B, L).
        assert losses.ndim == 2
        loss = losses[mask].mean()

        with torch.no_grad():
            metrics = {
                "current_influence": self.current_influence.item(),
                "bias-mean": biases[mask].mean().item()
            }
        return loss, metrics

    def predict_modes(self, predictions):
        assert predictions.shape[2] == 1
        biases = predictions.squeeze(2)  # (B, L).
        if self.current_influence < self.eps:
            modes = torch.zeros_like(biases)
        else:
            modes = (self.current_influence.log() - biases) / self.current_influence
        return modes.unsqueeze(2)  # (B, L, 1).

    def predict_means(self, predictions):
        if self.expectation_steps is None:
            raise ValueError("Need maximum expectation steps for mean estimation.")
        assert predictions.shape[2] == 1
        biases = predictions.squeeze(2)  # (B, L).
        b, l = biases.shape
        sample, mask = self._sample(biases.reshape(b * l), self.expectation_steps)  # (BL, N), (BL, N).
        empty = ~mask.any(1)  # (BL).
        if empty.any():
            sample[empty, 0] = self.max_delta
            mask[empty, 0] = True
        expectations = (sample * mask).sum(1) / mask.sum(1)  # (BL).
        return expectations.reshape(b, l, 1)

    def _sample(self, biases, max_steps):
        """Apply thinning algorithm.

        Args:
            biases: The biases tensor with shape (B).
            max_steps: The maximum number of steps in thinning algorithm.

        Returns:
            Samples tensor with shape (B, N) and acceptance tensor with shape (B, N).
        """
        if self.max_delta is None:
            raise ValueError("Need maximum time delta for sampling.")
        if self.current_influence > 0:
            raise RuntimeError("Can't sample with positive current influence.")
        bs, = biases.shape

        samples = torch.zeros(max_steps + 1, bs, dtype=biases.dtype, device=biases.device)  # (N, B).
        rejected = torch.ones(max_steps + 1, bs, dtype=torch.bool, device=biases.device)  # (N, B).
        for i in range(1, max_steps + 1):
            upper = (self.current_influence * samples[i - 1] + biases).exp()  # (B).
            tau = -torch.rand_like(upper).log() / upper  # (B).
            samples[i] = samples[i - 1] * rejected[i - 1] + tau
            rejected[i] = torch.rand_like(upper) * upper >= (self.current_influence * samples[i] + biases).exp()
            mask = samples[i] > self.max_delta
            samples[i, mask] = 0
            rejected[i, mask] = 1
        return samples[1:].T, (~rejected[1:]).T


class CrossEntropyLoss(BaseLoss):
    target_dim = 1

    def __init__(self, num_classes, grad_scale=None):
        super().__init__(grad_scale=grad_scale)
        self.input_dim = num_classes

    @property
    def num_classes(self):
        return self.input_dim

    def compute_loss(self, predictions, targets, mask):
        """Compute cross-entropy loss between predictions and targets.

        Args:
            predictions: Predicted values with shape (B, L, C).
            targets: Target values with shape (B, L), shifted w.r.t. predictions.
            mask: Sequence lengths mask with shape (B, L).

        Returns:
            Loss and metrics.
        """
        if targets.ndim != 2:
            raise ValueError(f"Expected targets with shape (B, L), got {targets.shape}.")
        losses = torch.nn.functional.cross_entropy(predictions.permute(0, 2, 1), targets.long(), reduction="none")  # (B, T).
        assert losses.ndim == 2
        return losses[mask].mean(), {}

    def predict_logits(self, predictions):
        return predictions  # (B, L, C).

    def predict_modes(self, predictions):
        return predictions.argmax(-1).unsqueeze(2)  # (B, L, 1).

    def predict_means(self, predictions):
        # There is no mean for a categorical distribution. Return modes.
        return predictions.argmax(-1).unsqueeze(2)  # (B, L, 1).


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
    def input_dim(self):
        return sum([loss.input_dim for loss in self._losses.values()])

    def forward(self, predictions, targets):
        """Compute loss between predictions and targets.

        Args:
            predictions: Predicted values with shape (B, L, C).
            targets: Target values with shape (B, L), shifted w.r.t. predictions.

        Returns:
            Losses dict and metrics dict.
        """
        # Align lengths.
        l = min(predictions.shape[1], targets.shape[1])
        lengths = torch.minimum(predictions.seq_lens, targets.seq_lens)
        predictions = PaddedBatch(predictions.payload[:, :l], lengths)
        targets = PaddedBatch({k: (v[:, :l] if k in targets.seq_names else v)
                               for k, v in targets.payload.items()},
                              lengths, targets.seq_names)

        # Compute losses.
        predictions = self._split_predictions(predictions)
        mask = targets.seq_len_mask.bool()
        losses = {}
        metrics = {}
        for name, output in predictions.items():
            target = targets.payload[name]
            loss, loss_metrics = self._losses[name](output, target, mask)
            losses[name] = loss
            for k, v in loss_metrics.items():
                metrics[f"{name}-{k}"] = v
        return losses, metrics

    def predict(self, predictions, fields=None):
        seq_lens = predictions.seq_lens
        predictions = self._split_predictions(predictions)
        result = {}
        for name in (fields or self._losses):
            if self._prediction == "mode":
                result[name] = self._losses[name].predict_modes(predictions[name]).squeeze(2)  # (B, L).
            elif self._prediction == "mean":
                result[name] = self._losses[name].predict_means(predictions[name]).squeeze(2)  # (B, L).
            else:
                raise ValueError(f"Unknown prediction type: {self._prediction}.")
        return PaddedBatch(result, seq_lens)

    def predict_category_logits(self, predictions, fields=None):
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
            result[name] = predictions.payload[:, :, offset:offset + loss.input_dim]
            offset += loss.input_dim
        if offset != self.input_dim:
            raise ValueError("Predictions tensor has inconsistent size.")
        return result
