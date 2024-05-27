import math
import torch

from .common import BaseLoss, compute_delta
from .tpp import thinning_expectation


class TimeRMTPPLoss(BaseLoss):
    """Temporal Point Process loss from RMTPP.

    See the original paper for details:

    Du, Nan, et al. "Recurrent marked temporal point processes:
    Embedding event history to vector." Proceedings of the 22nd ACM
    SIGKDD international conference on knowledge discovery and data
    mining. 2016.


    Args:
        delta: The type of time delta computation (`last` or `start`).
        grad_scale: Gradient scale to balance loss functions.
        init_influence: Initial value of the influence parameter.
        influence_dim: If greater than one, use individual influence for each time position (must be equal to L).
        force_negative_influence: Force current_influence is always negative.
        max_delta: Maximum time offset in sampling.
        max_intensity: Intensity threshold for preventing explosion.
        expectation_steps: The maximum sample size used for means prediction.
        eps: Small value used for influence thresholding in modes prediction.

    """
    def __init__(self, delta="last", max_delta=None, grad_scale=None,
                 init_influence=-1, influence_dim=1, force_negative_influence=True,
                 max_intensity=None, expectation_steps=None,
                 eps=1e-6):
        super().__init__(input_size=1, target_size=1,
                         grad_scale=grad_scale)
        self.delta = delta
        self.max_delta = max_delta
        self.eps = eps
        self.max_intensity = max_intensity
        self.force_negative_influence = force_negative_influence
        self.expectation_steps = expectation_steps
        # TODO: use predicted influence.
        # TODO: per-label parameter?
        self._hidden_current_influence = torch.nn.Parameter(torch.full([influence_dim], init_influence, dtype=torch.float))

    def get_current_influence(self, l=None):
        """Get influence vector.

        Args:
            l: Sequence length.

        Returns:
            Influence vector.
        """
        value = self._hidden_current_influence
        if self.force_negative_influence:
            value = -value.abs()
        if (l is not None) and (value.numel() != 1):
            if (l != 1) and (l != value.numel()):
                raise ValueError("Influence dimension and input length mismatch")
            value = value[:l]  # Return the first value for the next item prediction.
        return value

    def _log_intensity(self, influence, biases, deltas):
        log_intencities = influence * deltas + biases  # (B, L).
        if self.max_intensity is not None:
            log_intencities = log_intencities.clip(max=math.log(self.max_intensity))
        return log_intencities

    def compute_loss(self, inputs, predictions, mask=None):
        """Compute losses and metrics.

        NOTE: the model predicts next inputs.

        Args:
            inputs: Input features with shape (B, L).
            predictions: Model outputs with shape (B, L, P).
            mask: Sequence lengths mask with shape (B, L) or None.

        Returns:
            Losses tensor with shape (B, L'), (optional) mask tensor with shape (B, L') and metrics dictionary.
        """
        assert predictions.shape[2] == 1
        predictions = predictions[:, :-1].squeeze(2)  # (B, L - 1).
        deltas, mask = compute_delta(inputs, mask, delta=self.delta, max_delta=self.max_delta)

        log_intencities = self._log_intensity(self.get_current_influence(deltas.shape[1]), predictions, deltas)  # (B, L).
        log_densities = log_intencities - (log_intencities.exp() - predictions.exp()) / self.get_current_influence()  # (B, L).
        losses = -log_densities  # (B, L).

        with torch.no_grad():
            metrics = {
                "current_influence": self.get_current_influence().mean().item(),
                "bias-mean": (predictions[mask] if mask is not None else predictions).mean().item()
            }
        return losses, mask, metrics

    def predict_modes(self, predictions):
        """Predict distributions modes.

        Args:
            predictions: Parameters with shape (*, L, 1).

        Returns:
            Modes with shape (*, L, 1).
        """
        assert predictions.shape[-1] == 1
        biases = predictions.squeeze(-1)  # (*, L).
        influence = self.get_current_influence(biases.shape[-1])
        clipped_influence = influence.clip(min=self.eps)  # (L).
        modes = torch.where(influence < self.eps,
                            torch.zeros_like(biases),
                            (clipped_influence.log() - biases) / clipped_influence)  # (*, L).
        # Delta is always positive.
        return modes.unsqueeze(-1).clip(min=0)  # (*, L, 1).

    def predict_means(self, predictions):
        """Predict distributions means.

        Args:
            predictions: Parameters with shape (*, L, 1).

        Returns:
            Means with shape (*, L, 1).
        """
        if self.expectation_steps is None:
            raise ValueError("Need maximum expectation steps for the mean estimation.")
        if self.max_delta is None:
            raise ValueError("Need maximum time delta for sampling.")
        assert predictions.shape[-1] == 1
        biases = predictions.squeeze(-1).flatten(0, -2)  # (B, L).

        b, l = biases.shape
        influence = self.get_current_influence(l)
        if (influence > 0).any():
            raise RuntimeError("Can't sample with positive current influence.")
        expectations = thinning_expectation(b, l,
                                            intensity_fn=lambda deltas: self._log_intensity(influence, biases, deltas).exp(),
                                            max_steps=self.expectation_steps,
                                            max_delta=self.max_delta,
                                            dtype=biases.dtype, device=biases.device)  # (B, L).
        return expectations.unsqueeze(2)
