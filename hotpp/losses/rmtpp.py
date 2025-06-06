import math
import torch

from .common import BaseLoss, compute_delta
from .tpp import thinning_expectation, thinning_sample


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
        time_smoothing: The amount of noise to add to time deltas. Useful for discrete time to prevent spiky intensity.
        max_intensity: Intensity threshold for preventing explosion.
        thinning_params: A dictionary with thinning parameters.
        eps: Small value used for influence thresholding in modes prediction.

    """
    def __init__(self, delta="last", grad_scale=None,
                 init_influence=-1, influence_dim=1, force_negative_influence=True,
                 time_smoothing=None, max_intensity=None, thinning_params=None,
                 eps=1e-6):
        super().__init__(input_size=1, target_size=1,
                         grad_scale=grad_scale)
        self.delta = delta
        self.eps = eps
        self.time_smoothing = time_smoothing
        self.max_intensity = max_intensity
        self.force_negative_influence = force_negative_influence
        self.thinning_params = thinning_params
        # TODO: use predicted influence.
        # TODO: per-label parameter?
        self._hidden_current_influence = torch.nn.Parameter(torch.full([influence_dim], init_influence, dtype=torch.float))

    @property
    def need_interpolator(self):
        return False

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
        # influence: L, *.
        # biases: B, L, *.
        # deltas: B, L, *, S.
        log_intencities = influence.unsqueeze(-1) * deltas + biases.unsqueeze(-1)  # (B, L, S).
        if self.max_intensity is not None:
            log_intencities = log_intencities.clip(max=math.log(self.max_intensity))
        return log_intencities

    def compute_loss(self, inputs, predictions, mask=None):
        """Compute losses and metrics.

        NOTE: the model predicts next inputs.

        Args:
            inputs: Input features with shape (B, L, *).
            predictions: Model outputs with shape (B, L, *, P).
            mask: Sequence lengths mask with shape (B, L) or None.

        Returns:
            Losses tensor with shape (B, L', *), (optional) mask tensor with shape (B, L') and metrics dictionary.
        """

        assert predictions.shape[-1] == 1
        predictions = predictions.squeeze(-1)  # (B, L - 1, *).
        broadcast = (predictions.shape[1] != inputs.shape[1]) and (predictions.shape[1] == 1)
        predictions = predictions if broadcast else predictions[:, :-1]  # (B, L - 1, *).

        deltas, mask = compute_delta(inputs, mask, delta=self.delta, smoothing=self.time_smoothing)  # (B, L - 1, *)

        current_influence = self.get_current_influence(deltas.shape[1])
        current_influence = current_influence.reshape(*([len(current_influence)] + [1] * (deltas.ndim - 2)))  # (L - 1, *).
        log_intencities = self._log_intensity(current_influence, predictions, deltas.unsqueeze(-1)).squeeze(-1)  # (B, L - 1, *).
        log_densities = log_intencities - (log_intencities.exp() - predictions.exp()) / current_influence  # (B, L - 1, *).
        losses = -log_densities  # (B, L - 1, *).

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
        assert predictions.shape[-1] == 1
        biases = predictions.squeeze(-1).flatten(0, -2)  # (B, L).

        b, l = biases.shape
        influence = self.get_current_influence(l)
        if (influence > 0).any():
            raise RuntimeError("Can't sample with positive current influence.")
        expectations, _ = thinning_expectation(b, l,
                                               intensity_fn=lambda deltas: self._log_intensity(influence, biases, deltas).exp(),
                                               dtype=biases.dtype, device=biases.device,
                                               **self.thinning_params)  # (B, L).
        return expectations.unsqueeze(2)

    def predict_samples(self, predictions, temperature=1):
        """Sample from distributions.

        Args:
            predictions: Parameters with shape (*, L, 1).
            temperature (unused): Sampling temperature can't be applied to thinning.

        Returns:
            Means with shape (*, L, 1).
        """
        assert predictions.shape[-1] == 1
        biases = predictions.squeeze(-1).flatten(0, -2)  # (B, L).

        b, l = biases.shape
        influence = self.get_current_influence(l)
        if (influence > 0).any():
            raise RuntimeError("Can't sample with positive current influence.")
        samples, _ = thinning_sample(b, l,
                                     intensity_fn=lambda deltas: self._log_intensity(influence, biases, deltas).exp(),
                                     dtype=biases.dtype, device=biases.device,
                                     **self.thinning_params)  # (B, L).
        return samples.unsqueeze(2)
