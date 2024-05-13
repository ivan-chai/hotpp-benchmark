import math
import torch

from .common import BaseLoss, compute_delta


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
    def __init__(self, init_influence=1, max_intensity=None, force_negative_influence=True,
                 max_delta=None, expectation_steps=None,
                 grad_scale=None, eps=1e-6):
        super().__init__(input_dim=1, target_dim=1,
                         grad_scale=grad_scale)
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
        assert predictions.shape[2] == 1
        predictions = predictions[:, :-1].squeeze(2)  # (B, L - 1).
        deltas, mask = compute_delta(inputs, mask, delta="last")

        log_intencities = self._log_intensity(predictions, deltas)  # (B, L).
        log_densities = log_intencities - (log_intencities.exp() - predictions.exp()) / self.current_influence  # (B, L).
        losses = -log_densities  # (B, L).

        with torch.no_grad():
            metrics = {
                "current_influence": self.current_influence.item(),
                "bias-mean": (predictions[mask] if mask is not None else predictions).mean().item()
            }
        return losses, mask, metrics

    def predict_modes(self, predictions):
        assert predictions.shape[-1] == 1
        biases = predictions.squeeze(-1)  # (B, L).
        if self.current_influence < self.eps:
            modes = torch.zeros_like(biases)
        else:
            modes = (self.current_influence.log() - biases) / self.current_influence
        return modes.unsqueeze(-1)  # (B, L, 1).

    def predict_means(self, predictions):
        if self.expectation_steps is None:
            raise ValueError("Need maximum expectation steps for mean estimation.")
        assert predictions.shape[-1] == 1
        biases = predictions.squeeze(-1)  # (B, L).
        b, l = biases.shape
        sample, mask = self._sample(biases.flatten(), self.expectation_steps)  # (BL, N), (BL, N).
        empty = ~mask.any(1)  # (BL).
        if empty.any():
            sample[empty, 0] = self.max_delta
            mask[empty, 0] = True
        expectations = (sample * mask).sum(1) / mask.sum(1)  # (BL).
        return expectations.reshape(*biases.shape).unsqueeze(-1)  # (B, L, 1).

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
