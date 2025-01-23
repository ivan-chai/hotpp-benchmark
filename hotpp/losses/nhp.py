from functools import partial
import torch

from hotpp.data import PaddedBatch
from hotpp.utils.torch import module_mode, BATCHNORM_TYPES
from .common import compute_delta
from .tpp import thinning, thinning_expectation, thinning_sample


class NHPLoss(torch.nn.Module):
    """The loss for conditional time and label modeling.

    See the original paper for details:

    Mei, Hongyuan, and Jason M. Eisner. "The neural hawkes process: A neurally self-modulating
    multivariate point process." Advances in neural information processing systems 30 (2017).

    Args:
        num_classes: The number of possible event types.
        timestamps_field: The name of the timestamps field.
        labels_field: The name of the labels field.
        time_smoothing: The amount of noise to add to time deltas. Useful for discrete time to prevent spiky intensity.
        max_intensity: Intensity threshold for preventing explosion.
        likelihood_sample_size: The sample size per event to compute integral.
        thinning_params: A dictionary with thinning parameters.
        prediction: The type of prediction (either `mean`, `sample`, `sample-labels` or dictionary for each field).
        temperature: Labels sampling temperature (only when prediction is `sample`).
    """
    def __init__(self, num_classes,
                 timestamps_field="timestamps",
                 labels_field="labels",
                 time_smoothing=None,
                 max_intensity=None,
                 likelihood_sample_size=1,
                 thinning_params=None,
                 prediction="mean",
                 temperature=1):
        super().__init__()
        self._num_classes = num_classes
        self._timestamps_field = timestamps_field
        self._labels_field = labels_field
        self._time_smoothing = time_smoothing
        self._max_intensity = max_intensity
        self._likelihood_sample_size = likelihood_sample_size
        self._thinning_params = thinning_params or {}
        if prediction == "sample-labels":
            prediction = {timestamps_field: "mean", labels_field: "sample"}
        self._prediction = prediction
        self._temperature = temperature
        self._interpolator = None
        self.beta = torch.nn.Parameter(torch.ones(num_classes))

    @property
    def need_interpolator(self):
        return True

    @property
    def interpolator(self):
        return self._interpolator

    @interpolator.setter
    def interpolator(self, value):
        self._interpolator = value

    @property
    def fields(self):
        return [self._timestamps_field, self._labels_field]

    @property
    def input_size(self):
        return self._num_classes  # Intensity for each label.

    def get_delta_type(self, field):
        """Get time delta type."""
        return "last"

    def intensity(self, outputs, beta=None):
        if beta is None:
            beta = self.beta
        intensities = torch.nn.functional.softplus(outputs * beta) / beta / self._num_classes
        if self._max_intensity is not None:
            scale = (self._max_intensity / intensities.sum(-1, keepdim=True).clip(min=1e-6)).clip(max=1)
            intensities = intensities * scale
        return intensities

    def _forward_impl(self, inputs, outputs, states):
        # Align lengths.
        l = min(outputs.shape[1], inputs.shape[1])
        lengths = torch.minimum(outputs.seq_lens, inputs.seq_lens)
        inputs = PaddedBatch({k: (v[:, :l] if k in inputs.seq_names else v)
                              for k, v in inputs.payload.items()},
                             lengths, inputs.seq_names)

        # Extract targets.
        timestamps, mask = inputs.payload[self._timestamps_field], inputs.seq_len_mask  # (B, L), (B, L).
        lengths = (lengths - 1).clip(min=0)
        deltas, mask = compute_delta(timestamps, mask, smoothing=self._time_smoothing)
        labels = inputs.payload[self._labels_field][:, 1:].long().clip(min=0, max=self._num_classes - 1)  # (B, L).
        states = states[:, :, :l - 1]
        # states: (N, B, L, D).
        # deltas, labels: (B, L), shifted relative to states.

        outputs = self._interpolator(states, PaddedBatch(deltas.unsqueeze(2), lengths)).payload.squeeze(2)  # (B, L, D).
        outputs = outputs.take_along_dim(labels.unsqueeze(2), 2).squeeze(2)  # (B, L).
        betas = self.beta[None, None].take_along_dim(labels.unsqueeze(2), 2).squeeze(2)  # (B, L).
        log_intensities = self.intensity(outputs, betas).clip(min=1e-6).log()  # (B, L).

        sample_deltas = torch.rand(deltas.shape[0], deltas.shape[1], self._likelihood_sample_size,
                                   dtype=states.dtype, device=states.device)  # (B, L, S).
        sample_deltas *= deltas.unsqueeze(2)  # (B, L, S).
        sample_outputs = self._interpolator(states, PaddedBatch(sample_deltas, lengths)).payload  # (B, L, S, D).
        sample_intensities = self.intensity(sample_outputs)  # (B, L, S, D).
        integrals = sample_intensities.sum(3).mean(2) * deltas  # (B, L).
        # Negative Log likelihood, normalized by the number of events.
        loss = integrals.masked_select(mask).mean() - log_intensities.masked_select(mask).mean()
        losses = {
            "nhp": loss
        }
        metrics = {}
        return losses, metrics

    def forward(self, inputs, outputs, states):
        """Extract targets and compute loss between predictions and targets.

        Args:
            inputs: Input features with shape (B, L).
            outputs (unused): Model outputs with shape (B, L, D).
            states: Hidden model states with shape (N, B, L, D), where N is the number of layers.

        Returns:
            Losses dict and metrics dict.
        """
        # Don't hurt BatchNorm statistics during sampling.
        # We expect, that head statistics are updated during outputs computation in the base module.
        with module_mode(self._interpolator, training=False, layer_types=BATCHNORM_TYPES):
            losses, metrics = self._forward_impl(inputs, outputs, states)
        return losses, metrics

    def predict_next(self, outputs, states, fields=None, logits_fields_mapping=None):
        """Predict next events.

        Args:
            outputs: Model outputs with shape (B, L, D). It is used only for sequence lengths extraction.
            states: Hidden model states with shape (N, B, L, D), where N is the number of layers.
            fields: The fields to predict next values for. By default, predict all fields.
            logits_fields_mapping: A mapping from field to the output logits field to predict logits for.

        Returns:
            PaddedBatch with predictions with shape (B, L).
        """
        if (set(fields or []) | set(logits_fields_mapping or {})) - {self._timestamps_field, self._labels_field}:
            raise ValueError("Unexepcted field names")
        seq_lens = outputs.seq_lens
        _, b, l, _ = states.shape

        def intensity_fn(deltas):
            result = self._interpolator(states, PaddedBatch(deltas, seq_lens)).payload  # (B, L, S, D).
            assert result.ndim == 4
            intensities = self.intensity(result)  # (B, L, S, D).
            return intensities.sum(3)  # (B, L, S).

        prediction = self._prediction if isinstance(self._prediction, str) else self._prediction[self._timestamps_field]
        if prediction == "mean":
            timestamps, _ = thinning_expectation(b, l,
                                                 intensity_fn=intensity_fn,
                                                 dtype=states.dtype, device=states.device,
                                                 **self._thinning_params)  # (B, L), (B, L).
        elif prediction == "sample":
            timestamps, _ = thinning_sample(b, l,
                                            intensity_fn=intensity_fn,
                                            dtype=states.dtype, device=states.device,
                                            **self._thinning_params)  # (B, L), (B, L).
        else:
            raise ValueError(f"Unknown prediction type: {prediction}.")

        hiddens = self.interpolator(states, PaddedBatch(timestamps.unsqueeze(2), seq_lens)).payload.squeeze(2)  # (B, L, D).
        intensities = self.intensity(hiddens)  # (B, L, D).
        prediction = self._prediction if isinstance(self._prediction, str) else self._prediction[self._labels_field]
        if prediction == "mean":
            labels = intensities.argmax(2)  # (B, L).
        elif prediction == "sample":
            probs = intensities + torch.rand_like(intensities) * 1e-6  # (B, L, D).
            probs = probs / probs.sum(-1, keepdim=True).clip(min=1e-6)  # (B, L, D).
            if self._temperature != 1:
                log_probs = probs.clip(min=1e-6).log()
                probs = torch.nn.functional.softmax(log_probs / self._temperature, -1)  # (B, L, D).
            labels = torch.distributions.Categorical(probs).sample()  # (B, L).
        else:
            raise ValueError(f"Unknown prediction type: {prediction}.")

        result = {}
        if (fields is None) or (self._timestamps_field in fields):
            result[self._timestamps_field] = timestamps
        if (fields is None) or (self._labels_field in fields):
            result[self._labels_field] = labels
        if self._labels_field in (logits_fields_mapping or {}):
            probs = intensities / intensities.sum(2, keepdim=True)  # (B, L, D).
            log_probs = probs.clip(min=1e-6).log()
            if self._temperature != 1:
                log_probs = torch.nn.functional.log_softmax(log_probs / self._temperature, -1)  # (B, L, D).
            result[logits_fields_mapping[self._labels_field]] = log_probs

        return PaddedBatch(result, seq_lens)

    def compute_metrics(self, inputs, outputs, states):
        seq_lens = outputs.seq_lens
        _, b, l, _ = states.shape

        def intensity_fn(deltas):
            result = self._interpolator(states, PaddedBatch(deltas, seq_lens)).payload  # (B, L, S, D).
            assert result.ndim == 4
            intensities = self.intensity(result)  # (B, L, S, D).
            return intensities.sum(3)  # (B, L, S).

        _, accepted = thinning(b, l,
                               intensity_fn=intensity_fn,
                               dtype=states.dtype, device=states.device,
                               **self._thinning_params)  # (B, L, N).

        accepted = accepted.sum(2)  # (B, L).
        return {
            "thinning_accepted": accepted[outputs.seq_len_mask].float().mean().item()
        }
