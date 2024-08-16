from functools import partial
import torch

from hotpp.data import PaddedBatch


class NextItemLoss(torch.nn.Module):
    """Hybrid loss for next item prediction.

    Args:
        losses: Mapping from the feature name to the loss function.
        prediction: The type of prediction (either `mean`, `mode`, `sample`, or dictionary for each field).
        temperature: Temperature for the `sample` mode.
    """
    def __init__(self, losses, prediction="mean", temperature=1):
        super().__init__()
        self._losses = torch.nn.ModuleDict(losses)
        self._order = list(sorted(losses))
        self._prediction = prediction
        self._temperature = temperature
        self._interpolator = None

    @property
    def interpolator(self):
        return self._interpolator

    @interpolator.setter
    def interpolator(self, value):
        self._interpolator = value
        for name, loss in self._losses.items():
            loss.interpolator = partial(self._intepolate_field, field=name)

    def _intepolate_field(self, states, time_deltas, field):
        """Partial interpolator for a particular field."""
        outputs = self._interpolator(states, time_deltas)
        parameters = self._split_outputs(outputs)[field]
        return parameters

    @property
    def fields(self):
        return self._order

    @property
    def input_size(self):
        return sum([loss.input_size for loss in self._losses.values()])

    def get_delta_type(self, field):
        """Get time delta type."""
        return self._losses[field].delta

    def forward(self, inputs, outputs, states):
        """Extract targets and compute loss between predictions and targets.

        Args:
            inputs: Input features with shape (B, L).
            outputs: Model outputs with shape (B, L, D).
            states: Hidden model states with shape (N, B, L, D), where N is the number of layers.

        Returns:
            Losses dict and metrics dict.
        """
        # Align lengths.
        l = min(outputs.shape[1], inputs.shape[1])
        lengths = torch.minimum(outputs.seq_lens, inputs.seq_lens)
        inputs = PaddedBatch({k: (v[:, :l] if k in inputs.seq_names else v)
                              for k, v in inputs.payload.items()},
                             lengths, inputs.seq_names)
        outputs = PaddedBatch(outputs.payload[:, :l], lengths)

        # Compute losses. It is assumed that predictions lengths are equal to targets lengths.
        outputs = self._split_outputs(outputs)
        mask = inputs.seq_len_mask.bool() if (inputs.seq_lens != inputs.shape[1]).any() else None
        losses = {}
        metrics = {}
        for name, output in outputs.items():
            losses[name], loss_metrics = self._losses[name](inputs.payload[name], output, mask)
            for k, v in loss_metrics.items():
                metrics[f"{name}-{k}"] = v
        return losses, metrics

    def predict_next(self, outputs, states, fields=None, logits_fields_mapping=None):
        """Predict next events.

        Args:
            outputs: Model outputs with shape (B, L, D).
            states: Hidden model states with shape (N, B, L, D), where N is the number of layers.
            fields: The fields to predict next values for. By default, predict all fields.
            logits_fields_mapping: A mapping from field to the output logits field to predict logits for.

        Returns:
            PaddedBatch with predictions.
        """
        seq_lens = outputs.seq_lens
        outputs = self._split_outputs(outputs)
        result = {}
        for name in (fields or self._losses):
            prediction = self._prediction if isinstance(self._prediction, str) else self._prediction[name]
            if prediction == "mean":
                result[name] = self._losses[name].predict_means(outputs[name]).squeeze(-1)  # (B, L).
            elif prediction == "mode":
                result[name] = self._losses[name].predict_modes(outputs[name]).squeeze(-1)  # (B, L).
            elif prediction == "sample":
                result[name] = self._losses[name].predict_samples(outputs[name], temperature=self._temperature).squeeze(-1)  # (B, L).
            else:
                raise ValueError(f"Unknown prediction type: {prediction}.")
        for name, target_name in (logits_fields_mapping or {}).items():
            result[target_name] = self._losses[name].predict_logits(outputs[name])  # (B, L, C).
        return PaddedBatch(result, seq_lens)

    def _split_outputs(self, outputs):
        """Convert parameters tensor to the dictionary with parameters for each loss."""
        offset = 0
        result = {}
        for name in self._order:
            loss = self._losses[name]
            result[name] = outputs.payload[..., offset:offset + loss.input_size]
            offset += loss.input_size
        if offset != self.input_size:
            raise ValueError("Predictions tensor has inconsistent size.")
        return result
