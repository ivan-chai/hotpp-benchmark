import torch
from ..data import PaddedBatch
from ..utils.torch import deterministic, prefix_medians
from ..fields import LABELS_LOGITS
from .base_module import BaseModule


def exponential_average(x, momentum, dim=1):
    assert x.ndim == 3
    assert dim == 1
    b, l, d = x.shape
    results = torch.empty_like(x)  # (B, L, D).
    results[:, 0] = x[:, 0]
    for i in range(1, l):
        results[:, i] = momentum * results[:, i - 1] + (1 - momentum) * x[:, i]
    return results


class MostPopularEncoder(torch.nn.Module):
    """Compute mean time delta and labels distribution using historical data."""
    def __init__(self, num_classes, timestamps_field="timestamps", labels_field="labels",
                 time_aggregation="mean", max_time_delta=None,
                 amounts_field=None, log_amount=False,
                 exp_moving_avg=None):
        super().__init__()
        self._num_classes = num_classes
        self._timestamps_field = timestamps_field
        self._labels_field = labels_field
        self._time_aggregation = time_aggregation
        self._amounts_field = amounts_field
        self._max_time_delta = max_time_delta
        self._log_amount = log_amount
        self._exp_moving_avg = exp_moving_avg

    @property
    def need_states(self):
        return False

    @property
    def hidden_size(self):
        return 1 + self._num_classes * (2 if self._amounts_field else 1)

    def forward(self, x, return_states=False):
        timestamps = x.payload[self._timestamps_field].float()  # (B, L).
        deltas = timestamps.clone()
        deltas[:, 1:] -= timestamps[:, :-1]
        deltas[:, 0] = 0
        if self._max_time_delta is not None:
            deltas = deltas.clip(max=self._max_time_delta)
        parts = []
        with deterministic(False):
            arange = torch.arange(1, x.shape[1] + 1, device=x.device)[None, :, None]
            if self._time_aggregation == "mean":
                if self._exp_moving_avg:
                    deltas = exponential_average(deltas.unsqueeze(2), self._exp_moving_avg)  # (B, L, 1).
                else:
                    deltas_arange = arange - 1
                    deltas_arange[:, 0] = 1
                    deltas = deltas.cumsum(1).unsqueeze(2) / deltas_arange  # (B, L, 1).
            elif self._time_aggregation == "median":
                deltas = prefix_medians(deltas).unsqueeze(2)  # (B, L, 1).
            else:
                raise ValueError(f"Unknown time aggregation: {self._time_aggregation}")
            encoded_labels = torch.nn.functional.one_hot(x.payload[self._labels_field].long(), self._num_classes).float()  # (B, L, C).
            if self._exp_moving_avg:
                probabilities = exponential_average(encoded_labels, self._exp_moving_avg)  # (B, L, C).
                probabilities /= probabilities.sum(-1, keepdim=True)
            else:
                probabilities = encoded_labels.cumsum(1) / arange  # (B, L, C).
            parts.append(deltas)
            parts.append(probabilities)
            if self._amounts_field is not None:
                amounts = x.payload[self._amounts_field]
                if self._log_amount:
                    amounts = amounts.exp() - 1
                if self._exp_moving_avg:
                    amounts = exponential_average(amounts.unsqueeze(2) * encoded_labels, self._exp_moving_avg) / probabilities.clip(min=1e-6)  # (B, L, C).
                else:
                    amounts = (amounts.unsqueeze(2) * encoded_labels).cumsum(1) / encoded_labels.cumsum(1).clip(min=1)  # (B, L, C).
                if self._log_amount:
                    amounts = (amounts + 1).log()
                parts.append(amounts)
        hiddens = torch.concatenate(parts, dim=2)  # (B, L, D).
        hiddens = PaddedBatch(hiddens, x.seq_lens)
        return hiddens, hiddens.payload[None]  # (B, L, D), (N, B, L, D).


class Identity(torch.nn.Identity):
    def __init__(self, dim):
        super().__init__()
        self.input_size = dim

    @property
    def need_interpolator(self):
        return False


class MostPopularModule(BaseModule):
    """The model copies last seen events to the future.

    The model doesn't require training.

    Parameters.
        k: History length.
        num_classes: The number of classes in the dataset.
        prediction: Predict most popular label if "mode", sample labels if "sample", and approximate next-k distribution if "distribution".
        val_metric: Validation set metric.
        test_metric: Test set metric.
        kwargs: Ignored (keep for compatibility with base module).
    """
    def __init__(self, k, num_classes,
                 prediction="mode",
                 seq_encoder=None, loss=None,  # Ignored.
                 head_partial=None, optimizer_partial=None, lr_scheduler_partial=None,  # Ignored.
                 timestamps_field="timestamps",
                 labels_field="labels",
                 amounts_field=None,
                 time_aggregation="mean", max_time_delta=None,
                 exp_moving_avg=None,
                 log_amount=False,
                 **kwargs):
        super().__init__(seq_encoder=MostPopularEncoder(num_classes, timestamps_field=timestamps_field, labels_field=labels_field,
                                                        time_aggregation=time_aggregation, max_time_delta=max_time_delta,
                                                        exp_moving_avg=exp_moving_avg,
                                                        amounts_field=amounts_field, log_amount=log_amount),
                         loss=Identity(2),
                         timestamps_field=timestamps_field,
                         labels_field=labels_field,
                         amounts_field=amounts_field,
                         head_partial=lambda input_size, output_size: Identity(2),
                         optimizer_partial=lambda parameters: torch.optim.Adam(parameters, lr=0.001),  # Not used.
                         lr_scheduler_partial=lambda optimizer: torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=1),  # Not used.
                         **kwargs)
        self._k = k
        self._num_classes = num_classes
        self._prediction = prediction
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def predict_next(self, inputs, outputs, states, fields=None, logits_fields_mapping=None, predict_delta=False):
        """Predict events from head outputs.

        Args:
            inputs: Input features with shape (B, L).
            outputs: Output of the head module with shape (B, L, D).
            states (unused): Sequence model states with shape (N, B, L, D), where N is the number of layers.
            predict_delta: If True, return delta times. Generate absolute timestamps otherwise.
        """
        deltas, probabilities = outputs.payload[..., 0], outputs.payload[..., 1:1 + self._num_classes]  # (B, L), (B, L, C).
        if self._prediction in {"mode", "distribution"}:
            labels = probabilities.argmax(2)  # (B, L).
        elif self._prediction == "sample":
            labels = torch.distributions.Categorical(probabilities).sample()  # (B, L).
        else:
            raise NotImplementedError(f"{self._prediction} prediction.")
        results = {self._timestamps_field: deltas,
                   self._labels_field: labels}  # (B, L).
        if self._amounts_field:
            amounts = outputs.payload[..., 1 + self._num_classes:1 + 2 * self._num_classes]  # (B, L, C).
            results[self._amounts_field] = amounts.take_along_dim(labels.unsqueeze(2), 2).squeeze(2)  # (B, L).
        if not predict_delta:
            # Convert delta time to time.
            results[self._timestamps_field] += inputs.payload[self._timestamps_field]
        for name, logits_name in logits_fields_mapping.items():
            results[logits_name] = torch.nn.functional.one_hot(results[name], self._num_classes)  # (B, L, C).
        return PaddedBatch(results, outputs.seq_lens)

    def compute_loss(self, x, outputs, states):
        return {}, {}

    def generate_sequences(self, x, indices):
        """Generate future events.

        Args:
            x: Features with shape (B, L).
            indices: Indices with positions to start generation from with shape (B, I).

        Returns:
            Predicted sequences with shape (B, I, N).
        """
        init_times = x.payload[self._timestamps_field].take_along_dim(indices.payload, 1)  # (B, I).
        outputs, _ = self(x)  # (B, L, D).
        b, l = outputs.shape

        deltas, probabilities = outputs.payload[..., 0], outputs.payload[..., 1:1 + self._num_classes]  # (B, L), (B, L, C).
        deltas = deltas.take_along_dim(indices.payload, 1)  # (B, I).
        probabilities = probabilities.take_along_dim(indices.payload.unsqueeze(2), 1)  # (B, I, C).
        with deterministic(False):
            timestamps = deltas[:, :, None].repeat(1, 1, self._k).cumsum(2)  # (B, L, K).
        # Convert delta time to time.
        timestamps += init_times.unsqueeze(-1)
        if self._prediction == "mode":
            labels = probabilities.argmax(2)  # (B, I).
            labels = labels.unsqueeze(2).repeat(1, 1, self._k)  # (B, I, K).
        elif self._prediction == "sample":
            with deterministic(False):
                labels = torch.multinomial(probabilities.flatten(0, 1), self._k, replacement=True).reshape(b, indices.shape[1], self._k)
        elif self._prediction == "distribution":
            labels = torch.empty(b, indices.shape[1], self._k, dtype=torch.long, device=x.device)  # (B, I, K).
            fractions = torch.full_like(probabilities[:, :, :1], -1 / self._k)
            p = probabilities.clone()
            for i in range(self._k):
                ids = p.argmax(2)
                labels[:, :, i] = ids
                if i < self._k - 1:
                    k = self._k - i
                    p.scatter_add_(2, ids.unsqueeze(2), fractions) * k / (k - 1)
                    p.clip_(min=0)
                    p /= p.sum(-1, keepdim=True)
                    fractions *= k / (k - 1)
        else:
            raise NotImplementedError(f"{self._prediction} prediction.")

        logits = torch.nn.functional.one_hot(labels, self._num_classes).float()  # (B, L, C).
        sequences = {self._timestamps_field: timestamps,
                     self._labels_field: labels,
                     LABELS_LOGITS: logits}
        if self._amounts_field:
            amounts = outputs.payload[..., 1 + self._num_classes:1 + 2 * self._num_classes]  # (B, L, C).
            amounts = amounts.take_along_dim(indices.payload.unsqueeze(2), 1)  # (B, I, C).
            sequences[self._amounts_field] = amounts.take_along_dim(labels, 2)  # (B, I, K).
        return PaddedBatch(sequences, indices.seq_lens)
