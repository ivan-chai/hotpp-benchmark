import torch
from ..data import PaddedBatch
from ..utils.torch import deterministic, prefix_medians
from ..fields import PRESENCE_PROB, LABELS_LOGITS
from .base_module import BaseModule


class HistoryDensityEncoder(torch.nn.Module):
    """Compute labels and amounts density using historical data.
    Density is estimated with respect to time rather than indices.
    Amounts density is estimated for each label independently.

    Args:
        num_classes: The number of classes in the dataset.
        time_aggregation: One of "mean" and "median".
        max_time_delta: Truncate time deltas to this value during mean value estimation.
        log_amount: Whether amounts are preprocessed with log(x + 1).
    """

    def __init__(self, num_classes, timestamps_field="timestamps", labels_field="labels",
                 time_aggregation="mean", max_time_delta=None,
                 amounts_field=None, log_amount=False):
        super().__init__()
        self._num_classes = num_classes
        self._timestamps_field = timestamps_field
        self._labels_field = labels_field
        self._time_aggregation = time_aggregation
        self._max_time_delta = max_time_delta
        self._amounts_field = amounts_field
        self._log_amount = log_amount

    @property
    def need_states(self):
        return False

    @property
    def hidden_size(self):
        # 1 for time, C for labels density and (optionally) C for amounts density per label.
        return 1 + self._num_classes * (2 if self._amounts_field else 1)

    def forward(self, x, return_states=False):
        b, l = x.shape
        seq_mask = x.seq_len_mask.bool()  # (B, L).
        timestamps = x.payload[self._timestamps_field]  # (B, L).

        # Compute time offsets from the beginning. The first offset is the initial timestamp (to avoid zeros).
        offsets = timestamps.clone()
        offsets[:, 1:] -= timestamps[:, :1]  # (B, L).
        offsets.clip_(min=1e-6 * (1 if self._max_time_delta is None else self._max_time_delta))

        # Compute time offsets from previous events.
        deltas = timestamps.clone()
        deltas[:, 0] = 0
        deltas[:, 1:] -= timestamps[:, :-1]

        arange = torch.arange(1, l + 1, device=x.device)
        encoded_labels = torch.nn.functional.one_hot(x.payload[self._labels_field].long(), self._num_classes)  # (B, L, C).
        if self._max_time_delta is not None:
            deltas = deltas.clip(max=self._max_time_delta)
        with deterministic(False):
            if self._time_aggregation == "mean":
                agg_deltas = deltas.cumsum(1) / arange[None]  # (B, L).
            elif self._time_aggregation == "median":
                agg_deltas = prefix_medians(deltas)  # (B, L).
            else:
                raise ValueError(f"Unknown time aggregation: {self._time_aggregation}")
            label_densities = encoded_labels.cumsum(1) / offsets.unsqueeze(2)  # (B, L, C).

        hiddens = [agg_deltas.unsqueeze(2), label_densities]
        if self._amounts_field:
            amounts = x.payload[self._amounts_field]
            if self._log_amount:
                amounts = amounts.exp() - 1
            with deterministic(False):
                amount_densities = (amounts.unsqueeze(2) * encoded_labels).cumsum(1) / offsets.unsqueeze(2)  # (B, L, C).
            hiddens.append(amount_densities)
        hiddens = PaddedBatch(torch.cat(hiddens, dim=-1), x.seq_lens)  # (B, L, D).
        return hiddens, hiddens.payload[None]  # (B, L, D), (N, B, L, D).


class Identity(torch.nn.Identity):
    def __init__(self, dim):
        super().__init__()
        self.input_size = dim

    @property
    def need_interpolator(self):
        return False


class HistoryDensityModule(BaseModule):
    """Compute labels and amounts density using historical data.
    Density is estimated with respect to time rather than indices.
    Amounts density is estimated for each label independently.

    During inference, the module predicts average sequences for each horizon interval.

    The model doesn't require training.

    Parameters.
        num_classes: The number of classes in the dataset.
        horizons: The list of horizon intervals to model.
        time_aggregation: One of "mean" and "median".
        max_time_delta: Truncate time deltas to this value during mean value estimation.
        log_amount: Whether amounts are preprocessed with log(x + 1).
        kwargs: Ignored (keep for compatibility with base module).
    """
    def __init__(self, num_classes, horizons,
                 seq_encoder=None, loss=None,  # Ignored.
                 head_partial=None, optimizer_partial=None, lr_scheduler_partial=None,  # Ignored.
                 timestamps_field="timestamps",
                 labels_field="labels",
                 time_aggregation="mean", max_time_delta=None,
                 amounts_field=None, log_amount=False,
                 **kwargs):
        super().__init__(seq_encoder=HistoryDensityEncoder(num_classes,
                                                           timestamps_field=timestamps_field, labels_field=labels_field,
                                                           time_aggregation=time_aggregation, max_time_delta=max_time_delta,
                                                           amounts_field=amounts_field, log_amount=log_amount),
                         loss=Identity(2),
                         timestamps_field=timestamps_field,
                         labels_field=labels_field,
                         head_partial=lambda input_size, output_size: Identity(2),
                         optimizer_partial=lambda parameters: torch.optim.Adam(parameters, lr=0.001),  # Not used.
                         lr_scheduler_partial=lambda optimizer: torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=1),  # Not used.
                         **kwargs)
        self._horizons = list(sorted(horizons))
        self._amounts_field = amounts_field
        self._log_amount = log_amount
        self._num_classes = num_classes
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def predict_next(self, inputs, outputs, states, fields=None, logits_fields_mapping=None, predict_delta=False):
        """Predict events from head outputs.

        Args:
            inputs: Input features with shape (B, L).
            outputs: Output of the head module with shape (B, L, D).
            states (unused): Sequence model states with shape (N, B, L, D), where N is the number of layers.
            predict_delta: If True, return delta times. Generate absolute timestamps otherwise.
        """
        b, l = inputs.shape
        indices = PaddedBatch(torch.arange(l, device=inputs.device)[None].repeat(b, 1), inputs.seq_lens)
        sequences = self.generate_sequences(inputs, indices)  # (B, L, K).
        mask = sequences.payload[PRESENCE_PROB] > 0
        is_empty = mask.sum(2, keepdim=True) == 0  # (B, L, 1).
        mask.masked_fill_(is_empty, True)
        with deterministic(False):
            mask = torch.logical_and(mask, mask.cumsum(2) == 1)  # (B, L, K).

        fields = [PRESENCE_PROB, self._timestamps_field, self._labels_field]
        if self._amounts_field:
            fields.append(self._amounts_field)
        next_items = {}
        for field in fields:
            next_items[field] = sequences.payload[field][mask].reshape(b, l)  # (B, L).
        if self._labels_field in (logits_fields_mapping or {}):
            next_items[logits_fields_mapping[self._labels_field]] = sequences.payload[LABELS_LOGITS][mask].reshape(b, l, self._num_classes)
        return PaddedBatch(next_items, inputs.seq_lens)

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
        outputs = outputs.payload.take_along_dim(indices.payload.unsqueeze(2), 1)  # (B, I, D).
        seq_mask = indices.seq_len_mask.bool()  # (B, I).
        b, l, _ = outputs.shape

        deltas = outputs[..., 0]  # (B, I).
        label_densities = outputs[..., 1:1 + self._num_classes]  # (B, I, C).
        if self._amounts_field:
            amount_densities = outputs[..., 1 + self._num_classes:1 + 2 * self._num_classes]  # (B, I, C).

        active_classes = (label_densities[seq_mask] > 0).any(0)  # (C).
        active_indices = active_classes.nonzero()[:, 0]  # (A).
        assert (active_classes.ndim == 1) and (len(active_classes) == self._num_classes)
        n_active = active_classes.sum().item()
        k = len(self._horizons) * n_active  # The maximum output length.
        sequences = {
            PRESENCE_PROB: torch.zeros(b, l, k, device=x.device),
            self._timestamps_field: torch.zeros(b, l, k, device=x.device),
            self._labels_field: torch.zeros(b, l, k, device=x.device, dtype=torch.long)
        }
        if self._amounts_field:
            sequences[self._amounts_field] = torch.zeros(b, l, k, device=x.device)
        for i, h in enumerate(self._horizons):
            prev_h = self._horizons[i - 1] if i > 0 else 0
            timedelta = prev_h + (h - prev_h) / 2

            inter_counts = (h - prev_h) * label_densities  # (B, I, C).

            order = inter_counts[:, :, active_classes].argsort(2, descending=True)  # (B, I, C).
            start = i * n_active
            stop = (i + 1) * n_active
            probs = inter_counts.clip(max=1)[:, :, active_classes]
            sequences[PRESENCE_PROB][..., start:stop] = probs.take_along_dim(order, -1)
            sequences[self._timestamps_field][..., start:stop] = init_times[:, :, None] + timedelta
            sequences[self._labels_field][..., start:stop] = active_indices[None, None].expand(b, l, n_active).take_along_dim(order, -1)
            if self._amounts_field:
                inter_amounts = (h - prev_h) * amount_densities  # (B, I, C).
                sequences[self._amounts_field][..., start:stop] = (inter_amounts[:, :, active_classes] / probs.clip(min=1e-6)).take_along_dim(order, -1)

        sequences[LABELS_LOGITS] = (torch.nn.functional.one_hot(sequences[self._labels_field], self._num_classes) - 1) * 1000.0
        if self._amounts_field and self._log_amount:
            sequences[self._amounts_field] = (sequences[self._amounts_field] + 1).log()
        return PaddedBatch(sequences, indices.seq_lens)
