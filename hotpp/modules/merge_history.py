import torch
from hotpp.data import PaddedBatch
from hotpp.utils.torch import deterministic
from ..fields import PRESENCE_PROB, LABELS_LOGITS
from .base_module import BaseModule


class MergeHistoryEncoder(torch.nn.Module):
    """For each horizon compute labels frequency and (optionally) amounts per label.
    During inference generate all labels on the horizon with corresponding mean statistics.

    Args:
        horizons: The list of horizon intervals to model.
    """

    def __init__(self, num_classes, horizons,
                 timestamps_field="timestamps", labels_field="labels",
                 amounts_field=None, log_amount=False):
        super().__init__()
        self._num_classes = num_classes
        self._horizons = list(sorted(horizons))
        self._timestamps_field = timestamps_field
        self._labels_field = labels_field
        self._amounts_field = amounts_field
        self._log_amount = log_amount

    @property
    def need_states(self):
        return False

    @property
    def hidden_size(self):
        return len(self._horizons) * self._num_classes * (2 if self._amounts_field else 1)

    def forward(self, x, return_states=False):
        seq_mask = x.seq_len_mask.bool()  # (B, L).
        timestamps = x.payload[self._timestamps_field]  # (B, L).
        timestamps = timestamps.masked_fill(~seq_mask, timestamps[seq_mask].max().item())
        timestamps = torch.cat([timestamps[:, :1], timestamps], 1)  # (B, 1 + L).
        labels = x.payload[self._labels_field].masked_fill(~seq_mask, 0)  # (B, L).
        encoded_labels = torch.nn.functional.one_hot(x.payload[self._labels_field].long(), self._num_classes)  # (B, L, C).
        with deterministic(False):
            n_labels = encoded_labels.cumsum(1)  # (B, L, C).
            n_labels = torch.cat([torch.zeros_like(n_labels[:, :1]), n_labels], 1)  # (B, 1 + L, C).
            if self._amounts_field:
                amounts = x.payload[self._amounts_field]
                if self._log_amount:
                    amounts = amounts.exp() - 1
                label_sums = (encoded_labels * amounts.unsqueeze(2)).cumsum(1)  # (B, L, C).
                label_sums = torch.cat([torch.zeros_like(label_sums[:, :1]), label_sums], 1)  # (B, 1 + L, C).
        hiddens = []
        arange = 1 + torch.arange(x.shape[1], device=x.device)[None, :, None]  # (1, L, 1).
        for h in self._horizons:
            start = torch.searchsorted(timestamps, timestamps - h, side="left").unsqueeze(2)  # (B, 1 + L, 1).
            start = (start - 1).clip(min=0)
            end = torch.searchsorted(timestamps, timestamps, side="left").unsqueeze(2)  # (B, 1 + L, 1).
            end = torch.maximum(start, end - 1)  # (B, 1 + L, 1)
            # Ignore start == 0 (truncated horizon).
            counts = n_labels.take_along_dim(end, 1) - n_labels.take_along_dim(start, 1)  # (B, 1 + L, C).
            counts = counts[:, 1:]  # (B, L, C).
            assert counts.shape[2] == self._num_classes
            with deterministic(False):
                horizon_hiddens = counts.cumsum(1) / arange  # Average horizon sums with shape (B, L, C).
                if self._amounts_field is not None:
                    amounts = label_sums.take_along_dim(end, 1) - label_sums.take_along_dim(start, 1)  # (B, 1 + L, C).
                    amounts = amounts[:, 1:]  # (B, L, C).
                    horizon_hiddens = torch.cat([horizon_hiddens, amounts.cumsum(1) / arange], dim=-1)  # (B, L, 2C).
            hiddens.append(horizon_hiddens)
        hiddens = PaddedBatch(torch.cat(hiddens, dim=-1), x.seq_lens)  # (B, L, H2C).
        return hiddens, hiddens.payload[None]  # (B, L, H2C), (N, B, L, H2C).


class Identity(torch.nn.Identity):
    def __init__(self, dim):
        super().__init__()
        self.input_size = dim

    @property
    def need_interpolator(self):
        return False


class MergeHistoryModule(BaseModule):
    """The model computes per-label average activations and predicts the full set of labels with average horizon statistics.

    The model doesn't require training.

    Parameters.
        horizons: The list of horizon intervals to model.
        val_metric: Validation set metric.
        test_metric: Test set metric.
        kwargs: Ignored (keep for compatibility with base module).
    """
    def __init__(self, num_classes, horizons,
                 seq_encoder=None, loss=None,  # Ignored.
                 head_partial=None, optimizer_partial=None, lr_scheduler_partial=None,  # Ignored.
                 timestamps_field="timestamps",
                 labels_field="labels",
                 amounts_field=None, log_amount=False,
                 **kwargs):
        super().__init__(seq_encoder=MergeHistoryEncoder(num_classes, horizons,
                                                         timestamps_field=timestamps_field, labels_field=labels_field,
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
        outputs, _ = self(x)  # (B, L, H2C).
        outputs = outputs.payload.take_along_dim(indices.payload.unsqueeze(2), 1)  # (B, I, H2C).
        seq_mask = indices.seq_len_mask.bool()  # (B, I).
        b, l, _ = outputs.shape
        r = 2 if self._amounts_field else 1
        outputs = outputs.reshape(b, l, len(self._horizons), r, self._seq_encoder._num_classes)  # (B, I, H, 2, C).
        counts = outputs[:, :, :, 0, :]  # (B, I, H, C).
        if self._amounts_field:
            amounts = outputs[:, :, :, 1, :]  # (B, I, H, C).
        else:
            amounts = None
        active_classes = (counts[seq_mask] > 0).flatten(0, 1).any(0)  # (C).
        active_indices = active_classes.nonzero()[:, 0]  # (A).
        assert (active_classes.ndim == 1) and (len(active_classes) == self._num_classes)
        n_active = active_classes.sum().item()
        k = len(self._horizons) * n_active  # The maximum output length.
        sequences = {
            PRESENCE_PROB: torch.zeros(b, l, k, device=x.device),
            self._timestamps_field: torch.zeros(b, l, k, device=x.device),
            self._labels_field: torch.zeros(b, l, k, device=x.device, dtype=torch.long)
        }
        if amounts is not None:
            sequences[self._amounts_field] = torch.zeros(b, l, k, device=x.device)
        for i, h in enumerate(self._horizons):
            prev_h = self._horizons[i - 1] if i > 0 else 0
            timedelta = prev_h + (h - prev_h) / 2
            inter_counts = counts[:, :, i]  # (B, I, C).
            inter_amounts = amounts[:, :, i] if amounts is not None else None  # (B, I, C).
            if i > 0:
                inter_counts = (inter_counts - counts[:, :, i - 1]).clip(min=0)
                inter_amounts = (inter_amounts - amounts[:, :, i - 1]).clip(min=0) if amounts is not None else None
            start = i * n_active
            stop = (i + 1) * n_active
            probs = inter_counts.clip(max=1)[:, :, active_classes]
            sequences[PRESENCE_PROB][..., start:stop] = probs
            sequences[self._timestamps_field][..., start:stop] = init_times[:, :, None] + timedelta
            sequences[self._labels_field][..., start:stop] = active_indices
            if amounts is not None:
                sequences[self._amounts_field][..., start:stop] = inter_amounts[:, :, active_classes] / probs.clip(min=1e-6)
        sequences[LABELS_LOGITS] = (torch.nn.functional.one_hot(sequences[self._labels_field], self._num_classes) - 1) * 1000.0
        if (amounts is not None) and self._log_amount:
            sequences[self._amounts_field] = (sequences[self._amounts_field] + 1).log()
        return PaddedBatch(sequences, indices.seq_lens)
