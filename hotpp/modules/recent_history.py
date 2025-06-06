import torch
from hotpp.data import PaddedBatch
from hotpp.utils.torch import deterministic
from ..fields import LABELS_LOGITS
from .base_module import BaseModule


class SlidingEncoder(torch.nn.Module):
    def __init__(self, k, fields, timestamps_field="timestamps"):
        super().__init__()
        self._k = k
        self._fields = list(sorted(fields))
        self._timestamps_field = timestamps_field

    @property
    def need_states(self):
        return False

    @property
    def fields(self):
        return self._fields

    @property
    def hidden_size(self):
        return len(self._fields) * self._k

    def forward(self, x, return_states=False):
        timestamps = x.payload[self._timestamps_field]  # (B, L).
        deltas = timestamps.clone()
        deltas[:, 1:] -= timestamps[:, :-1]
        deltas[:, 0] = 0
        deltas.clip_(min=0)
        x = x.clone()
        x.payload[self._timestamps_field] = deltas
        merged = torch.stack([x.payload[name].float() for name in self._fields], 2)  # (B, L, D).
        b, l, d = merged.shape
        windows = torch.stack([merged.roll(i, 1) for i in range(self._k - 1, -1, -1)], 2)  # (B, L, K, D).
        hiddens = PaddedBatch(windows.reshape(b, l, -1), x.seq_lens)
        return hiddens, hiddens.payload[None]  # (B, L, KD), (N, B, L, KD).


class Identity(torch.nn.Identity):
    def __init__(self, dim):
        super().__init__()
        self.input_size = dim

    @property
    def need_interpolator(self):
        return False


class RecentHistoryModule(BaseModule):
    """The model copies last seen events to the future.

    The model doesn't require training.

    Parameters.
        k: History length.
        val_metric: Validation set metric.
        test_metric: Test set metric.
        kwargs: Ignored (keep for compatibility with base module).
    """
    def __init__(self, k, num_classes,
                 seq_encoder=None, loss=None,  # Ignored.
                 head_partial=None, optimizer_partial=None, lr_scheduler_partial=None,  # Ignored.
                 timestamps_field="timestamps",
                 labels_field="labels",
                 **kwargs):
        super().__init__(seq_encoder=SlidingEncoder(k, [timestamps_field, labels_field], timestamps_field=timestamps_field),
                         loss=Identity(2),
                         timestamps_field=timestamps_field,
                         labels_field=labels_field,
                         head_partial=lambda input_size, output_size: Identity(2),
                         optimizer_partial=lambda parameters: torch.optim.Adam(parameters, lr=0.001),  # Not used.
                         lr_scheduler_partial=lambda optimizer: torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=1),  # Not used.
                         **kwargs)
        self._k = k
        self._num_classes = num_classes
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def predict_next(self, inputs, outputs, states, fields=None, logits_fields_mapping=None, predict_delta=False):
        """Predict events from head outputs.

        Args:
            inputs: Input features with shape (B, L).
            outputs: Output of the head module with shape (B, L, D).
            states: Sequence model states with shape (N, B, L, D), where N is the number of layers.
            predict_delta: If True, return delta times. Generate absolute timestamps otherwise.
        """
        b, l = inputs.shape
        next_k = outputs.payload.reshape(b, l, self._k, -1)
        next_1 = next_k[:, :, 0]  # (B, L, D).
        results = {name: next_1[:, :, i] for i, name in enumerate(self._seq_encoder.fields)}  # (B, L).
        results[self._labels_field] = results[self._labels_field].long()
        if not predict_delta:
            # Convert delta time to time.
            results[self._timestamps_field] += inputs.payload[self._timestamps_field]
        for name, logits_name in logits_fields_mapping.items():
            results[logits_name] = torch.nn.functional.one_hot(results[name], self._num_classes)  # (B, L, C).
        return PaddedBatch(results, outputs.seq_lens)

    def compute_loss(self, x, outputs, states):
        return {}, {}

    def predict_next_k(self, inputs, outputs, states, fields=None, logits_fields_mapping=None, predict_delta=False):
        """Predict K next events from head outputs.

        Args:
            inputs: Input features with shape (B, L).
            outputs: Output of the head module with shape (B, L, D).
            states: Sequence model states with shape (N, B, L, D), where N is the number of layers.
            predict_delta: If True, return delta times. Generate absolute timestamps otherwise.
        """
        b, l = inputs.shape
        next_k = outputs.payload.reshape(b, l, self._k, -1)  # (B, L, K, F).
        results = {name: next_k[..., i] for i, name in enumerate(self._seq_encoder.fields)}  # (B, L, K).
        results[self._labels_field] = results[self._labels_field].long()
        for name, logits_name in logits_fields_mapping.items():
            if name != self._labels_field:
                raise ValueError(f"Can't compute logits for {name}")
            results[logits_name] = torch.nn.functional.one_hot(results[name], self._num_classes).float()  # (B, L, K, C).
        results = PaddedBatch(results, outputs.seq_lens)
        with deterministic(False):
            results.payload[self._timestamps_field].cumsum_(2)
        if not predict_delta:
            # Convert delta time to time.
            results.payload[self._timestamps_field] += inputs.payload[self._timestamps_field].unsqueeze(2)
        return results

    def generate_sequences(self, x, indices):
        """Generate future events.

        Args:
            x: Features with shape (B, L).
            indices: Indices with positions to start generation from with shape (B, I).

        Returns:
            Predicted sequences with shape (B, I, N).
        """
        init_times = x.payload[self._timestamps_field].take_along_dim(indices.payload, 1)  # (B, I).
        init_times = PaddedBatch({self._timestamps_field: init_times}, indices.seq_lens)
        outputs, states = self(x)  # (B, L, D), (N, B, L, D).
        outputs = PaddedBatch(outputs.payload.take_along_dim(indices.payload.unsqueeze(2), 1),
                              indices.seq_lens)  # (B, I, D).
        states = states.take_along_dim(indices.payload[None, :, :, None], 2)  # (N, B, I, D).
        sequences = self.predict_next_k(init_times, outputs, states, logits_fields_mapping={self._labels_field: LABELS_LOGITS})  # (B, I, K) or (B, I, K, C).
        return sequences  # (B, I, K) or (B, I, K, C).
