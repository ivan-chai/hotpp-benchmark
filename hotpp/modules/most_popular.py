import torch
from hotpp.data import PaddedBatch
from hotpp.utils.torch import deterministic
from ..nn import Head
from .base_module import BaseModule


class MostPopularEncoder(torch.nn.Module):
    def __init__(self, num_classes, timestamps_field="timestamps", labels_field="labels"):
        super().__init__()
        self._num_classes = num_classes
        self._timestamps_field = timestamps_field
        self._labels_field = labels_field

    @property
    def hidden_size(self):
        return 2

    def forward(self, x, return_full_states=False):
        timestamps = x.payload[self._timestamps_field]  # (B, L).
        deltas = timestamps.clone()
        deltas[:, 1:] -= timestamps[:, :-1]
        deltas[:, 0] = 0
        with deterministic(False):
            deltas = deltas.cumsum(1) / (1 + torch.arange(deltas.shape[1], dtype=deltas.dtype, device=deltas.device))
            encoded_labels = torch.nn.functional.one_hot(x.payload[self._labels_field].long(), self._num_classes)  # (B, L, C).
            top_labels = encoded_labels.cumsum(1).argmax(-1)  # (B, L).
        hiddens = torch.stack([deltas, top_labels.to(deltas.dtype)], dim=2)  # (B, L, 2).
        hiddens = PaddedBatch(hiddens, x.seq_lens)
        return hiddens, hiddens.payload[None]  # (B, L, KD), (N, B, L, KD).


class Identity(torch.nn.Identity):
    def __init__(self, dim):
        super().__init__()
        self.input_size = dim


class MostPopularModule(BaseModule):
    """The model copies last seen events to the future.

    The model doesn't require training.

    Parameters.
        k: History length.
        val_metric: Validation set metric.
        test_metric: Test set metric.
        kwargs: Ignored (keep for compatibility with base module).
    """
    def __init__(self, k, num_classes,
                 timestamps_field="timestamps",
                 labels_field="labels",
                 val_metric=None,
                 test_metric=None,
                 **kwargs):
        super().__init__(seq_encoder=MostPopularEncoder(num_classes, timestamps_field=timestamps_field, labels_field=labels_field),
                         loss=Identity(2),
                         timestamps_field=timestamps_field,
                         labels_field=labels_field,
                         head_partial=lambda input_size, output_size: Identity(2),
                         optimizer_partial=lambda parameters: torch.optim.Adam(parameters, lr=0.001),  # Not used.
                         lr_scheduler_partial=lambda optimizer: torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=1),  # Not used.
                         val_metric=val_metric,
                         test_metric=test_metric)
        self._k = k
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
        deltas, labels = outputs.payload[..., 0], outputs.payload[..., 1].long()  # (B, L), (B, L).
        results = {self._timestamps_field: deltas,
                   self._labels_field: labels}  # (B, L).
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
        hiddens, _ = self.encode(x)  # (B, L, D), (N, B, L, D).
        init_times = x.payload[self._timestamps_field].take_along_dim(indices.payload, 1)  # (B, I).
        outputs = self.apply_head(hiddens)  # (B, L, D).
        b, l = outputs.shape

        deltas, labels = outputs.payload[..., 0], outputs.payload[..., 1].long()  # (B, L), (B, L).
        deltas = deltas.take_along_dim(indices.payload, 1)  # (B, I).
        labels = labels.take_along_dim(indices.payload, 1)  # (B, I).
        with deterministic(False):
            timestamps = deltas[:, :, None].repeat(1, 1, self._k).cumsum(2)  # (B, L, K).
        # Convert delta time to time.
        timestamps += init_times.unsqueeze(-1)
        labels = labels[:, :, None].repeat(1, 1, self._k)  # (B, L, K).
        logits = torch.nn.functional.one_hot(labels, self._num_classes).float()  # (B, L, C).
        sequences = {self._timestamps_field: timestamps,
                     self._labels_field: labels,
                     self._labels_logits_field: logits}
        return PaddedBatch(sequences, indices.seq_lens)
