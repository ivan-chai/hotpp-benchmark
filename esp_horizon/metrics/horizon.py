import torch

from ..data import PaddedBatch
from .next_item import NextItemMetric
from .map import MAPMetric


class HorizonMetric:
    """A common interface to all future prediction metrics.

    Args:
        horizon: Prediction horizon.
        horizon_evaluation_step: The period for horizon metrics evaluation.
        map_thresholds: The list of time delta thresholds for mAP evaluation.
        max_target_length: The maximum target length for mAP evaluation.
            Must be large enough to include all horizon events.
    """
    def __init__(self, horizon, horizon_evaluation_step=1,
                 map_thresholds=None, max_target_length=None):
        self.horizon = horizon
        self.horizon_evaluation_step = horizon_evaluation_step

        self.next_item = NextItemMetric()
        if map_thresholds is not None:
            if max_target_length is None:
                raise ValueError("Need the max target sequence length for mAP computation")
            self.max_target_length = max_target_length
            self.map = MAPMetric(time_delta_thresholds=map_thresholds)
        else:
            self.map = None

        self.reset()

    @property
    def horizon_prediction(self):
        return self.map is not None

    def reset(self):
        self.next_item.reset()
        if self.map is not None:
            self.map.reset()

    def select_horizon_indices(self, lengths):
        """Select indices for horizon metrics evaluation."""
        step = self.horizon_evaluation_step
        l = lengths.max().item()
        # Skip first `step` events.
        indices = torch.arange(step, l, step, device=lengths.device)  # (I).
        indices_lens = (indices[None] < lengths[:, None]).sum(1)  # (B).
        return PaddedBatch(indices[None].repeat(len(lengths), 1), indices_lens)  # (B, I).

    def update_next_item(self, seq_lens, timestamps, labels, predicted_timestamps, predicted_labels_logits):
        """Update next-item metrics with new observations.

        NOTE: Timestamps and labels must be provided without offset w.r.t. predictions, i.e. input features.

        Args:
            seq_lens: Sequence lengths with shape (B).
            timestamps: Dataset timestamps with shape (B, T) (without offset).
            labels: Dataset labels with shape (B, T) (without offset).
            predicted_timestamps: Predicted next-item timestamps with shape (B, T).
            predicted_labels_logits: Predicted next-item labels logits with shape (B, T, C).
        """
        # Targets are features shifted w.r.t. prediction.
        mask = PaddedBatch(timestamps, seq_lens).seq_len_mask
        self.next_item.update(mask=mask[:, 1:],
                              target_labels=labels[:, 1:],
                              predicted_labels_logits=predicted_labels_logits[:, :-1])

    def update_horizon(self, seq_lens, timestamps, labels,
                       indices, indices_lens, seq_predicted_timestamps, seq_predicted_labels_logits):
        """Update sequence metrics with new observations.

        NOTE: Timestamps and labels must be provided without offset w.r.t. predictions, i.e. input features.

        Args:
            seq_lens: Sequence lengths with shape (B).
            timestamps: Dataset timestamps with shape (B, T) (without offset).
            labels: Dataset labels with shape (B, T) (without offset).
            indices: Sequence prediction initial feature positions with shape (B, I).
            indices_lens: The number of indices for each element in the batch with shape (B).
            seq_predicted_timestamps: Predicted timestamps with shape (B, I, N).
            seq_predicted_labels_logits: Predicted labels logits with shape (B, I, N, C).
        """
        features = PaddedBatch({"timestamps": timestamps, "labels": labels}, seq_lens)
        indices = PaddedBatch(indices, indices_lens)
        predictions = PaddedBatch({"timestamps": seq_predicted_timestamps, "labels_logits": seq_predicted_labels_logits},
                                  indices_lens)
        if not features.seq_len_mask.take_along_dim(indices.payload, 1).masked_select(indices.seq_len_mask).all():
            raise ValueError("Some indices are out of sequence lengths")

        initial_timestamps = features.payload["timestamps"].take_along_dim(indices.payload, 1)  # (B, I).
        targets = self._extract_target_sequences(features, indices)  # (B, I, K).

        # Align lengths.
        lengths = torch.minimum(targets.seq_lens, predictions.seq_lens)  # (B).
        targets = PaddedBatch(targets.payload, lengths)
        predictions = PaddedBatch(predictions.payload, lengths)
        seq_mask = targets.seq_len_mask.bool()  # (B, I).

        # Apply horizon.
        targets_mask = self._get_horizon_mask(initial_timestamps, targets)  # (B, I, K).
        predictions_mask = self._get_horizon_mask(initial_timestamps, predictions)  # (B, I, N).

        # Update mAP.
        if self.map is not None:
            # Truncate sequences to the specified horizon.
            self.map.update(
                target_mask=targets_mask[seq_mask],  # (V, K).
                target_times=targets.payload["timestamps"][seq_mask],  # (V, K).
                target_labels=targets.payload["labels"][seq_mask],  # (V, K).
                predicted_mask=predictions_mask[seq_mask],  # (V, N).
                predicted_times=predictions.payload["timestamps"][seq_mask],  # (V, N).
                predicted_labels_logits=predictions.payload["labels_logits"][seq_mask],  # (V, N, C).
            )

    def compute(self):
        values = {}
        values.update(self.next_item.compute())
        if self.map is not None:
            values.update(self.map.compute())
        return values

    def _extract_target_sequences(self, features, indices):
        """Extract target sequence for each index."""
        b, i = indices.shape
        n = self.max_target_length
        # Each sequence starts from the next item.
        offsets = torch.arange(1, n + 1, device=features.device)  # (N).
        gather_indices = indices.payload.unsqueeze(2) + offsets[None, None]  # (B, I, N).

        valid_mask = (gather_indices < features.seq_lens[:, None, None]).all(2)  # (B, I).
        valid_mask.logical_and_(indices.seq_len_mask)
        gather_indices.clip_(max=features.shape[1] - 1)

        lengths = valid_mask.sum(1)  # (B).
        sequences = PaddedBatch({k: v.take_along_dim(gather_indices.flatten(1), 1).reshape(b, i, n)  # (B, I, N).
                                 for k, v in features.payload.items()},
                                lengths)
        return sequences

    def _get_horizon_mask(self, initial_timestamps, features):
        offsets = features.payload["timestamps"] - initial_timestamps.unsqueeze(2)  # (B, L, N).
        return torch.logical_and(features.seq_len_mask.unsqueeze(2), offsets < self.horizon)  # (B, L, N).
