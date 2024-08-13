import torch

from ..data import PaddedBatch
from .map import MAPMetric
from .next_item import NextItemMetric
from .otd import OTDMetric


class HorizonMetric:
    """A common interface to all future prediction metrics.

    Args:
        horizon: Prediction horizon.
        horizon_evaluation_step: The period for horizon metrics evaluation.
        map_deltas: The list of time delta thresholds for mAP evaluation.
        map_target_length: The maximum target length for mAP evaluation.
            Must be large enough to include all horizon events.
        otd_steps: The number of steps for optimal transport distance evaluation.
        otd_insert_cost: OTD insert cost.
        otd_delete_cost: OTD delete cost.
    """
    def __init__(self, horizon, horizon_evaluation_step=1,
                 map_deltas=None, map_target_length=None,
                 otd_steps=None, otd_insert_cost=None, otd_delete_cost=None):
        self.horizon = horizon
        self.horizon_evaluation_step = horizon_evaluation_step

        self.next_item = NextItemMetric()
        if map_deltas is not None:
            if map_target_length is None:
                raise ValueError("Need the max target sequence length for mAP computation")
            self.map_target_length = map_target_length
            self.map = MAPMetric(time_delta_thresholds=map_deltas)
        else:
            self.map_target_length = None
            self.map = None
        if otd_steps is not None:
            if (otd_insert_cost is None) or (otd_delete_cost is None):
                raise ValueError("Need insertion and deletion costs for the OTD metric.")
            self.otd_steps = otd_steps
            self.otd = OTDMetric(insert_cost=otd_insert_cost,
                                 delete_cost=otd_delete_cost)
        else:
            self.otd_steps = None
            self.otd = None

        self.reset()

    @property
    def horizon_prediction(self):
        return (self.map is not None) or (self.otd is not None)

    def reset(self):
        self._target_lengths = []
        self._predicted_lengths = []
        self._horizon_predicted_deltas_sums = []
        self._horizon_n_predicted_deltas = 0
        self.next_item.reset()
        if self.map is not None:
            self.map.reset()
        if self.otd is not None:
            self.otd.reset()

    def select_horizon_indices(self, seq_lens):
        """Select indices for horizon metrics evaluation."""
        step = self.horizon_evaluation_step
        l = seq_lens.max().item()
        # Skip first `step` events.
        indices = torch.arange(step, l, step, device=seq_lens.device)  # (I).
        indices_lens = (indices[None] < seq_lens[:, None]).sum(1)  # (B).
        return PaddedBatch(indices[None].repeat(len(seq_lens), 1), indices_lens)  # (B, I).

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
        self.next_item.update(mask=mask[:, 1:],  # Same as logical_and(mask[:, 1:], mask[:, :-1]).
                              target_timestamps=timestamps[:, 1:],
                              target_labels=labels[:, 1:],
                              predicted_timestamps=predicted_timestamps[:, :-1],
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
        self._target_lengths.append(targets_mask[seq_mask].sum(1).cpu().flatten())  # (V).
        self._predicted_lengths.append(predictions_mask[seq_mask].sum(1).cpu().flatten())  # (BI).

        # Update deltas stats.
        predicted_timestamps = predictions.payload["timestamps"][seq_mask]  # (V, N).
        deltas = predicted_timestamps[:, 1:] - predicted_timestamps[:, :-1]
        self._horizon_predicted_deltas_sums.append(deltas.float().mean().cpu() * deltas.numel())
        self._horizon_n_predicted_deltas += deltas.numel()

        # Update mAP.
        if self.map is not None:
            self.map.update(
                target_mask=targets_mask[seq_mask][:, :self.map_target_length],  # (V, K).
                target_times=targets.payload["timestamps"][seq_mask][:, :self.map_target_length],  # (V, K).
                target_labels=targets.payload["labels"][seq_mask][:, :self.map_target_length],  # (V, K).
                predicted_mask=predictions_mask[seq_mask],  # (V, N).
                predicted_times=predicted_timestamps,  # (V, N).
                predicted_labels_scores=predictions.payload["labels_logits"][seq_mask],  # (V, N, C).
            )

        # Update OTD.
        if self.otd is not None:
            if predictions_mask.shape[-1] < self.otd_steps:
                raise RuntimeError("Need more predicted events for OTD evaluation.")
            predicted_labels = predictions.payload["labels_logits"].argmax(-1)  # (B, I, N).
            self.otd.update(
                target_times=targets.payload["timestamps"][seq_mask][:, :self.otd_steps],  # (V, S).
                target_labels=targets.payload["labels"][seq_mask][:, :self.otd_steps],  # (V, S).
                predicted_times=predictions.payload["timestamps"][seq_mask][:, :self.otd_steps],  # (V, S).
                predicted_labels=predicted_labels[seq_mask][:, :self.otd_steps],  # (V, S).
            )

    def compute(self):
        values = {}
        if self._target_lengths:
            target_lengths = torch.cat(self._target_lengths)
            predicted_lengths = torch.cat(self._predicted_lengths)
            values.update({
                "mean-target-length": target_lengths.sum().item() / target_lengths.numel(),
                "mean-predicted-length":predicted_lengths.sum().item() / predicted_lengths.numel(),
                "horizon-mean-time-step": torch.stack(self._horizon_predicted_deltas_sums).sum() / self._horizon_n_predicted_deltas
            })
        values.update(self.next_item.compute())
        if self.map is not None:
            values.update(self.map.compute())
        if self.otd is not None:
            values.update(self.otd.compute())
        return values

    def _extract_target_sequences(self, features, indices):
        """Extract target sequence for each index."""
        b, i = indices.shape
        n = max(self.map_target_length or 0, self.otd_steps or 0)
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
