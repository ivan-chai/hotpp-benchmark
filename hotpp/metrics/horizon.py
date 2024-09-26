import torch

from ..data import PaddedBatch
from .tmap import TMAPMetric
from .next_item import NextItemMetric
from .otd import OTDMetric, batch_bincount


class HorizonMetric:
    """A common interface to all future prediction metrics.

    Args:
        horizon: Prediction horizon.
        horizon_evaluation_step: The period for horizon metrics evaluation.
        max_time_delta: Maximum time delta for next item metrics.
        map_deltas: The list of time delta thresholds for mAP evaluation.
        map_target_length: The maximum target length for mAP evaluation.
            Must be large enough to include all horizon events.
        otd_steps: The number of steps for optimal transport distance evaluation.
        otd_insert_cost: OTD insert cost.
        otd_delete_cost: OTD delete cost.
    """
    def __init__(self, horizon, horizon_evaluation_step=1, max_time_delta=None,
                 map_deltas=None, map_target_length=None,
                 otd_steps=None, otd_insert_cost=None, otd_delete_cost=None):
        self.horizon = horizon
        self.horizon_evaluation_step = horizon_evaluation_step

        self.next_item = NextItemMetric(max_time_delta=max_time_delta)
        if map_deltas is not None:
            if map_target_length is None:
                raise ValueError("Need the max target sequence length for mAP computation")
            self.map_target_length = map_target_length
            self.tmap = TMAPMetric(time_delta_thresholds=map_deltas)
        else:
            self.map_target_length = None
            self.tmap = None
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
        return (self.tmap is not None) or (self.otd is not None)

    def reset(self):
        self._target_lengths = []
        self._predicted_lengths = []
        self._horizon_predicted_deltas_sums = []
        self._horizon_n_predicted_deltas = 0
        self._sequence_labels_entropies = []
        self.next_item.reset()
        if self.tmap is not None:
            self.tmap.reset()
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

    def update_next_item(self, seq_lens, timestamps, labels,
                         predicted_timestamps, predicted_labels, predicted_labels_logits):
        """Update next-item metrics with new observations.

        NOTE: Timestamps and labels must be provided without offset w.r.t. predictions, i.e. input features.

        Args:
            seq_lens: Sequence lengths with shape (B).
            timestamps: Dataset timestamps with shape (B, T) (without offset).
            labels: Dataset labels with shape (B, T) (without offset).
            predicted_timestamps: Predicted next-item timestamps with shape (B, T).
            predicted_labels: Predicted next-item labels with shape (B, T).
            predicted_labels_logits: Predicted next-item labels logits with shape (B, T, C).
        """
        # Targets are features shifted w.r.t. prediction.
        mask = PaddedBatch(timestamps, seq_lens).seq_len_mask
        self.next_item.update(mask=mask[:, 1:],  # Same as logical_and(mask[:, 1:], mask[:, :-1]).
                              target_timestamps=timestamps[:, 1:],
                              target_labels=labels[:, 1:],
                              predicted_timestamps=predicted_timestamps[:, :-1],
                              predicted_labels=predicted_labels[:, :-1],
                              predicted_labels_logits=predicted_labels_logits[:, :-1])

    def update_horizon(self, seq_lens, timestamps, labels,
                       indices, indices_lens,
                       seq_predicted_timestamps, seq_predicted_labels, seq_predicted_labels_logits,
                       seq_predicted_weights=None):
        """Update sequence metrics with new observations.

        NOTE: Timestamps and labels must be provided without offset w.r.t. predictions, i.e. input features.

        Args:
            seq_lens: Sequence lengths with shape (B).
            timestamps: Dataset timestamps with shape (B, T) (without offset).
            labels: Dataset labels with shape (B, T) (without offset).
            indices: Sequence prediction initial feature positions with shape (B, I).
            indices_lens: The number of indices for each element in the batch with shape (B).
            seq_predicted_timestamps: Predicted timestamps with shape (B, I, N).
            seq_predicted_labels: Predicted labels with shape (B, I, N).
            seq_predicted_labels_logits: Predicted labels logits with shape (B, I, N, C).
            seq_predicted_weights (optional): Choose > 0 during OTD computation and use top-K if > 0 doesn't produce the required number of events.
        """
        features = PaddedBatch({"timestamps": timestamps, "labels": labels}, seq_lens)
        indices = PaddedBatch(indices, indices_lens)
        predictions = {"timestamps": seq_predicted_timestamps,
                       "labels": seq_predicted_labels,
                       "labels_logits": seq_predicted_labels_logits}
        if seq_predicted_weights is not None:
            predictions["_weights"] = seq_predicted_weights
        predictions = PaddedBatch(predictions, indices_lens)
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
        horizon_predictions_mask = self._get_horizon_mask(initial_timestamps, predictions)  # (B, I, N).
        if seq_predicted_weights is not None:
            predictions_mask = horizon_predictions_mask.logical_and(seq_predicted_weights > 0)
        else:
            predictions_mask = horizon_predictions_mask
        self._target_lengths.append(targets_mask[seq_mask].sum(1).cpu().flatten())  # (V).
        self._predicted_lengths.append(predictions_mask[seq_mask].sum(1).cpu().flatten())  # (BI).

        # Update deltas stats.
        predicted_timestamps = predictions.payload["timestamps"][seq_mask]  # (V, N).
        predicted_labels = predictions.payload["labels"][seq_mask]  # (V, N).
        if (len(predicted_timestamps) > 0) and (predicted_timestamps.shape[1] >= 2):
            deltas = predicted_timestamps[:, 1:] - predicted_timestamps[:, :-1]
            self._horizon_predicted_deltas_sums.append(deltas.float().mean().cpu() * deltas.numel())
            self._horizon_n_predicted_deltas += deltas.numel()

        # Update entropies.
        not_event = predicted_labels.max().item() + 1
        predicted_labels_masked = predictions.payload["labels"].masked_fill(~predictions_mask, not_event).flatten(1, 2)  # (B, IN).
        assert predicted_labels_masked.ndim == 2
        counts = batch_bincount(predicted_labels_masked, not_event + 1)[:, :-1]  # (B, C).
        probs = (counts / counts.sum(dim=1, keepdim=True).clip(min=1))
        entropies = -(probs * probs.clip(min=1e-6).log()).sum(1)  # (B).
        self._sequence_labels_entropies.append(entropies.cpu())

        # Update T-mAP.
        if self.tmap is not None:
            self.tmap.update(
                target_mask=targets_mask[seq_mask][:, :self.map_target_length],  # (V, K).
                target_times=targets.payload["timestamps"][seq_mask][:, :self.map_target_length],  # (V, K).
                target_labels=targets.payload["labels"][seq_mask][:, :self.map_target_length],  # (V, K).
                predicted_mask=horizon_predictions_mask[seq_mask],  # (V, N).
                predicted_times=predicted_timestamps,  # (V, N).
                predicted_labels_scores=predictions.payload["labels_logits"][seq_mask],  # (V, N, C).
            )

        # Update OTD.
        if self.otd is not None:
            if predicted_timestamps.shape[-1] < self.otd_steps:
                raise RuntimeError("Need more predicted events for OTD evaluation.")
            if seq_predicted_weights is not None:
                otd_weights = predictions.payload["_weights"][seq_mask]  # (V, S).
                otd_mask = otd_weights > 0  # (V, S).
                otd_mask.logical_and_(otd_mask.cumsum(1) <= self.otd_steps)
                not_enough = otd_mask.sum(1) < self.otd_steps  # (V).
                top_indices = otd_weights[not_enough].topk(self.otd_steps, dim=1)[1]  # (E, R).
                otd_mask[not_enough] = otd_mask[not_enough].scatter(1, top_indices, torch.full_like(top_indices, True, dtype=torch.bool))
                otd_predicted_timestamps = predicted_timestamps[otd_mask].reshape(-1, self.otd_steps)  # (V, S).
                otd_predicted_labels = predicted_labels[otd_mask].reshape(-1, self.otd_steps)  # (V, S).
            else:
                otd_predicted_timestamps = predicted_timestamps[:, :self.otd_steps]  # (V, S).
                otd_predicted_labels = predicted_labels[:, :self.otd_steps]  # (V, S).
            self.otd.update(
                target_times=targets.payload["timestamps"][seq_mask][:, :self.otd_steps],  # (V, S).
                target_labels=targets.payload["labels"][seq_mask][:, :self.otd_steps],  # (V, S).
                predicted_times=otd_predicted_timestamps,  # (V, S).
                predicted_labels=otd_predicted_labels,  # (V, S).
            )

    def compute(self):
        values = {}
        if self._target_lengths:
            target_lengths = torch.cat(self._target_lengths)
            predicted_lengths = torch.cat(self._predicted_lengths)
            sequence_labels_entropies = torch.cat(self._sequence_labels_entropies)
            values.update({
                "mean-target-length": target_lengths.sum().item() / target_lengths.numel(),
                "mean-predicted-length": predicted_lengths.sum().item() / predicted_lengths.numel(),
                "horizon-mean-time-step": torch.stack(self._horizon_predicted_deltas_sums).sum().item() / self._horizon_n_predicted_deltas,
                "sequence-labels-entropy": sequence_labels_entropies.mean().item()
            })
        values.update(self.next_item.compute())
        if self.tmap is not None:
            values.update(self.tmap.compute())
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
