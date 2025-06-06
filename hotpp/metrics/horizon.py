import torch
from torchmetrics.aggregation import MeanMetric

from ..data import PaddedBatch
from .horizon_binary_targets import HorizonBinaryTargetsMetric
from .next_item import NextItemMetric
from .otd import OTDMetric, batch_bincount
from .tmap import TMAPMetric


class HorizonMetric(torch.nn.Module):
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
        horizon_binary_targets: Horizon targets in the form of dictionaries with "horizon", "label", "threshold", and "is_less" fields.
        log_amount: Set this flag if input amount is processed with log(1 + x).
    """
    def __init__(self, horizon, horizon_evaluation_step=1, max_time_delta=None,
                 map_deltas=None, map_target_length=None,
                 otd_steps=None, otd_insert_cost=None, otd_delete_cost=None,
                 horizon_binary_targets=None, log_amount=False,
                 compute_on_cpu=False):
        super().__init__()
        self.horizon = horizon
        self.horizon_evaluation_step = horizon_evaluation_step

        self._target_lengths = MeanMetric(compute_on_cpu=compute_on_cpu)
        self._predicted_lengths = MeanMetric(compute_on_cpu=compute_on_cpu)
        self._horizon_predicted_deltas = MeanMetric(compute_on_cpu=compute_on_cpu)
        self._sequence_labels_entropies = MeanMetric(compute_on_cpu=compute_on_cpu)
        self._target_sequence_labels_entropies = MeanMetric(compute_on_cpu=compute_on_cpu)

        self.next_item = NextItemMetric(max_time_delta=max_time_delta, compute_on_cpu=compute_on_cpu)
        if map_deltas is not None:
            if map_target_length is None:
                raise ValueError("Need the max target sequence length for mAP computation")
            self.map_target_length = map_target_length
            self.tmap = TMAPMetric(horizon=horizon,
                                   time_delta_thresholds=map_deltas,
                                   compute_on_cpu=compute_on_cpu)
        else:
            self.map_target_length = None
            self.tmap = None
        if otd_steps is not None:
            if (otd_insert_cost is None) or (otd_delete_cost is None):
                raise ValueError("Need insertion and deletion costs for the OTD metric.")
            self.otd_steps = otd_steps
            self.otd = OTDMetric(insert_cost=otd_insert_cost,
                                 delete_cost=otd_delete_cost,
                                 compute_on_cpu=compute_on_cpu)
        else:
            self.otd_steps = None
            self.otd = None

        if horizon_binary_targets is not None:
            self.horizon_binary_targets = HorizonBinaryTargetsMetric(horizon_binary_targets, log_amount=log_amount)
        else:
            self.horizon_binary_targets = None

        self.reset()

    @property
    def need_amount(self):
        return self.horizon_binary_targets is not None

    @property
    def horizon_prediction(self):
        return (self.tmap is not None) or (self.otd is not None)

    def reset(self):
        self._target_lengths.reset()
        self._predicted_lengths.reset()
        self._horizon_predicted_deltas.reset()
        self._sequence_labels_entropies.reset()
        self._target_sequence_labels_entropies.reset()

        self.next_item.reset()
        if self.tmap is not None:
            self.tmap.reset()
        if self.otd is not None:
            self.otd.reset()
        if self.horizon_binary_targets is not None:
            self.horizon_binary_targets.reset()

    def select_horizon_indices(self, seq_lens):
        """Select indices for horizon metrics evaluation."""
        step = self.horizon_evaluation_step
        l = seq_lens.max().item()
        # Skip first `step` events.
        if l > step:
            indices = torch.arange(step, l, step, device=seq_lens.device)  # (I).
        else:
            indices = torch.zeros([], dtype=torch.long, device=seq_lens.device)  # (I).
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
                       seq_predicted_mask=None, seq_predicted_probabilities=None,
                       amounts=None, seq_predicted_amounts=None):
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
            seq_predicted_mask (optional): Mask of predicted events with shape (B, I, N).
            seq_predicted_probabilities (optional): Occurrence probabilities for each prediction with shape (B, I, N).
                By default all probabilities are equal to one. If probabilities are provided, they are used in T-mAP
                computation instead of seq_predicted_mask.
            amounts: Dataset amounts with shape (B, T).
            seq_predicted_amounts (optional): Predicted amount values for the horizon binary targets metric with shape (B, I, N).
        """
        b, l, n = seq_predicted_timestamps.shape
        device = seq_predicted_timestamps.device
        features = {"timestamps": timestamps, "labels": labels}
        if amounts is not None:
            features["amounts"] = amounts
        features = PaddedBatch(features, seq_lens)
        indices = PaddedBatch(indices, indices_lens)
        if not features.seq_len_mask.take_along_dim(indices.payload, 1).masked_select(indices.seq_len_mask).all():
            raise ValueError("Some indices are out of sequence lengths")

        # Extract target windows.
        seq_initial_timestamps = features.payload["timestamps"].take_along_dim(indices.payload, 1)  # (B, I).
        targets = self._extract_target_sequences(features, indices)  # (B, I, K).
        if targets.seq_lens.sum() == 0:
            return
        if (indices_lens < targets.seq_lens).any():
            raise ValueError("Some predictions are missing.")

        # Extract valid evaluation positions.
        seq_mask = targets.seq_len_mask.bool()  # (B, I).
        v = seq_mask.sum().item()
        initial_timestamps = seq_initial_timestamps[:, :l][seq_mask]  # (V).
        target_timestamps = targets.payload["timestamps"][seq_mask]  # (V, K).
        target_labels = targets.payload["labels"][seq_mask]  # (V, K).
        if "amounts" in targets.payload:
            target_amounts = targets.payload["amounts"][seq_mask]
        else:
            target_amounts = None
        predicted_timestamps = seq_predicted_timestamps[:, :l][seq_mask]  # (V, N).
        predicted_labels = seq_predicted_labels[:, :l][seq_mask]  # (V, N).
        predicted_labels_logits = seq_predicted_labels_logits[:, :l][seq_mask]  # (V, N, C).
        if seq_predicted_mask is None:
            seq_predicted_mask = torch.ones(b, l, n, dtype=torch.bool, device=device)  # (B, I, N).
        predicted_mask = seq_predicted_mask[:, :l][seq_mask]  # (V, N)
        if seq_predicted_probabilities is None:
            predicted_probabilities = None
        else:
            predicted_probabilities = seq_predicted_probabilities[:, :l][seq_mask]  # (V, N).
        if seq_predicted_amounts is None:
            predicted_amounts = None
        else:
            predicted_amounts = seq_predicted_amounts[:, :l][seq_mask]  # (V, N).

        # Compute simple horizon metrics.
        horizon_targets_mask = target_timestamps - initial_timestamps.unsqueeze(1) < self.horizon  # (V, N).
        horizon_predicted_mask = torch.logical_and(predicted_timestamps - initial_timestamps.unsqueeze(1) < self.horizon,
                                                   predicted_mask)  # (V, N).
        self._target_lengths.update(horizon_targets_mask.sum(1))  # (V).
        self._predicted_lengths.update(horizon_predicted_mask.sum(1).flatten())  # (BI).

        # Update deltas stats.
        if (len(predicted_timestamps) > 0) and (predicted_timestamps.shape[1] >= 2):
            deltas = predicted_timestamps[:, 1:] - predicted_timestamps[:, :-1]
            self._horizon_predicted_deltas.update(deltas.float())

        # Update entropies.
        predicted_entropies = self._eval_entropies(seq_initial_timestamps,
                                                   torch.logical_and(seq_predicted_mask, seq_mask.unsqueeze(2)),
                                                   seq_predicted_timestamps, seq_predicted_labels)
        self._sequence_labels_entropies.update(predicted_entropies)
        target_entropies = self._eval_entropies(seq_initial_timestamps, seq_mask.unsqueeze(2).expand(*targets.payload["timestamps"].shape),
                                                targets.payload["timestamps"], targets.payload["labels"])
        self._target_sequence_labels_entropies.update(target_entropies)

        # Update T-mAP.
        if self.tmap is not None:
            tmap_predicted_labels_scores = predicted_labels_logits
            tmap_predicted_mask = predicted_mask
            if predicted_probabilities is not None:
                predicted_logprobs = predicted_probabilities.log().clip(min=-100)  # (V, N).
                tmap_predicted_labels_scores = tmap_predicted_labels_scores + predicted_logprobs.unsqueeze(2)  # (V, N, C).
                tmap_predicted_mask = torch.ones_like(tmap_predicted_mask)
            self.tmap.update(
                initial_times=initial_timestamps,
                target_mask=torch.ones(v, self.map_target_length, dtype=torch.bool, device=initial_timestamps.device),  # (V, K).
                target_times=target_timestamps[:, :self.map_target_length],  # (V, K).
                target_labels=target_labels[:, :self.map_target_length],  # (V, K).
                predicted_mask=tmap_predicted_mask,  # (V, N).
                predicted_times=predicted_timestamps,  # (V, N).
                predicted_labels_scores=tmap_predicted_labels_scores  # (V, N, C).
            )

        # Update OTD.
        if self.otd is not None:
            if predicted_timestamps.shape[-1] < self.otd_steps:
                raise RuntimeError("Need more predicted events for OTD evaluation.")
            otd_mask = torch.logical_and(predicted_mask, predicted_mask.cumsum(1) <= self.otd_steps)  # (V, N).
            not_enough = otd_mask.sum(1) < self.otd_steps  # (V).
            if not_enough.any():
                if predicted_probabilities is None:
                    raise ValueError("Not enough predictions for OTD, need predicted probabilities to select predictions")
                top_indices = predicted_probabilities[not_enough].topk(self.otd_steps, dim=1)[1]  # (E, R).
                otd_mask[not_enough] = False
                otd_mask[not_enough] = otd_mask[not_enough].scatter(1, top_indices, torch.full_like(top_indices, True, dtype=torch.bool))
            otd_predicted_timestamps = predicted_timestamps[otd_mask].reshape(v, self.otd_steps)  # (V, S).
            otd_predicted_labels = predicted_labels[otd_mask].reshape(v, self.otd_steps)  # (V, S).
            self.otd.update(
                target_times=targets.payload["timestamps"][seq_mask][:, :self.otd_steps],  # (V, S).
                target_labels=targets.payload["labels"][seq_mask][:, :self.otd_steps],  # (V, S).
                predicted_times=otd_predicted_timestamps,  # (V, S).
                predicted_labels=otd_predicted_labels,  # (V, S).
            )

        # Update horizon binary targets classification.
        if self.horizon_binary_targets is not None:
            if predicted_probabilities is not None:
                binary_targets_predicted_probabilities = predicted_probabilities
            else:
                binary_targets_predicted_probabilities = predicted_mask.float()
            self.horizon_binary_targets.update(
                initial_times=initial_timestamps,
                target_mask=torch.ones(*target_timestamps.shape, dtype=torch.bool, device=target_timestamps.device),  # (V, K).
                target_times=target_timestamps,  # (V, K).
                target_labels=target_labels,  # (V, K).
                target_amounts=target_amounts,  # (V, K).
                predicted_probabilities=binary_targets_predicted_probabilities,  # (V, K).
                predicted_times=predicted_timestamps,  # (V, N).
                predicted_labels_logits=predicted_labels_logits,  # (V, N, C).
                predicted_amounts=predicted_amounts,  # (V, K).
            )

    def compute(self):
        values = {}
        target_length = self._target_lengths.compute()
        if not target_length.isnan():
            values.update({
                "mean-target-length": target_length,
                "mean-predicted-length": self._predicted_lengths.compute(),
                "sequence-labels-entropy": self._sequence_labels_entropies.compute(),
                "target-sequence-labels-entropy": self._target_sequence_labels_entropies.compute()
            })
            mean_time_step = self._horizon_predicted_deltas.compute()
            if not mean_time_step.isnan():
                values["horizon-mean-time-step"] = mean_time_step
        values.update(self.next_item.compute())
        if self.tmap is not None:
            values.update(self.tmap.compute())
        if self.otd is not None:
            values.update(self.otd.compute())
        if self.horizon_binary_targets is not None:
            values.update(self.horizon_binary_targets.compute())
        return values

    def _extract_target_sequences(self, features, indices):
        """Extract target sequence for each index."""
        b, i = indices.shape
        # TODO: remove dependencies between metrics computation.
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

    def _eval_entropies(self, seq_initial_timestamps, seq_mask, seq_timestamps, seq_labels):
        if seq_mask.sum() > 0:
            not_event = seq_labels[seq_mask].max().item() + 1
        else:
            not_event = 0
        horizon_seq_mask = torch.logical_and(seq_mask,
                                             seq_timestamps - seq_initial_timestamps.unsqueeze(2) < self.horizon)
        labels_masked = seq_labels.masked_fill(~horizon_seq_mask, not_event).flatten(1, 2)  # (B, IN).
        counts = batch_bincount(labels_masked, not_event + 1)[:, :-1]  # (B, C).
        probs = (counts / counts.sum(dim=1, keepdim=True).clip(min=1))
        entropies = -(probs * probs.clip(min=1e-6).log()).sum(1)  # (B).
        return entropies
