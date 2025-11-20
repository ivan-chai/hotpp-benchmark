import torch
from torchmetrics import Metric
from torchmetrics.utilities import dim_zero_cat
from torch_linear_assignment import batch_linear_assignment


class HorizonEditDistanceMetric(Metric):
    """Edit distance and F1 for horizon predictions.

    Args:
        horizon: Prediction horizon.
        time_delta_thresholds: A list of time difference thresholds to average metric for.
    """
    def __init__(self, horizon, time_delta_thresholds, compute_on_cpu=False):
        super().__init__(compute_on_cpu=compute_on_cpu)
        self.horizon = horizon
        self.time_delta_thresholds = time_delta_thresholds
        self._device = torch.device("cpu")

        self.add_state(f"_predictions", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state(f"_targets", default=torch.tensor(0), dist_reduce_fx="sum")
        for i in range(len(time_delta_thresholds)):
            self.add_state(f"_true_positives_{i}", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, initial_times, target_mask, target_times, target_labels, predicted_mask, predicted_times, predicted_labels):
        """Update metric statistics.

        Args:
            initial_times: Last event time seen by the model with shape (B).
            target_mask: Mask of valid targets with shape (B, T).
            target_times: Target timestamps with shape (B, T).
            target_labels: Target labels with shape (B, T).
            predicted_mask: Mask of valid predictions with shape (B, P).
            predicted_times: Event timestamps with shape (B, P).
            predicted_labels: Event labels with shape (B, P).
        """
        device = predicted_labels.device
        b, p = predicted_labels.shape
        if b == 0:
            return
        predicted_mask = torch.logical_and(predicted_mask.bool(),
                                           predicted_times - initial_times[:, None] < self.horizon)
        target_mask = torch.logical_and(target_mask.bool(),
                                        target_times - initial_times[:, None] < self.horizon)
        target_labels = target_labels.long()

        time_delta_thresholds = torch.tensor(list(self.time_delta_thresholds), device=device)  # (D).
        time_deltas = (predicted_times[:, :, None] - target_times[:, None, :]).abs()  # (B, P, T).
        time_mismatch = time_deltas[None, :, :, :] > time_delta_thresholds[:, None, None, None]  # (D, B, P, T).
        label_mismatch = predicted_labels[:, :, None] != target_labels[:, None, :]  # (B, P, T).
        costs = torch.logical_or(time_mismatch, label_mismatch[None]).long()  # (D, B, P, T).
        costs = torch.logical_or(costs, torch.logical_or(~predicted_mask[:, :, None].bool(), ~target_mask[:, None, :].bool())[None])

        for i in range(len(self.time_delta_thresholds)):
            matching = batch_linear_assignment(costs[i])  # (B, P).
            matching_mask = matching >= 0  # (B, P).
            matching_costs = costs[i].take_along_dim(matching.clip(min=0)[:, :, None], 2).squeeze(2)  # (B, P).
            result = getattr(self, f"_true_positives_{i}")
            result += (matching_costs[matching_mask] == 0).sum().item()
        self._predictions += predicted_mask.sum().item()
        self._targets += target_mask.sum().item()

    def compute(self):
        n_pred = self._predictions.item()
        n_gt = self._targets.item()
        edit_distances = []
        f1_scores = []
        for i in range(len(self.time_delta_thresholds)):
            n_tp = getattr(self, f"_true_positives_{i}").item()
            n_fp = n_pred - n_tp
            n_fn = n_gt - n_tp
            f1_scores.append(2 * n_tp / (2 * n_tp + n_fp + n_fn))
            edit_distances.append(max(n_fp, n_fn) / max(n_pred, n_gt))
        return {
            "horizon-f1-score-micro": sum(f1_scores) / len(f1_scores),
            "horizon-edit-distance": sum(edit_distances) / len(edit_distances)
        }
