import torch
from torchmetrics import Metric
from torchmetrics.utilities import dim_zero_cat
from torch_linear_assignment import batch_linear_assignment


def compute_map(targets, scores, device=None, cuda_buffer_size=10**7):
    """Compute APs.

    Args:
        targets: Ground truth one-hot encoded labels with shape (B, C).
        scores: Predicted label scores with shape (B, C).
        device: Internal computation device.
        cuda_buffer_size: Maximum CUDA memory usage.

    Returns:
        AP values for each of C classes.
    """
    # Targets: (B, C).
    # Scores: (B, C).
    b, c = targets.shape
    device = targets.device if device is None else device
    if (b == 0) or (c == 0):
        return torch.zeros([c], device=device), torch.zeros([c], device=device)
    if targets.dtype != torch.bool:
        if targets.dtype.is_floating_point() or (targets > 1).any() or (targets < 0).any():
            raise ValueError("Expected boolean target or 0-1 values.")
        targets = targets.round().bool()
    if device.type == "cuda":
        # Compute large tasks step-by-step.
        batch_size = max(cuda_buffer_size // int(b), 1) if cuda_buffer_size is not None else c
        if batch_size < c:
            aps, f_scores = [], []
            for start in range(0, c, batch_size):
                ap, f_score = compute_map(targets[:, start:start + batch_size].to(device),
                                          scores[:, start:start + batch_size].to(device),
                                          cuda_buffer_size=cuda_buffer_size)
                aps.append(ap)
                f_scores.append(f_score)
            return torch.cat(aps), torch.cat(f_scores)
        else:
            targets = targets.to(device)
            scores = scores.to(device)
    order = scores.argsort(dim=0, descending=True)  # (B, C).
    targets = targets.take_along_dim(order, dim=0)  # (B, C).
    arange = torch.arange(1, 1 + len(targets), device=device)  # (B).
    n_positives = targets.sum(0)  # (C).
    tp = targets.cumsum(0)  # (B, C).

    recalls = tp / n_positives.clip(min=1)
    precisions = tp / arange[:, None]
    aps = ((recalls[1:] - recalls[:-1]) * precisions[1:]).sum(0)
    aps += precisions[0] * recalls[0]

    f_scores = 2 * precisions * recalls / (precisions + recalls).clip(min=1e-6)  # (B, C).
    f_scores = f_scores.max(0)[0]  # (C).
    return aps, f_scores


class TMAPMetric(Metric):
    """Average mAP metric among different time difference thresholds.

    Args:
        horizon: Prediction horizon.
        time_delta_thresholds: A list of time difference thresholds to average metric for.
    """
    def __init__(self, horizon, time_delta_thresholds, compute_on_cpu=False):
        super().__init__(compute_on_cpu=compute_on_cpu)
        self.horizon = horizon
        self.time_delta_thresholds = time_delta_thresholds
        self._device = torch.device("cpu")

        # The total number of targets of the specified class. A list of tensors, each with shape (C).
        self.add_state("_total_targets", default=[], dist_reduce_fx="cat")
        # For each delta: a list of tensor, each with shape (C), containing the number of unmatched targets for each label.
        for i in range(len(time_delta_thresholds)):
            self.add_state(f"_n_unmatched_targets_delta_{i}", default=[], dist_reduce_fx="cat")
        # Scores of matched predictions for each class. A list of tensors, each with shape (B, C).
        self.add_state("_matched_scores", default=[], dist_reduce_fx="cat")
        # For each delta: a list of tensors, each with shape (B, C), containing the mask of matched predictions for each class.
        for i in range(len(time_delta_thresholds)):
            self.add_state(f"_matched_delta_{i}", default=[], dist_reduce_fx="cat")

    def update(self, initial_times, target_mask, target_times, target_labels, predicted_mask, predicted_times, predicted_labels_scores):
        """Update metric statistics.

        NOTE: If predicted scores contain log probabilities, then total cost is equal to likelihood.

        Args:
            initial_times: Last event time seen by the model with shape (B).
            target_mask: Mask of valid targets with shape (B, T).
            target_times: Target timestamps with shape (B, T).
            target_labels: Target labels with shape (B, T).
            predicted_mask: Mask of valid predictions with shape (B, P).
            predicted_times: Event timestamps with shape (B, P).
            predicted_labels_scores: Event labels scores with shape (B, P, C).
        """
        device = predicted_labels_scores.device
        b, p, c = predicted_labels_scores.shape
        if b == 0:
            return
        predicted_mask = torch.logical_and(predicted_mask.bool(),
                                           predicted_times - initial_times[:, None] < self.horizon)
        target_mask = torch.logical_and(target_mask.bool(),
                                        target_times - initial_times[:, None] < self.horizon)
        target_labels = target_labels.long()
        sorted_time_delta_thresholds = torch.tensor(list(sorted(self.time_delta_thresholds, reverse=True)), device=device)  # (D).
        time_deltas = (predicted_times[:, :, None] - target_times[:, None, :]).abs()  # (B, P, T).
        diff_horizon = time_deltas[None, :, :, :] > sorted_time_delta_thresholds[:, None, None, None]  # (D, B, P, T).

        all_labels = target_labels[target_mask].unique(sorted=False).cpu().tolist()
        target_labels.masked_fill_(~target_mask, -1)
        target_labels_one_hot = torch.nn.functional.one_hot(target_labels + 1, num_classes=c + 1)  # (B, T, C + 1), zero is reserved for "no event".
        target_labels_counts = target_labels_one_hot.sum(1)[:, 1:]  # (B, C).
        target_labels_mask = target_labels_counts > 0  # (B, C).
        target_labels_one_hot = target_labels_one_hot[:, :, 1:]  # (B, T, C).
        target_labels_counts = target_labels_counts.sum(0)  # (C).

        costs = -predicted_labels_scores.take_along_dim(target_labels.clip(min=0)[:, None, :], 2)  # (B, P, T).
        predicted_costs = costs[predicted_mask]
        inf_cost = predicted_costs.max().item() + 2 if len(predicted_costs) > 0 else 1e6
        valid_cost_threshold = inf_cost - 1
        costs.masked_fill_(~predicted_mask.unsqueeze(2), inf_cost)
        costs.masked_fill_(~target_mask.unsqueeze(1), inf_cost)

        # Precompute special cost matrix for each label.
        costs = costs.unsqueeze(3).repeat(1, 1, 1, c)  # (B, P, T, C).
        costs.masked_fill_(~target_labels_one_hot.bool().unsqueeze(1), inf_cost)

        predicted_scores = predicted_labels_scores.masked_select(predicted_mask.unsqueeze(2)).reshape(-1, c)  # (V, C).
        n_valid = len(predicted_scores)
        matching = torch.empty(b, p, c, dtype=torch.long, device=device)  # (B, P, C).
        for i in range(len(self.time_delta_thresholds)):
            # As time_delta_thresholds are sorted in a descending order, the previous mask is always a subset of the next mask.
            # We thus can update costs inplace.
            costs.masked_fill_(diff_horizon[i].unsqueeze(3), inf_cost)  # (B, P, T, C).
            matching.fill_(-1)  # Reset.
            for label in all_labels:
                label_mask = target_labels_mask[:, label]  # (B).
                label_costs = costs[label_mask, :, :, label]  # (M, P, T).
                matching[label_mask, :, label] = batch_linear_assignment(label_costs)  # (M, P).
            matching_costs = costs.take_along_dim(matching.unsqueeze(2).clip(min=0), 2).squeeze(2)  # (B, P, C).
            matching.masked_fill_(matching_costs > valid_cost_threshold, -1)
            n_unmatched_targets = target_labels_counts - (matching.masked_select(predicted_mask.unsqueeze(2)).reshape(n_valid, c) >= 0).sum(0)  # (C).
            predicted_targets = matching.masked_select(predicted_mask.unsqueeze(2)).reshape(n_valid, c) >= 0  # (V, C).
            getattr(self, f"_n_unmatched_targets_delta_{i}").append(n_unmatched_targets[None])  # (1, C).
            getattr(self, f"_matched_delta_{i}").append(predicted_targets)  # (V, C).
        self._total_targets.append(target_labels_counts[None])  # (1, C).
        self._matched_scores.append(predicted_scores)  # (V, C).
        self._device = device

    def compute(self):
        total_targets = dim_zero_cat(self._total_targets)
        if len(total_targets) == 0:
            return {}
        total_targets = total_targets.sum(0)  # (C).
        c = len(total_targets)
        device = total_targets.device
        matched_scores = dim_zero_cat(self._matched_scores)  # (B, C).
        micro_weights = total_targets / total_targets.sum()  # (C).

        maps = []
        micro_maps = []
        f_scores = []
        micro_f_scores = []
        for i in range(len(self.time_delta_thresholds)):
            n_unmatched_targets = dim_zero_cat(getattr(self, f"_n_unmatched_targets_delta_{i}")).sum(0)  # (C).
            matched_targets = dim_zero_cat(getattr(self, f"_matched_delta_{i}"))  # (B, C).
            max_recalls = 1 - n_unmatched_targets / total_targets.clip(min=1)
            assert (max_recalls >= 0).all() and (max_recalls <= 1).all()
            label_mask = torch.logical_and(~matched_targets.all(0), matched_targets.any(0))  # (C).
            aps, max_f_scores = compute_map(matched_targets[:, label_mask],
                                            matched_scores[:, label_mask],
                                            device=self._device)  # (C').
            aps = torch.zeros(c, device=device).masked_scatter_(label_mask, aps.to(device))
            aps *= max_recalls
            maps.append(aps.sum().item() / c)
            micro_maps.append((aps * micro_weights).sum().item())

            max_f_scores = torch.zeros(c, device=device).masked_scatter_(label_mask, max_f_scores.to(device))
            f_scores.append(max_f_scores.sum().item() / c)
            micro_f_scores.append((max_f_scores * micro_weights).sum().item())
        return {
            "T-mAP": sum(maps) / len(maps),
            "T-mAP-weighted": sum(micro_maps) / len(micro_maps),
            "horizon-max-f-score": sum(f_scores) / len(f_scores),
            "horizon-max-f-score-weighted": sum(micro_f_scores) / len(micro_f_scores)
        }
