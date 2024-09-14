import torch
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
        return torch.zeros([c], device=device)
    if targets.dtype != torch.bool:
        if targets.dtype.is_floating_point() or (targets > 1).any() or (targets < 0).any():
            raise ValueError("Expected boolean target on 0-1 values.")
        targets = targets.bool()
    if device.type == "cuda":
        # Compute large tasks step-by-step.
        batch_size = max(cuda_buffer_size // int(b), 1) if cuda_buffer_size is not None else c
        if batch_size < c:
            return torch.cat([compute_map(targets[:, start:start + batch_size].to(device),
                                          scores[:, start:start + batch_size].to(device),
                                          cuda_buffer_size=cuda_buffer_size)
                              for start in range(0, c, batch_size)])
        else:
            targets = targets.to(device)
            scores = scores.to(device)
    order = scores.argsort(dim=0, descending=True)  # (B, C).
    targets = targets.take_along_dim(order, dim=0)  # (B, C).
    cumsum = targets.cumsum(0)
    recalls = cumsum / targets.sum(0).clip(min=1)
    precisions = cumsum / torch.arange(1, 1 + len(targets), device=device)[:, None]
    aps = ((recalls[1:] - recalls[:-1]) * precisions[1:]).sum(0)
    aps += precisions[0] * recalls[0]
    return aps


class TMAPMetric:
    """Average mAP metric among different time difference thresholds.

    Args:
        time_delta_thresholds: A list of time difference thresholds to average metric for.
    """
    def __init__(self, time_delta_thresholds):
        self.time_delta_thresholds = time_delta_thresholds
        self.reset()

    def update(self, target_mask, target_times, target_labels, predicted_mask, predicted_times, predicted_labels_scores):
        """Update metric statistics.

        NOTE: If predicted scores contain log probabilities, then total cost is equal to likelihood.

        Args:
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
        predicted_mask = predicted_mask.bool()
        target_mask = target_mask.bool()
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
        inf_cost = costs[predicted_mask].max().item() + 2
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
            self._n_unmatched_targets_by_delta[i].append(n_unmatched_targets.cpu())  # (C).
            self._matched_by_delta[i].append(predicted_targets.cpu())  # (V, C).
        self._total_targets.append(target_labels_counts.cpu())  # (C).
        self._matched_scores.append(predicted_scores.cpu())  # (V, C).
        self._device = device

    def reset(self):
        self._device = "cpu"
        # The total number of targets of the specified class. A list of tensors, each with shape (C).
        self._total_targets = []
        # For each delta: a list of tensor, each with shape (C), containing the number of unmatched targets for each label.
        self._n_unmatched_targets_by_delta = [list() for _ in self.time_delta_thresholds]
        # Scores of matched predictions for each class. A list of tensors, each with shape (B, C).
        self._matched_scores = []
        # For each delta: a list of tensors, each with shape (B, C), containing the mask of matched predictions for each class.
        self._matched_by_delta = [list() for _ in self.time_delta_thresholds]

    def compute(self):
        if len(self._total_targets) == 0:
            return {}

        # Fix zero-length predictions.
        c = max(map(len, self._total_targets))
        total_targets = torch.stack([v.reshape(c) for v in self._total_targets]).sum(0)  # (C).
        micro_weights = total_targets / total_targets.sum()  # (C).
        matched_scores = torch.cat([v.reshape(len(v), c) for v in self._matched_scores])  # (B, C).
        losses = []
        micro_losses = []
        for i in range(len(self.time_delta_thresholds)):
            n_unmatched_targets = torch.stack([v.reshape(c) for v in self._n_unmatched_targets_by_delta[i]]).sum(0)  # (C).
            matched_targets = torch.cat([v.reshape(v.shape[0], c) for v in self._matched_by_delta[i]])  # (B, C).
            max_recalls = 1 - n_unmatched_targets / total_targets.clip(min=1)
            assert (max_recalls >= 0).all() and (max_recalls <= 1).all()
            label_mask = torch.logical_and(~matched_targets.all(0), matched_targets.any(0))  # (C).
            aps = compute_map(matched_targets[:, label_mask],
                              matched_scores[:, label_mask],
                              device=self._device).cpu()  # (C').
            aps = torch.zeros(c).masked_scatter_(label_mask, aps)
            aps *= max_recalls
            losses.append(aps.sum().item() / c)
            micro_losses.append((aps * micro_weights).sum().item())
        return {
            "T-mAP": sum(losses) / len(losses),
            "T-mAP-micro": sum(micro_losses) / len(micro_losses)
        }
