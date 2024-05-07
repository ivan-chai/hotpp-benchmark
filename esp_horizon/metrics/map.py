import torch
from torch_linear_assignment import batch_linear_assignment


def compute_map(targets, scores, cuda_buffer_size=10**7):
    # Targets: (B, C).
    # Scores: (B, C).
    b, c = targets.shape
    if b == 0:
        return torch.zeros_like(scores)
    if torch.cuda.is_available():
        batch_size = max(cuda_buffer_size // int(b), 1) if cuda_buffer_size is not None else c
        # Compute large tasks step-by-step.
        if batch_size < c:
            return torch.cat([compute_map(targets[:, start:start + batch_size],
                                          scores[:, start:start + batch_size])
                              for start in range(0, c, batch_size)])
        targets = targets.cuda()
        scores = scores.cuda()
    order = scores.argsort(dim=0, descending=True)  # (B, C), (B, C).
    targets = targets.take_along_dim(order, dim=0)  # (B, C).
    cumsum = targets.cumsum(0)
    recalls = cumsum / targets.sum(0)
    precisions = cumsum / torch.arange(1, 1 + len(targets), device=targets.device)[:, None]
    aps = ((recalls[1:] - recalls[:-1]) * precisions[1:]).sum(0)
    aps += precisions[0] * recalls[0]
    return aps.cpu()


class MAPMetric:
    """Average mAP metric among different time difference thresholds.

    Args:
        time_delta_thresholds: A list of time difference thresholds.
    """
    def __init__(self, time_delta_thresholds):
        self.time_delta_thresholds = time_delta_thresholds
        self.reset()

    def update(self, target_mask, target_times, target_labels, predicted_mask, predicted_times, predicted_labels_logits):
        """Update metric statistics.

        Args:
            target_mask: Mask of valid targets with shape (B, K).
            target_times: Target timestamps with shape (B, K).
            target_labels: Target labels with shape (B, K).
            predicted_mask: Mask of valid predictions with shape (B, N).
            predicted_times: Event timestamps with shape (B, N).
            predicted_labels_logits: Event labels log probabilities or logits with shape (B, N, C).
        """
        device = predicted_labels_logits.device
        b, n, c = predicted_labels_logits.shape
        predicted_mask = predicted_mask.bool()
        target_mask = target_mask.bool()
        target_labels = target_labels.long()
        sorted_time_delta_thresholds = torch.tensor(list(sorted(self.time_delta_thresholds, reverse=True)), device=device)  # (H).
        h = len(sorted_time_delta_thresholds)
        time_deltas = (predicted_times[:, :, None] - target_times[:, None, :]).abs()  # (B, N, K).
        diff_horizon = time_deltas[None, :, :, :] >= sorted_time_delta_thresholds[:, None, None, None]  # (H, B, N, K).

        all_labels = target_labels[target_mask].unique(sorted=False).cpu().tolist()
        target_labels.masked_fill_(~target_mask, -1)
        target_labels_one_hot = torch.nn.functional.one_hot(target_labels + 1, num_classes=c + 1)  # (B, K, C + 1).
        target_labels_counts = target_labels_one_hot.sum(1)[:, 1:]  # (B, C).
        target_labels_mask = target_labels_counts > 0  # (B, C).
        target_labels_one_hot = target_labels_one_hot[:, :, 1:]  # (B, K, C).
        target_labels_counts = target_labels_counts.sum(0)  # (C).

        predicted_labels_log_probs = torch.nn.functional.log_softmax(predicted_labels_logits, dim=-1)  # (B, N, C).
        costs = -predicted_labels_log_probs.take_along_dim(target_labels.clip(min=0)[:, None, :], 2)  # (B, N, K).
        inf_cost = max(costs.max().item() * 100, 1e6)
        valid_cost_threshold = inf_cost - 1
        costs.masked_fill_(~predicted_mask.unsqueeze(2), inf_cost)
        costs.masked_fill_(~target_mask.unsqueeze(1), inf_cost)
        costs = costs.unsqueeze(3).repeat(1, 1, 1, c)  # (B, N, K, C).
        costs.masked_fill_(~target_labels_one_hot.bool().unsqueeze(1), inf_cost)

        predicted_scores = predicted_labels_log_probs.masked_select(predicted_mask.unsqueeze(2)).reshape(-1, c)  # (V, C).
        n_valid = len(predicted_scores)
        matching = torch.empty(b, n, c, dtype=torch.long, device=device)  # (B, N, C).
        for i in range(h):
            # As time_delta_thresholds are sorted in a descending order, the next mask is always included to the previous mask.
            # We thus can update costs inplace.
            costs.masked_fill_(diff_horizon[i].unsqueeze(3), inf_cost)  # (B, N, K, C).
            matching.fill_(-1)  # Reset.
            for label in all_labels:
                label_mask = target_labels_mask[:, label]  # (B).
                label_costs = costs[label_mask, :, :, label]  # (M, N, K).
                matching[label_mask, :, label] = batch_linear_assignment(label_costs)  # (M, N).
            matching_costs = costs.take_along_dim(matching.unsqueeze(2).clip(min=0), 2).squeeze(2)  # (B, N, C).
            matching.masked_fill_(matching_costs >= valid_cost_threshold, -1)
            n_unmatched_targets = target_labels_counts - (matching.masked_select(predicted_mask.unsqueeze(2)).reshape(n_valid, c) >= 0).sum(0)  # (C).
            predicted_targets = matching.masked_select(predicted_mask.unsqueeze(2)).reshape(n_valid, c) >= 0  # (V, C).
            self._n_unmatched_targets_by_horizon[i].append(n_unmatched_targets.cpu())  # (C).
            self._matched_targets_by_horizon[i].append(predicted_targets.cpu())  # (V, C).
        self._total_targets.append(target_labels_counts.cpu())  # (C).
        self._matched_scores.append(predicted_scores.cpu())  # (V, C).

    def reset(self):
        self._total_targets = []
        self._n_unmatched_targets_by_horizon = [list() for _ in self.time_delta_thresholds]
        self._matched_scores = []
        self._matched_targets_by_horizon = [list() for _ in self.time_delta_thresholds]

    def compute(self):
        if len(self._total_targets) == 0:
            return {}

        # Fix zero-length predictions.
        nc = max(map(len, self._total_targets))
        total_targets = torch.stack([v.reshape(nc) for v in self._total_targets]).sum(0)  # (C).
        matched_scores = torch.cat([v.reshape(v.shape[0], nc) for v in self._matched_scores])  # (B, C).
        c = len(total_targets)
        losses = []
        for h in range(len(self.time_delta_thresholds)):
            n_unmatched_targets = torch.stack([v.reshape(nc) for v in self._n_unmatched_targets_by_horizon[h]]).sum(0)  # (C).
            matched_targets = torch.cat([v.reshape(v.shape[0], nc) for v in self._matched_targets_by_horizon[h]])  # (B, C).
            max_recalls = 1 - n_unmatched_targets / total_targets.clip(min=1)
            assert (max_recalls >= 0).all() and (max_recalls <= 1).all()
            label_mask = torch.logical_and(~matched_targets.all(0), matched_targets.any(0)).numpy()  # (C).
            aps = compute_map(matched_targets[:, label_mask],
                              matched_scores[:, label_mask])  # (C').
            aps *= max_recalls.numpy()[label_mask]
            losses.append(aps.sum().item() / c)
        return {
            "detection-mAP": sum(losses) / len(losses)
        }
