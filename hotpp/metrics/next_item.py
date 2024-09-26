import torch
from .tmap import compute_map


class NextItemMetric(torch.nn.Module):
    """Next item (event) prediction evaluation metrics."""

    def __init__(self, max_time_delta=None):
        super().__init__()
        self.max_time_delta = max_time_delta
        self.reset()

    def update(self, mask, target_timestamps, target_labels,
               predicted_timestamps, predicted_labels, predicted_labels_logits):
        """Update metrics with new data.

        Args:
            mask: Valid targets and predictions mask with shape (B, L).
            target_timestamps: Valid target timestamps with shape (B, L).
            target_labels: True labels with shape (B, L).
            predicted_timestamps: Predicted timestamps with shape (B, L).
            predicted_labels: Predicted labels with shape (B, L).
            predicted_labels_logits: Predicted class logits with shape (B, L, C).
        """
        is_correct = predicted_labels == target_labels  # (B, L).
        is_correct = is_correct.masked_select(mask)  # (V).
        self._n_correct_labels += is_correct.sum().item()
        self._n_labels += is_correct.numel()
        self._scores.append(predicted_labels_logits[mask].cpu())  # (V, C).
        self._labels.append(target_labels[mask].cpu())  # (V).

        ae = (target_timestamps - predicted_timestamps).abs()  # (B, L).
        if self.max_time_delta is not None:
            ae = ae.clip(max=self.max_time_delta)
        ae = ae.masked_select(mask)  # (V).
        assert ae.numel() == is_correct.numel()
        self._ae_sums.append(ae.float().mean().cpu() * ae.numel())
        self._se_sums.append(ae.square().float().mean().cpu() * ae.numel())

        deltas = predicted_timestamps[:, 1:] - predicted_timestamps[:, :-1]  # (B, L - 1).
        deltas = deltas.clip(min=0)
        deltas = deltas.masked_select(torch.logical_and(mask[:, 1:], mask[:, :-1]))  # (V).
        self._delta_sums.append(deltas.float().mean().cpu() * deltas.numel())
        self._n_deltas += deltas.numel()

        self._device = mask.device

    def reset(self):
        self._device = "cpu"
        self._n_correct_labels = 0
        self._n_labels = 0
        self._scores = []
        self._labels = []
        self._ae_sums = []
        self._se_sums = []
        self._delta_sums = []
        self._n_deltas = 0

    def compute(self):
        if not self._delta_sums:
            return {}
        scores = torch.cat(self._scores)
        nc = scores.shape[-1]
        labels = torch.cat(self._labels)
        one_hot_labels = torch.nn.functional.one_hot(labels.long(), nc).bool()  # (B, C).
        micro_weights = one_hot_labels.sum(0) / one_hot_labels.sum()  # (C).
        aps = compute_map(one_hot_labels, scores, device=self._device).cpu()  # (C).
        return {
            "next-item-mean-time-step": torch.stack(self._delta_sums).sum() / self._n_deltas,
            "next-item-mae": torch.stack(self._ae_sums).sum() / self._n_labels,
            "next-item-rmse": (torch.stack(self._se_sums).sum() / self._n_labels).sqrt(),
            "next-item-accuracy": self._n_correct_labels / self._n_labels,
            "next-item-map": aps.mean(),
            "next-item-map-micro": (aps * micro_weights).sum()
        }
