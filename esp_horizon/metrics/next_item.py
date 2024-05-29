import torch


class NextItemMetric(torch.nn.Module):
    """Next item (event) prediction evaluation metrics.

    Args:
        max_time_delta: Limit the maximum difference to stabilize MAE estimation.
    """

    def __init__(self, max_time_delta=None):
        super().__init__()
        self.max_time_delta = max_time_delta
        self.reset()

    def update(self, mask, target_timestamps, target_labels,
               predicted_timestamps, predicted_labels_logits):
        """Update metrics with new data.

        Args:
            mask: Valid targets and predictions mask with shape (B, L).
            target_timestamps: Valid target timestamps with shape (B, L).
            target_labels: True labels with shape (B, L).
            predicted_timestamps: Predicted timestamps with shape (B, L).
            predicted_labels_logits: Predicted class logits with shape (B, L, C).
        """
        predictions = predicted_labels_logits.argmax(2)  # (B, L).
        is_correct = predictions == target_labels  # (B, L).
        is_correct = is_correct.masked_select(mask)  # (V).
        self._n_correct_labels += is_correct.sum().item()
        self._n_labels += is_correct.numel()

        ae = (target_timestamps - predicted_timestamps).abs()  # (B, L).
        ae = ae.masked_select(mask)  # (V).
        if self.max_time_delta is not None:
            ae = ae.clip(max=self.max_time_delta)
        self._ae_sums.append(ae.float().mean().cpu() * ae.numel())

        deltas = predicted_timestamps[:, 1:] - predicted_timestamps[:, :-1]  # (B, L - 1).
        deltas = deltas.masked_select(torch.logical_and(mask[:, 1:], mask[:, :-1]))  # (V).
        self._delta_sums.append(deltas.float().mean().cpu() * deltas.numel())
        self._n_deltas += deltas.numel()

    def reset(self):
        self._n_correct_labels = 0
        self._n_labels = 0
        self._ae_sums = []
        self._delta_sums = []
        self._n_deltas = 0

    def compute(self):
        if not self._delta_sums:
            return {}
        return {
            "next-item-mean-time-step": torch.stack(self._delta_sums).sum() / self._n_deltas,
            "next-item-mae": torch.stack(self._ae_sums).sum() / self._n_labels,
            "next-item-accuracy": self._n_correct_labels / self._n_labels
        }
