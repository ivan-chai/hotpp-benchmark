import torch
from torchmetrics import Metric
from torchmetrics.utilities import dim_zero_cat
from .tmap import compute_map


class NextItemMetric(Metric):
    """Next item (event) prediction evaluation metrics."""

    def __init__(self, max_time_delta=None, compute_on_cpu=False):
        super().__init__(compute_on_cpu=compute_on_cpu)
        self.max_time_delta = max_time_delta
        self._device = torch.device("cpu")

        # There is a bug with tensor states, when computed on CPU. Use lists instead.
        self.add_state("_n_correct_labels", default=[], dist_reduce_fx="cat")
        self.add_state("_n_labels", default=[], dist_reduce_fx="cat")
        self.add_state("_scores", default=[], dist_reduce_fx="cat")
        self.add_state("_labels", default=[], dist_reduce_fx="cat")
        self.add_state("_ae_sums", default=[], dist_reduce_fx="cat")
        self.add_state("_se_sums", default=[], dist_reduce_fx="cat")
        self.add_state("_delta_sums", default=[], dist_reduce_fx="cat")
        self.add_state("_n_deltas", default=[], dist_reduce_fx="cat")

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
        device = mask.device
        is_correct = predicted_labels == target_labels  # (B, L).
        is_correct = is_correct.masked_select(mask)  # (V).
        self._n_correct_labels.append(torch.tensor([is_correct.sum()], device=device))
        self._n_labels.append(torch.tensor([is_correct.numel()], device=device))
        self._scores.append(predicted_labels_logits[mask])  # (V, C).
        self._labels.append(target_labels[mask])  # (V).

        ae = (target_timestamps - predicted_timestamps).abs()  # (B, L).
        if self.max_time_delta is not None:
            ae = ae.clip(max=self.max_time_delta)
        ae = ae.masked_select(mask)  # (V).
        assert ae.numel() == is_correct.numel()
        self._ae_sums.append(ae.float().mean(0, keepdim=True) * ae.numel())
        self._se_sums.append(ae.square().float().mean(0, keepdim=True) * ae.numel())

        deltas = predicted_timestamps[:, 1:] - predicted_timestamps[:, :-1]  # (B, L - 1).
        deltas = deltas.clip(min=0)
        deltas = deltas.masked_select(torch.logical_and(mask[:, 1:], mask[:, :-1]))  # (V).
        self._delta_sums.append(deltas.float().mean(0, keepdim=True) * deltas.numel())
        self._n_deltas.append(torch.tensor([deltas.numel()], device=device))

        self._device = mask.device

    def compute(self):
        delta_sums = dim_zero_cat(self._delta_sums)
        if len(delta_sums) == 0:
            return {}
        device = delta_sums.device
        ae_sums = dim_zero_cat(self._ae_sums)
        se_sums = dim_zero_cat(self._se_sums)
        scores = dim_zero_cat(self._scores)
        n_deltas = dim_zero_cat(self._n_deltas).sum().item()
        n_labels = dim_zero_cat(self._n_labels).sum().item()
        n_correct_labels = dim_zero_cat(self._n_correct_labels).sum().item()

        nc = scores.shape[-1]
        labels = dim_zero_cat(self._labels)
        one_hot_labels = torch.nn.functional.one_hot(labels.long(), nc).bool()  # (B, C).
        micro_weights = one_hot_labels.sum(0) / one_hot_labels.sum()  # (C).
        aps, max_f_scores = compute_map(one_hot_labels, scores, device=self._device)  # (C).
        aps = aps.to(device)
        max_f_scores = max_f_scores.to(device)
        return {
            "next-item-mean-time-step": delta_sums.sum().item() / n_deltas,
            "next-item-mae": ae_sums.sum().item() / n_labels,
            "next-item-rmse": (se_sums.sum() / n_labels).sqrt().item(),
            "next-item-accuracy": n_correct_labels / n_labels,
            "next-item-max-f-score": max_f_scores.mean().item(),
            "next-item-max-f-score-weighted": (max_f_scores * micro_weights).sum().item(),
            "next-item-map": aps.mean().item(),
            "next-item-map-weighted": (aps * micro_weights).sum().item()
        }
