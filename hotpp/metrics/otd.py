import torch
from torchmetrics import Metric
from torchmetrics.utilities import dim_zero_cat
from torch_linear_assignment import batch_linear_assignment


def batch_bincount(x, minlength=0):
    # x: (B, N).
    # returns: (B, C).
    b, n = x.shape
    c = max(x.max().item() + 1, minlength)
    counts = x.new_zeros(b, c)
    counts.scatter_add_(dim=1, index=x.long(), src=torch.ones_like(x))
    return counts


class OTDMetric(Metric):
    """Optimal Transport Distance (OTD) for event sequences.

    See the original paper for details:
    Mei, Hongyuan, Guanghui Qin, and Jason Eisner. "Imputing missing events in continuous-time
    event streams." International Conference on Machine Learning. PMLR, 2019.

    Args:
        k: The number of future events to evaluate.
        insert_cost: The cost of inserting new event to the prediction.
        delete_cost: The cost of deleting an event from the prediction.
    """

    def __init__(self, insert_cost, delete_cost, compute_on_cpu=False):
        super().__init__(compute_on_cpu=compute_on_cpu)
        self.insert_cost = insert_cost
        self.delete_cost = delete_cost
        self.add_state("_costs", default=[], dist_reduce_fx="cat")
        self.add_state("_label_distribution_deltas", default=[], dist_reduce_fx="cat")

    @torch.no_grad()
    def update(self, target_times, target_labels, predicted_times, predicted_labels):
        """Update metric statistics.

        Args:
            target_times: Target timestamps with shape (B, N).
            target_labels: Target labels with shape (B, N).
            predicted_times: Event timestamps with shape (B, N).
            predicted_labels: Event labels with shape (B, N).
        """
        assert target_times.shape == target_labels.shape == predicted_times.shape == predicted_labels.shape
        b, n = predicted_times.shape
        if b == 0:
            return
        infinity = self.insert_cost + self.delete_cost
        costs = (predicted_times[:, :, None] - target_times[:, None, :]).abs().float().clip(max=infinity)  # (B, N, N).
        costs.masked_fill_(predicted_labels[:, :, None] != target_labels[:, None, :], infinity)
        self._costs.append(self._get_min_distance(costs))  # (B).

        max_labels = max(target_labels.max().item(), predicted_labels.max().item()) + 1
        target_counts = batch_bincount(target_labels, max_labels)
        predicted_counts = batch_bincount(predicted_labels, max_labels)
        self._label_distribution_deltas.append((target_counts - predicted_counts).abs().sum(1))  # (B).

    def compute(self):
        costs = dim_zero_cat(self._costs)
        if len(costs) == 0:
            return {}
        label_distribution_deltas = dim_zero_cat(self._label_distribution_deltas)

        return {
            "optimal-transport-distance": costs.mean().item(),
            "next-k-label-distribution-delta": label_distribution_deltas.float().mean().item()
        }

    def _get_min_distance(self, costs):
        """Get the minimum cost.

        Args:
            costs: The costs matrix with shape (B, N, K).

        Returns:
            Minimum costs with shape (B).
        """
        matching = batch_linear_assignment(costs)  # (B, N).
        mask = matching >= 0  # (B, N).
        matched_costs = costs.take_along_dim(matching.clip(min=0).unsqueeze(2), 2).squeeze(2)  # (B, N).
        matched_costs = (matched_costs * mask).sum(1)  # (B).
        n_delete = (~mask).sum(1)  # (B).
        n_insert = costs.shape[2] - mask.sum(1)  # (B).
        min_costs = matched_costs + self.insert_cost * n_insert + self.delete_cost * n_delete
        return min_costs
