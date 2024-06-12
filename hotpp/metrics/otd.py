import torch


class OTDMetric:
    """Optimal Transport Distance (OTD) for event sequences.

    See the original paper for details:
    Mei, Hongyuan, Guanghui Qin, and Jason Eisner. "Imputing missing events in continuous-time
    event streams." International Conference on Machine Learning. PMLR, 2019.

    Args:
        k: The number of future events to evaluate.
        insert_cost: The cost of inserting new event to the prediction.
        delete_cost: The cost of deleting an event from the prediction.
    """

    def __init__(self, insert_cost, delete_cost):
        self.insert_cost = insert_cost
        self.delete_cost = delete_cost
        self.reset()

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
        costs = (predicted_times[:, :, None] - target_times[:, None, :]).abs().float()  # (B, N, N).
        infinity = torch.finfo(costs.dtype).max
        costs.masked_fill_(predicted_labels[:, :, None] != target_labels[:, None, :], infinity)
        self._costs.append(self._get_min_distance(costs).cpu())  # (B).

    def reset(self):
        self._costs = []

    def compute(self):
        if len(self._costs) == 0:
            return {}
        costs = torch.cat(self._costs)
        return {
            "optimal-transport-distance": costs.mean().item()
        }

    def _get_min_distance(self, costs):
        """Get the minimum cost.

        Args:
            costs: The costs matrix with shape (B, N, K).

        Returns:
            Minimum costs with shape (B).
        """
        b, n, k = costs.shape
        min_costs = torch.empty(b, n + 1, k + 1, dtype=costs.dtype, device=costs.device)
        min_costs[:, :, 0] = torch.arange(n + 1, device=costs.device) * self.insert_cost
        min_costs[:, 0, :] = torch.arange(k + 1, device=costs.device) * self.delete_cost
        for i in range(n):
            for j in range(k):
                c = torch.minimum(min_costs[:, i, j + 1] + self.insert_cost, min_costs[:, i + 1, j] + self.delete_cost)   # (B).
                min_costs[:, i + 1, j + 1] = torch.minimum(c, min_costs[:, i, j] + costs[:, i, j])
        return min_costs[:, n, k]
