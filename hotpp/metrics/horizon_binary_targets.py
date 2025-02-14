import torch
from torch_linear_assignment import batch_linear_assignment
from sklearn.metrics import roc_auc_score


class HorizonBinaryTargetsMetric:
    """Average mAP metric among different time difference thresholds.

    Args:
        targets: Horizon targets in the form of dictionaries with "horizon", "label", "threshold", and "is_less" fields.
        log_amount: Set this flag if input amount is processed with log(1 + x).
    """
    def __init__(self, targets, log_amount=False):
        self.targets = list(targets)
        self.log_amount = log_amount
        self.reset()

    def update(self, initial_times, target_mask, target_times, target_labels,
               predicted_probabilities, predicted_times, predicted_labels_logits,
               target_amounts=None, predicted_amounts=None):
        """Update metric statistics.

        NOTE: If predicted scores contain log probabilities, then total cost is equal to likelihood.

        Args:
            initial_times: Last event time seen by the model with shape (B).
            target_mask: Mask of valid targets with shape (B, T).
            target_times: Target timestamps with shape (B, T).
            target_labels: Target labels with shape (B, T).
            predicted_probabilities: Occurrence probability of each event with shape (B, P).
            predicted_times: Event timestamps with shape (B, P).
            predicted_labels_logits: Event labels logits (log-probabilities) with shape (B, P, C).
            target_amounts: The "amount" of event (like wastes). By default all amounts are equal to 1.
            predicted_amounts: The predicted "amount" of event (like wastes). By default all amounts are equal to 1.
        """
        device = predicted_labels_logits.device
        b, p, c = predicted_labels_logits.shape
        if b == 0:
            return
        if target_amounts is None:
            target_amounts = torch.ones(b, p, device=device)
        elif self.log_amount:
            target_amounts = target_amounts.exp() - 1
        if predicted_amounts is None:
            predicted_amounts = torch.ones(b, p, device=device)
        elif self.log_amount:
            predicted_amounts = predicted_amounts.exp() - 1
        horizons = {target["horizon"] for target in self.targets}
        by_horizon = {h: [i for i, t in enumerate(self.targets) if t["horizon"] == h] for h in horizons}
        target_labels = target_labels.masked_fill(~target_mask.bool(), 0)
        target_labels_probabilities = torch.nn.functional.one_hot(target_labels.long(), c)  # (B, P, C).
        predicted_labels_probabilities = torch.nn.functional.softmax(predicted_labels_logits, dim=-1)  # (B, P, C).
        binary_labels = torch.empty(b, len(self.targets), device=device)
        binary_scores = torch.empty(b, len(self.targets), device=device)
        for horizon in horizons:
            target_weights = torch.logical_and(target_mask, target_times - initial_times.unsqueeze(1) < horizon).float()  # (B, T).
            predicted_weights = predicted_probabilities * (predicted_times - initial_times.unsqueeze(1) < horizon).float()  # (B, P).
            for i in by_horizon[horizon]:
                target = self.targets[i]
                labels = torch.tensor(target["label"], device=device, dtype=torch.long)  # (R).
                target_cumsum = (target_labels_probabilities[..., labels]
                                 * target_weights[:, :, None]
                                 * target_amounts[:, :, None]).flatten(1, 2).sum(1)  # (B).
                predicted_cumsum = (predicted_labels_probabilities[..., labels]
                                    * predicted_weights[:, :, None]
                                    * predicted_amounts[:, :, None]).flatten(1, 2).sum(1)  # (B).
                binary_labels[:, i] = (target_cumsum < target["threshold"] if target["is_less"]
                                       else target_cumsum >= target["threshold"])
                binary_scores[:, i] = predicted_cumsum
        self._targets.append(binary_labels.cpu())
        self._scores.append(binary_scores.cpu())

    def reset(self):
        # Binary classification results for all targets (lists of arrays with length N).
        self._targets = []
        self._scores = []

    def compute(self):
        if not self._targets:
            return {}
        targets = torch.cat(self._targets)  # (B, T).
        scores = torch.cat(self._scores)  # (B, T).
        active_classes = torch.logical_and(targets.any(0), ~targets.all(0))  # (T).
        targets = targets[:, active_classes]
        scores = scores[:, active_classes]
        is_less = torch.tensor([t["is_less"] for t in self.targets], dtype=torch.bool)[active_classes]
        scores[:, is_less] *= -1
        return {
            "horizon-binary-targets-roc-auc": roc_auc_score(targets, scores, average="macro"),
            "horizon-binary-targets-roc-auc-weighted": roc_auc_score(targets, scores, average="weighted")
        }
