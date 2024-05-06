import torch


class NextItemMetric(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.reset()

    def update(self, target_mask, target_labels, predicted_labels_logits):
        """Update metrics with new data.

        Args:
            target_mask: Valid targets mask with shape (B, L).
            target_labels: True labels with shape (B, L).
            predicted_labels_logits: Predicted class logits with shape (B, L, C).
        """
        predictions = predicted_labels_logits.argmax(2)  # (B, L).
        is_correct = predictions == target_labels  # (B, L).
        is_correct = is_correct.masked_select(target_mask)  # (V).
        self._n_correct_labels += is_correct.sum().item()
        self._n_labels += is_correct.numel()

    def reset(self):
        self._n_correct_labels = 0
        self._n_labels = 0

    def compute(self):
        return {
            "next-item-accuracy": self._n_correct_labels / self._n_labels
        }
