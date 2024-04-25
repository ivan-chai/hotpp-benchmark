import torch
from torchmetrics.classification import Accuracy


class NextItemMetric(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.accuracy = Accuracy(task="multiclass",
                                 num_classes=num_classes)
        self.reset()

    def update(self, labels_probs, targets, mask):
        """Update metrics with new data.

        Args:
            labels_probs: Predicted class probabilities with shape (B, L, C).
            targets: True labels with shape (B, L).
            mask: Valid targets mask with shape (B, L).
        """
        b, l, c = labels_probs.shape
        valid_probs = labels_probs.masked_select(mask.unsqueeze(2)).reshape(-1, c)  # (V, C).
        valid_targets = targets.masked_select(mask)  # (V).
        self.accuracy.update(valid_probs, valid_targets)

    def reset(self):
        self.accuracy.reset()

    def compute(self):
        return {
            "next-item-accuracy": self.accuracy.compute()
        }
