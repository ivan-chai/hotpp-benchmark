import torch
import torch.nn.functional as F
from hotpp.data import PaddedBatch


class HyproBCELoss(torch.nn.Module):
    def forward(self, target_energies, noise_energies):
        """Compute losses and metrics.

        Args:
            target_energies: Target energies with shape (B, I).
            noise_energies: Target energies with shape (B, I, S).

        Returns:
            Losses dict and metrics dict.
        """
        mask = target_energies.seq_len_mask.bool()
        target_ls = F.logsigmoid(-target_energies.payload)  # (B, I).
        noise_ls = F.logsigmoid(noise_energies.payload)  # (B, I, S).
        likelihoods = target_ls + noise_ls.sum(2)  # (B, I).
        losses = {
            "hypro-bce": -likelihoods[mask].mean()
        }
        with torch.no_grad():
            metrics = {
                "mean-target-energy": target_energies.payload[mask].mean(),
                "mean-noise-energy": noise_energies.payload[mask].mean()
            }
        return losses, metrics

    def get_weights(self, energies):
        """Convert scores to sequence probabilities.

        Args:
            energies: Tensor with shape (B, I, S).

        Returns:
            Probabilities with shape (B, I, S).
        """
        return PaddedBatch(F.softmax(-energies.payload, dim=-1),  # (B, I)
                           energies.seq_lens)
