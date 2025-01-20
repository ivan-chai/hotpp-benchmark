import torch


class MeanAggregator(torch.nn.Module):
    """Compute embeddings average."""
    def forward(self, embeddings):
        embeddings, mask, lengths = embeddings.payload, embeddings.seq_len_mask.bool(), embeddings.seq_lens
        embeddings = embeddings.masked_fill(~mask.unsqueeze(2), 0)  # (B, L, D).
        sums = embeddings.sum(1)  # (B, D).
        means = sums / lengths.unsqueeze(1).clip(min=1)
        return means  # (B, D).


class LastAggregator(torch.nn.Module):
    """Extract last embedding from each sequence."""
    def forward(self, embeddings):
        embeddings, lengths = embeddings.payload, embeddings.seq_lens
        empty_mask = lengths == 0
        indices = (lengths - 1).clip(min=0)  # (B).
        last = embeddings.take_along_dim(indices[:, None, None], 1).squeeze(1)  # (B, D).
        last.masked_fill_(empty_mask.unsqueeze(1), 0)
        return last  # (B, D).


class MiddleAggregator(torch.nn.Module):
    """Extract middle embedding from each sequence."""
    def forward(self, embeddings):
        embeddings, lengths = embeddings.payload, embeddings.seq_lens
        empty_mask = lengths == 0
        indices = lengths // 2  # (B).
        last = embeddings.take_along_dim(indices[:, None, None], 1).squeeze(1)  # (B, D).
        last.masked_fill_(empty_mask.unsqueeze(1), 0)
        return last  # (B, D).


class MeanLastAggregator(torch.nn.Module):
    """Average the required amount of embeddings at the end of the sequence."""
    def __init__(self, n):
        super().__init__()
        self.n = n

    def forward(self, embeddings):
        embeddings, lengths = embeddings.payload, embeddings.seq_lens
        rng = torch.arange(embeddings.shape[1], device=embeddings.device)[None]  # (1, L).
        mask = torch.logical_and(
            rng < lengths[:, None],
            rng >= lengths[:, None] - self.n
        )  # (B, L).
        embeddings = embeddings.masked_fill(~mask.unsqueeze(2), 0)  # (B, L, D).
        sums = embeddings.sum(1)  # (B, D).
        means = sums / mask.sum(1, keepdim=True).clip(min=1)
        return means  # (B, D).
