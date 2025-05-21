import torch


class BaseAggregator(torch.nn.Module):
    def __init__(self, layers=None):
        super().__init__()
        self.layers = layers

    @property
    def need_states(self):
        return self.layers is not None

    def get_activations(self, embeddings, states=None):
        """Get activations sequence with shape (B, L, D).

        Args:
            embeddings: PaddedBatch with shape (B, L, D).
            states (optional): Layer activations with shape (N, B, L, D) or a list of (B, L, D).
        """
        if self.layers is None:
            return embeddings.payload  # (B, L, D).
        else:
            if states is None:
                raise ValueError("Need states for hiddens aggregation")
            if isinstance(states, torch.Tensor):
                return states[self.layers].mean(0)
            else:
                return torch.stack([states[i] for i in self.layers]).mean(0)


class MeanAggregator(BaseAggregator):
    """Compute embeddings average.

    Average activations among positions and (optionally) among layres.

    Args:
        layers: A list of layer indices to average activations for. By default use only outputs.
    """
    def forward(self, embeddings, states=None):
        mask, lengths = embeddings.seq_len_mask.bool(), embeddings.seq_lens
        embeddings = self.get_activations(embeddings, states)  # (B, L, D).
        embeddings = embeddings.masked_fill(~mask.unsqueeze(2), 0)  # (B, L, D).
        sums = embeddings.sum(1)  # (B, D).
        means = sums / lengths.unsqueeze(1).clip(min=1)
        return means  # (B, D).


class LastAggregator(BaseAggregator):
    """Extract last embedding from each sequence."""
    def forward(self, embeddings, states=None):
        lengths = embeddings.seq_lens
        embeddings = self.get_activations(embeddings, states)  # (B, L, D).
        empty_mask = lengths == 0
        indices = (lengths - 1).clip(min=0)  # (B).
        last = embeddings.take_along_dim(indices[:, None, None], 1).squeeze(1)  # (B, D).
        last.masked_fill_(empty_mask.unsqueeze(1), 0)
        return last  # (B, D).


class MiddleAggregator(BaseAggregator):
    """Extract middle embedding from each sequence."""
    def forward(self, embeddings, states=None):
        lengths = embeddings.seq_lens
        embeddings = self.get_activations(embeddings, states)  # (B, L, D).
        empty_mask = lengths == 0
        indices = lengths // 2  # (B).
        last = embeddings.take_along_dim(indices[:, None, None], 1).squeeze(1)  # (B, D).
        last.masked_fill_(empty_mask.unsqueeze(1), 0)
        return last  # (B, D).


class MeanLastAggregator(BaseAggregator):
    """Average the required amount of embeddings at the end of the sequence."""
    def __init__(self, n):
        super().__init__()
        self.n = n

    def forward(self, embeddings, states=None):
        lengths = embeddings.seq_lens
        embeddings = self.get_activations(embeddings, states)  # (B, L, D).
        rng = torch.arange(embeddings.shape[1], device=embeddings.device)[None]  # (1, L).
        mask = torch.logical_and(
            rng < lengths[:, None],
            rng >= lengths[:, None] - self.n
        )  # (B, L).
        embeddings = embeddings.masked_fill(~mask.unsqueeze(2), 0)  # (B, L, D).
        sums = embeddings.sum(1)  # (B, D).
        means = sums / mask.sum(1, keepdim=True).clip(min=1)
        return means  # (B, D).
