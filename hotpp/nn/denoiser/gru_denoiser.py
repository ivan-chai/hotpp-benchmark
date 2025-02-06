import torch
from hotpp.data import PaddedBatch


class GRUDenoiser(torch.nn.Module):
    def __init__(self, input_size, length, hidden_size, generation_steps, num_layers=1, bidirectional=True):
        super().__init__()
        self._steps_embedder = torch.nn.Embedding(generation_steps + 1, input_size)
        self._hidden_size = hidden_size
        self._bidirectional = bidirectional
        self._total_layers = num_layers * (2 if bidirectional else 1)
        self._model = torch.nn.GRU(input_size, hidden_size,
                                   num_layers=num_layers, batch_first=True,
                                   bidirectional=bidirectional)
        self._projection = torch.nn.Linear(hidden_size, input_size)

    @property
    def condition_size(self):
        return self._hidden_size

    def forward(self, embeddings, conditions, steps):
        """Apply denoiser model.

        Args:
            embeddings: Input embeddings with shape (B, L, D).
            conditions: Generation conditions with shape (B, D).
            steps: Denoising steps with shape (B).

        Returns:
            Cleaned embeddings with shape (B, L, D).
        """
        b, l, d = embeddings.payload.shape
        if (embeddings.seq_lens != l).any():
            raise NotImplementedError(f"Can't use padding in denoiser.")
        step_embeddings = self._steps_embedder(steps)  # (B, D).
        inputs = embeddings.payload + step_embeddings[:, None, :]  # (B, L, D).
        states = conditions[None].expand(self._total_layers, b, self._hidden_size).contiguous()
        result, _ = self._model(inputs, states)  # (B, L, N * D).
        output_layers = 2 if self._bidirectional else 1
        result = result.reshape(b, l, output_layers, self._hidden_size)  # (B, L, N, D).
        result = result.sum(2)  # (B, L, D).
        result = self._projection(result)
        return PaddedBatch(result, embeddings.seq_lens)
