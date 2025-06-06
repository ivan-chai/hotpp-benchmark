import torch
from hotpp.data import PaddedBatch


class TransformerDenoiser(torch.nn.Module):
    def __init__(self, input_size, length, hidden_size, generation_steps, n_heads=1, num_layers=1,
                 dim_feedforward=None, dropout=0):
        super().__init__()
        if dim_feedforward is None:
            dim_feedforward = hidden_size
        self._hidden_size = hidden_size
        self._input_projection = torch.nn.Linear(input_size, hidden_size)
        self._position_embedder = torch.nn.Embedding(length, hidden_size)
        self._steps_embedder = torch.nn.Embedding(generation_steps + 1, hidden_size)
        layer = torch.nn.TransformerEncoderLayer(hidden_size, nhead=n_heads,
                                                 dim_feedforward=dim_feedforward,
                                                 dropout=dropout,
                                                 batch_first=True)
        norm = torch.nn.LayerNorm(hidden_size)
        self._transformer = torch.nn.TransformerEncoder(layer, num_layers, norm)
        self._output_projection = torch.nn.Linear(hidden_size, input_size)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

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
        if (embeddings.seq_lens != self._position_embedder.num_embeddings).any():
            raise ValueError(f"Expected input with shape (B, L, D) with L equal to {self._position_embedder.num_embeddings}")
        b, l, d = embeddings.payload.shape
        projected = self._input_projection(embeddings.payload)  # (B, L, D).
        steps_embeddings = self._steps_embedder(steps)  # (B, D).
        position_embeddings = self._position_embedder.weight  # (L, D).
        x = projected + (conditions + steps_embeddings).unsqueeze(1) + position_embeddings.unsqueeze(0)  # (B, L, D).
        x = self._transformer(x)
        assert x.ndim == 3 and x.shape[:2] == (b, l)
        x = self._output_projection(x)
        return PaddedBatch(x, embeddings.seq_lens)
