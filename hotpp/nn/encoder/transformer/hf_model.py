import torch
from torch import Tensor
from typing import Tuple, Optional

from hotpp.data import PaddedBatch


class HuggingFaceTransformer(torch.nn.Module):
    """Transformers interface."""
    def __init__(
            self,
            input_size,
            model):
        super().__init__()
        self.model = model
        self._hidden_size = self.model.config.n_embd
        self.input_projection = torch.nn.Linear(input_size, self.model.config.n_embd) #Layer to get right size

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def delta_time(self):
        return False

    def forward(self, x: PaddedBatch, timestamps: PaddedBatch,
                states: Optional[Tensor]=None, return_states=False) -> Tuple[PaddedBatch, Optional[Tensor]]:
        """Apply Transformer.

        Args:
            x: Batch with shape (B, L, D).
            timestamps (unused): Relative inputs timestamps.
            states: Initial states with shape (N, B, D), where N is the number of layers.
            return_states: Whether to return final states with shape (B, D), full states with shape (B, T, D)
                or no states (either False, "last" or "full"). Must be False. Exists only for interface compatibility.

        Returns:
            Outputs with shape (B, L, D) and None (states are not supported).
        """
        if return_states:
            raise ValueError("Transformers encoder doesn't support states return")

        embeddings = self.input_projection(x.payload)
        outputs = self.model(inputs_embeds = embeddings, attention_mask = x.seq_len_mask)  # (B, L, D).

        return PaddedBatch(outputs.last_hidden_state, x.seq_lens), None

    def interpolate(self, states: Tensor, time_deltas: PaddedBatch) -> PaddedBatch:
        raise NotImplementedError("Transformers do not support continuous-time interpolation in this setup")
