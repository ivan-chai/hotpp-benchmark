import torch
from torch import Tensor
from typing import Tuple, Optional

from hotpp.data import PaddedBatch

class PublicTransformer(torch.nn.Module):
    """Transformers interface."""
    def __init__(
            self,
            emb_dim,
            model):
        super().__init__()
        self.model = model
        self.model.config
        self._hidden_size = self.model.config.n_embd
        self.delta_time = False
        self.input_proj = torch.nn.Linear(emb_dim, self.model.config.n_embd) #Layer to get right size

    @property
    def output_size(self) -> int:
        return self._hidden_size

    @property
    def init_state(self) -> Tensor:
        p = next(iter(self.parameters()))
        return torch.zeros(1, self._hidden_size, dtype=p.dtype, device=p.device)  # (1, D).

    def forward(self, x: PaddedBatch, time_deltas: PaddedBatch,
                states: Optional[Tensor]=None, return_states=False) -> Tuple[PaddedBatch, Tensor]:
        """Apply Transformer.

        Args:
            x: Batch with shape (B, L, D).
            time_deltas (unused): Relative inputs timestamps.
            states: Initial states with shape (N, B, D), where N is the number of layers.
            return_full_states: Whether to return full states with shape (B, L, D)
                or only output states with shape (B, D).

        Returns:
            Outputs with shape (B, L, D) and states with shape (N, B, D) or (N, B, L, D), where
            N is the number of layers.
        """
        x_emb = self.input_proj(x.payload)
        outputs = self.model(inputs_embeds = x_emb)  # (B, L, D).
        
        all_hidden_states = torch.stack(outputs.hidden_states)
        
        if return_states == "last":
            output_states = all_hidden_states[:, :, -1, :]  # (N, B, D)
        elif return_states == "full":
            output_states = all_hidden_states  # (N, B, L, D)
        elif not return_states:
            output_states = None
        else:
            raise ValueError(f"Unknown return_states flag: {return_states}")
        
        return PaddedBatch(outputs.last_hidden_state, x.seq_lens), output_states
    
    def interpolate(self, states: Tensor, time_deltas: PaddedBatch) -> PaddedBatch:
        """Compute model outputs in continuous time.

        Args:
            states: Model states with shape (N, B, L, 3D + 1), where N is the number of layers.
            time_deltas: Relative timestamps with shape (B, L, S), where S is a sample size.

        Returns:
            Outputs with shape (B, L, S, D).
        """
        # Transformer output is constant between events.
        s = time_deltas.payload.shape[2]
        outputs = states[-1].unsqueeze(2).repeat(1, 1, s, 1)  # (B, L, S, D).
        return PaddedBatch(outputs, time_deltas.seq_lens)
