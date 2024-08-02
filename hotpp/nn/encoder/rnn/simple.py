import torch
from torch import Tensor
from typing import Tuple, Optional

from hotpp.data import PaddedBatch


class GRU(torch.nn.GRU):
    """GRU interface."""
    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self._hidden_size = hidden_size

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def init_state(self):
        p = next(iter(self.parameters()))
        return torch.zeros(1, self.hidden_size, dtype=p.dtype, device=p.device)  # (1, D).

    def forward(self, x: PaddedBatch, time_deltas: PaddedBatch,
                states: Optional[Tensor]=None, return_full_states=False) -> Tuple[PaddedBatch, Tensor]:
        """Apply RNN.

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
        outputs, _ = super().forward(x.payload, states)  # (B, L, D).
        if return_full_states:
            if self.num_layers == 1:
                # In GRU output and states are equal.
                states = outputs[None]  # (1, B, L, D).
            else:
                raise NotImplementedError("Multilayer GRU states")
        else:
            states = outputs.take_along_dim((x.seq_lens - 1).clip(min=0)[:, None, None], 1).squeeze(1)[None]  # (1, B, D).
        outputs = PaddedBatch(outputs, x.seq_lens)
        return outputs, states

    def interpolate(self, states: Tensor, time_deltas: PaddedBatch) -> PaddedBatch:
        """Compute model outputs in continuous time.

        Args:
            states: Model states with shape (N, B, L, 3D + 1), where N is the number of layers.
            time_deltas: Relative timestamps with shape (B, L, S), where S is a sample size.

        Returns:
            Outputs with shape (B, L, S, D).
        """
        # GRU output is constant between events.
        s = time_deltas.payload.shape[2]
        outputs = states[-1].unsqueeze(2).repeat(1, 1, s, 1)  # (B, L, S, D).
        return PaddedBatch(outputs, time_deltas.seq_lens)
