import torch


class GRU(torch.nn.GRU):
    """GRU interface."""
    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self._num_layers = num_layers

    def forward(self, x, time_deltas, states=None, return_full_states=False):
        """Apply RNN.

        Args:
            x: Batch with shape (B, L, D).
            time_deltas (unused): Relative inputs timestamps.
            states: Initial states with shape (N, B, D), where N is the number of layers.
            return_full_states: Whether to return states for each iteration or only output states.

        Returns:
            Outputs with shape (B, T, D) and states with shape (N, B, D) or (N, B, T, D), where
            N is the number of layers.
        """
        outputs, states = super().forward(x, states)
        if return_full_states:
            if self._num_layers == 1:
                # In GRU output and states are equal.
                states = outputs[None]  # (1, B, L, D).
            else:
                raise NotImplementedError("Multilayer GRU states")
        return outputs, states

    def interpolate(self, states, time_deltas):
        # GRU output is constant between events.
        return states[-1]  # (B, L, D).
