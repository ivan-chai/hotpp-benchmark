import torch
import torch.nn.functional as F
from torch import Tensor
from torchdiffeq import odeint
from typing import Tuple, Optional

from hotpp.data import PaddedBatch


class Scale(torch.nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return x * self.scale


def odernn(x, time_deltas, states, cell, diff_func, method="rk4", grid_size=2, rtol=1e-3, atol=1e-4):
    """Apply ODE RNN.

    Args:
        x: Batch with shape (B, L, I).
        time_deltas: Relative timestamps with shape (B, L).
        states: States, each with shape (B, D).
        cell: RNN Cell which maps (input, state) to (output, state).
        diff_func: Model for estimating derivatives: state (B, D) -> derivatives (B, D).
            We don't use timestamp in this model, following the original implementation.
        method: ODE solver (`euler` or `rk4`).
        grid_size: The grid size for ODE solving.

    Returns:
        Model outputs with shape (B, L, D) and states with shape (B, L, D).
    """
    if grid_size < 2:
        raise ValueError("Grid size must be greater or equal to 2 to include starting and end points.")
    b, l, _ = x.shape

    outputs = []
    output_states = []
    times = torch.linspace(0, 1, grid_size, device=x.device, dtype=x.dtype)
    for step in range(l):
        # Extrapolate state with ODE.
        #
        # NOTE.
        # The odeint function requires the same set of timestamps for all elements in the batch.
        # We use change of variables to align time points.
        # x'_t(t) = f(t, x(t)), t in [0, T].
        # t = Ts, s in [0, 1].
        # x'_s(Ts) = T x'_t(Ts) = T f(Ts, x(Ts)).
        step_diff_func = lambda t, h: diff_func(h) * time_deltas[:, step:step + 1]  # (B, D) -> (B, D).
        h = odeint(step_diff_func, states, times, method=method, rtol=rtol, atol=atol)  # (T, B, D).
        h = h[-1]  # (B, D).
        # Apply RNN cell.
        result = cell(x[:, step], h)
        if not isinstance(result, tuple):
            result = (result, result)  # GRU.
        output, states = result
        outputs.append(output)  # (B, D).
        output_states.append(states)  # (B, D).
    return torch.stack(outputs, 1), torch.stack(output_states, 1)  # (B, L, D), (B, L, D).


class ODEGRU(torch.nn.Module):
    """ODE-based continuous-time GRU recurrent network.

    See the original paper for details:

    Rubanova, Yulia, Ricky TQ Chen, and David K. Duvenaud. "Latent ordinary differential equations for irregularly-sampled time series." Advances in neural information processing systems 32 (2019).

    Args:
        num_diff_layers: The number of layers in the derivative computation model.
        diff_hidden_size: The size of the hidden layer in the derivative computation model.
            Use RNN hidden_size by default.
        lipschitz: The maximum derivative magnitude. Use `None` to disable the constraint.
        method: ODE solver (`euler` or `rk4`).
        grid_size: The grid size for ODE solving.
    """
    def __init__(self, input_size, hidden_size, num_layers=1,
                 num_diff_layers=2, diff_hidden_size=None, lipschitz=1,
                 method="rk4", grid_size=2, rtol=1e-3, atol=1e-4):
        super().__init__()
        if num_layers != 1:
            raise NotImplementedError("Cont-LSTM with multiple layers")
        diff_hidden_size = diff_hidden_size or hidden_size
        self._hidden_size = hidden_size
        self._num_layers = num_layers
        self._method = method
        self._grid_size = grid_size
        self._atol = atol
        self._rtol = rtol
        self.cell = torch.nn.GRUCell(input_size, hidden_size)
        diff_layers = []
        for i in range(num_diff_layers):
            layer = torch.nn.Linear(diff_hidden_size if i > 0 else hidden_size,
                                    diff_hidden_size if i < num_diff_layers - 1 else hidden_size)
            torch.nn.init.xavier_uniform_(layer.weight)
            torch.nn.init.constant_(layer.bias, 0)
            diff_layers.append(layer)
            diff_layers.append(torch.nn.Tanh())
        if lipschitz is None:
            # Remove last Tanh.
            diff_layers = diff_layers[:-1]
        else:
            diff_layers.append(Scale(lipschitz))
        self.diff_func = torch.nn.Sequential(*diff_layers)
        self.h0 = torch.nn.Parameter(torch.zeros(num_layers, hidden_size))  # (N, D).

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def num_layers(self):
        return self._num_layers

    @property
    def init_state(self):
        return self.h0

    def forward(self, x: PaddedBatch, time_deltas: PaddedBatch,
                states: Optional[Tensor]=None, return_full_states=False) -> Tuple[PaddedBatch, Tensor]:
        """Apply RNN.

        Args:
            x: PaddedBatch with shape (B, L, D).
            time_deltas: Relative timestamps with shape (B, L).
            states: Initial states with shape (1, B, D).
                State output gate, context_start, context_end, and delta parameter.
            return_full_states: Whether to return full states with shape (B, T, D)
                or only final states with shape (B, D).

        Returns:
            Output with shape (B, L, D) and optional states tensor with shape (1, B, L, D).
        """
        if self._num_layers != 1:
            raise NotImplementedError("Multiple layers.")
        b, l = x.shape
        s = self._hidden_size
        seq_lens, mask = x.seq_lens, x.seq_len_mask  # (B), (B, L).
        x = x.payload * mask.unsqueeze(2)  # (B, L, D).
        time_deltas = time_deltas.payload * mask  # (B, L).
        if states is None:
            states = self.init_state.repeat(b, 1)  # (B, D).
        else:
            states = states.squeeze(0)  # Remove layer dim, (B, D).
        outputs, output_states = odernn(x, time_deltas, states, self.cell, self.diff_func,
                                        method=self._method, grid_size=self._grid_size,
                                        rtol=self._rtol, atol=self._atol)  # (B, L, D), (B, L, D).
        outputs = PaddedBatch(outputs, seq_lens)
        if not return_full_states:
            output_states = output_states.take_along_dim((seq_lens - 1).clip(min=0)[:, None, None], 1).squeeze(1)  # (B, D).
        return outputs, output_states[None]  # (1, B, D) or (1, B, L, D).

    def interpolate(self, states: Tensor, time_deltas: PaddedBatch) -> PaddedBatch:
        """Compute model outputs in continuous time.

        Args:
            states: Model states with shape (N, B, L, D), where N is the number of layers.
            time_deltas: Relative timestamps with shape (B, L, S), where S is a sample size.

        Returns:
            Outputs with shape (B, L, S, D).
        """
        if self._num_layers != 1:
            raise NotImplementedError("Multiple layers.")
        if time_deltas.payload.ndim != 3:
            raise ValueError("Expected time_deltas with shape (B, L, S).")
        seq_lens, mask = time_deltas.seq_lens, time_deltas.seq_len_mask  # (B), (B, L).

        assert len(states) == 1
        states = states.squeeze(0)  # (B, L, D).
        states = states * mask.unsqueeze(2)
        states = states.unsqueeze(2).repeat(1, 1, time_deltas.payload.shape[-1], 1)  # (B, L, S, D).
        b, l, s, d = states.shape
        time_deltas = time_deltas.payload * mask.unsqueeze(2)  # (B, L, S).

        states = states.flatten(0, 2)  # (BLS, D).
        time_deltas = time_deltas.flatten()  # (BLS).
        times = torch.linspace(0, 1, self._grid_size, device=states.device, dtype=states.dtype)

        step_diff_func = lambda t, h: self.diff_func(h) * time_deltas[:, None]  # (B, D) -> (B, D).
        h = odeint(step_diff_func, states, times, method=self._method, rtol=self._rtol, atol=self._atol)  # (T, BLS, D).
        h = h[-1]  # (BLS, D).
        return PaddedBatch(h.reshape(b, l, s, d), seq_lens)
