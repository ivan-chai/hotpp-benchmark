import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Optional

from hotpp.data import PaddedBatch


class ODEDNN(torch.jit.ScriptModule):
    """The Runge-Kutta neural ODE solver.

    NOTE: DNN doesn't use timestamp for prediction following the original implementation of
          "Latent ODEs for Irregularly-Sampled Time Series" by Yulia Rubanova, Ricky Chen, and David Duvenaud.
    """
    def __init__(self, size, hidden_size=None, num_layers=2,
                 lipschitz=1, n_steps=1):
        super().__init__()
        if hidden_size is None:
            hidden_size = size
        layers = []
        for i in range(num_layers):
            layer = torch.nn.Linear(hidden_size if i > 0 else size,
                                    hidden_size if i < num_layers - 1 else size)
            torch.nn.init.xavier_uniform_(layer.weight)
            torch.nn.init.constant_(layer.bias, 0)
            layers.append(layer)
            if i < num_layers - 1:
                layers.append(torch.nn.Tanh())
        if lipschitz is not None:
            layers.append(torch.nn.Tanh())
            scale = lipschitz
        else:
            scale = 1
        self.nn = torch.nn.Sequential(*layers)
        self.n_steps = n_steps
        self.s_dt = scale / n_steps
        self.s_hdt = scale / n_steps / 2
        self.s_sdt = scale / n_steps / 6

    @torch.jit.script_method
    def step(self, x: Tensor, time_scales: Tensor) -> Tensor:
        k1 = self.nn(x) * time_scales
        k2 = self.nn(x + self.s_hdt * k1) * time_scales
        k3 = self.nn(x + self.s_hdt * k2) * time_scales
        k4 = self.nn(x + self.s_dt * k3) * time_scales
        return x + self.s_sdt * (k1 + 2 * (k2 + k3) + k4)

    @torch.jit.script_method
    def forward(self, x0: Tensor, ts: Tensor) -> Tensor:
        """Solve ODE.

        Args:
            x0: Value at zero with shape (B, D).
            ts: Positions for function evaluation with shape (B).

        Returns:
            Values at positions with shape (B, D).
        """
        # Use change of variables and integrate from 0 to 1.
        x = x0
        time_scales = ts.unsqueeze(1)
        for _ in range(self.n_steps):
            x = self.step(x, time_scales)
        return x


class ODERNNCell(torch.jit.ScriptModule):
    def __init__(self, input_size, hidden_size,
                 diff_hidden_size=None, num_diff_layers=2,
                 lipschitz=1, n_steps=1):
        super().__init__()
        self.cell = torch.nn.GRUCell(input_size, hidden_size)
        self.diff_func = ODEDNN(hidden_size,
                                hidden_size=diff_hidden_size, num_layers=num_diff_layers,
                                lipschitz=lipschitz, n_steps=n_steps)

    @torch.jit.script_method
    def forward(self,
                input: Tensor,
                time_deltas: Tensor,
                state: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Make single step.

        NOTE: input must be preprocessed before call to forward.

        Args:
            input: Preprocessed input tensor with shape (B, D).
            time_deltas: Time steps with shape (B).
            state: Tuple of 4 states, each with shape (B, D).

        Returns:
            Output and a tuple of new states, each with shape (B, D).
        """
        # Apply ODE.
        h = self.diff_func(state, time_deltas)  # (B, D).
        # Apply RNN cell.
        result = self.cell(input, h)
        return (result, result)

    @torch.jit.script_method
    def interpolate(self, states: Tensor, time_deltas: Tensor, mask: Tensor) -> Tensor:
        # States: (B, L, D).
        # time_deltas: (B, L, S).
        # mask: (B, L).
        # output: (B, L, S, D).
        states = states * mask.unsqueeze(2)
        time_deltas = time_deltas * mask.unsqueeze(2)
        extended_states = states.unsqueeze(2).repeat(1, 1, time_deltas.shape[2], 1)  # (B, L, S, D).
        interpolated = self.diff_func(extended_states.flatten(0, 2), time_deltas.flatten())  # (BLS, D).
        b, l, s, d = extended_states.shape
        return interpolated.reshape(b, l, s, d)  # (B, L, S, D).


class ODEGRU(torch.jit.ScriptModule):
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
                 diff_hidden_size=None, num_diff_layers=2,
                 lipschitz=1, n_steps=1):
        super().__init__()
        if num_layers != 1:
            raise NotImplementedError("Cont-LSTM with multiple layers")
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cell = ODERNNCell(input_size, hidden_size,
                               diff_hidden_size=diff_hidden_size, num_diff_layers=num_diff_layers,
                               lipschitz=lipschitz, n_steps=n_steps)
        self.h0 = torch.nn.Parameter(torch.zeros(num_layers, hidden_size))  # (N, D).

    @property
    def delta_time(self):
        """Whether to take delta time or raw timestamps at input."""
        # Need time_deltas.
        return True

    @property
    def output_size(self):
        return self.hidden_size

    @property
    def init_state(self):
        return self.h0

    @torch.jit.script_method
    def _forward_loop(self,
                      x: Tensor,
                      time_deltas: Tensor,
                      state: Tensor,
                      ) -> Tuple[Tensor, Tensor]:
        inputs = x.unbind(1)
        dt = time_deltas.unbind(1)
        outputs = torch.jit.annotate(List[Tensor], [])
        output_states = torch.jit.annotate(List[Tensor], [])
        for i in range(len(inputs)):
            out, state = self.cell(inputs[i], dt[i], state)
            outputs += [out]
            output_states += [state]
        return torch.stack(outputs, 1), torch.stack(output_states, 1)

    def forward(self, x: PaddedBatch, time_deltas: PaddedBatch,
                states: Optional[Tensor]=None, return_states=False) -> Tuple[PaddedBatch, Tensor]:
        """Apply RNN.

        Args:
            x: PaddedBatch with shape (B, L, D).
            time_deltas: Relative timestamps with shape (B, L).
            states: Initial states with shape (1, B, D).
                State output gate, context_start, context_end, and delta parameter.
            return_states: Whether to return final states with shape (B, D), full states with shape (B, T, D)
                or no states (either False, "last" or "full").

        Returns:
            Output with shape (B, L, D) and optional states tensor with shape (1, B, L, D).
        """
        b, l = x.shape
        dim = self.hidden_size
        seq_lens, mask = x.seq_lens, x.seq_len_mask  # (B), (B, L).
        x = x.payload * mask.unsqueeze(2)  # (B, L, D).
        time_deltas = time_deltas.payload * mask  # (B, L).
        if states is None:
            states = self.init_state.repeat(b, 1)  # (B, D).
        else:
            states = states.squeeze(0)  # Remove layer dim, (B, D).
        outputs, output_states = self._forward_loop(x, time_deltas, states)  # (B, L, D), (B, L, D).
        outputs = PaddedBatch(outputs, seq_lens)
        if not return_states:
            output_states = None
        elif return_states == "last":
            output_states = output_states.take_along_dim((seq_lens - 1).clip(min=0)[:, None, None], 1).squeeze(1)[None]  # (1, B, 4D).
        elif return_states == "full":
            output_states = output_states[None]  # (1, B, L, 4D)
        else:
            raise ValueError(f"Unknown states flag: {return_states}")
        return outputs, output_states  # None or (1, B, D) or (1, B, L, D).

    def interpolate(self, states: Tensor, time_deltas: PaddedBatch) -> PaddedBatch:
        """Compute model outputs in continuous time.

        Args:
            states: Model states with shape (N, B, L, D), where N is the number of layers.
            time_deltas: Relative timestamps with shape (B, L, S), where S is a sample size.

        Returns:
            Outputs with shape (B, L, S, D).
        """
        if time_deltas.payload.ndim != 3:
            raise ValueError("Expected time_deltas with shape (B, L, S).")
        if len(states) != self.num_layers:
            raise ValueError("Incompatible states shape.")
        assert len(states) == 1
        h = self.cell.interpolate(states.squeeze(0), time_deltas.payload, time_deltas.seq_len_mask)  # (B, L, S, D).
        return PaddedBatch(h, time_deltas.seq_lens)
