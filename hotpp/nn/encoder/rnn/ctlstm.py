import math
import torch
import torch.nn.functional as F
from torch import Tensor
from typing import List, Tuple, Optional

from hotpp.data import PaddedBatch


class ContTimeLSTMCell(torch.jit.ScriptModule):
    # 7 states: i, f, ie, fe, o, z, d.
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.projection = torch.nn.Linear(input_size, 7 * hidden_size, bias=False)
        self.weight = torch.nn.Parameter(torch.empty(hidden_size, 7 * hidden_size))  # (D, 7D).
        self.bias = torch.nn.Parameter(torch.empty(7 * hidden_size))  # (7D).
        self.reset_parameters()

    def reset_parameters(self):
        self.projection.reset_parameters()
        stdv = 1.0 / math.sqrt(self.hidden_size) if self.hidden_size > 0 else 0
        torch.nn.init.uniform_(self.weight, -stdv, stdv)
        torch.nn.init.uniform_(self.bias, -stdv, stdv)

    def preprocess(self, input):
        """Prepare input for recurrent processing.

        Args:
            input: Input data with shape (*, I).

        Returns:
            Projected input with shape (*, D).
        """
        return self.projection(input)

    @torch.jit.script_method
    def forward(
            self,
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
        dim = self.hidden_size
        cs_state, ce_state, decay, o_gate = state.chunk(4, 1)  # (B, D) each.
        c = ce_state + (cs_state - ce_state) * (-decay * time_deltas.unsqueeze(1)).exp()  # (B, D).
        h = o_gate * torch.tanh(c)  # (B, D).
        proj = input + torch.mm(h, self.weight) + self.bias  # (B, 7D).
        sigmoid_proj = torch.sigmoid(proj[:, :5 * dim])  # (B, 5D).
        i_gate, f_gate, ie_gate, fe_gate, o_gate = sigmoid_proj.chunk(5, 1)  # (B, D).
        z = torch.tanh(proj[:, 5 * dim:6 * dim])  # (B, D).
        decay = torch.nn.functional.softplus(proj[:, 6 * dim:7 * dim])
        cs_state = f_gate * c + i_gate * z
        ce_state = fe_gate * ce_state + ie_gate * z
        return h, torch.cat((cs_state, ce_state, decay, o_gate), 1)


class ContTimeLSTM(torch.jit.ScriptModule):
    """Continuous time LSTM from NHP method.

    See the original paper for details:

    Mei, Hongyuan, and Jason M. Eisner. "The neural hawkes process: A neurally self-modulating
    multivariate point process." Advances in neural information processing systems 30 (2017).

    NOTE: Our implementation is slightly different because our first time delta is always zero.
    This way we encode time from a previous event rather than time to a future event.
    """
    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        if num_layers != 1:
            raise NotImplementedError("Cont-LSTM with multiple layers")
        self.num_layers = num_layers
        self.cell = ContTimeLSTMCell(input_size, hidden_size)
        self.bos = torch.nn.Parameter(torch.randn(input_size))  # (D).

    @property
    def delta_time(self):
        """Whether to take delta time or raw timestamps at input."""
        # Need time_deltas.
        return True

    @property
    def output_size(self):
        return self.cell.hidden_size

    @property
    def init_state(self):
        dtype, device = self.bos.dtype, self.bos.device
        bos = self.cell.preprocess(self.bos[None, None]).squeeze(0)  # (1, D).
        dt = torch.zeros(1, dtype=dtype, device=device)  # (1).
        zeros = torch.zeros(1, self.cell.hidden_size, dtype=dtype, device=device)
        state = torch.cat((zeros, zeros, zeros, zeros), 1)
        _, bos_state = self.cell(bos, dt, state)
        return bos_state  # (1, 4D).

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
            states: Initial states with shape (1, B, 4D).
                State output gate, context_start, context_end, and delta parameter.
            return_states: Whether to return final states with shape (B, D), full states with shape (B, T, D)
                or no states (either False, "last" or "full").

        Returns:
            Output with shape (B, L, D) and optional states tensor with shape (1, B, L, 4D).
        """
        b, l = x.shape
        dim = self.cell.hidden_size
        seq_lens, mask = x.seq_lens, x.seq_len_mask  # (B), (B, L).
        x = x.payload * mask.unsqueeze(2)  # (B, L, D).
        time_deltas = time_deltas.payload * mask  # (B, L).
        if states is None:
            states = self.init_state.repeat(b, 1)  # (B, 4D).
        else:
            states = states.squeeze(0)  # Remove layer dim, (B, 4D).
        x = self.cell.preprocess(x)  # (B, L, D).
        outputs, output_states = self._forward_loop(x, time_deltas, states)  # (B, L, D), (B, L, 4D).
        outputs = PaddedBatch(outputs, seq_lens)
        if not return_states:
            output_states = None
        elif return_states == "last":
            output_states = output_states.take_along_dim((seq_lens - 1).clip(min=0)[:, None, None], 1).squeeze(1)[None]  # (1, B, 4D).
        elif return_states == "full":
            output_states = output_states[None]  # (1, B, L, 4D)
        else:
            raise ValueError(f"Unknown states flag: {return_states}")
        return outputs, output_states  # None or (1, B, 4D) or (1, B, L, 4D).

    @torch.jit.script_method
    def _interpolate_impl(self, states: Tensor, time_deltas: Tensor) -> Tensor:
        # STATES: (B, L, D).
        # time_deltas: (B, L, S).
        # output: (B, L, S, D).
        cs_state, ce_state, decay, o_gate = states.unsqueeze(2).chunk(4, -1)  # (B, L, 1, D).
        c = ce_state + (cs_state - ce_state) * (-decay * time_deltas.unsqueeze(3)).exp()  # (B, L, S, D).
        h = o_gate * torch.tanh(c)  # (B, L, S, D).
        return h

    def interpolate(self, states: Tensor, time_deltas: PaddedBatch) -> PaddedBatch:
        """Compute model outputs in continuous time.

        Args:
            states: Model states with shape (N, B, L, 4D), where N is the number of layers.
            time_deltas: Relative timestamps with shape (B, L, S), where S is a sample size.

        Returns:
            Outputs with shape (B, L, S, D).
        """
        h = self._interpolate_impl(states[-1], time_deltas.payload)  # (B, L, S, D).
        return PaddedBatch(h, time_deltas.seq_lens)
