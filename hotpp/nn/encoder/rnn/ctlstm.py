import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Optional

from hotpp.data import PaddedBatch


@torch.jit.script
def cont_time_lstm(x, time_deltas, states, weight, bias):
    """Apply continuos-time LSTM using a simple PyTorch loop.

    Args:
        x: Batch with shape (B, L, I).
        time_deltas: Relative timestamps with shape (B, L).
        states: States, each with shape (B, 4D).
        weight: Layer weight matrix with shape (7D, I + D).
        bias: Layer bias vector with shape (7D).

    Returns:
        Model outputs with shape (B, L, D) and states with shape (B, L, 4D).
    """
    b, l, x_dim = x.shape
    assert states.shape[1] % 4 == 0
    cs_state, ce_state, decay, o_gate = states.chunk(4, 1)
    h_dim = o_gate.shape[1]
    x_proj = F.linear(x, weight[:, :x_dim])  # (B, L, 7D).
    h_weight = weight[:, x_dim:]  # (7D, D).

    outputs = []
    output_states = []
    for step in range(l):
        c = ce_state + (cs_state - ce_state) * (-decay * time_deltas[:, step, None]).exp()  # (B, D).
        h = o_gate * torch.tanh(c)  # (B, D).
        proj = x_proj[:, step] + F.linear(h, h_weight, bias)  # (B, 7D).
        sigmoid_proj = torch.sigmoid(proj[:, :5 * h_dim])  # (B, 5D).
        i_gate, f_gate, ie_gate, fe_gate, o_gate = sigmoid_proj.chunk(5, 1)  # (B, D).
        z = torch.tanh(proj[:, 5 * h_dim:6 * h_dim])  # (B, D).
        cs_state = f_gate * c + i_gate * z
        ce_state = fe_gate * ce_state + ie_gate * z
        decay = torch.nn.functional.softplus(proj[:, 6 * h_dim:7 * h_dim])
        outputs.append(h)  # (B, D).
        output_states.append(torch.cat([cs_state, ce_state, decay, o_gate], 1))  # (B, 4D).
    return torch.stack(outputs, 1), torch.stack(output_states, 1)  # (B, L, D), (B, L, 4D).


class ContTimeLSTM(torch.nn.Module):
    """Continuous time LSTM from NHP method.

    See the original paper for details:

    Mei, Hongyuan, and Jason M. Eisner. "The neural hawkes process: A neurally self-modulating
    multivariate point process." Advances in neural information processing systems 30 (2017).

    NOTE: Our implementation is slightly different because our first time_delta is always zero.
    This way we encode time from a previous event rather than time to a future event.
    """
    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        if num_layers != 1:
            raise NotImplementedError("Cont-LSTM with multiple layers")
        self._hidden_size = hidden_size
        self._num_layers = num_layers
        self.bos = torch.nn.Parameter(torch.randn(input_size))  # (D).
        layer = torch.nn.Linear(input_size + hidden_size, 7 * hidden_size)  # i, f, ie, fe, o, z, d.
        self.weight = layer.weight
        self.bias = layer.bias

    @property
    def init_state(self):
        bos = self.bos[None, None]  # (1, 1, D).
        zero_state = torch.zeros(1, 4 * self._hidden_size, dtype=self.bos.dtype, device=self.bos.device)  # (1, 4D).
        zero_dt = torch.zeros(1, 1, dtype=self.bos.dtype, device=self.bos.device)  # (1, 1).
        _, states = cont_time_lstm(bos, zero_dt, zero_state, self.weight, self.bias)  # (1, 1, 4D).
        return states.squeeze(1)  # (1, 4D).

    def forward(self, x: PaddedBatch, time_deltas: PaddedBatch,
                states: Optional[Tensor]=None, return_full_states=False) -> Tuple[PaddedBatch, Tensor]:
        """Apply RNN.

        Args:
            x: PaddedBatch with shape (B, L, D).
            time_deltas: Relative timestamps with shape (B, L).
            states: Initial states with shape (1, B, 4D).
                State output gate, context_start, context_end, and delta parameter.
            return_full_states: Whether to return full states with shape (B, T, D)
                or only final states with shape (B, D).

        Returns:
            Output with shape (B, L, D) and optional states tensor with shape (1, B, L, 4D).
        """
        if self._num_layers != 1:
            raise NotImplementedError("Multiple layers.")
        b, l = x.shape
        s = self._hidden_size
        seq_lens, mask = x.seq_lens, x.seq_len_mask  # (B), (B, L).
        x = x.payload * mask.unsqueeze(2)  # (B, L, D).
        time_deltas = time_deltas.payload * mask  # (B, L).
        if states is None:
            states = self.init_state.repeat(b, 1)  # (B, 4D).
        else:
            states = states.squeeze(0)  # Remove layer dim, (B, 4D).
        outputs, output_states = cont_time_lstm(x, time_deltas, states, self.weight, self.bias)  # (B, L, D), (B, L, 4D).
        outputs = PaddedBatch(outputs, seq_lens)
        if not return_full_states:
            output_states = output_states.take_along_dim((seq_lens - 1).clip(min=0)[:, None, None], 1).squeeze(1)  # (B, 4D).
        return outputs, output_states[None]  # (1, B, 4D) or (1, B, L, 4D).

    def interpolate(self, states: Tensor, time_deltas: PaddedBatch) -> PaddedBatch:
        """Compute model outputs in continuous time.

        Args:
            states: Model states with shape (N, B, L, 4D), where N is the number of layers.
            time_deltas: Relative timestamps with shape (B, L, S), where S is a sample size.

        Returns:
            Outputs with shape (B, L, S, D).
        """
        states = states[-1].unsqueeze(2)  # (B, L, 1, 4D).
        cs_state, ce_state, decay, o_gate = states.chunk(4, -1)  # (B, L, 1, D).
        c = ce_state + (cs_state - ce_state) * (-decay * time_deltas.payload.unsqueeze(3)).exp()  # (B, L, S, D).
        h = o_gate * torch.tanh(c)  # (B, L, S, D).
        return PaddedBatch(h, time_deltas.seq_lens)
