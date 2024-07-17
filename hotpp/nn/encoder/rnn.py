import torch
import torch.nn.functional as F
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
        self._num_layers = num_layers

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
            if self._num_layers == 1:
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


# JIT increases memory usage without significant speedup.
#@torch.jit.script
def cont_time_lstm(x, time_deltas, o_state, cs_state, ce_state, d_state,
                   weight, bias):
    """Apply ContTimeLSTM.

    Args:
        x: Batch with shape (L, B, D).
        time_deltas: Relative timestamps with shape (L, B, 1).
        o_state, cs_state, ce_state, d_state: States, each with shape (B, D).
        weight, bias: Layer parameters.

    Returns:
        Model outputs with shape (B, L, D) and states with shape (B, L, 4D).
    """
    outputs = []
    output_states = []
    b, s = o_state.shape
    for step in range(len(x)):
        c = cs_state + (ce_state - cs_state) * (-d_state * time_deltas[step]).exp()  # (B, D).
        h = o_state * torch.tanh(c)  # (B, D).
        x_s = torch.cat([x[step], h], dim=1)  # (B, 2D).
        proj = F.linear(x_s, weight, bias)  # (B, 7D).
        sigmoid_proj = torch.sigmoid(proj[:, :5 * s])  # (B, 5D).
        i_gate = sigmoid_proj[:, :s]  # (B, D).
        f_gate = sigmoid_proj[:, s:2 * s]  # (B, D).
        ie_gate = sigmoid_proj[:, 2 * s:3 * s]  # (B, D).
        fe_gate = sigmoid_proj[:, 3 * s:4 * s]  # (B, D).
        o_state = sigmoid_proj[:, 4 * s:5 * s]  # (B, D).
        z = torch.tanh(proj[:, 5 * s:6 * s])  # (B, D).
        cs_state = f_gate * c + i_gate * z
        ce_state = fe_gate * ce_state + ie_gate * z
        d_state = torch.nn.functional.softplus(proj[:, 6 * s:7 * s])
        # Layer output: i, f, ie, fe, z, o, d: 6D + 1.
        outputs.append(h)  # (B, D).
        output_states.append(torch.cat([o_state, cs_state, ce_state, d_state], 1))  # (B, 4D).
    return torch.stack(outputs, 1), torch.stack(output_states, 1)  # (B, L, D), (B, L, 4D).


class ContTimeLSTM(torch.nn.Module):
    """Continuous time LSTM from NHP method.

    See the original paper for details:

    Mei, Hongyuan, and Jason M. Eisner. "The neural hawkes process: A neurally self-modulating
    multivariate point process." Advances in neural information processing systems 30 (2017).
    """
    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        if num_layers != 1:
            raise NotImplementedError("Cont-LSTM with multiple layers")
        self._hidden_size = hidden_size
        self._num_layers = num_layers
        self.start = torch.nn.Parameter(torch.randn(input_size))  # (D).
        self.layer = torch.nn.Linear(input_size + hidden_size, 7 * hidden_size)  # i, f, ie, fe, o, z, d.

    @property
    def init_state(self):
        s = self._hidden_size
        proj = F.linear(self.start[None], self.layer.weight[:, :len(self.start)], self.layer.bias).squeeze(0)  # (7D).
        sigmoid_proj = torch.sigmoid(proj[:5 * s])  # (6D).
        i_gate = sigmoid_proj[:s]  # (D).
        f_gate = sigmoid_proj[s:2 * s]  # (D).
        ie_gate = sigmoid_proj[2 * s:3 * s]  # (D).
        fe_gate = sigmoid_proj[3 * s:4 * s]  # (D).
        o_state = sigmoid_proj[4 * s:5 * s]  # (D).
        z = torch.tanh(proj[5 * s:6 * s])  # (D).
        cs_state = i_gate * z
        ce_state = ie_gate * z
        d_state = torch.nn.functional.softplus(proj[6 * s:7 * s])
        return torch.cat([o_state, cs_state, ce_state, d_state])[None]  # (1, 4D).

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
        o_state = states[:, :s]
        cs_state = states[:, s:2 * s]
        ce_state = states[:, 2 * s:3 * s]
        d_state = states[:, 3 * s:]
        outputs, output_states = cont_time_lstm(x.permute(1, 0, 2), time_deltas.T.unsqueeze(2),  # (L, B, D), (L, B, 1).
                                                o_state, cs_state, ce_state, d_state,
                                                self.layer.weight, self.layer.bias)  # (B, L, D), (B, L, 4D).
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
        s = self._hidden_size
        states = states[-1].unsqueeze(2)  # (B, L, 1, 4D).
        states_o = states[..., :s]  # (B, L, 1, D).
        states_cs = states[..., s:2 * s]  # (B, L, 1, D).
        states_ce = states[..., 2 * s:3 * s]  # (B, L, 1, D).
        states_d = states[..., 3 * s:]  # (B, L, 1, D).

        c = states_cs + (states_ce - states_cs) * (-states_d * time_deltas.payload.unsqueeze(3)).exp()  # (B, L, S, D).
        h = states_o * torch.tanh(c)  # (B, L, S, D).
        return PaddedBatch(h, time_deltas.seq_lens)
