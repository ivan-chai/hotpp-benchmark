import torch
import torch.nn.functional as F


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


@torch.jit.script
def cont_time_lstm(x, time_deltas, o_state, cs_state, ce_state, d_state,
                   weight, bias, d_weight, d_bias, d_beta):
    """Apply ContTimeLSTM.

    Args:
        x: Batch with shape (L, B, D).
        time_deltas: Relative timestamps with shape (L, B, 1).
        o_state, cs_state, ce_state, d_state: States with shapes (B, D), (B, D), (B, D), and (B, 1).
        weight, bias, d_weight, d_bias, d_beta: Layer parameters.

    Returns:
        Model outputs with shape (B, L, D) and states with shape (B, L, 3D + 1).
    """
    outputs = []
    output_states = []
    b, s = o_state.shape
    for step in range(len(x)):
        c = cs_state + (ce_state - cs_state) * (-d_state * time_deltas[step]).exp()  # (B, D).
        h = o_state * (2 * torch.sigmoid(2 * c) - 1)  # (B, D).
        x_s = torch.cat([x[step], h], dim=1)  # (B, 2D).
        proj = torch.sigmoid(F.linear(x_s, weight, bias))  # (B, 6D + 1).
        i_gate = proj[:, :s]  # (B, D).
        f_gate = proj[:, s:2 * s]  # (B, D).
        ie_gate = proj[:, 2 * s:3 * s]  # (B, D).
        fe_gate = proj[:, 3 * s:4 * s]  # (B, D).
        z = 2 * proj[:, 4 * s:5 * s] - 1  # (B, D).
        o_state = proj[:, 5 * s:6 * s]  # (B, D).
        cs_state = f_gate * c + i_gate * z
        ce_state = fe_gate * ce_state + ie_gate * z
        d_state = torch.nn.functional.softplus(d_beta * F.linear(x_s, d_weight, d_bias)) / d_beta
        # Layer output: i, f, ie, fe, z, o, d: 6D + 1.
        outputs.append(h)  # (B, D).
        output_states.append(torch.cat([o_state, cs_state, ce_state, d_state], 1))  # (B, 3D + 1).
    return torch.stack(outputs, 1), torch.stack(output_states, 1)  # (B, L, 3D + 1).


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
        self.layer = torch.nn.Linear(input_size + hidden_size, 6 * hidden_size + 1)
        self.d = torch.nn.Linear(input_size + hidden_size, 1)
        self.beta = torch.nn.Parameter(torch.ones([]))

    def forward(self, x, time_deltas, states=None, return_full_states=False):
        """Apply RNN.

        Args:
            x: Batch with shape (B, L, D).
            time_deltas: Relative timestamps with shape (B, L).
            states: Initial states with shape (1, B, 3D + 1).
                State output gate, context_start, context_end, and delta parameter.
            return_full_states: Whether to return states for each iteration or only output states.

        Returns:
            Output with shape (B, L, D) and optional states tensor with shape (1, B, L, 3D + 1).
        """
        if self._num_layers != 1:
            raise NotImplementedError("Multiple layers.")
        b, l, _ = x.shape
        s = self._hidden_size
        if states is not None:
            states = states.squeeze(0)  # Remove layer dim.
            o_state = states[:, :s]
            cs_state = states[:, s:2 * s]
            ce_state = states[:, 2 * s:3 * s]
            d_state = states[:, 3 * s:]
        else:
            o_state = torch.zeros(b, s, dtype=x.dtype, device=x.device)
            cs_state = torch.zeros(b, s, dtype=x.dtype, device=x.device)
            ce_state = torch.zeros(b, s, dtype=x.dtype, device=x.device)
            d_state = torch.zeros(b, 1, dtype=x.dtype, device=x.device)
        outputs, output_states = cont_time_lstm(x.permute(1, 0, 2), time_deltas.T.unsqueeze(2),  # (L, B, D), (L, B, 1).
                                                o_state, cs_state, ce_state, d_state,
                                                self.layer.weight, self.layer.bias,
                                                self.d.weight, self.d.bias, self.beta)  # (B, L, D), (B, L, 3D + 1).
        if not return_full_states:
            output_states = output_states[:, -1]
        return outputs, output_states[None]

    def interpolate(self, states, time_deltas):
        # GRU output is constant between events.
        h = self._hidden_size
        states = states[-1]  # (B, L, 3D + 2).
        states_o = states[..., :h]  # (B, L, D).
        states_cs = states[..., h:2 * h]  # (B, L, D).
        states_ce = states[..., 2 * h:3 * h]  # (B, L, D).
        states_d = states[..., 3 * h].unsqueeze(-1)  # (B, L, 1).

        c = states_cs + (states_ce - states_cs) * (-states_d * time_deltas).exp()  # (B, L, D).
        h = states_o * (2 * torch.sigmoid(2 * c) - 1)  # (B, L, D).
        return h
