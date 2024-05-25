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

    def forward(self, x, timestamps, states=None, return_states=False):
        """Apply RNN.

        Args:
            x: Batch with shape (B, L, D).
            timestamps (unused): Inputs timestamps.
            states: Initial states with shape (N, B, D), where N is the number of layers.
            return_states: Whether to return next hiddens states or not.

        Returns:
            Output with shape (B, L, D) and optional states tensor with shape (N, B, L, D).
        """
        outputs, _ = super().forward(x, states)
        if not return_states:
            return outputs
        # Compute and return states.
        if self._num_layers == 1:
            # In GRU output and states are equal.
            states = outputs[None]  # (1, B, L, D).
        else:
            raise NotImplementedError("Multilayer GRU states")
        return outputs, states

    def interpolate(self, states, time_deltas):
        # GRU output is constant between events.
        return states[-1]  # (B, L, D).


class ContTimeLSTM(torch.nn.Module):
    """Continuous time LSTM from NHP method.

    See the original paper for details:

    Mei, Hongyuan, and Jason M. Eisner. "The neural hawkes process: A neurally self-modulating
    multivariate point process." Advances in neural information processing systems 30 (2017).

    The implementation is inspired by EasyTPP.
    """

    def __init__(self, input_size, hidden_size, num_layers=1):
        if num_layers != 1:
            raise NotImplementedError("Cont-LSTM with multiple layers")
        super().__init__()
        self._hidden_size = hidden_size
        self.i = nn.Linear(input_size + hidden_size, hidden_size)
        self.f = nn.Linear(input_size + hidden_size, hidden_size)
        self.ie = nn.Linear(input_size + hidden_size, hidden_size)
        self.fe = nn.Linear(input_size + hidden_size, hidden_size)
        self.z = nn.Linear(input_size + hidden_size, hidden_size)
        self.o = nn.Linear(input_size + hidden_size, hidden_size)
        self.d = nn.Linear(input_size + hidden_size, 1)
        self.beta = torch.nn.Parameter(torch.ones([]))

    def forward(self, x, timestamps, states=None, return_states=False):
        """Apply RNN.

        Args:
            x: Batch with shape (B, L, D).
            timestamps: Inputs timestamps with shape (B, L).
            states: Initial states with shape (1, B, 3D + 2).
                State output gate, context_start, context_end, delta and previous timestamp.
            return_states: Whether to return next hiddens states or not.

        Returns:
            Output with shape (B, L, D) and optional states tensor with shape (1, B, L, 3D + 2).
        """
        b, l, d = x.shape
        h = self._hidden_size
        if states is None:
            states = torch.zeros(b, 3 * h + 2, device=x.device, dtype=x.dtype)
        else:
            if (states.ndim != 3) or (states.shape[2] != 3 * h + 2):
                raise ValueError("The state shape must be equal to (N, B, 3D + 2).")
            if states.shape[0] != 1:
                raise ValueError("The first states dimension must be equal to the number of layers (1).")
            states = states.squeeze(0)  # (B, 3D + 2).
        # States: (B, 3D + 2).
        prev_o = states[:, :h]  # (B, D).
        prev_cs = states[:, h:2 * h]  # (B, D).
        prev_ce = states[:, 2 * h:3 * h]  # (B, D).
        prev_d = states[:, 3 * h].unsqueeze(1)  # (B, 1).
        prev_t = states[:, 3 * h + 1].unsqueeze(1)  # (B, 1).

        outputs = []
        output_states = []
        for step in range(l):
            t = timestamps[:, step].unsqueeze(1)  # (B, 1).
            c = prev_cs + (prev_ce - prev_cs) * (prev_d * (prev_t - t)).exp()  # (B, D).
            h = prev_o * (2 * torch.sigmoid(2 * c) - 1)  # (B, D).
            x_s = torch.cat([x[:, step], h], dim=1)  # (B, 2D).
            i_gate = torch.sigmoid(self.i(x_s))  # (B, D).
            f_gate = torch.sigmoid(self.f(x_s))  # (B, D).
            ie_gate = torch.sigmoid(self.ie(x_s))  # (B, D).
            fe_gate = torch.sigmoid(self.fe(x_s))  # (B, D).
            z = 2 * torch.sigmoid(self.z(x_s)) - 1  # (B, D).
            o_gate = torch.sigmoid(self.o(x_s))  # (B, D).
            cs = f * c + i * z
            ce = fe * prev_ce + ie * z
            d = torch.nn.functional.softplus(self.d(x_s), self.beta)  # (B, 1).
            outputs.append(h)  # (B, D).
            if return_states:
                output_states.append(torch.cat([o_gate, cs, ce, d, t], dim=1))  # (B, 3D + 2).
            prev_o, prev_cs, prev_ce, prev_d, prev_t = o_gate, cs, ce, d, t
        outputs = torch.stack(outputs, dim=1)  # (B, L, D).
        if not return_states:
            return outputs
        else:
            states = torch.stack(output_states, dim=1)[None]  # (1, B, L, 3D + 2).
            return outputs, states

    def interpolate(self, states, time_deltas):
        # GRU output is constant between events.
        h = self._hidden_size
        states = states[-1]  # (B, L, 3D + 2).
        prev_o = states[..., :h]  # (B, L, D).
        prev_cs = states[..., h:2 * h]  # (B, L, D).
        prev_ce = states[..., 2 * h:3 * h]  # (B, L, D).
        prev_d = states[..., 3 * h].unsqueeze(-1)  # (B, L, 1).

        c = prev_cs + (prev_ce - prev_cs) * (-prev_d * time_deltas).exp()  # (B, L, D).
        h = prev_o * (2 * torch.sigmoid(2 * c) - 1)  # (B, L, D).
        return h
