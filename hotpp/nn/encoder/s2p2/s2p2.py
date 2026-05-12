from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from hotpp.data import PaddedBatch
from .models import LLH, Int_Forward_LLH, Int_Backward_LLH


class ScaledSoftplus(nn.Module):
    def __init__(self, num_features: int, threshold: float = 20.0):
        super().__init__()
        self.threshold = threshold
        self.log_beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: Tensor) -> Tensor:
        import math
        beta = self.log_beta.exp()
        beta_x = beta * x
        return torch.where(
            beta_x <= self.threshold,
            torch.log1p(beta_x.clamp(max=math.log(1e5)).exp()) / beta,
            x,
        )


class IntensityNet(nn.Module):
    def __init__(self, input_dim: int, num_event_types: int, bias: bool = True):
        super().__init__()
        self.intensity_net = nn.Linear(input_dim, num_event_types, bias=bias)
        self.softplus = ScaledSoftplus(num_event_types)

    def forward(self, x: Tensor) -> Tensor:
        if x.is_complex():
            x = x.real
        return self.softplus(self.intensity_net(x))


class S2P2(nn.Module):
    def __init__(
        self,
        hidden_size: int = 64,
        state_dim: int = 64,
        num_layers: int = 4,
        num_event_types: int = 1,
        dt_init_min: float = 1e-4,
        dt_init_max: float = 0.1,
        act_func: str = "full_glu",
        dropout_rate: float = 0.0,
        for_loop: bool = False,
        pre_norm: bool = True,
        post_norm: bool = False,
        simple_mark: bool = True,
        relative_time: bool = False,
        complex_values: bool = True,
        int_forward_variant: bool = False,
        int_backward_variant: bool = False,
        bias: bool = True,
    ):
        super().__init__()

        self.H = hidden_size
        self.P = state_dim
        self.n_layers = num_layers
        self.num_event_types = num_event_types
        self.complex_values = complex_values
        self.backward_variant = int_backward_variant

        assert int_forward_variant + int_backward_variant <= 1
        if int_forward_variant:
            llh_layer = Int_Forward_LLH
        elif int_backward_variant:
            llh_layer = Int_Backward_LLH
        else:
            llh_layer = LLH

        layer_kwargs = dict(
            P=state_dim, H=hidden_size,
            dt_init_min=dt_init_min, dt_init_max=dt_init_max,
            act_func=act_func, dropout_rate=dropout_rate,
            for_loop=for_loop, pre_norm=pre_norm, post_norm=post_norm,
            simple_mark=simple_mark, relative_time=relative_time,
            complex_values=complex_values,
        )

        self.layers = nn.ModuleList([
            llh_layer(**layer_kwargs, is_first_layer=(i == 0))
            for i in range(num_layers)
        ])

    def _get_hidden(self, x_LP: Union[Tensor, List[Tensor]], right_us_BNH) -> Tensor:
        left_u_H = None
        for i, layer in enumerate(self.layers):
            if isinstance(x_LP, list):
                left_u_H = layer.depth_pass(x_LP[i], current_left_u_H=left_u_H, prev_right_u_H=right_us_BNH[i])
            else:
                left_u_H = layer.depth_pass(x_LP[..., i, :], current_left_u_H=left_u_H, prev_right_u_H=right_us_BNH[i])
        return left_u_H

    def _evolve_and_get_hidden_at_sampled_dts(self, x_LP, dt_G, right_us_H):
        left_u_GH = None
        for i, layer in enumerate(self.layers):
            x_GP = layer.get_left_limit(
                right_limit_P=x_LP[..., i, :],
                dt_G=dt_G,
                next_left_u_GH=left_u_GH,
                current_right_u_H=right_us_H[i],
            )
            left_u_GH = layer.depth_pass(
                current_left_x_P=x_GP,
                current_left_u_H=left_u_GH,
                prev_right_u_H=right_us_H[i],
            )
        return left_u_GH

    def forward_core(self, dt_BN, alpha_BNH):
        return self.forward_core_with_alpha(dt_BN, alpha_BNH)

    def forward_core_with_alpha(self, dt_BN, alpha_BNH, ssm_initial_states=None):
        right_xs_BNP = []
        left_xs_BNm1P = []
        right_us_BNH = [None]
        left_u_BNH, right_u_BNH = None, None

        for l_i, layer in enumerate(self.layers):
            init_state = ssm_initial_states[l_i] if ssm_initial_states is not None else None
            x_BNP, next_layer_left_u_BNH, next_layer_right_u_BNH = layer.forward(
                left_u_BNH, right_u_BNH, alpha_BNH, dt_BN, init_state
            )

            right_xs_BNP.append(x_BNP)
            if next_layer_left_u_BNH is None:
                left_xs_BNm1P.append(
                    layer.get_left_limit(
                        x_BNP[..., :-1, :],
                        dt_BN[..., 1:].unsqueeze(-1),
                        current_right_u_H=right_u_BNH if right_u_BNH is None else right_u_BNH[..., :-1, :],
                        next_left_u_GH=left_u_BNH if left_u_BNH is None else left_u_BNH[..., 1:, :].unsqueeze(-2),
                    ).squeeze(-2)
                )
            right_us_BNH.append(next_layer_right_u_BNH)
            left_u_BNH, right_u_BNH = next_layer_left_u_BNH, next_layer_right_u_BNH

        right_xs_BNLP = torch.stack(right_xs_BNP, dim=-2)

        ret = {
            "right_xs_BNLP": right_xs_BNLP,
            "right_us_BNH": right_us_BNH,
        }
        if left_u_BNH is not None:
            ret["left_u_BNm1H"] = left_u_BNH[..., 1:, :]
        else:
            ret["left_xs_BNm1LP"] = torch.stack(left_xs_BNm1P, dim=-2)
        return ret


class S2P2Encoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        state_dim: int = 64,
        num_layers: int = 4,
        num_event_types: int = 1,
        dt_init_min: float = 1e-4,
        dt_init_max: float = 0.1,
        act_func: str = "full_glu",
        dropout_rate: float = 0.0,
        for_loop: bool = False,
        pre_norm: bool = True,
        post_norm: bool = False,
        simple_mark: bool = True,
        relative_time: bool = False,
        complex_values: bool = True,
        int_forward_variant: bool = False,
        int_backward_variant: bool = False,
    ):
        super().__init__()

        self.model = S2P2(
            hidden_size=hidden_size, state_dim=state_dim, num_layers=num_layers,
            num_event_types=num_event_types, dt_init_min=dt_init_min, dt_init_max=dt_init_max,
            act_func=act_func, dropout_rate=dropout_rate, for_loop=for_loop,
            pre_norm=pre_norm, post_norm=post_norm, simple_mark=simple_mark,
            relative_time=relative_time, complex_values=complex_values,
            int_forward_variant=int_forward_variant, int_backward_variant=int_backward_variant,
        )

        self._hidden_size = hidden_size
        self._num_layers = num_layers
        self._state_dim = state_dim
        self._complex_values = complex_values
        self._int_backward_variant = int_backward_variant
        self.bos = nn.Parameter(torch.randn(input_size))

        self.embed_proj = nn.Linear(input_size, hidden_size)

        if input_size == hidden_size:
            nn.init.eye_(self.embed_proj.weight)
            nn.init.zeros_(self.embed_proj.bias)

        P_stored = 2 * state_dim if complex_values else state_dim
        self._state_size = num_layers * (P_stored + hidden_size + 1) + hidden_size

    @property
    def delta_time(self) -> bool:
        return True

    @property
    def output_size(self) -> int:
        return self._hidden_size

    @property
    def num_layers(self) -> int:
        return 1

    @property
    def init_state(self) -> Tensor:
        P_s = 2 * self._state_dim if self._complex_values else self._state_dim
        sz = P_s + self._hidden_size + 1
        s = torch.zeros(self._state_size, device=self.bos.device, dtype=self.bos.dtype)
        for i, layer in enumerate(self.model.layers):
            o = i * sz
            x = layer.initial_state_P
            s[o:o + self._state_dim] = x.real if self._complex_values else x
            if self._complex_values:
                s[o + self._state_dim:o + P_s] = x.imag
            s[o + P_s + self._hidden_size] = 1.0
        return s.unsqueeze(0)

    def forward(self, x: PaddedBatch, time_deltas: PaddedBatch, states=None, return_states=False):
        dt_BN = time_deltas.payload
        seq_lens = x.seq_lens
        B, N = dt_BN.shape
        alpha_BNH = self.embed_proj(x.payload)

        ssm_init = None
        if states is not None:
            s = states.squeeze(0)
            P_s = 2 * self._state_dim if self._complex_values else self._state_dim
            sz = P_s + self._hidden_size + 1
            chunks = s[:, :self._num_layers * sz].reshape(-1, self._num_layers, sz)
            xs = chunks[..., :P_s]
            if self._complex_values:
                xs = torch.complex(xs[..., :self._state_dim], xs[..., self._state_dim:])
            ssm_init = [xs[:, i] for i in range(self._num_layers)]

        fwd = self.model.forward_core_with_alpha(dt_BN, alpha_BNH, ssm_initial_states=ssm_init)

        outputs = fwd["right_us_BNH"][-1]
        if outputs.is_complex():
            outputs = outputs.real

        output_states = None
        if return_states:
            right_xs = fwd["right_xs_BNLP"]
            if right_xs.is_complex():
                right_xs = torch.cat([right_xs.real, right_xs.imag], dim=-1)

            right_us = torch.stack([
                u if u is not None else torch.zeros(B, N, self.model.H, device=dt_BN.device)
                for u in fwd["right_us_BNH"][1:]
            ], dim=2).float()

            valid = torch.tensor([1.0 if u is not None else 0.0 for u in fwd["right_us_BNH"][1:]],
                                 device=dt_BN.device)[None, None, :, None].expand(B, N, -1, 1)

            combined = torch.cat([right_xs.float(), right_us, valid], dim=-1)
            combined_flat = combined.flatten(-2, -1)

            if "left_u_BNm1H" in fwd:
                left_u = fwd["left_u_BNm1H"]
                if left_u.is_complex():
                    left_u = left_u.real
                left_u_padded = torch.cat([
                    left_u,
                    torch.zeros(B, 1, left_u.shape[-1], device=left_u.device, dtype=left_u.dtype)
                ], dim=1)
            else:
                left_u_padded = torch.zeros(B, N, self._hidden_size, device=dt_BN.device)

            combined_flat = torch.cat([combined_flat, left_u_padded], dim=-1)

            if return_states == "full":
                output_states = combined_flat.unsqueeze(0)
            elif return_states == "last":
                last_idx = (seq_lens - 1).clip(min=0)[:, None, None]
                output_states = combined_flat.take_along_dim(last_idx, 1).squeeze(1).unsqueeze(0)

        return PaddedBatch(outputs, seq_lens), output_states

    def interpolate(self, states: Tensor, time_deltas: PaddedBatch) -> PaddedBatch:
        dt = time_deltas.payload
        seq_lens = time_deltas.seq_lens
        B, N, S = dt.shape

        states_flat = states.squeeze(0)
        P_stored = 2 * self._state_dim if self._complex_values else self._state_dim
        size_per_layer = P_stored + self._hidden_size + 1
        layer_state_size = self._num_layers * size_per_layer

        layer_states = states_flat[..., :layer_state_size]
        left_u_stored = states_flat[..., layer_state_size:layer_state_size + self._hidden_size]

        states_BNLD = layer_states.reshape(B, N, self._num_layers, size_per_layer)

        if self._complex_values:
            right_xs = torch.complex(states_BNLD[..., :self._state_dim],
                                     states_BNLD[..., self._state_dim:2*self._state_dim])
            right_us = states_BNLD[..., 2*self._state_dim:2*self._state_dim + self._hidden_size]
        else:
            right_xs = states_BNLD[..., :self._state_dim]
            right_us = states_BNLD[..., self._state_dim:self._state_dim + self._hidden_size]

        right_us_list = [None] + [right_us[:, :, i, :] for i in range(self._num_layers)]

        outputs_list = []
        for s in range(S):
            dt_s = dt[:, :, s:s+1]

            left_u_GH = None
            for i, layer in enumerate(self.model.layers):
                x_GP = layer.get_left_limit(
                    right_limit_P=right_xs[..., i, :],
                    dt_G=dt_s,
                    next_left_u_GH=left_u_GH,
                    current_right_u_H=right_us_list[i],
                )
                left_u_GH = layer.depth_pass(
                    current_left_x_P=x_GP,
                    current_left_u_H=left_u_GH,
                    prev_right_u_H=right_us_list[i],
                )

            hidden = left_u_GH.squeeze(-2)
            if hidden.is_complex():
                hidden = hidden.real
            outputs_list.append(hidden)

        outputs = torch.stack(outputs_list, dim=2)
        return PaddedBatch(outputs, seq_lens)
