"""S2P2 Model for hotpp-benchmark - Direct port from EasyTPP."""
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from hotpp.data import PaddedBatch
from .models import LLH, Int_Forward_LLH, Int_Backward_LLH


class ScaledSoftplus(nn.Module):
    """Scaled Softplus from EasyTPP - CORRECT formula: softplus(beta * x) / beta."""
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
            x,  # linear above threshold for numerical stability
        )


class IntensityNet(nn.Module):
    """Intensity network from EasyTPP."""
    def __init__(self, input_dim: int, num_event_types: int, bias: bool = True):
        super().__init__()
        self.intensity_net = nn.Linear(input_dim, num_event_types, bias=bias)
        self.softplus = ScaledSoftplus(num_event_types)

    def forward(self, x: Tensor) -> Tensor:
        if x.is_complex():
            x = x.real
        return self.softplus(self.intensity_net(x))


class S2P2(nn.Module):
    """S2P2: State Space Model for Point Processes - Direct port from EasyTPP."""
    
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
        
        # Select layer type (same as EasyTPP)
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
        
        # Note: layers_mark_emb is NOT used in hotpp flow
        # hotpp uses: embed_proj (in S2P2Encoder) + Head + NHPLoss.intensity()
        # Removed to avoid "no grad" warning
    
    def _get_hidden(self, x_LP: Union[Tensor, List[Tensor]], right_us_BNH) -> Tensor:
        """Compute hidden state at left limit (for depth pass). Returns hidden, not intensity."""
        left_u_H = None
        for i, layer in enumerate(self.layers):
            if isinstance(x_LP, list):
                left_u_H = layer.depth_pass(x_LP[i], current_left_u_H=left_u_H, prev_right_u_H=right_us_BNH[i])
            else:
                left_u_H = layer.depth_pass(x_LP[..., i, :], current_left_u_H=left_u_H, prev_right_u_H=right_us_BNH[i])
        return left_u_H  # Return hidden, intensity computed by Head+NHPLoss
    
    def _evolve_and_get_hidden_at_sampled_dts(self, x_LP, dt_G, right_us_H):
        """Evolve state and get hidden at sampled times. Returns hidden, not intensity."""
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
        return left_u_GH  # Return hidden, intensity computed by Head+NHPLoss

    def forward_core(self, dt_BN, alpha_BNH):
        """Core forward pass - same as EasyTPP forward. Alpha should be pre-computed."""
        return self.forward_core_with_alpha(dt_BN, alpha_BNH)
    
    def forward_core_with_alpha(self, dt_BN, alpha_BNH, ssm_initial_states=None):
        """Core forward pass with pre-computed alpha (mark embeddings).

        Args:
            dt_BN: Time deltas (B, N).
            alpha_BNH: Mark embeddings (B, N, H).
            ssm_initial_states: Optional list of per-layer SSM initial states, each (B, P).
                When provided, these override the learnable initial_state_P for each layer.
        """
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
            if next_layer_left_u_BNH is None:  # NOT backward variant
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
    """S2P2 Encoder for hotpp-benchmark."""
    
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
        
        # Project embedder output (input_size) to hidden_size for mark input
        # This replaces layers_mark_emb since hotpp embedder already embeds labels
        self.embed_proj = nn.Linear(input_size, hidden_size)
        
        # Initialize embed_proj closer to identity when sizes match
        # (EasyTPP uses direct embedding lookup, no extra linear)
        if input_size == hidden_size:
            nn.init.eye_(self.embed_proj.weight)
            nn.init.zeros_(self.embed_proj.bias)
        
        # State size for storing forward results
        # Includes: right_xs (P_stored per layer), right_us (H per layer), valid flags (1 per layer), left_u (H)
        P_stored = 2 * state_dim if complex_values else state_dim
        self._state_size = num_layers * (P_stored + hidden_size + 1) + hidden_size  # +H for left_u
    
    @property
    def delta_time(self) -> bool:
        return True
    
    @property
    def output_size(self) -> int:
        return self._hidden_size
    
    @property
    def num_layers(self) -> int:
        return 1  # Single-layer from RNN interface perspective
    
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
            s[o + P_s + self._hidden_size] = 1.0  # valid flag
        return s.unsqueeze(0)
    
    def forward(self, x: PaddedBatch, time_deltas: PaddedBatch, states=None, return_states=False):
        dt_BN = time_deltas.payload
        seq_lens = x.seq_lens
        B, N = dt_BN.shape
        alpha_BNH = self.embed_proj(x.payload)
        
        # Unpack per-layer SSM states from carry-over states (for autoregressive generation).
        ssm_init = None
        if states is not None:
            s = states.squeeze(0)  # (B, D)
            P_s = 2 * self._state_dim if self._complex_values else self._state_dim
            sz = P_s + self._hidden_size + 1
            chunks = s[:, :self._num_layers * sz].reshape(-1, self._num_layers, sz)
            xs = chunks[..., :P_s]
            if self._complex_values:
                xs = torch.complex(xs[..., :self._state_dim], xs[..., self._state_dim:])
            ssm_init = [xs[:, i] for i in range(self._num_layers)]
        
        fwd = self.model.forward_core_with_alpha(dt_BN, alpha_BNH, ssm_initial_states=ssm_init)
        
        # For backward variant, use left_u directly (already computed in forward)
        # For non-backward variant, use right_u (will be evolved in interpolate)
        if "left_u_BNm1H" in fwd:
            # Backward variant: left_u at [t_1, ..., t_N], pad with zeros at t_0
            left_u = fwd["left_u_BNm1H"]  # (B, N-1, H)
            if left_u.is_complex():
                left_u = left_u.real
            # Pad to match sequence length: prepend zeros for t_0
            outputs = torch.cat([torch.zeros(B, 1, left_u.shape[-1], device=left_u.device, dtype=left_u.dtype), left_u], dim=1)
        else:
            # Non-backward variant: use right_u, will evolve in interpolate
            outputs = fwd["right_us_BNH"][-1]  # (B, N, H)
            if outputs.is_complex():
                outputs = outputs.real
        
        output_states = None
        if return_states:
            # Pack states for interpolation
            right_xs = fwd["right_xs_BNLP"]  # (B, N, L, P)
            if right_xs.is_complex():
                right_xs = torch.cat([right_xs.real, right_xs.imag], dim=-1)
            
            right_us = torch.stack([
                u if u is not None else torch.zeros(B, N, self.model.H, device=dt_BN.device)
                for u in fwd["right_us_BNH"][1:]
            ], dim=2).float()
            
            valid = torch.tensor([1.0 if u is not None else 0.0 for u in fwd["right_us_BNH"][1:]], 
                                 device=dt_BN.device)[None, None, :, None].expand(B, N, -1, 1)
            
            # Flatten layer states first
            combined = torch.cat([right_xs.float(), right_us, valid], dim=-1)
            combined_flat = combined.flatten(-2, -1)  # (B, N, L * (P_stored + H + 1))
            
            # For backward variant, also store left_u for direct use at event times
            if "left_u_BNm1H" in fwd:
                # left_u_BNm1H is at [t_1, ..., t_{N-1}] (N-1 positions)
                # When interpolating state[i] by delta[i], we get hidden at t_{i+1}
                # So left_u_BNm1H[i] = left_u at t_{i+1} is exactly what we need!
                # Pad at END to match shape (B, N, H), the last position won't be used
                left_u = fwd["left_u_BNm1H"]  # (B, N-1, H)
                if left_u.is_complex():
                    left_u = left_u.real
                # Pad: append zeros at end to match shape (B, N, H)
                left_u_padded = torch.cat([
                    left_u,
                    torch.zeros(B, 1, left_u.shape[-1], device=left_u.device, dtype=left_u.dtype)
                ], dim=1)  # (B, N, H) - position i has left_u at t_{i+1}
            else:
                # No left_u for non-backward variant, add zeros placeholder
                left_u_padded = torch.zeros(B, N, self._hidden_size, device=dt_BN.device)
            
            # Append left_u after flattened layer states
            combined_flat = torch.cat([combined_flat, left_u_padded], dim=-1)  # (B, N, D)
            
            if return_states == "full":
                output_states = combined_flat.unsqueeze(0)  # (1, B, N, D)
            elif return_states == "last":
                last_idx = (seq_lens - 1).clip(min=0)[:, None, None]
                output_states = combined_flat.take_along_dim(last_idx, 1).squeeze(1).unsqueeze(0)  # (1, B, D)
        
        return PaddedBatch(outputs, seq_lens), output_states
    
    def interpolate(self, states: Tensor, time_deltas: PaddedBatch) -> PaddedBatch:
        """Compute intensity at sampled times - returns hidden states for head."""
        dt = time_deltas.payload  # (B, N, S)
        seq_lens = time_deltas.seq_lens
        B, N, S = dt.shape
        
        # Unpack states
        states_flat = states.squeeze(0)  # (B, N, D)
        P_stored = 2 * self._state_dim if self._complex_values else self._state_dim
        size_per_layer = P_stored + self._hidden_size + 1
        layer_state_size = self._num_layers * size_per_layer
        
        # Split: layer states and left_u
        layer_states = states_flat[..., :layer_state_size]
        left_u_stored = states_flat[..., layer_state_size:layer_state_size + self._hidden_size]  # (B, N, H)
        
        states_BNLD = layer_states.reshape(B, N, self._num_layers, size_per_layer)
        
        if self._complex_values:
            right_xs = torch.complex(states_BNLD[..., :self._state_dim], 
                                     states_BNLD[..., self._state_dim:2*self._state_dim])
            right_us = states_BNLD[..., 2*self._state_dim:2*self._state_dim + self._hidden_size]
        else:
            right_xs = states_BNLD[..., :self._state_dim]
            right_us = states_BNLD[..., self._state_dim:self._state_dim + self._hidden_size]
        
        # Build right_us list
        right_us_list = [None] + [right_us[:, :, i, :] for i in range(self._num_layers)]
        
        # NOTE: We always evolve from right limit, even for backward variant
        # Using stored left_u would only work for original event times during training,
        # but interpolate() is also called during thinning/generation with arbitrary times.
        # The loss difference is small (~0.2), but generation quality is more important.
        
        # Evolve from right limit for all cases
        outputs_list = []
        for s in range(S):
            dt_s = dt[:, :, s:s+1]
            
            # Evolve and get hidden (not intensity - head handles that)
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
        
        outputs = torch.stack(outputs_list, dim=2)  # (B, N, S, H)
        return PaddedBatch(outputs, seq_lens)
