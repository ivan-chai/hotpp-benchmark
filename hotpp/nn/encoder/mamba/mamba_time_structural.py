import math
from typing import Optional, Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from hotpp.data import PaddedBatch
from hotpp.nn.encoder.transformer.simple import PositionalEncoding
from hotpp.utils.torch import deterministic


_PURE_TIME_MODES = (
    "structural", "structural_channel",
    "lin_scalar", "lin_channel",
    "log_scalar", "log_channel",
    "power_scalar", "power_channel",
)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(self.weight.dtype)


class StructuralMambaMixer(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        time_mode: Optional[str] = None,
        time_scale_init: float = 0.1,
        jump: bool = False,
        jump_mode: str = "b_jump",
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)
        self.time_mode = time_mode
        self.jump = jump
        self.jump_mode = jump_mode if jump else None

        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv,
                                groups=self.d_inner, padding=d_conv - 1, bias=True)
        self.activation = nn.SiLU()
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        if time_mode in _PURE_TIME_MODES:
            self.x_proj = nn.Linear(self.d_inner, self.d_state * 2, bias=False)

            if time_mode == "structural":
                self.log_step = nn.Parameter(
                    torch.linspace(math.log(1e-4), math.log(0.1), self.d_state)
                )
            elif time_mode == "structural_channel":
                self.log_step = nn.Parameter(
                    torch.linspace(math.log(1e-4), math.log(0.1), self.d_state)[None, :].expand(self.d_inner, -1).clone()
                )
            elif time_mode in ("lin_scalar", "log_scalar", "power_scalar"):
                self.log_step = nn.Parameter(torch.tensor([math.log(1e-2)]))
            elif time_mode in ("lin_channel", "log_channel", "power_channel"):
                self.log_step = nn.Parameter(
                    torch.linspace(math.log(1e-4), math.log(0.1), self.d_inner)
                )

            if time_mode in ("power_scalar", "power_channel"):
                self.p_raw = nn.Parameter(torch.zeros(1))
        elif time_mode == "exp_decay":
            self.dt_rank = math.ceil(d_model / 16)
            self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
            self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
            self.time_scale = nn.Parameter(torch.full((self.d_inner,), time_scale_init))
        elif time_mode in ("bc_time", "bc_time_gate"):
            self.dt_rank = math.ceil(d_model / 16)
            self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
            self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
            time_hidden = max(16, self.d_state)
            if time_mode == "bc_time":
                self.time_proj_B = nn.Sequential(
                    nn.Linear(1, time_hidden),
                    nn.SiLU(),
                    nn.Linear(time_hidden, self.d_state),
                )
                self.time_proj_C = nn.Sequential(
                    nn.Linear(1, time_hidden),
                    nn.SiLU(),
                    nn.Linear(time_hidden, self.d_state),
                )
            else:
                self.time_gate_B = nn.Sequential(
                    nn.Linear(1, time_hidden),
                    nn.SiLU(),
                    nn.Linear(time_hidden, self.d_state),
                )
                self.time_gate_C = nn.Sequential(
                    nn.Linear(1, time_hidden),
                    nn.SiLU(),
                    nn.Linear(time_hidden, self.d_state),
                )
        elif time_mode == "selective_time":
            self.dt_rank = math.ceil(d_model / 16)
            self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
            self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
            self.time_selection = nn.Sequential(
                nn.Linear(1, self.d_inner // 4),
                nn.SiLU(),
                nn.Linear(self.d_inner // 4, self.d_inner),
            )
            self.selection_proj = nn.Linear(self.d_inner * 2, self.d_inner)
        elif time_mode == "full_time":
            self.dt_rank = math.ceil(d_model / 16)
            self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
            self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
            self.time_scale = nn.Parameter(torch.full((self.d_inner,), time_scale_init))
            time_hidden = max(16, self.d_state)
            self.time_proj_B = nn.Sequential(
                nn.Linear(1, time_hidden),
                nn.SiLU(),
                nn.Linear(time_hidden, self.d_state),
            )
            self.time_proj_C = nn.Sequential(
                nn.Linear(1, time_hidden),
                nn.SiLU(),
                nn.Linear(time_hidden, self.d_state),
            )
        else:
            self.dt_rank = math.ceil(d_model / 16)
            self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)

            if time_mode == "additive":
                self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
                self.time_scale = nn.Parameter(torch.full((self.d_inner,), time_scale_init))
            elif time_mode == "multiplicative":
                self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
                self.time_scale = nn.Parameter(torch.full((self.d_inner,), time_scale_init))
            elif time_mode == "gated":
                self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
                self.time_gate = nn.Sequential(
                    nn.Linear(1, self.d_inner // 4),
                    nn.SiLU(),
                    nn.Linear(self.d_inner // 4, self.d_inner),
                )
            elif time_mode == "concat":
                self.time_embed_dim = 16
                self.time_embed = nn.Linear(1, self.time_embed_dim)
                self.dt_proj = nn.Linear(self.dt_rank + self.time_embed_dim, self.d_inner, bias=True)
            else:
                self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32)[None, :].repeat(self.d_inner, 1)))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        if jump:
            if self.jump_mode == "b_jump":
                self.B_jump = nn.Parameter(torch.zeros(self.d_inner, d_state))
            elif self.jump_mode in ("e_shared", "e_channel"):
                self.E_proj = nn.Linear(d_model, d_state, bias=False)
            elif self.jump_mode == "no_dt_input":
                pass
            elif self.jump_mode == "e_both":
                self.B_jump = nn.Parameter(torch.zeros(self.d_inner, d_state))
                self.E_proj = nn.Linear(d_model, d_state, bias=False)
            elif self.jump_mode == "e_gate":
                self.E_proj = nn.Linear(d_model, d_state, bias=False)
                self.E_gate = nn.Linear(d_model, d_state, bias=True)
            else:
                raise ValueError(f"Unknown jump_mode: {self.jump_mode}")

    def apply_hf_initialization(self):
        nn.init.normal_(self.in_proj.weight, std=0.1)
        nn.init.normal_(self.x_proj.weight, std=0.1)

        if self.time_mode not in _PURE_TIME_MODES:
            nn.init.normal_(self.dt_proj.weight, std=0.1)
            nn.init.zeros_(self.dt_proj.bias)

        nn.init.normal_(self.out_proj.weight, std=0.1)

        if self.time_mode not in _PURE_TIME_MODES:
            dt_init_std = self.dt_rank ** -0.5
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)

            dt = torch.exp(torch.rand(self.d_inner) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)).clamp(min=1e-4)
            with torch.no_grad():
                self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))

        if self.time_mode == "bc_time":
            for module in [self.time_proj_B, self.time_proj_C]:
                for m in module:
                    if isinstance(m, nn.Linear):
                        nn.init.normal_(m.weight, std=0.02)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)
        elif self.time_mode == "bc_time_gate":
            for module in [self.time_gate_B, self.time_gate_C]:
                for m in module:
                    if isinstance(m, nn.Linear):
                        nn.init.normal_(m.weight, std=0.02)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)
        elif self.time_mode == "selective_time":
            for m in self.time_selection:
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
            nn.init.normal_(self.selection_proj.weight, std=0.02)
            nn.init.zeros_(self.selection_proj.bias)
        elif self.time_mode == "full_time":
            for module in [self.time_proj_B, self.time_proj_C]:
                for m in module:
                    if isinstance(m, nn.Linear):
                        nn.init.normal_(m.weight, std=0.02)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)

        nn.init.kaiming_uniform_(self.conv1d.weight, a=math.sqrt(5))
        nn.init.zeros_(self.conv1d.bias)
        nn.init.kaiming_uniform_(self.out_proj.weight, a=math.sqrt(5))

        if self.jump:
            if self.jump_mode in ("e_shared", "e_channel"):
                nn.init.zeros_(self.E_proj.weight)
            elif self.jump_mode == "e_both":
                nn.init.zeros_(self.E_proj.weight)
            elif self.jump_mode == "e_gate":
                nn.init.zeros_(self.E_proj.weight)
                nn.init.zeros_(self.E_gate.weight)
                nn.init.constant_(self.E_gate.bias, -5.0)

    def forward(self, hidden_states: Tensor, time_deltas: Tensor,
                attention_mask: Optional[Tensor] = None, save_ssm_state: bool = False):
        B, L, _ = hidden_states.shape
        dtype = hidden_states.dtype
        attn = attention_mask.to(dtype).unsqueeze(-1) if attention_mask is not None else None

        projected = self.in_proj(hidden_states).transpose(1, 2)
        x, gate = projected.chunk(2, dim=1)

        if attn is not None:
            x = x * attn.transpose(1, 2)

        x_conv = self.activation(self.conv1d(x)[:, :, :L])
        if attn is not None:
            x_conv = x_conv * attn.transpose(1, 2)

        x_conv_t = x_conv.transpose(1, 2)

        time_deltas = torch.clamp(time_deltas, min=0.0, max=100.0)

        if self.time_mode in _PURE_TIME_MODES:
            B_param, C_param = self.x_proj(x_conv_t).split(self.d_state, dim=-1)
            step = torch.exp(self.log_step)
            discrete_time_step = None

        elif self.time_mode == "additive":
            dt_input, B_param, C_param = self.x_proj(x_conv_t).split(
                [self.dt_rank, self.d_state, self.d_state], dim=-1
            )
            dt_from_input = self.dt_proj(dt_input)
            time_bias = time_deltas.unsqueeze(-1) * self.time_scale
            discrete_time_step = F.softplus(dt_from_input + time_bias).transpose(1, 2)

        elif self.time_mode == "multiplicative":
            dt_input, B_param, C_param = self.x_proj(x_conv_t).split(
                [self.dt_rank, self.d_state, self.d_state], dim=-1
            )
            dt_from_input = F.softplus(self.dt_proj(dt_input))
            time_factor = 1 + torch.tanh(self.time_scale) * time_deltas.unsqueeze(-1)
            discrete_time_step = (dt_from_input * time_factor).transpose(1, 2)

        elif self.time_mode == "gated":
            dt_input, B_param, C_param = self.x_proj(x_conv_t).split(
                [self.dt_rank, self.d_state, self.d_state], dim=-1
            )
            dt_from_input = F.softplus(self.dt_proj(dt_input))
            time_gate = torch.sigmoid(self.time_gate(time_deltas.unsqueeze(-1)))
            discrete_time_step = (dt_from_input * time_gate).transpose(1, 2)

        elif self.time_mode == "concat":
            dt_input, B_param, C_param = self.x_proj(x_conv_t).split(
                [self.dt_rank, self.d_state, self.d_state], dim=-1
            )
            time_emb = self.time_embed(time_deltas.unsqueeze(-1))
            dt_input_with_time = torch.cat([dt_input, time_emb], dim=-1)
            discrete_time_step = F.softplus(self.dt_proj(dt_input_with_time)).transpose(1, 2)

        elif self.time_mode == "exp_decay":
            dt_input, B_param, C_param = self.x_proj(x_conv_t).split(
                [self.dt_rank, self.d_state, self.d_state], dim=-1
            )
            dt_base = F.softplus(self.dt_proj(dt_input))
            exp_factor = torch.exp(torch.tanh(self.time_scale) * time_deltas.unsqueeze(-1)) - 1 + 0.1
            discrete_time_step = (dt_base * exp_factor).transpose(1, 2)

        elif self.time_mode == "bc_time":
            dt_input, B_param, C_param = self.x_proj(x_conv_t).split(
                [self.dt_rank, self.d_state, self.d_state], dim=-1
            )
            discrete_time_step = F.softplus(self.dt_proj(dt_input)).transpose(1, 2)
            time_B = self.time_proj_B(time_deltas.unsqueeze(-1))
            time_C = self.time_proj_C(time_deltas.unsqueeze(-1))
            B_param = B_param + time_B
            C_param = C_param + time_C

        elif self.time_mode == "bc_time_gate":
            dt_input, B_param, C_param = self.x_proj(x_conv_t).split(
                [self.dt_rank, self.d_state, self.d_state], dim=-1
            )
            discrete_time_step = F.softplus(self.dt_proj(dt_input)).transpose(1, 2)
            gate_B = torch.sigmoid(self.time_gate_B(time_deltas.unsqueeze(-1)))
            gate_C = torch.sigmoid(self.time_gate_C(time_deltas.unsqueeze(-1)))
            B_param = B_param * gate_B
            C_param = C_param * gate_C

        elif self.time_mode == "selective_time":
            dt_input, B_param, C_param = self.x_proj(x_conv_t).split(
                [self.dt_rank, self.d_state, self.d_state], dim=-1
            )
            dt_base = F.softplus(self.dt_proj(dt_input))
            time_sel = self.time_selection(time_deltas.unsqueeze(-1))
            combined = torch.cat([dt_base, time_sel], dim=-1)
            selection = torch.sigmoid(self.selection_proj(combined))
            discrete_time_step = (dt_base * selection).transpose(1, 2)

        elif self.time_mode == "full_time":
            dt_input, B_param, C_param = self.x_proj(x_conv_t).split(
                [self.dt_rank, self.d_state, self.d_state], dim=-1
            )
            dt_from_input = self.dt_proj(dt_input)
            time_bias = time_deltas.unsqueeze(-1) * self.time_scale
            discrete_time_step = F.softplus(dt_from_input + time_bias).transpose(1, 2)
            time_B = self.time_proj_B(time_deltas.unsqueeze(-1))
            time_C = self.time_proj_C(time_deltas.unsqueeze(-1))
            B_param = B_param + time_B
            C_param = C_param + time_C

        else:
            dt_input, B_param, C_param = self.x_proj(x_conv_t).split(
                [self.dt_rank, self.d_state, self.d_state], dim=-1
            )
            discrete_time_step = F.softplus(self.dt_proj(dt_input)).transpose(1, 2)

        A = -torch.exp(self.A_log.float())
        ssm_state = hidden_states.new_zeros(B, self.d_inner, self.d_state)
        outputs = []
        ssm_state_list = [] if save_ssm_state else None
        is_structural = self.time_mode in _PURE_TIME_MODES

        if self.jump and self.jump_mode in ("e_shared", "e_channel", "e_both", "e_gate"):
            with torch.amp.autocast('cuda', enabled=False):
                jump_e = self.E_proj(hidden_states.float())
            if self.jump_mode == "e_gate":
                with torch.amp.autocast('cuda', enabled=False):
                    gate = torch.sigmoid(self.E_gate(hidden_states.float()))
                jump_e = gate * jump_e

        if self.time_mode in ("power_scalar", "power_channel"):
            p_exp = 2.0 * torch.sigmoid(self.p_raw)

        for t in range(L):
            if is_structural:
                t_val = time_deltas[:, t]
                if self.time_mode in ("log_scalar", "log_channel"):
                    f_t = torch.log1p(t_val)
                elif self.time_mode in ("power_scalar", "power_channel"):
                    f_t = (t_val + 1e-4).pow(p_exp)
                else:
                    f_t = t_val
                if self.time_mode in ("lin_channel", "log_channel", "power_channel"):
                    dt_t = f_t.view(-1, 1, 1) * step.view(1, -1, 1)
                elif self.time_mode in ("lin_scalar", "log_scalar", "power_scalar"):
                    dt_t = f_t.view(-1, 1, 1) * step.view(1, 1, 1)
                else:
                    dt_t = f_t.view(-1, 1, 1) * step
            else:
                dt_t = discrete_time_step[:, :, t].unsqueeze(-1)
            x_conv_t_col = x_conv[:, :, t].unsqueeze(-1).float()
            if self.jump and self.jump_mode == "no_dt_input":
                ssm_state = torch.exp(A.unsqueeze(0) * dt_t) * ssm_state + \
                            B_param[:, t].unsqueeze(1).float() * x_conv_t_col
            else:
                ssm_state = torch.exp(A.unsqueeze(0) * dt_t) * ssm_state + \
                            (dt_t * B_param[:, t].unsqueeze(1).float()) * x_conv_t_col
            if self.jump:
                if self.jump_mode == "b_jump":
                    ssm_state = ssm_state + self.B_jump.unsqueeze(0).float() * x_conv_t_col
                elif self.jump_mode in ("e_shared", "e_gate"):
                    ssm_state = ssm_state + jump_e[:, t].unsqueeze(1)
                elif self.jump_mode == "e_channel":
                    ssm_state = ssm_state + x[:, :, t].unsqueeze(-1).float() * jump_e[:, t].unsqueeze(1)
                elif self.jump_mode == "e_both":
                    ssm_state = ssm_state + self.B_jump.unsqueeze(0).float() * x_conv_t_col + jump_e[:, t].unsqueeze(1)
            if ssm_state_list is not None:
                ssm_state_list.append(ssm_state)
            outputs.append((ssm_state.to(dtype) * C_param[:, t].unsqueeze(1)).sum(dim=-1))

        scan_output = torch.stack(outputs, dim=-1) + x_conv * self.D.view(1, -1, 1)
        out = self.out_proj((scan_output * self.activation(gate)).transpose(1, 2))
        out = out * attn if attn is not None else out

        if save_ssm_state:
            return out, torch.stack(ssm_state_list, dim=2), C_param, x_conv, gate
        return out


class StructuralMambaBlock(nn.Module):
    def __init__(self, hidden_size: int, **mixer_kwargs):
        super().__init__()
        self.norm = RMSNorm(hidden_size)
        self.mixer = StructuralMambaMixer(hidden_size, **mixer_kwargs)

    def forward(self, hidden_states: Tensor, time_deltas: Tensor,
                attention_mask: Optional[Tensor] = None, save_ssm_state: bool = False):
        if save_ssm_state:
            out, ssm_states, C_param, x_conv, gate = self.mixer(
                self.norm(hidden_states), time_deltas, attention_mask, save_ssm_state=True
            )
            return hidden_states + out, ssm_states, C_param, x_conv, gate
        return hidden_states + self.mixer(self.norm(hidden_states), time_deltas, attention_mask)


class StructuralMambaTimeEmbedding(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        time_mode: Optional[str] = None,
        time_scale_init: float = 0.1,
        pos_type: str = "none",
        max_duration: float = 15.0,
        n_positions: int = 1024,
        dropout: float = 0.1,
        vocab_size: int = 0,
        jump: bool = False,
        jump_mode: str = "b_jump",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.time_mode = time_mode
        self.pos_type = pos_type

        _dummy = nn.Embedding(vocab_size, hidden_size) if vocab_size > 0 else None

        self.layers = nn.ModuleList([
            StructuralMambaBlock(
                hidden_size,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                time_mode=time_mode,
                time_scale_init=time_scale_init,
                jump=jump,
                jump_mode=jump_mode,
            )
            for _ in range(num_layers)
        ])
        self.norm_f = RMSNorm(hidden_size)

        if _dummy is not None:
            nn.init.normal_(_dummy.weight, std=0.1)
            del _dummy

        for layer in self.layers:
            layer.mixer.apply_hf_initialization()

        self.input_projection = nn.Linear(input_size, hidden_size)

        if pos_type != "none":
            self.positional = PositionalEncoding(
                n_embd=hidden_size,
                n_positions=n_positions,
                pos_type=pos_type,
                max_duration=max_duration,
                dropout=dropout,
            )
        else:
            self.positional = None

    @property
    def output_size(self) -> int:
        return self.hidden_size

    @property
    def delta_time(self) -> bool:
        return False

    @property
    def _supports_nhp(self) -> bool:
        return self.time_mode in ("structural", "structural_channel")

    def forward(self, x: PaddedBatch, timestamps: PaddedBatch, states: Optional[Tensor] = None,
                return_states=False, attention_mask: Optional[Tensor] = None) -> Tuple[PaddedBatch, Optional[Tensor]]:
        if return_states and return_states != "last" and not self._supports_nhp:
            raise ValueError(f"return_states='full' only supported for structural/structural_channel, got time_mode={self.time_mode}")

        x_payload, ts = x.payload, timestamps.payload
        B, L = ts.shape

        if attention_mask is None:
            attention_mask = torch.zeros(B, L, dtype=torch.bool, device=x_payload.device)
            for i, l in enumerate(x.seq_lens):
                attention_mask[i, :l] = True

        deltas = torch.zeros_like(ts)
        deltas[:, 1:] = (ts[:, 1:] - ts[:, :-1])

        hidden = self.input_projection(x_payload)
        if self.positional is not None:
            hidden = self.positional(hidden, ts)

        need_ssm = (return_states == "full")
        for i, layer in enumerate(self.layers):
            if need_ssm and i == len(self.layers) - 1:
                last_residual = hidden
                hidden, ssm_states, C_param, x_conv, gate = layer(
                    hidden, deltas, attention_mask, save_ssm_state=True
                )
            else:
                hidden = layer(hidden, deltas, attention_mask)

        output = self.norm_f(hidden)
        output_states = None

        if return_states == "full":
            mixer = self.layers[-1].mixer
            d_inner, d_state = mixer.d_inner, mixer.d_state
            ssm_flat = ssm_states.permute(0, 2, 1, 3).reshape(B, L, d_inner * d_state)
            state = torch.cat([
                ssm_flat,
                C_param.float(),
                x_conv.transpose(1, 2).float(),
                gate.transpose(1, 2).float(),
                last_residual.float(),
            ], dim=-1)
            output_states = state.unsqueeze(0)
        elif return_states == "last":
            last_idx = (x.seq_lens - 1).clip(min=0)[:, None, None]
            output_states = output.take_along_dim(last_idx, 1).squeeze(1).unsqueeze(0)
        elif return_states and return_states is not False:
            raise ValueError(f"Unknown return_states: {return_states}")

        return PaddedBatch(output, x.seq_lens), output_states

    def interpolate(self, states: Tensor, time_deltas: PaddedBatch) -> PaddedBatch:
        state = states[0]
        dt = time_deltas.payload
        B, L, S = dt.shape

        mixer = self.layers[-1].mixer
        d_inner, d_state = mixer.d_inner, mixer.d_state
        step = torch.exp(mixer.log_step)
        A = -torch.exp(mixer.A_log.float())

        idx = 0
        ssm_state = state[..., idx:idx + d_inner * d_state].reshape(B, L, d_inner, d_state)
        idx += d_inner * d_state
        C = state[..., idx:idx + d_state]
        idx += d_state
        x_conv = state[..., idx:idx + d_inner]
        idx += d_inner
        gate = state[..., idx:idx + d_inner]
        idx += d_inner
        residual = state[..., idx:]

        A_eff = A * step

        results = []
        for s in range(S):
            delta = dt[:, :, s]
            decay = torch.exp(A_eff[None, None] * delta[:, :, None, None])
            evolved = ssm_state * decay

            ssm_out = (evolved * C[:, :, None, :]).sum(-1)
            scan_out = ssm_out + x_conv * mixer.D[None, None]
            gated = scan_out * mixer.activation(gate)
            mixer_out = mixer.out_proj(gated)

            block_out = residual + mixer_out
            results.append(self.norm_f(block_out))

        outputs = torch.stack(results, dim=2)
        return PaddedBatch(outputs, time_deltas.seq_lens)
