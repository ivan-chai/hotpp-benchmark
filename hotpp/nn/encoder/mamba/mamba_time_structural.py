import math
from typing import Optional, Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from hotpp.data import PaddedBatch
from hotpp.nn.encoder.transformer.simple import PositionalEncoding
from hotpp.utils.torch import deterministic


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
    """
    Mamba SSM with configurable time integration.
    
    Args:
        time_mode: How to incorporate time into dt:
            - None: HF-compatible mode, dt computed from input only (for equivalence tests)
            - "structural": dt = time_delta * softplus(scale) (time-only approach)
            - "structural_channel": same but with per-channel scale
            - "additive": dt = softplus(dt_proj(x) + time_scale * time_delta)
            - "multiplicative": dt = softplus(dt_proj(x)) * (1 + tanh(scale) * time_delta)
            - "gated": dt = softplus(dt_proj(x)) * sigmoid(time_gate(time_delta))
            - "concat": dt = softplus(dt_proj(concat(x, time_emb)))
            
            NEW MODES:
            - "bc_time": B and C parameters modulated by time (B' = B + time_proj_B(dt), C' = C + time_proj_C(dt))
            - "bc_time_gate": B and C gated by time (B' = B * sigmoid(gate_B(dt)), similar for C)
            - "selective_time": selectivity depends on time (selection = f(x, time_delta))
            - "exp_decay": exponential time decay dt = exp(time_delta * scale) - 1 instead of linear
            - "full_time": combines additive dt + bc_time for maximum time awareness
            
        time_scale_init: Initial value for time scaling parameters (default 0.1)
    """
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        time_mode: Optional[str] = None,
        time_scale_init: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)
        self.time_mode = time_mode

        # Layer creation order matches HF MambaMixer
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv,
                                groups=self.d_inner, padding=d_conv - 1, bias=True)
        self.activation = nn.SiLU()
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # Structural modes: dt comes purely from time_delta
        if time_mode in ("structural", "structural_channel"):
            self.x_proj = nn.Linear(self.d_inner, self.d_state * 2, bias=False)
            if time_mode == "structural_channel":
                self.dt_scale = nn.Parameter(torch.full((self.d_inner,), time_scale_init))
            else:
                self.dt_scale = nn.Parameter(torch.tensor(time_scale_init))
        elif time_mode == "exp_decay":
            # Exponential decay mode: dt = exp(time_delta * scale) - 1
            self.dt_rank = math.ceil(d_model / 16)
            self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
            self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
            self.time_scale = nn.Parameter(torch.full((self.d_inner,), time_scale_init))
        elif time_mode in ("bc_time", "bc_time_gate"):
            # B/C time modulation modes
            self.dt_rank = math.ceil(d_model / 16)
            self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
            self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
            # Time projections for B and C
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
            else:  # bc_time_gate
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
            # Selectivity depends on both x and time
            self.dt_rank = math.ceil(d_model / 16)
            # x_proj outputs: dt_rank (for dt) + d_state (B) + d_state (C) + d_inner (selection)
            self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
            self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
            # Time-aware selection
            self.time_selection = nn.Sequential(
                nn.Linear(1, self.d_inner // 4),
                nn.SiLU(),
                nn.Linear(self.d_inner // 4, self.d_inner),
            )
            self.selection_proj = nn.Linear(self.d_inner * 2, self.d_inner)
        elif time_mode == "full_time":
            # Full time awareness: additive dt + bc_time
            self.dt_rank = math.ceil(d_model / 16)
            self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
            self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
            self.time_scale = nn.Parameter(torch.full((self.d_inner,), time_scale_init))
            # Time projections for B and C
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
            # All other modes need dt_rank and dt_proj
            self.dt_rank = math.ceil(d_model / 16)
            self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
            
            # Time-aware modes need additional parameters
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
                # None: HF-compatible mode
                self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32)[None, :].repeat(self.d_inner, 1)))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def apply_hf_initialization(self):
        """Apply HF-style initialization. Called by parent after all blocks created."""
        nn.init.normal_(self.in_proj.weight, std=0.1)
        nn.init.normal_(self.x_proj.weight, std=0.1)
        
        # dt_proj exists for all modes except structural/structural_channel
        if self.time_mode not in ("structural", "structural_channel"):
            nn.init.normal_(self.dt_proj.weight, std=0.1)
            nn.init.zeros_(self.dt_proj.bias)
        
        nn.init.normal_(self.out_proj.weight, std=0.1)
        
        # MambaMixer-specific init
        if self.time_mode not in ("structural", "structural_channel"):
            dt_init_std = self.dt_rank ** -0.5
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
            
            dt = torch.exp(torch.rand(self.d_inner) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)).clamp(min=1e-4)
            with torch.no_grad():
                self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))
        
        # Initialize new time-aware components
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

    def forward(self, hidden_states: Tensor, time_deltas: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
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
        
        # Clamp time_deltas for stability
        time_deltas = torch.clamp(time_deltas, min=0.0, max=100.0)

        # Compute dt based on time_mode
        if self.time_mode == "structural":
            # dt = time_delta * softplus(scale)
            B_param, C_param = self.x_proj(x_conv_t).split(self.d_state, dim=-1)
            scale = F.softplus(self.dt_scale)
            discrete_time_step = (time_deltas * scale).unsqueeze(1).expand(B, self.d_inner, L)
            
        elif self.time_mode == "structural_channel":
            # dt = time_delta * softplus(scale[channel])
            B_param, C_param = self.x_proj(x_conv_t).split(self.d_state, dim=-1)
            scale = F.softplus(self.dt_scale)
            discrete_time_step = time_deltas.unsqueeze(1) * scale.unsqueeze(0).unsqueeze(-1)
            
        elif self.time_mode == "additive":
            # dt = softplus(dt_proj(x) + time_scale * time_delta)
            dt_input, B_param, C_param = self.x_proj(x_conv_t).split(
                [self.dt_rank, self.d_state, self.d_state], dim=-1
            )
            dt_from_input = self.dt_proj(dt_input)
            time_bias = time_deltas.unsqueeze(-1) * self.time_scale
            discrete_time_step = F.softplus(dt_from_input + time_bias).transpose(1, 2)
            
        elif self.time_mode == "multiplicative":
            # dt = softplus(dt_proj(x)) * (1 + tanh(scale) * time_delta)
            dt_input, B_param, C_param = self.x_proj(x_conv_t).split(
                [self.dt_rank, self.d_state, self.d_state], dim=-1
            )
            dt_from_input = F.softplus(self.dt_proj(dt_input))
            time_factor = 1 + torch.tanh(self.time_scale) * time_deltas.unsqueeze(-1)
            discrete_time_step = (dt_from_input * time_factor).transpose(1, 2)
            
        elif self.time_mode == "gated":
            # dt = softplus(dt_proj(x)) * sigmoid(time_gate(time_delta))
            dt_input, B_param, C_param = self.x_proj(x_conv_t).split(
                [self.dt_rank, self.d_state, self.d_state], dim=-1
            )
            dt_from_input = F.softplus(self.dt_proj(dt_input))
            time_gate = torch.sigmoid(self.time_gate(time_deltas.unsqueeze(-1)))
            discrete_time_step = (dt_from_input * time_gate).transpose(1, 2)
            
        elif self.time_mode == "concat":
            # dt = softplus(dt_proj(concat(x, time_emb)))
            dt_input, B_param, C_param = self.x_proj(x_conv_t).split(
                [self.dt_rank, self.d_state, self.d_state], dim=-1
            )
            time_emb = self.time_embed(time_deltas.unsqueeze(-1))
            dt_input_with_time = torch.cat([dt_input, time_emb], dim=-1)
            discrete_time_step = F.softplus(self.dt_proj(dt_input_with_time)).transpose(1, 2)
            
        elif self.time_mode == "exp_decay":
            # dt = softplus(dt_proj(x)) * (exp(time_scale * time_delta) - 1 + eps)
            dt_input, B_param, C_param = self.x_proj(x_conv_t).split(
                [self.dt_rank, self.d_state, self.d_state], dim=-1
            )
            dt_base = F.softplus(self.dt_proj(dt_input))
            # Exponential decay factor: exp(scale * dt) - 1 (approaches 0 for small dt, grows for large)
            exp_factor = torch.exp(torch.tanh(self.time_scale) * time_deltas.unsqueeze(-1)) - 1 + 0.1
            discrete_time_step = (dt_base * exp_factor).transpose(1, 2)
            
        elif self.time_mode == "bc_time":
            # Standard dt, but B and C are modulated by time
            dt_input, B_param, C_param = self.x_proj(x_conv_t).split(
                [self.dt_rank, self.d_state, self.d_state], dim=-1
            )
            discrete_time_step = F.softplus(self.dt_proj(dt_input)).transpose(1, 2)
            # Time modulation of B and C: additive bias
            time_B = self.time_proj_B(time_deltas.unsqueeze(-1))  # (B, L, d_state)
            time_C = self.time_proj_C(time_deltas.unsqueeze(-1))  # (B, L, d_state)
            B_param = B_param + time_B
            C_param = C_param + time_C
            
        elif self.time_mode == "bc_time_gate":
            # Standard dt, but B and C are gated by time
            dt_input, B_param, C_param = self.x_proj(x_conv_t).split(
                [self.dt_rank, self.d_state, self.d_state], dim=-1
            )
            discrete_time_step = F.softplus(self.dt_proj(dt_input)).transpose(1, 2)
            # Time gating of B and C
            gate_B = torch.sigmoid(self.time_gate_B(time_deltas.unsqueeze(-1)))  # (B, L, d_state)
            gate_C = torch.sigmoid(self.time_gate_C(time_deltas.unsqueeze(-1)))  # (B, L, d_state)
            B_param = B_param * gate_B
            C_param = C_param * gate_C
            
        elif self.time_mode == "selective_time":
            # Selectivity depends on both x and time
            dt_input, B_param, C_param = self.x_proj(x_conv_t).split(
                [self.dt_rank, self.d_state, self.d_state], dim=-1
            )
            dt_base = F.softplus(self.dt_proj(dt_input))
            # Time-aware selection
            time_sel = self.time_selection(time_deltas.unsqueeze(-1))  # (B, L, d_inner)
            # Combine content and time for selection
            combined = torch.cat([dt_base, time_sel], dim=-1)  # (B, L, 2*d_inner)
            selection = torch.sigmoid(self.selection_proj(combined))  # (B, L, d_inner)
            discrete_time_step = (dt_base * selection).transpose(1, 2)
            
        elif self.time_mode == "full_time":
            # Full time awareness: additive dt + bc_time modulation
            dt_input, B_param, C_param = self.x_proj(x_conv_t).split(
                [self.dt_rank, self.d_state, self.d_state], dim=-1
            )
            # Additive time for dt
            dt_from_input = self.dt_proj(dt_input)
            time_bias = time_deltas.unsqueeze(-1) * self.time_scale
            discrete_time_step = F.softplus(dt_from_input + time_bias).transpose(1, 2)
            # Time modulation of B and C
            time_B = self.time_proj_B(time_deltas.unsqueeze(-1))
            time_C = self.time_proj_C(time_deltas.unsqueeze(-1))
            B_param = B_param + time_B
            C_param = C_param + time_C
            
        else:
            # None/HF-compatible: dt from input only (no time)
            dt_input, B_param, C_param = self.x_proj(x_conv_t).split(
                [self.dt_rank, self.d_state, self.d_state], dim=-1
            )
            discrete_time_step = F.softplus(self.dt_proj(dt_input)).transpose(1, 2)

        # Save dt for debugging/analysis
        self._last_dt = discrete_time_step.detach()
        self._last_time_deltas = time_deltas.detach()

        # SSM recurrence
        A = -torch.exp(self.A_log.float())
        ssm_state = hidden_states.new_zeros(B, self.d_inner, self.d_state)
        outputs = []
        
        for t in range(L):
            dt_t = discrete_time_step[:, :, t]
            ssm_state = torch.exp(A.unsqueeze(0) * dt_t.unsqueeze(-1)) * ssm_state + \
                        (dt_t.unsqueeze(-1) * B_param[:, t].unsqueeze(1).float()) * x_conv[:, :, t].unsqueeze(-1).float()
            outputs.append((ssm_state.to(dtype) * C_param[:, t].unsqueeze(1)).sum(dim=-1))

        scan_output = torch.stack(outputs, dim=-1) + x_conv * self.D.view(1, -1, 1)
        out = self.out_proj((scan_output * self.activation(gate)).transpose(1, 2))
        
        return out * attn if attn is not None else out


class StructuralMambaBlock(nn.Module):
    def __init__(self, hidden_size: int, **mixer_kwargs):
        super().__init__()
        self.norm = RMSNorm(hidden_size)
        self.mixer = StructuralMambaMixer(hidden_size, **mixer_kwargs)

    def forward(self, hidden_states: Tensor, time_deltas: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        return hidden_states + self.mixer(self.norm(hidden_states), time_deltas, attention_mask)


class StructuralMambaTimeEmbedding(nn.Module):
    """
    Mamba with configurable time-awareness.
    
    Args:
        time_mode: How to incorporate time into SSM dt:
            - None: HF-compatible (no time in dt)
            - "structural": dt = time_delta * scale
            - "structural_channel": dt = time_delta * scale[channel]
            - "additive": dt = softplus(dt_proj(x) + scale * time_delta)
            - "multiplicative": dt = softplus(dt_proj(x)) * (1 + tanh(scale) * time_delta)
            - "gated": dt = softplus(dt_proj(x)) * sigmoid(gate(time_delta))
            - "concat": dt = softplus(dt_proj([x, time_emb]))
            - "bc_time": B, C modulated by time (B' = B + proj_B(dt), C' = C + proj_C(dt))
            - "bc_time_gate": B, C gated by time (B' = B * sigmoid(gate_B(dt)))
            - "selective_time": selectivity depends on time (selection = f(x, dt))
            - "exp_decay": exponential time decay dt = base * (exp(scale*dt) - 1)
            - "full_time": combines additive dt + bc_time for maximum time awareness
            
        pos_type: Positional encoding type for INPUT (optional, default "none"):
            - "none": no positional encoding on input
            - "time-angular-rel": relative time angular encoding (like MambaTimeEmbedding)
            - "time-angular-abs": absolute time angular encoding
            
        time_scale_init: Initial value for time scaling parameters
        max_duration: Maximum time span for positional encoding
        n_positions: Number of positions for encoding
        dropout: Dropout rate for positional encoding
        vocab_size: For HF RNG compatibility (creates dummy embedding)
    """
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
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.pos_type = pos_type
        
        # Match HF RNG order: embedding creation -> blocks creation -> embedding init -> blocks init -> input_proj
        _dummy = nn.Embedding(vocab_size, hidden_size) if vocab_size > 0 else None
        
        self.layers = nn.ModuleList([
            StructuralMambaBlock(
                hidden_size,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                time_mode=time_mode,
                time_scale_init=time_scale_init,
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
        
        # Optional positional encoding on input
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

    def forward(self, x: PaddedBatch, timestamps: PaddedBatch, states: Optional[Tensor] = None,
                return_states: bool = False, attention_mask: Optional[Tensor] = None) -> Tuple[PaddedBatch, Optional[Tensor]]:
        if return_states:
            raise ValueError("StructuralMambaTimeEmbedding does not support return_states")

        x_payload, ts = x.payload, timestamps.payload
        B, L = ts.shape

        if attention_mask is None:
            attention_mask = torch.zeros(B, L, dtype=torch.bool, device=x_payload.device)
            for i, l in enumerate(x.seq_lens):
                attention_mask[i, :l] = True

        # ts = absolute timestamps (since delta_time=False)
        # Compute deltas
        deltas = torch.zeros_like(ts)
        deltas[:, 1:] = (ts[:, 1:] - ts[:, :-1])
        
        # Compute "delta of deltas" for SSM - this was the "bug" that worked well
        delta_of_deltas = torch.zeros_like(ts)
        delta_of_deltas[:, 1:] = deltas[:, 1:] - deltas[:, :-1]
        delta_of_deltas = delta_of_deltas * attention_mask

        # Input projection + optional positional encoding
        hidden = self.input_projection(x_payload)
        if self.positional is not None:
            hidden = self.positional(hidden, ts)  # positional gets absolute timestamps
        
        for layer in self.layers:
            hidden = layer(hidden, deltas, attention_mask)

        return PaddedBatch(self.norm_f(hidden), x.seq_lens), None
