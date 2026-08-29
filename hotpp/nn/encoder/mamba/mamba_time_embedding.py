import torch
from torch import nn, Tensor
from typing import Optional, Tuple

from transformers import MambaConfig, MambaModel
from hotpp.data import PaddedBatch
from hotpp.nn.encoder.transformer.simple import PositionalEncoding


class MambaTimeEmbedding(nn.Module):
    def __init__(
        self,
        input_size: int,
        n_positions: int = 1024,
        hidden_size: int = 128,
        num_layers: int = 4,
        pos_type: str = "time-angular-rel",
        max_duration: float = 1.0,
        min_time_step: Optional[float] = None,
        dropout: float = 0.1,
        vocab_size: int = 1,
    ):
        super().__init__()

        config = MambaConfig(
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            vocab_size=vocab_size,
        )
        self.mamba = MambaModel(config)
        self.hidden_size = hidden_size

        self.input_projection = nn.Linear(input_size, hidden_size)

        self.positional = PositionalEncoding(
            n_embd=hidden_size,
            n_positions=n_positions,
            pos_type=pos_type,     
            max_duration=max_duration,
            min_time_step=min_time_step,
            dropout=dropout,
        )

    @property
    def output_size(self) -> int:
        return self.hidden_size

    @property
    def delta_time(self) -> bool:
        return False

    def forward(
        self,
        x: PaddedBatch,
        timestamps: PaddedBatch,
        states: Optional[Tensor] = None,
        return_states: bool = False,
        attention_mask: Optional[Tensor] = None,
    ) -> Tuple[PaddedBatch, Optional[Tensor]]:

        if return_states not in {False}:
            raise ValueError("MambaTimeTransformer не поддерживает return_states")

        token_emb = self.input_projection(x.payload)
        emb_with_pos = self.positional(token_emb, timestamps.payload)
        attn_mask = x.seq_len_mask

        outputs = self.mamba(
            inputs_embeds=emb_with_pos,
            attention_mask=attn_mask,
        )

        seq = outputs.last_hidden_state
        return PaddedBatch(seq, x.seq_lens), None
