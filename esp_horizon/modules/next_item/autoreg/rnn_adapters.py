from abc import ABC, abstractmethod
from typing import Tuple

import torch
from torch import Tensor

from ptls.nn.seq_encoder import RnnEncoder
from ptls.nn.seq_encoder.containers import SeqEncoderContainer

from esp_horizon.data import PaddedBatch
from .base_adapter import BaseRNNAdapter
from .window import apply_windows


class NextItemRNNAdapter(BaseRNNAdapter):
    """Adapter based on MLE module.

    Args:
      inference_mode: The type of inference. Use `mode` for a fixed prediction.
      max_context: Maximum RNN context for long sequences.
      context_step: Window step when max_context is provided.
      dump_category_logits: A mapping from category field names to output logits field names.
    """
    def __init__(self, model, time_field="timestamps", time_int=False,
                 time_input_delta=True, time_output_delta=True,
                 category_mapping=None, inference_mode="mode",
                 max_context=None, context_step=None,
                 dump_category_logits=None):
        try:
            if ((not isinstance(model.seq_encoder.seq_encoder, RnnEncoder))
                or (model.seq_encoder.seq_encoder.rnn_type != "gru")
                or (model.seq_encoder.seq_encoder.num_layers != 1)
                or (model.seq_encoder.seq_encoder.bidirectional)
                or (model.seq_encoder.seq_encoder.trainable_starter != "static")):
                raise NotImplementedError("Only uni-directional single-layer GRU RNN encoder with trainable initial state is supported.")
        except AttributeError:
            raise ValueError(f"Need Next-item module capable of features predictions, got {type(model)}.")
        if (max_context is not None) and (context_step is None):
            raise ValueError("Need context_step for sliding window.")

        super().__init__(model, time_field, time_int,
                         time_input_delta, time_output_delta,
                         category_mapping)
        self.inference_mode = inference_mode
        self.max_context = max_context
        self.context_step = context_step
        self.dump_category_logits = dump_category_logits or {}

    @property
    def output_seq_features(self):
        return super().output_seq_features | set(self.dump_category_logits.values())

    def eval_states(self, x: PaddedBatch) -> PaddedBatch:
        """Apply encoder to the batch of features and produce batch of input hidden states for each iteration.

        Args:
          - x: Payload contains dictionary of features with shapes (B, T, D_i).

        Returns:
          PaddedBatch with payload containing hidden states with shape (B, T, D).
        """
        if x.left:
            raise NotImplementedError("Left-padded batches are not implemented.")
        init_state = self.model.seq_encoder.seq_encoder.starter_h  # (1, 1, D).
        assert init_state.shape[0] == 1
        init_state = init_state.repeat(len(x), 1, 1)  # (B, 1, D).
        # GRU outputs are also states.
        encoder = self.model.seq_encoder
        embeddings = encoder.trx_encoder(x)
        next_states = apply_windows(embeddings, encoder.seq_encoder, self.max_context, self.context_step)  # (B, T, D).
        payload = torch.cat([init_state, next_states.payload[:, :-1]], dim=1)  # (B, T, D).
        return PaddedBatch(payload, next_states.seq_lens)

    def forward(self, x: PaddedBatch, states: Tensor):
        """Predict features given inputs and states.

        Args:
          - x: Payload contains dictionary of features with shapes (B, T, D_i).
          - states: Initial states with shape (B, D).

        Returns:
          Next token features.
        """
        if x.left:
            raise NotImplementedError("Left-padded batches are not implemented.")
        trx_embeddings = self.model.seq_encoder.trx_encoder(x)
        embeddings = self.model.seq_encoder.seq_encoder(trx_embeddings,
                                                        states.unsqueeze(0) if states is not None else None)
        last = (embeddings.seq_lens - 1).unsqueeze(1).unsqueeze(1)  # (B, 1, 1).
        embeddings = PaddedBatch(embeddings.payload.take_along_dim(last, 1),
                                 torch.ones_like(embeddings.seq_lens))  # (B, 1, D).
        head_outputs = self.model.apply_head(embeddings)
        new_states = embeddings.payload.squeeze(1)  # (B, D).
        assert new_states.ndim == 2
        if self.inference_mode == "mode":
            features = self.model.get_modes(head_outputs)
        else:
            raise ValueError(f"Unknown inference mode: {self.inference_mode}")

        outputs = {}
        for k, v in features.payload.items():
            if v.ndim > x.payload[k].ndim:
                # Remove extra dims.
                assert v.shape[-1] == 1
                v = v.squeeze(-1)
            # Remove time dimension.
            assert v.shape[1] == 1
            v = v.squeeze(1)
            outputs[k] = v

        if self.dump_category_logits:
            logits = self.model.get_logits(head_outputs)
            for field, logits_field in self.dump_category_logits.items():
                outputs[logits_field] = logits.payload[field].squeeze(1)  # (B, C).

        return outputs, new_states
