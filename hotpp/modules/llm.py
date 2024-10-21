import torch
import hydra
from hotpp.data import PaddedBatch
from hotpp.utils.torch import deterministic
from ..nn import Head
from .base_module import BaseModule
from transllm.summary_embed import get_embeddings


class LLMEncoder(torch.nn.Module):
    def __init__(self, encoder, model,
                 max_summary_tokens,
                 max_output_tokens,
                 summary_input_prompt,
                 summary_query_prompt,
                 input_prompt,
                 query_prompt,
                 max_length=None,
                 temperature=0,
                 timestamps_field="timestamps",
                 labels_field="labels",
                 embeddings=None,  # Unused.
                 max_time_delta=None,  # Unused.
                 ):
        super().__init__()
        self._timestamps_field = timestamps_field
        self._labels_field = labels_field
        self.encoder = encoder
        self.model = model
        self.max_length = max_length
        self.max_summary_tokens = max_summary_tokens
        self.max_output_tokens = max_output_tokens
        self.summary_input_prompt = summary_input_prompt
        self.summary_query_prompt = summary_query_prompt
        self.input_prompt = input_prompt
        self.query_prompt = query_prompt
        self.temperature = temperature

    @property
    def hidden_size(self):
        config = self.model.llm_engine.model_config.hf_config
        return config.num_key_value_heads * config.head_dim

    def forward(self, x, return_full_states=False):
        timestamps = x.payload[self._timestamps_field]  # (B, L).
        labels = x.payload[self._labels_field]  # (B, L).
        prompts = []
        for i in range(len(x)):
            l = x.seq_lens[i]
            s_ts = timestamps[i:i + 1, :l]
            s_l = labels[i:i + 1, :l]
            if (self.max_length is not None) and (l > self.max_length):
                l = self.max_length
                s_ts = s_ts[:, -l:]
                s_l = s_l[:, -l:]
            text = self.encoder(PaddedBatch({self._timestamps_field: s_ts, self._labels_field: s_l}, torch.full_like(x.seq_lens[i:i + 1], l)))[0]
            prompts.append(text)
        embeddings = get_embeddings(self.model, prompts, conf=self).payload  # (B, L, N, D).
        assert embeddings.ndim == 4
        # TODO: Check emb lengths.
        assert embeddings.shape[1] == 1, "Only single-token embeddings are supported."
        embeddings = embeddings.squeeze(1)  # (B, N, D).
        embeddings = PaddedBatch(embeddings.float(), torch.full_like(x.seq_lens, embeddings.shape[1]))
        return embeddings, None  # (B, N, D).


class Identity(torch.nn.Identity):
    def __init__(self, dim):
        super().__init__()
        self.input_size = dim


class LLMModule(BaseModule):
    """The model copies last seen events to the future.

    The model doesn't require training.

    Parameters.
        k: History length.
        val_metric: Validation set metric.
        test_metric: Test set metric.
        kwargs: Ignored (keep for compatibility with base module).
    """
    def __init__(self, seq_encoder,
                 timestamps_field="timestamps",
                 labels_field="labels",
                 val_metric=None,
                 test_metric=None,
                 **kwargs):
        super().__init__(seq_encoder=seq_encoder,
                         loss=Identity(2),
                         timestamps_field=timestamps_field,
                         labels_field=labels_field,
                         head_partial=lambda input_size, output_size: Identity(2),
                         optimizer_partial=lambda parameters: torch.optim.Adam(parameters, lr=0.001),  # Not used.
                         lr_scheduler_partial=lambda optimizer: torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=1),  # Not used.
                         val_metric=val_metric,
                         test_metric=test_metric)
        self.dummy = torch.nn.Parameter(torch.zeros(1))
