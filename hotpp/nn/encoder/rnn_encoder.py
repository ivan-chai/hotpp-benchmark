from collections import defaultdict

import torch
from .base_encoder import BaseEncoder

from hotpp.data import PaddedBatch
from hotpp.utils.torch import deterministic
from .window import apply_windows


class RnnEncoder(BaseEncoder):
    """Optimized RNN sequence encoder.

    The encoder is primarily used with a single-layer GRU backbone, which allows
    effective intermediate state computation and multi-prefix inference. In other
    cases prefere general Encoder class.

    Args:
        embedder: An instance of embedder Class for input events encoding.
        rnn_partial: RNN constructor with a single `input_dim` parameter.
        timestamps_field: The name of the timestamps field.
        max_time_delta: Limit maximum time delta at the model input.
        max_inference_context: Maximum RNN context for long sequences.
        inference_context_step: Window step when max_context is provided.
    """
    def __init__(self,
                 embedder,
                 rnn_partial,
                 timestamps_field="timestamps",
                 max_time_delta=None,
                 max_inference_context=None, inference_context_step=None,
                 ):
        super().__init__(
            embedder=embedder,
            timestamps_field=timestamps_field,
            max_time_delta=max_time_delta
        )
        self._max_context = max_inference_context
        self._context_step = inference_context_step
        self.rnn = rnn_partial(self.embedder.output_size)
        if not self.rnn.delta_time:
            raise NotImplementedError("Only RNNs with time delta at input are supported.")

    @property
    def need_states(self):
        return True

    @property
    def hidden_size(self):
        return self.rnn.output_size

    @property
    def num_layers(self):
        return self.rnn.num_layers

    def forward(self, x, return_states=False):
        """Apply RNN.

        Args:
            x: PaddedBatch with input features.
            return_states: Whether to return final states with shape (B, D), full states with shape (B, T, D)
                or no states (either False, "last" or "full").

        Returns:
            Outputs is with shape (B, T, D) and states with shape (N, B, D) or (N, B, T, D).
        """
        x = self.compute_time_deltas(x)
        time_deltas = x[self._timestamps_field]
        embeddings = self.apply_embedder(x, compute_time_deltas=False)
        outputs, states = self.rnn(embeddings, time_deltas, return_states=return_states)
        return outputs, states

    def interpolate(self, states, time_deltas):
        """Compute layer output for continous time.

        Args:
            states: Last model states with shape (N, B, L, D).
            time_deltas: Relative timestamps with shape (B, L, S), where S is a sample size.

        Returns:
            Outputs with shape (B, L, S, D).
        """
        return self.rnn.interpolate(states, time_deltas)

    def generate(self, x, indices, predict_fn, n_steps):
        """Use auto-regression to generate future sequence.

        Args:
            x: Batch of inputs with shape (B, T).
            indices: Output prediction indices for each element of the batch with shape (B, I).
            predict_fn: A mapping from embedding to input features.
            n_steps: The maximum number of generated items.

        Returns:
            Predicted sequences as a batch with the shape (B, I, N), where I is the number of indices and N is the number of steps.
        """
        batch_size, index_size = indices.shape

        # Compute time deltas and save initial times.
        initial_timestamps = x.payload[self._timestamps_field].take_along_dim(indices.payload, dim=1)  # (B, I).
        x = self.compute_time_deltas(x)

        # Select input state and initial feature for each index position.
        initial_states, initial_features = self._get_initial_states(x, indices)  # (N, B, I, D), (B, I).

        # Flatten batches.
        mask = initial_features.seq_len_mask.bool()  # (B, I).
        initial_timestamps = initial_timestamps[mask].unsqueeze(1)  # (B * I, 1).
        initial_states = initial_states.masked_select(mask[None, :, :, None]).reshape(
            len(initial_states), -1, 1, initial_states.shape[-1])  # (N, B * I, 1, D).
        lengths = torch.ones(initial_states.shape[1], device=indices.device, dtype=torch.long)
        initial_features = PaddedBatch({k: (v[mask].unsqueeze(1) if k in initial_features.seq_names else v)
                                        for k, v in initial_features.payload.items()},
                                       lengths, initial_features.seq_names)  # (B * I, 1).

        # Predict.
        sequences = self._generate_autoreg(initial_states, initial_features, predict_fn, n_steps)  # (B * I, N).

        # Revert deltas.
        with deterministic(False):
            sequences.payload[self._timestamps_field].cumsum_(1)
        sequences.payload[self._timestamps_field] += initial_timestamps

        # Gather results.
        mask = indices.seq_len_mask.bool()  # (B, I).
        payload = {}
        for k, v in sequences.payload.items():
            if k not in sequences.seq_names:
                payload[k] = v
                continue
            dims = [batch_size, index_size, n_steps] + list(v.shape[2:])
            zeros = torch.zeros(*dims, device=v.device, dtype=v.dtype)
            broad_mask = mask.reshape(*(list(mask.shape) + [1] * (zeros.ndim - mask.ndim)))  # (B, I, *).
            payload[k] = zeros.masked_scatter_(broad_mask, v)
        return PaddedBatch(payload, indices.seq_lens, sequences.seq_names)

    def _get_initial_states(self, batch, indices):
        time_deltas = batch[self._timestamps_field]
        indices, seq_lens = indices.payload, indices.seq_lens

        if self.num_layers != 1:
            raise NotImplementedError("Only single-layer RNN is supported.")
        embeddings = self.apply_embedder(batch, compute_time_deltas=False)  # (B, T, D).
        next_states = apply_windows((embeddings, time_deltas),
                                    lambda xe, xt: PaddedBatch(self.rnn(xe, xt, return_states="full")[1].squeeze(0),
                                                               xe.seq_lens),
                                    self._max_context, self._context_step).payload[None]  # (N, B, T, D).

        initial_state = self.rnn.init_state[:, None, None, :].repeat(1, len(batch), 1, 1)  # (N, B, 1, LD).
        input_states = torch.cat([initial_state, next_states[:, :, :-1]], dim=2)  # (N, B, T, LD).
        input_states = input_states.take_along_dim(indices[None, :, :, None], 2)  # (N, B, I, LD).

        input_features = PaddedBatch({k: (v.take_along_dim(indices, 1) if k in batch.seq_names else v)
                                      for k, v in batch.payload.items()},
                                     seq_lens,
                                     batch.seq_names)  # (B, I).
        return input_states, input_features

    def _generate_autoreg(self, states, features, predict_fn, n_steps):
        # states: (N, B, 1, D), where N is the number of layers.
        # features: (B, 1).
        assert states.shape[2] == 1
        assert features.shape[1] == 1
        batch_size = states.shape[1]
        device = states.device
        seq_names = set(features.seq_names)
        states = states.squeeze(2)  # (N, B, D).

        static_outputs = {k: v for k, v in features.payload.items()
                          if k not in seq_names}
        outputs = defaultdict(list)
        for _ in range(n_steps):
            time_deltas = features[self._timestamps_field]
            embeddings = self.apply_embedder(features, compute_time_deltas=False)  # (B, 1, D).
            embeddings, states = self.rnn(embeddings, time_deltas, states=states, return_states="last")  # (B, 1, D), (N, B, D).
            features = predict_fn(embeddings, states.unsqueeze(2))  # (B, 1).
            for k, v in features.payload.items():
                outputs[k].append(v.squeeze(1))  # (B).
        for k in list(outputs):
            outputs[k] = torch.stack(outputs[k], dim=1)  # (B, T, D).
        lengths = torch.full([batch_size], n_steps, device=device)  # (B).
        outputs = PaddedBatch(static_outputs | dict(outputs), lengths, seq_names=list(outputs))
        return outputs
