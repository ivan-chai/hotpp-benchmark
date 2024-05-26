from collections import defaultdict

import torch
from .base_encoder import BaseEncoder

from esp_horizon.data import PaddedBatch
from esp_horizon.utils.torch import deterministic
from .window import apply_windows
from .rnn import GRU, ContTimeLSTM


class RnnEncoder(BaseEncoder):
    """RNN sequecne encoder.

    Args:
        embeddings: Dict with categorical feature names. Values must be like this `{'in': dictionary_size, 'out': embedding_size}`.
        timestamps_field: The name of the timestamps field.
        rnn_type: Type of the model (`gru` or `cont-lstm`).
        hidden_size: The size of the hidden layer.
        num_layers: The number of layers.
        max_inference_context: Maximum RNN context for long sequences.
        inference_context_step: Window step when max_context is provided.
    """
    def __init__(self,
                 embeddings,
                 timestamps_field="timestamps",
                 rnn_type="gru",
                 hidden_size=None,
                 num_layers=1,
                 max_inference_context=None, inference_context_step=None,
                 ):
        if hidden_size is None:
            raise ValueError("Hidden size must be provided.")
        super().__init__(
            embeddings=embeddings,
            timestamps_field=timestamps_field
        )
        self._hidden_size = hidden_size
        self._num_layers = num_layers
        self._max_context = max_inference_context
        self._context_step = inference_context_step
        if rnn_type == "gru":
            self.rnn = GRU(
                self.embedder.output_size,
                hidden_size,
                num_layers=num_layers
            )
        elif rnn_type == "cont-time-lstm":
            self.rnn = ContTimeLSTM(
                self.embedder.output_size,
                hidden_size,
                num_layers=num_layers
            )
        else:
            raise ValueError(f"Unknown RNN type: {rnn_type}")

    @property
    def hidden_size(self):
        return self._hidden_size

    @property
    def num_layers(self):
        return self._num_layers

    def forward(self, x, return_full_states=False):
        """Apply RNN model.

        Args:
            x: PaddedBatch with input features.
            return_full_states: Whether to return full states with shape (B, T, D)
                or only final states with shape (B, D).

        Returns:
            Outputs is with shape (B, T, D) and states with shape (N, B, D) or (N, B, T, D).
        """
        x = self.compute_time_deltas(x)
        time_deltas = x.payload[self._timestamps_field]
        embeddings = self.embed(x, compute_time_deltas=False)
        outputs, states = self.rnn(embeddings.payload, time_deltas, return_full_states=return_full_states)
        return PaddedBatch(outputs, embeddings.seq_lens), states

    def interpolate(self, states, time_deltas):
        """Compute layer output for continous time.

        Args:
            states: Last model states with shape (N, B, L, D).
            time_deltas: Relative timestamps with shape (B, L).

        Returns:
            Outputs with shape (B, L, D).
        """
        result = self.rnn.interpolate(states.payload, time_deltas.payload)
        return PaddedBatch(result, states.seq_lens)

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

        initial_timestamps = x.payload[self._timestamps_field].take_along_dim(indices.payload, dim=1)  # (B, I).

        # Compute time deltas and save initial times.
        x = self.compute_time_deltas(x)

        # Select input state and initial feature for each index position.
        initial_states, initial_features = self._get_initial_states(x, indices)  # (N, B, I, D), (B, I).

        # Flatten batches.
        mask = initial_features.seq_len_mask.bool()  # (B, I).
        initial_timestamps = initial_timestamps[mask].unsqueeze(1)  # (B * I, 1).
        initial_states = initial_states.masked_select(mask[None, :, :, None]).reshape(
            len(initial_states), -1, 1, initial_states.shape[-1])  # (N, B * I, 1, D).
        lengths = torch.ones(len(initial_states), device=indices.device, dtype=torch.long)
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
        time_deltas = PaddedBatch(batch.payload[self._timestamps_field], batch.seq_lens)
        indices, seq_lens = indices.payload, indices.seq_lens

        if self.num_layers != 1:
            raise NotImplementedError("Only single-layer RNN is supported.")
        # GRU states are equal to GRU outputs.
        embeddings = self.embed(batch, compute_time_deltas=False)  # (B, T, D).
        next_states = apply_windows((embeddings, time_deltas),
                                    lambda xe, xt: PaddedBatch(self.rnn(xe.payload, xt.payload, return_full_states=True)[1].squeeze(0),
                                                               xe.seq_lens),
                                    self._max_context, self._context_step).payload[None]  # (N, B, T, D).

        initial_state = torch.zeros_like(next_states[:, :, :1])  # (N, B, 1, LD).
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
            time_deltas = features.payload[self._timestamps_field]
            embeddings = self.embed(features, compute_time_deltas=False).payload  # (B, 1, D).
            embeddings, states = self.rnn(embeddings, time_deltas, states=states)  # (B, 1, D), (N, B, D).
            embeddings = PaddedBatch(embeddings, features.seq_lens)
            features = predict_fn(embeddings, states.unsqueeze(2))  # (B, 1).
            for k, v in features.payload.items():
                outputs[k].append(v.squeeze(1))  # (B).
        for k in list(outputs):
            outputs[k] = torch.stack(outputs[k], dim=1)  # (B, T, D).
        lengths = torch.full([batch_size], n_steps, device=device)  # (B).
        outputs = PaddedBatch(static_outputs | dict(outputs), lengths, seq_names=list(outputs))
        return outputs
