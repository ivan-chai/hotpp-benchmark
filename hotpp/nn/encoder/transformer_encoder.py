from collections import defaultdict

import torch
from hotpp.data import PaddedBatch
from hotpp.utils.torch import deterministic
from .base_encoder import BaseEncoder
from .transformer import TransformerState


def limit_history(x, seq_lens, max_length=None):
    """Limit maximum history size.

    Args:
        x: Embeddings with shape (B, L, D) or (N, B, L, D).
        seq_lens: Input sequence lengths with shape (B).
        max_length: Maximum output sequence length.

    Returns:
        New embeddings with shape (B, L', D) and new lengths with shape (B).
    """
    if max_length is None:
        return x, seq_lens
    max_length = min(max_length, seq_lens.max().item())
    b, l = x.shape[-3:-1]
    exclude = (seq_lens - max_length).clip(min=0)  # (B).
    indices = (torch.arange(max_length, device=x.device)[None] + exclude[:, None]).clip(max=l - 1)  # (B, L').
    shape = [1] * (x.ndim - 3) + [b, max_length, 1]
    indices = indices.reshape(*shape)  # (*, B, L', 1).
    result = x.take_along_dim(indices, -2)  # (*, B, L, D).
    return result, seq_lens.clip(max=max_length)


class TransformerEncoder(BaseEncoder):
    """Transformer sequence encoder.

    Args:
        embeddings: Dict with categorical feature names. Values must be like this `{'in': dictionary_size, 'out': embedding_size}`.
        transformer_partial: transformer decoder constructor with a single `input_dim` parameter.
        timestamps_field: The name of the timestamps field.
        max_time_delta: Limit maximum time delta at the model input.
        max_context: Maximum prefix length.
        embedder_batch_norm: Use batch normalization in embedder.
        autoreg_batch_size: Apply auto-reg in the batched mode.
    """
    def __init__(self,
                 embeddings,
                 transformer_partial,
                 timestamps_field="timestamps",
                 max_time_delta=None,
                 max_context=None,
                 embedder_batch_norm=True,
                 autoreg_batch_size=None
                 ):
        super().__init__(
            embeddings=embeddings,
            timestamps_field=timestamps_field,
            max_time_delta=max_time_delta,
            embedder_batch_norm=embedder_batch_norm
        )
        self.transformer = transformer_partial(self.embedder.output_size)
        self.max_context = max_context
        self.autoreg_batch_size = autoreg_batch_size

    @property
    def hidden_size(self):
        return self.transformer.output_size

    def forward(self, x, return_full_states=False):
        """Apply encoder network.

        Args:
            x: PaddedBatch with input features.
            return_full_states: Whether to return full states with shape (B, T, D)
                or only final states with shape (B, D).

        Returns:
            Outputs is with shape (B, T, D) and states with shape (N, B, D) or (N, B, T, D).
        """
        times = x[self._timestamps_field]  # (B, L).
        embeddings = self.embed(x, compute_time_deltas=False)
        outputs, states = self.transformer(embeddings, times)   # (B, L, D), (N, B, L, D).
        if not return_full_states:
            states = states.take_along_dim((states.seq_lens - 1)[None, :, None, None], 2).squeeze(2)  # (N, B, D).
        return outputs, states

    def interpolate(self, states, time_deltas):
        """Apply decoder.

        Args:
            states: Last model states with shape (N, B, L, D).
            time_deltas: Relative timestamps with shape (B, L, S), where S is a sample size.

        Returns:
            Outputs with shape (B, L, S, D).
        """
        b, l, s = time_deltas.payload.shape
        times = PaddedBatch(time_deltas.payload + states.index_times.payload.unsqueeze(-1), time_deltas.seq_lens)  # (B, L, S).
        if l > 1:
            last_history_index = torch.arange(l, device=time_deltas.device)  # (L).
        else:
            last_history_index = None
        outputs = self.transformer.interpolate(times, states, last_history_index)  # (B, L, S, D).
        return outputs

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
        initial_timestamps = x.payload[self._timestamps_field].take_along_dim(indices.payload, dim=1)  # (B, I).
        index_mask = indices.seq_len_mask  # (B, I).

        # Create buckets by sequence length.
        prefix_indices = torch.nonzero(index_mask)  # (V, 2).
        order = torch.argsort(indices.payload[prefix_indices[:, 0], prefix_indices[:, 1]])  # (V).
        prefix_indices = prefix_indices[order]  # (V, 2).
        prefix_lengths = indices.payload[prefix_indices[:, 0], prefix_indices[:, 1]] + 1  # (V).

        # Generate.
        outputs, states = self(x, return_full_states=True)  # (B, L, D), (N, B, L, D).
        predictions = defaultdict(list)
        sequences = []
        lengths = []
        batch_size = self.autoreg_batch_size if self.autoreg_batch_size is not None else len(prefix_indices)
        for start in range(0, len(prefix_indices), batch_size):
            stop = start + batch_size
            bucket_indices = prefix_indices[start:stop, 0]
            bucket_lengths = prefix_lengths[start:stop]
            max_length = bucket_lengths.max().item()
            # Note, that prefix_lengths are strictly positive.
            bucket_outputs = PaddedBatch(outputs.payload[bucket_indices].take_along_dim(bucket_lengths[:, None, None] - 1, 1),
                                         torch.ones_like(bucket_indices))  # (V, 1, D).
            bucket_times = limit_history(states.times[bucket_indices][:, :max_length].unsqueeze(-1), bucket_lengths, self.max_context)[0].squeeze(-1)
            bucket_states, states_lengths = limit_history(states.payload[:, bucket_indices, :max_length], bucket_lengths, self.max_context)
            bucket_states = TransformerState(
                bucket_times, bucket_states, states_lengths,
                index=states_lengths[:, None] - 1,
                index_lens=torch.ones_like(states_lengths)
            )
            bucket_predictions = self._generate_autoreg(bucket_outputs, bucket_states, predict_fn, n_steps)  # (V, N) or (V, N, C).
            assert (bucket_predictions.seq_lens == n_steps).all()
            for k, v in bucket_predictions.payload.items():
                predictions[k].append(v)
        predictions = {k: torch.cat(v) for k, v in predictions.items()}  # (V, N) or (V, N, C).

        # Gather results.
        iorder = torch.argsort(order)
        predictions = {k: v[iorder] for k, v in predictions.items()}  # (V, N) or (V, N, C).
        sequences = {k: torch.zeros(indices.shape[0], indices.shape[1], n_steps,
                                    dtype=v.dtype, device=v.device).masked_scatter_(index_mask.unsqueeze(-1), v)
                     for k, v in predictions.items() if v.ndim == 2}  # (B, I, N).
        sequences |= {k: torch.zeros(indices.shape[0], indices.shape[1], n_steps, v.shape[2],
                                     dtype=v.dtype, device=v.device).masked_scatter_(index_mask.unsqueeze(-1).unsqueeze(-1), v)
                      for k, v in predictions.items() if v.ndim == 3}  # (B, I, N, C).

        # Revert deltas.
        with deterministic(False):
            sequences[self._timestamps_field].cumsum_(2)
        sequences[self._timestamps_field] += initial_timestamps.unsqueeze(-1)
        return PaddedBatch(sequences, indices.seq_lens)

    def _generate_autoreg(self, last_outputs, last_states, predict_fn, n_steps):
        # outputs: (B, 1, D).
        # states: (N, B, 1, D).
        # predict_fn: (B, L, D), (N, B, L, D) -> (B, L).
        assert last_outputs.payload.ndim == 3 and last_outputs.payload.shape[1] == 1
        assert last_states.ndim == 4 and last_states.shape[2] == 1
        outputs = defaultdict(list)
        times = last_states.index_times
        for i in range(n_steps):
            features = predict_fn(last_outputs, last_states)  # (B, 1).
            for k, v in features.payload.items():
                outputs[k].append(v.squeeze(1))  # (B).
            if i == n_steps - 1:
                break
            # Convert predicted delta to absolute time for the next input.
            times.payload += features.payload[self._timestamps_field]
            features.payload[self._timestamps_field] = times.payload
            embeddings = self.embed(features, compute_time_deltas=False)  # (B, 1, D).
            last_outputs, new_states = self.transformer.decode(embeddings, times, last_states)  # (B, 1, D), (N, B, 1, D).
            # Append to the beginning, since transformer doesn't take order into account.
            last_states = TransformerState(
                times=torch.cat([new_states.times, last_states.times], 1),  # (B, 1 + L).
                states=torch.cat([new_states.payload, last_states.payload], 2),  # (N, B, 1 + L, D).
                seq_lens=last_states.seq_lens + 1,
                index=last_states.index + 1,
                index_lens=last_states.index_lens)
        for k in list(outputs):
            outputs[k] = torch.stack(outputs[k], dim=1)  # (B, T, D).
        lengths = torch.full([len(last_outputs)], n_steps, device=last_outputs.device)  # (B).
        outputs = PaddedBatch(dict(outputs), lengths, seq_names=list(outputs))
        return outputs
