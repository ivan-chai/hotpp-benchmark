from collections import defaultdict

import torch
from hotpp.data import PaddedBatch
from hotpp.utils.torch import deterministic
from .base_encoder import BaseEncoder


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


class Encoder(BaseEncoder):
    """Simple sequence encoder.

    Args:
        embedder: An instance of embedder Class for input events encoding.
        model_partial: Model constructor with a single `input_dim` parameter.
        timestamps_field: The name of the timestamps field.
        max_time_delta: Limit maximum time delta at the model input.
        max_context: Maximum prefix length.
        autoreg_batch_size: Apply auto-reg in the batched mode.
    """
    def __init__(self,
                 embedder,
                 model_partial,
                 timestamps_field="timestamps",
                 time_to_delta=True,
                 max_time_delta=None,
                 max_context=None,
                 autoreg_batch_size=None
                 ):
        super().__init__(
            embedder=embedder,
            timestamps_field=timestamps_field,
            max_time_delta=max_time_delta
        )
        self.model = model_partial(self.embedder.output_size)
        self.max_context = max_context
        self.autoreg_batch_size = autoreg_batch_size

    @property
    def need_states(self):
        return False

    @property
    def hidden_size(self):
        return self.model.output_size

    def embed(self, x):
        """Extract embeddings with shape (B, D)."""
        if not hasattr(self.model, "embed"):
            raise NotImplementedError("The model doesn't support embeddings extraction.")
        times = (self.compute_time_deltas(x) if self.model.delta_time else x)[self._timestamps_field]  # (B, L).
        x = self.apply_embedder(x)
        return self.model.embed(x, times)  # (B, D).

    def forward(self, x, return_states=False):
        """Apply the encoder network.

        Args:
            x: PaddedBatch with input features.
            return_states: Whether to return final states with shape (B, D), full states with shape (B, T, D)
                or no states (either False, "last" or "full").

        Returns:
            Outputs is with shape (B, T, D) and states with shape (N, B, D) or (N, B, T, D).
        """
        times = (self.compute_time_deltas(x) if self.model.delta_time else x)[self._timestamps_field]  # (B, L).
        embeddings = self.apply_embedder(x)
        outputs, states = self.model(embeddings, times, return_states=return_states)   # (B, L, D), (N, B, L, D).
        return outputs, states

    def interpolate(self, states, time_deltas):
        """Apply decoder.

        Args:
            states: Last model states with shape (N, B, L, D).
            time_deltas: Relative timestamps with shape (B, L, S), where S is a sample size.

        Returns:
            Outputs with shape (B, L, S, D).
        """
        if not self.model.delta_time:
            raise NotImplementedError("Interpolation is not supported for general models with raw time input.")
        return self.model.interpolate(states, time_deltas)

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
        predictions = defaultdict(list)
        sequences = []
        lengths = []
        batch_size = self.autoreg_batch_size if self.autoreg_batch_size is not None else len(prefix_indices)
        for start in range(0, len(prefix_indices), batch_size):
            stop = start + batch_size
            bucket_indices = prefix_indices[start:stop, 0]
            bucket_lengths = prefix_lengths[start:stop]
            max_length = bucket_lengths.max().item()
            # Note, that bucket_lengths are strictly positive.
            bucket_prefixes = PaddedBatch({k: (v[bucket_indices][:, :max_length] if k in x.seq_names else v[bucket_indices]) for k, v in x.payload.items()},
                                          bucket_lengths, seq_names=x.seq_names)  # (B, L).
            if self.max_context is not None:
                bucket_prefixes = PaddedBatch({k: (limit_history(v.unsqueeze(-1), bucket_prefixes.seq_lens, self.max_context)[0].squeeze(-1)
                                                   if k in bucket_prefixes.seq_names else v)
                                               for k, v in bucket_prefixes.payload.items()},
                                              bucket_prefixes.seq_lens.clip(max=self.max_context),
                                              seq_names=bucket_prefixes.seq_names)  # (B, L).
            bucket_predictions = self._generate_autoreg(bucket_prefixes, predict_fn, n_steps)  # (V, N) or (V, N, C).
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

    def _generate_autoreg(self, prefixes, predict_fn, n_steps):
        # prefixes: (B, L).
        # predict_fn: (B, L, D), None -> (B, L).
        embeddings = self.apply_embedder(prefixes)  # (B, L, D).
        b, l, d = embeddings.payload.shape
        extra_space = torch.zeros(b, n_steps, d, dtype=embeddings.payload.dtype, device=embeddings.device)
        embeddings = PaddedBatch(torch.cat([embeddings.payload, extra_space], dim=1), embeddings.seq_lens.clone())  # (B, L + N, D).

        last_times = prefixes.payload[self._timestamps_field].take_along_dim((prefixes.seq_lens - 1).clip(min=0)[:, None], dim=1)  # (B, 1).
        times = (self.compute_time_deltas(prefixes) if self.model.delta_time else prefixes)[self._timestamps_field]  # (B, L).
        extra_space = torch.zeros(b, n_steps, dtype=times.payload.dtype, device=times.device)
        times = PaddedBatch(torch.cat([times.payload, extra_space], dim=1), times.seq_lens)  # (B, L + N).

        lengths1 = torch.ones_like(prefixes.seq_lens)

        outputs = defaultdict(list)
        for i in range(n_steps):
            if self.max_context is not None:
                start = max(l + i - self.max_context, 0)
            else:
                start = 0
            truncated_embeddings = PaddedBatch(embeddings.payload[:, start:l + i], embeddings.seq_lens - start)
            truncated_times = PaddedBatch(times.payload[:, start:l + i], times.seq_lens - start)
            model_outputs, _ = self.model(truncated_embeddings, truncated_times, return_states=False)  # (B, L', D).
            last_outputs = model_outputs.payload.take_along_dim((model_outputs.seq_lens - 1).clip(min=0)[:, None, None], dim=1)  # (B, 1, D).
            last_outputs = PaddedBatch(last_outputs, lengths1)
            features = predict_fn(last_outputs, None)  # (B, 1).
            for k, v in features.payload.items():
                outputs[k].append(v.squeeze(1))  # (B).
            if i == n_steps - 1:
                break
            # Convert predicted delta to absolute time for the next input if necessary.
            last_times += features.payload[self._timestamps_field]

            new_times =  features.payload[self._timestamps_field] if self.model.delta_time else last_times
            times.payload.scatter_(1, times.seq_lens[:, None], new_times)
            times.seq_lens.add_(1)

            # The timestamps field at the model output already contains deltas.
            new_embeddings = self.apply_embedder(features, compute_time_deltas=False)  # (B, 1, D).
            embeddings.payload.scatter_(1, embeddings.seq_lens[:, None, None].expand(b, 1, d), new_embeddings.payload)
            embeddings.seq_lens.add_(1)

        for k in list(outputs):
            outputs[k] = torch.stack(outputs[k], dim=1)  # (B, T, D).
        lengths = torch.full([len(last_outputs)], n_steps, device=last_outputs.device)  # (B).
        outputs = PaddedBatch(dict(outputs), lengths, seq_names=list(outputs))
        return outputs
