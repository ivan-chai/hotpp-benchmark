from collections import defaultdict

import torch
from hotpp.data import PaddedBatch
from .base_encoder import BaseEncoder
from .transformer import TransformerState


class TransformerEncoder(BaseEncoder):
    """Transformer sequence encoder.

    Args:
        embeddings: Dict with categorical feature names. Values must be like this `{'in': dictionary_size, 'out': embedding_size}`.
        transformer_partial: transformer decoder constructor with a single `input_dim` parameter.
        timestamps_field: The name of the timestamps field.
        max_time_delta: Limit maximum time delta at the model input.
        max_context: Maximum prefix length.
        embedder_batch_norm: Use batch normalization in embedder.
    """
    def __init__(self,
                 embeddings,
                 transformer_partial,
                 timestamps_field="timestamps",
                 labels_field="labels",
                 max_time_delta=None,
                 max_context=None,
                 embedder_batch_norm=True,
                 ):
        super().__init__(
            embeddings=embeddings,
            timestamps_field=timestamps_field,
            max_time_delta=max_time_delta,
            embedder_batch_norm=embedder_batch_norm
        )
        self._labels_field = labels_field
        self.transformer = transformer_partial(self.embedder.output_size)
        self.max_context = max_context

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
        outputs, states = self(x, return_full_states=True)  # (B, L, D), (N, B, L, D).

        sequences = []
        masks = []
        max_index = indices.seq_lens.max().item() - 1
        lengths1 = torch.ones_like(outputs.seq_lens)
        for i in range(indices.shape[1]):
            valid_mask = i < indices.seq_lens  # (B).
            masks.append(valid_mask)  # (B).
            if i > max_index:
                continue
            subset_indices = indices.payload[valid_mask, i]
            subset_lens = subset_indices + 1
            max_length = subset_lens.max().item()
            subset_outputs = outputs.payload[valid_mask].take_along_dim(subset_indices[:, None, None], 1)  # (V, 1, D).
            subset_outputs = PaddedBatch(subset_outputs, lengths1[:len(subset_outputs)])  # (V, 1, D).
            subset_states = states[:, valid_mask].take_along_dim(subset_indices[None, :, None, None], 2)  # (N, V, 1, D).
            subset_states.seq_lens = subset_indices + 1
            sequences.append(self._generate_autoreg(subset_outputs, subset_states, predict_fn, n_steps))  # (V, N) or (V, N, C).
        masks = torch.stack(masks)  # (I, B).
        joined_sequences = {k: torch.cat([s.payload[k] for s in sequences]) for k in sequences[0].payload}  # (IV, N) or (IV, N, C).
        sequences = {k: torch.zeros(indices.shape[1], indices.shape[0], n_steps,
                                    dtype=v.dtype, device=v.device).masked_scatter_(masks.unsqueeze(-1), v)
                     for k, v in joined_sequences.items() if v.ndim == 2}  # (I, B, N).
        sequences |= {k: torch.zeros(indices.shape[1], indices.shape[0], n_steps, v.shape[2],
                                     dtype=v.dtype, device=v.device).masked_scatter_(masks.unsqueeze(-1).unsqueeze(-1), v)
                      for k, v in joined_sequences.items() if v.ndim == 3}  # (I, B, N).
        sequences = {k: (v.permute(1, 0, 2) if v.ndim == 3 else v.permute(1, 0, 2, 3)) for k, v in sequences.items()}  # (B, I, N) or (B, I, N, C).
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
