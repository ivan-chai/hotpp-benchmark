import torch
from collections import defaultdict
from typing import List

from esp_horizon.data import PaddedBatch

from .base_adapter import BaseRNNAdapter
from .base_predictor import BaseSequencePredictor


def take_along_dim(features, indices, dim):
    if features.ndim == 3:
        return features.take_along_dim(indices.unsqueeze(2), dim)
    return features.take_along_dim(indices, dim)


class RNNSequencePredictor(BaseSequencePredictor):
    """Autoregressive sequence predictor.

    Args:
        max_steps: The maximum number of generated items.

    Inputs:
        batch: Batch of inputs with shape (B, T).
        indices: Output prediction indices for each element of the batch with shape (B, I).

    Outputs:
        Predicted sequences as a batch with the shape (B, I, N), where I is the number of indices and N is the number of steps.
    """

    def __init__(self, model: BaseRNNAdapter, max_steps):
        super().__init__(max_steps)
        self.model = model

    def forward(self, batch: PaddedBatch, indices: PaddedBatch) -> PaddedBatch:
        if batch.left:
            raise NotImplementedError("Left-padded batches are not implemented.")

        batch_size, index_size = indices.shape

        batch = self.model.prepare_features(batch)
        initial_states, initial_features = self._get_initial_states(batch, indices)

        # Flatten batches.
        mask = initial_states.seq_len_mask.bool()
        initial_states = initial_states.payload[mask].unsqueeze(1)  # (B * I, 1, D).
        lengths = torch.ones(len(initial_states), device=indices.device, dtype=torch.long)
        initial_states = PaddedBatch(initial_states, lengths)
        initial_features = PaddedBatch({k: (v[mask].unsqueeze(1) if k in initial_features.seq_names else v)
                                        for k, v in initial_features.payload.items()},
                                       lengths, initial_features.seq_names)  # (B * I, 1).

        # Predict.
        sequences = self._generate(initial_states, initial_features)  # (B * I, T).
        sequences = self.model.revert_features(sequences)  # (B * I, T).

        # Gather results.
        mask = indices.seq_len_mask.bool()  # (B, I).
        payload = {}
        for k, v in sequences.payload.items():
            if k not in sequences.seq_names:
                payload[k] = v
                continue
            dims = [batch_size, index_size, self.max_steps] + list(v.shape[2:])
            zeros = torch.zeros(*dims, device=batch.device, dtype=v.dtype)
            broad_mask = mask.reshape(*(list(mask.shape) + [1] * (zeros.ndim - mask.ndim)))  # (B, I, *).
            payload[k] = zeros.masked_scatter_(broad_mask, v)
        return PaddedBatch(payload, indices.seq_lens, sequences.seq_names)

    def _get_initial_states(self, batch, indices):
        indices, seq_lens = indices.payload, indices.seq_lens

        states = self.model.eval_states(batch)  # PaddedBatch with (B, T, D).
        assert states.shape == batch.shape
        initial_states = states.payload.take_along_dim(indices.unsqueeze(2), 1)  # (B, I, D).
        assert initial_states.shape[2] == states.payload.shape[2]
        initial_states = PaddedBatch(initial_states, seq_lens)

        initial_features = PaddedBatch({k: (take_along_dim(v, indices, 1) if k in batch.seq_names else v)
                                        for k, v in batch.payload.items()},
                                       seq_lens,
                                       batch.seq_names)  # (B, I, D).
        return initial_states, initial_features

    def _generate(self, states, features):
        batch_size, t, dim = states.payload.shape
        seq_names = set(features.seq_names) | self.model.output_seq_features
        assert t == 1
        states = states.payload.squeeze(1)  # (B, D).
        outputs = defaultdict(list)
        outputs.update({k: v for k, v in features.payload.items()
                        if k not in seq_names})
        for _ in range(self.max_steps):
            lengths = features.seq_lens
            predictions, states = self.model(features, states)
            _, features = self.model.output_to_next_input(features, predictions)
            for k in seq_names & set(features):
                outputs[k].append(features[k])
                features[k] = features[k].unsqueeze(1)  # (B, 1).
            features = PaddedBatch(features, torch.ones_like(lengths), seq_names)  # (B, 1).
        for k in list(outputs):
            if isinstance(outputs[k], list):
                outputs[k] = torch.stack(outputs[k], dim=1)  # (B, T, D).
        lengths = torch.full([batch_size], self.max_steps, device=states.device)  # (B).
        outputs = PaddedBatch(dict(outputs), lengths, seq_names)
        return outputs
