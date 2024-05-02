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
      batch: Batch of inputs.
      indices: Output prediction indices for each element of the batch (list of lists).

    Outputs:
      Predicted sequences as a list of PaddedBatches each with shape (I, T), where I is the number of indices.
    """

    def __init__(self, model: BaseRNNAdapter, max_steps):
        super().__init__(max_steps)
        self.model = model

    def forward(self, batch: PaddedBatch, indices: PaddedBatch) -> List[PaddedBatch]:
        if batch.left:
            raise NotImplementedError("Left-padded batches are not implemented.")
        batch = self.model.prepare_features(batch)
        initial_states, initial_features = self._get_initial_states(batch, indices)

        # Flatten batches.
        mask = initial_states.seq_len_mask.bool()
        initial_states = initial_states.payload[mask].unsqueeze(1)  # (B * I, 1, D).
        lengths = torch.ones(len(initial_states), device=indices.device, dtype=torch.long)
        initial_states = PaddedBatch(initial_states, lengths)
        initial_features = PaddedBatch({k: (v[mask].unsqueeze(1) if k in initial_features.seq_names else v)
                                        for k, v in initial_features.payload.items()},
                                       lengths, initial_features.seq_names)  # (B * I, 1, D).

        # Predict.
        sequences = self._generate(initial_states, initial_features)  # (B * I, T, D).
        sequences = self.model.revert_features(sequences)  # (B * I, T, D).

        # Split batches
        results = []
        start = 0
        for l in indices.seq_lens:
            results.append(PaddedBatch({k: v[start:start + l] for k, v in sequences.payload.items()},
                                       sequences.seq_lens[start:start + l],
                                       sequences.seq_names))
            start += l
        assert start == len(sequences)
        return results

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
        seq_names = features.seq_names
        assert t == 1
        states = states.payload.squeeze(1)  # (B, D).
        outputs = defaultdict(list)
        outputs.update({k: v for k, v in features.payload.items()
                        if k not in seq_names})
        for _ in range(self.max_steps):
            lengths = features.seq_lens
            predictions, states = self.model(features, states)
            time_deltas, features = self.model.output_to_next_input(features, predictions)
            for k in seq_names:
                outputs[k].append(predictions.get(k, features[k]))
                if k in features:
                    features[k] = features[k].unsqueeze(1)  # (B, T).
            features = PaddedBatch(features, torch.ones_like(lengths), seq_names)
        for k in list(outputs):
            if isinstance(outputs[k], list):
                outputs[k] = torch.stack(outputs[k], dim=1)  # (B, T, D).
        lengths = torch.full([batch_size], self.max_steps, device=states.device)  # (B).
        outputs = PaddedBatch(dict(outputs), lengths, features.seq_names)
        return outputs
