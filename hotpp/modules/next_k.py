import torch
from hotpp.data import PaddedBatch
from ..fields import LABELS_LOGITS, PRESENCE
from .base_module import BaseModule


class NextKModule(BaseModule):
    """Train the model for next K events prediction.

    The model is composed of the following modules:
    1. input encoder, responsible for input-to-vector conversion,
    2. sequential encoder, which captures time dependencies,
    3. fc head for embeddings projection (optional),
    4. loss, which estimates likelihood and predictions.

    Input encoder and sequential encoder are combined within SeqEncoder from Pytorch Lifestream.

    Parameters
        seq_encoder: Backbone model, which includes input encoder and sequential encoder.
        loss: Training loss.
        max_predicitions: Limit the maximum number of predictions.
        recurrent_steps: Apply next-k prediction multiple times using autoregression.
    """
    def __init__(self, seq_encoder, loss,
                 max_predictions=None,
                 recurrent_steps=1,
                 **kwargs):
        if recurrent_steps < 1:
            raise ValueError(f"Recurrent steps must be positive, got {recurrent_steps}")
        super().__init__(
            seq_encoder=seq_encoder,
            loss=loss,
            **kwargs
        )
        self._max_predictions = max_predictions
        self._recurrent_steps = recurrent_steps

    @property
    def delta_type(self):
        """Returns the type of time delta computation: `last` or `start`."""
        return self._loss.get_delta_type(self._timestamps_field)

    def predict_next_k(self, inputs, outputs, states, fields=None, logits_fields_mapping=None):
        """Predict K next events from head outputs.

        Args:
            inputs: Input features with shape (B, L).
            outputs: Output of the head module with shape (B, L, D).
            states: Sequence model states with shape (N, B, L, D), where N is the number of layers.
        """
        results = self._loss.predict_next_k(outputs, states,
                                            fields=fields,
                                            logits_fields_mapping=logits_fields_mapping)  # (B, L, K) or (B, L, K, C).
        # Convert delta time to time.
        results.payload[self._timestamps_field] += inputs.payload[self._timestamps_field].unsqueeze(2)
        return results

    def generation_step(self, x, indices):
        """Generate future events (single next-k step).

        Args:
            x: Features with shape (B, L).
            indices: Indices with positions to start generation from with shape (B, I).

        Returns:
            Predicted sequences with shape (B, I, N).
        """
        init_times = x.payload[self._timestamps_field].take_along_dim(indices.payload, 1)  # (B, I).
        init_times = PaddedBatch({self._timestamps_field: init_times}, indices.seq_lens)
        outputs, states = self(x, return_states="full" if self._need_states else False)  # (B, L, D), (N, B, L, D).
        outputs = PaddedBatch(outputs.payload.take_along_dim(indices.payload.unsqueeze(2), 1),
                              indices.seq_lens)  # (B, I, D).
        states = states.take_along_dim(indices.payload[None, :, :, None], 2) if states is not None else states  # (N, B, I, D).
        sequences = self.predict_next_k(init_times, outputs, states, logits_fields_mapping={self._labels_field: LABELS_LOGITS})  # (B, I, K) or (B, I, K, C).
        return sequences  # (B, I, K) or (B, I, K, C).

    def generate_sequences_recurrent(self, x, indices):
        """Generate future events.

        Args:
            x: Features with shape (B, L).
            indices: Indices with positions to start generation from with shape (B, I).

        Returns:
            Predicted sequences with shape (B, I, N).
        """
        input_fields = {field for field in x.payload if field in self._loss.fields}
        # Extract prefixes (BI, L).
        b, l = x.shape
        il = indices.shape[1]
        mask = indices.seq_len_mask.bool()  # (B, I).
        prefixes = {}
        for field in input_fields:
            v = x.payload[field]  # (B, L).
            assert v.ndim == 2
            v = v.unsqueeze(1).expand(b, il, l)  # (B, I, L).
            prefixes[field] = v[mask]  # (BI, L).
        prefixes = PaddedBatch(prefixes, (indices.payload + 1)[mask])  # (BI, L).
        assert (prefixes.seq_lens >= 1).all()

        use_presence = False
        predictions = None
        for step in range(self._recurrent_steps):
            # Utilize the predicted "presence" flag.
            if use_presence:
                # Select valid events.
                presence = prefixes.payload[PRESENCE]  # (BI, L).
                assert presence.dtype == torch.bool
                assert presence.ndim == 2
                # Put valid events to the beginning.
                order = torch.argsort(presence.long(), dim=1, descending=True, stable=True)  # (BI, N).
                input_prefixes = PaddedBatch({k: v.take_along_dim(order, 1) for k, v in prefixes.payload.items()},
                                             presence.sum(1))  # (BI, N).
            else:
                input_prefixes = prefixes

            # Predict (BI, N).
            last = PaddedBatch(input_prefixes.seq_lens.unsqueeze(1) - 1, torch.full_like(input_prefixes.seq_lens, 1))  # (BI, 1).
            sequences = self.generation_step(input_prefixes, last)  # (BI, 1, N).
            n = sequences.payload[self._timestamps_field].shape[2]
            if not predictions:
                predictions = {k: [v.squeeze(1)] for k, v in sequences.payload.items()}  # (BI, N, *).
            else:
                for k in list(predictions):
                    predictions[k].append(sequences.payload[k].squeeze(1))  # (BI, N, *).

            if step == self._recurrent_steps - 1:
                # Skip prefixes update.
                break

            if (not use_presence) and (PRESENCE in sequences.payload):
                # Initialize presence tensor.
                use_presence = True
                prefixes = PaddedBatch(prefixes.payload | {PRESENCE: prefixes.seq_len_mask.bool()},
                                       prefixes.seq_lens)  # (BI, L).

            # Extend (BI, N + L).
            new_prefixes = {}
            for field, v in prefixes.payload.items():
                s = sequences.payload[field]
                assert s.shape[1] == 1
                s = s.squeeze(1)  # (BI, N).
                new_prefixes[field] = torch.cat([v, s], 1)  # (BI, L + N).
            prefixes = PaddedBatch(new_prefixes, prefixes.seq_lens + n)

        # Gather results.
        predictions = {k: torch.cat(v, 1) for k, v in predictions.items()}  # (BI, N, *).
        n = predictions[self._timestamps_field].shape[1]
        sequences = {k: torch.zeros(*([b, il, n] + list(v.shape[2:])), dtype=v.dtype, device=v.device)
                     for k, v in predictions.items()}  # (B, I, N, *).
        for k, v in predictions.items():
            sequences[k].masked_scatter_(mask.reshape(*([b, il, 1] + [1] * (v.ndim - 2))),  # (B, I, 1, *).
                                         v)
        return PaddedBatch(sequences, indices.seq_lens) # (B, I, N).

    def generate_sequences(self, x, indices):
        """Generate future events.

        Args:
            x: Features with shape (B, L).
            indices: Indices with positions to start generation from with shape (B, I).

        Returns:
            Predicted sequences with shape (B, I, N).
        """
        if self._recurrent_steps == 1:
            sequences = self.generation_step(x, indices)
        else:
            sequences = self.generate_sequences_recurrent(x, indices)
        if self._max_predictions is not None:
            sequences = PaddedBatch({k: (v[:, :, :self._max_predictions] if k in sequences.seq_names else v)
                                     for k, v in sequences.payload.items()},
                                    sequences.seq_lens)
        return sequences
