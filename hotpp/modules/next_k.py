from hotpp.data import PaddedBatch
from ..fields import LABELS_LOGITS
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
    """
    def __init__(self, seq_encoder, loss,
                 max_predictions=None,
                 **kwargs):

        super().__init__(
            seq_encoder=seq_encoder,
            loss=loss,
            **kwargs
        )
        self._max_predictions = max_predictions

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

    def generate_sequences(self, x, indices):
        """Generate future events.

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
        if self._max_predictions is not None:
            sequences = PaddedBatch({k: (v[:, :, :self._max_predictions] if k in sequences.seq_names else v)
                                     for k, v in sequences.payload.items()},
                                    sequences.seq_lens)
        return sequences  # (B, I, K) or (B, I, K, C).
