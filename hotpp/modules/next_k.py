from hotpp.data import PaddedBatch
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
        timestamps_field: The name of the timestamps field.
        labels_field: The name of the labels field.
        head_partial: FC head model class which accepts input and output dimensions.
        optimizer_partial:
            optimizer init partial. Network parameters are missed.
        lr_scheduler_partial:
            scheduler init partial. Optimizer are missed.
        val_metric: Validation set metric.
        test_metric: Test set metric.
    """

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
        hiddens, states = self.encode(x)  # (B, L, D), (N, B, L, D).
        init_times = x.payload[self._timestamps_field].take_along_dim(indices.payload, 1)  # (B, I).
        init_times = PaddedBatch({self._timestamps_field: init_times}, indices.seq_lens)
        outputs = self.apply_head(hiddens)  # (B, L, D).
        outputs = PaddedBatch(outputs.payload.take_along_dim(indices.payload.unsqueeze(2), 1),
                              indices.seq_lens)  # (B, I, D).
        states = states.take_along_dim(indices.payload[None, :, :, None], 2)  # (N, B, I, D).
        sequences = self.predict_next_k(init_times, outputs, states, logits_fields_mapping={self._labels_field: self._labels_logits_field})  # (B, I, K) or (B, I, K, C).
        return sequences  # (B, I, K) or (B, I, K, C).
