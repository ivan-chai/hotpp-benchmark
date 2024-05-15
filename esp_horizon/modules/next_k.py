from esp_horizon.data import PaddedBatch
from esp_horizon.utils.torch import deterministic
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
        encode_time_as_delta: Encode input NN time as a delta feature.
        head_partial: FC head model class which accepts input and output dimensions.
        optimizer_partial:
            optimizer init partial. Network parameters are missed.
        lr_scheduler_partial:
            scheduler init partial. Optimizer are missed.
        dev_metric: Dev set metric.
        test_metric: Test set metric.
    """

    def generate_sequences(self, x, indices):
        """Generate future events.

        Args:
            x: Features with shape (B, L).
            indices: Indices with positions to start generation from with shape (B, I).

        Returns:
            Predicted sequences with shape (B, I, N).
        """
        outputs = self.forward(x)  # (B, L, D).
        outputs = PaddedBatch(outputs.payload.take_along_dim(indices.payload.unsqueeze(2), 1),
                              indices.seq_lens)  # (B, I, D).
        sequences = self._loss.predict_next_k(outputs, dump_category_logits={self._labels_field: self._labels_logits_field})  # (B, I, N)

        # Deltas to times.
        init_times = x.payload[self._timestamps_field].take_along_dim(indices.payload, 1)  # (B, I).
        delta_type = self._loss.get_delta_type(self._timestamps_field)
        if delta_type == "last":
            with deterministic(False):
                sequences.payload[self._timestamps_field].cumsum_(2)
        elif delta_type != "start":
            raise ValueError(f"Unknown delta type: {delta_type}.")
        sequences.payload[self._timestamps_field] += init_times.unsqueeze(2)  # (B, I, N).

        # Sort sequences.
        order = sequences.payload[self._timestamps_field].argsort(dim=2)  # (B, I, N).
        for k in sequences.seq_names:
            shaped_order = order.reshape(*(list(order.shape) + [1] * (sequences.payload[k].ndim - order.ndim)))  # (B, I, N, *).
            sequences.payload[k] = sequences.payload[k].take_along_dim(shaped_order, dim=2)  # (B, I, N, *).
        return sequences
