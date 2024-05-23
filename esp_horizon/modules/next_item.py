from esp_horizon.data import PaddedBatch
from .base_module import BaseModule


class NextItemModule(BaseModule):
    """Train for the next token prediction.

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
        autoreg_max_step: The maximum number of future predictions.
    """
    def __init__(self, seq_encoder, loss,
                 timestamps_field="timestamps",
                 labels_field="labels",
                 encode_time_as_delta=False,
                 head_partial=None,
                 optimizer_partial=None,
                 lr_scheduler_partial=None,
                 dev_metric=None,
                 test_metric=None,
                 autoreg_max_steps=None):

        super().__init__(
            seq_encoder=seq_encoder,
            loss=loss,
            timestamps_field=timestamps_field,
            labels_field=labels_field,
            encode_time_as_delta=encode_time_as_delta,
            head_partial=head_partial,
            optimizer_partial=optimizer_partial,
            lr_scheduler_partial=lr_scheduler_partial,
            dev_metric=dev_metric,
            test_metric=test_metric
        )
        self._autoreg_max_steps = autoreg_max_steps

    def generate_sequences(self, x, indices):
        """Generate future events.

        Args:
            x: Features with shape (B, L).
            indices: Indices with positions to start generation from with shape (B, I).

        Returns:
            Predicted sequences with shape (B, I, N).
        """
        def predict_fn(embeddings):
            self.apply_head(embeddings)
            predictions = self.loss.predict_next(embeddings)
            if hasattr(self.loss, "predict_next_category_logits"):
                logits = self.loss.predict_next_category_logits(embeddings, fields=[self._labels_field]).payload[self._labels_field]
                predictions.payload.update({self._labels_logits_field: logits})
                predictions.seq_names |= {self._labels_logits_field}
            return predictions
        return self.seq_encoder.generate(x, indices, predict_fn, self._autoreg_max_steps)  # (B, I, N).
