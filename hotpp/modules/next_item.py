from hotpp.data import PaddedBatch
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
        head_partial: FC head model class which accepts input and output dimensions.
        optimizer_partial:
            optimizer init partial. Network parameters are missed.
        lr_scheduler_partial:
            scheduler init partial. Optimizer are missed.
        val_metric: Validation set metric.
        test_metric: Test set metric.
        autoreg_max_steps: The maximum number of future predictions.
    """
    def __init__(self, seq_encoder, loss,
                 timestamps_field="timestamps",
                 labels_field="labels",
                 head_partial=None,
                 optimizer_partial=None,
                 lr_scheduler_partial=None,
                 val_metric=None,
                 test_metric=None,
                 autoreg_max_steps=None):

        super().__init__(
            seq_encoder=seq_encoder,
            loss=loss,
            timestamps_field=timestamps_field,
            labels_field=labels_field,
            head_partial=head_partial,
            optimizer_partial=optimizer_partial,
            lr_scheduler_partial=lr_scheduler_partial,
            val_metric=val_metric,
            test_metric=test_metric
        )
        self._autoreg_max_steps = autoreg_max_steps

    def generate_sequences(self, x, indices, n_steps=None):
        """Generate future events.

        Args:
            x: Features with shape (B, L).
            indices: Indices with positions to start generation from with shape (B, I).
            n_steps: The number of steps to generate. Use autoreg_max_steps by default.

        Returns:
            Predicted sequences with shape (B, I, N).
        """
        if n_steps is None:
            n_steps = self._autoreg_max_steps
        def predict_fn(hiddens, states):
            outputs = self.apply_head(hiddens)  # (B, L, D).
            return self.predict_next(None, outputs, states,
                                     predict_delta=True,
                                     logits_fields_mapping={self._labels_field: self._labels_logits_field})  # (B, L).
        return self._seq_encoder.generate(x, indices, predict_fn, n_steps)  # (B, I, N).
