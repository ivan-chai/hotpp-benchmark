from esp_horizon.data import PaddedBatch
from .base_module import BaseModule
from .autoreg import RNNSequencePredictor


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
        head_partial: FC head model class which accepts input and output dimensions.
        optimizer_partial:
            optimizer init partial. Network parameters are missed.
        lr_scheduler_partial:
            scheduler init partial. Optimizer are missed.
        labels_field: The name of the labels field.
        dev_metric: Dev set metric.
        test_metric: Test set metric.
        autoreg_max_step: The maximum number of future predictions.
        autoreg_adapter_partial: An autoregressive adapter constructor (see `autoreg` submodule).
    """
    def __init__(self, seq_encoder, loss,
                 timestamps_field="timestamps",
                 labels_field="labels",
                 head_partial=None,
                 optimizer_partial=None,
                 lr_scheduler_partial=None,
                 dev_metric=None,
                 test_metric=None,
                 autoreg_max_steps=None,
                 autoreg_adapter_partial=None):

        super().__init__(
            seq_encoder=seq_encoder,
            loss=loss,
            timestamps_field=timestamps_field,
            labels_field=labels_field,
            head_partial=head_partial,
            optimizer_partial=optimizer_partial,
            lr_scheduler_partial=lr_scheduler_partial,
            dev_metric=dev_metric,
            test_metric=test_metric
        )

        if (autoreg_adapter_partial is None) ^ (autoreg_max_steps is None):
            raise ValueError("Autoreg adapter and autoreg max steps must be provided together.")

        if autoreg_adapter_partial is not None:
            logits_field_mapping = {self._labels_field: self._labels_logits_field}
            self._autoreg_adapter = autoreg_adapter_partial(self, dump_category_logits=logits_field_mapping)
            self._autoreg_max_steps = autoreg_max_steps
        else:
            self._autoreg_adapter = None

    def generate_sequences(self, x, indices):
        """Generate future events.

        Args:
            x: Features with shape (B, L).
            indices: Indices with positions to start generation from with shape (B, I).

        Returns:
            Predicted sequences with shape (B, I, N).
        """
        if self._autoreg_adapter is None:
            raise RuntimeError("Need autoregressive adapter for prediction.")
        predictor = RNNSequencePredictor(self._autoreg_adapter, max_steps=self._autoreg_max_steps)
        return predictor(x, indices)
