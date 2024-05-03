import pytorch_lightning as pl
import torch

from esp_horizon.data import PaddedBatch
from .autoreg import NextItemRNNAdapter, RNNSequencePredictor


class NextItemModule(pl.LightningModule):
    """Train next token prediction.

    Parameters
        seq_encoder: Backbone model.
        loss: Training loss
        head_partial: FC head model class which accepts input and output dimensions.
        optimizer_partial:
            optimizer init partial. Network parameters are missed.
        lr_scheduler_partial:
            scheduler init partial. Optimizer are missed.
        labels_field: The name of the labels field.
        metric_partial: Metric for logging.
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

        super().__init__()
        self._timestamps_field = timestamps_field
        self._labels_field = labels_field
        self._labels_logits_field = f"{labels_field}_logits"

        self._loss = loss
        self._seq_encoder = seq_encoder
        self._seq_encoder.is_reduce_sequence = False
        self._dev_metric = dev_metric
        self._test_metric = test_metric
        self._optimizer_partial = optimizer_partial
        self._lr_scheduler_partial = lr_scheduler_partial

        if head_partial is not None:
            embedding_dim = self._seq_encoder.embedding_size
            output_dim = loss.input_dim
            self._head = head_partial(embedding_dim, output_dim)
        else:
            self._head = None

        if (autoreg_adapter_partial is None) ^ (autoreg_max_steps is None):
            raise ValueError("Autoreg adapter and autoreg max steps must be provided together.")

        if autoreg_adapter_partial is not None:
            logits_field_mapping = {self._labels_field: self._labels_logits_field}
            self._autoreg_adapter = autoreg_adapter_partial(self, dump_category_logits=logits_field_mapping)
            self._autoreg_max_steps = autoreg_max_steps
        else:
            self._autoreg_adapter = None

    @property
    def seq_encoder(self):
        return self._seq_encoder

    def apply_head(self, encoder_output):
        payload, seq_lens  = encoder_output.payload, encoder_output.seq_lens
        if self._head is not None:
            payload = self._head(payload)
        return PaddedBatch(payload, seq_lens)

    def forward(self, x):
        encoder_output = self._seq_encoder(x)
        predictions = self.apply_head(encoder_output)
        return predictions

    def get_embeddings(self, x):
        """Get embedding for each position."""
        return self._seq_encoder(x)  # (B, L, D).

    def get_modes(self, predictions):
        return self._loss.get_modes(predictions)  # (B, L).

    def get_logits(self, predictions):
        return self._loss.get_logits(predictions)  # (B, L).

    def generate_sequences(self, x, indices):
        """Generate future events.

        Args:
            x: Features with shape (B, L, D).
            indices: Indices with positions to start generation from with shape (B, I).

        Returns:
            A list of batches with generated sequences for each input sequence. Each batch has shape (I, N, D).
        """
        if self._autoreg_adapter is None:
            raise RuntimeError("Need autoregressive adapter for prediction.")
        predictor = RNNSequencePredictor(self._autoreg_adapter, max_steps=self._autoreg_max_steps)
        return predictor(x, indices)

    def training_step(self, batch, _):
        x, _ = batch
        predictions = self.forward(x)  # (B, L, D).
        losses = self._loss(predictions, self._get_targets(x))
        loss = sum(losses.values())

        # Log statistics.
        for k, v in losses.items():
            self.log(f"train/loss_{k}", v)
        self.log("train/loss", loss, prog_bar=True)
        self.log("sequence_length", x.seq_lens.float().mean(), prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        x, _ = batch
        predictions = self.forward(x)  # (B, L, D).
        losses = self._loss(predictions, self._get_targets(x))
        loss = sum(losses.values())

        # Log statistics.
        for k, v in losses.items():
            self.log(f"dev/loss_{k}", v, batch_size=len(x))
        self.log("dev/loss", loss, batch_size=len(x))
        if self._dev_metric is not None:
            self._update_metric(self._dev_metric, predictions, x)

    def on_validation_epoch_end(self):
        if self._dev_metric is not None:
            metrics = self._dev_metric.compute()
            for k, v in metrics.items():
                self.log(f"dev/{k}", v, prog_bar=True)
            self._dev_metric.reset()

    def test_step(self, batch, _):
        x, _ = batch
        predictions = self.forward(x)  # (B, L, D).
        losses = self._loss(predictions, self._get_targets(x))
        loss = sum(losses.values())

        # Log statistics.
        for k, v in losses.items():
            self.log(f"test/loss_{k}", v)
        self.log("test/loss", loss)
        if self._test_metric is not None:
            self._update_metric(self._test_metric, predictions, x)

    def on_test_epoch_end(self):
        if self._test_metric is not None:
            metrics = self._test_metric.compute()
            for k, v in metrics.items():
                self.log(f"test/{k}", v, prog_bar=True)
            self._test_metric.reset()

    def configure_optimizers(self):
        optimizer = self._optimizer_partial(self.parameters())
        scheduler = self._lr_scheduler_partial(optimizer)

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler = {
                "scheduler": scheduler,
                "monitor": "dev/loss",
            }
        return [optimizer], [scheduler]

    def _get_targets(self, x):
        """Shift targets w.r.t. predictions."""
        lengths = (x.seq_lens - 1).clip(min=0)
        targets = PaddedBatch({k: x.payload[k][:, 1:] for k in self._loss.loss_names},
                              lengths, x.seq_names)
        return targets

    def _update_metric(self, metric, predictions, features):
        lengths = torch.minimum(predictions.seq_lens, features.seq_lens)
        parameters = self._loss.split_predictions(predictions)  # (B, L, P).
        predicted_timestamps = self._loss[self._timestamps_field].get_modes(parameters[self._timestamps_field])  # (B, L).
        predicted_labels_logits = self._loss[self._labels_field].get_log_proba(parameters[self._labels_field])  # (B, L, C).

        metric.update_next_item(lengths,
                                features.payload[self._timestamps_field],
                                features.payload[self._labels_field],
                                predicted_timestamps,
                                predicted_labels_logits)

        if metric.horizon_prediction:
            indices = metric.select_horizon_indices(features.seq_lens)
            sequences = self.generate_sequences(features, indices)
            metric.update_horizon(features.seq_lens,
                                  features.payload[self._timestamps_field],
                                  features.payload[self._labels_field],
                                  indices.payload,
                                  indices.seq_lens,
                                  sequences.payload[self._timestamps_field],
                                  sequences.payload[self._labels_logits_field])
