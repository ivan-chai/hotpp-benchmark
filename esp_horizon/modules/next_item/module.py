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
                 head_partial=None,
                 optimizer_partial=None,
                 lr_scheduler_partial=None,
                 labels_field="labels",
                 metric_partial=None,
                 autoreg_adapter_partial=None):

        super().__init__()
        # self.save_hyperparameters()

        self._loss = loss
        self._seq_encoder = seq_encoder
        self._seq_encoder.is_reduce_sequence = False
        self._labels_field = labels_field
        if metric_partial is not None:
            self._metric = metric_partial(
                num_classes=self._loss[labels_field].num_classes
            )
        else:
            self._metric = None
        self._optimizer_partial = optimizer_partial
        self._lr_scheduler_partial = lr_scheduler_partial

        if head_partial is not None:
            embedding_dim = self._seq_encoder.embedding_size
            output_dim = loss.input_dim
            self._head = head_partial(embedding_dim, output_dim)
        else:
            self._head = None

        if autoreg_adapter_partial is not None:
            self._autoreg_adapter = autoreg_adapter_partial(self)
        else:
            self._autoreg_adapter = None

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

    def get_modes(self, encoder_output):
        head_output = self.apply_head(encoder_output)
        return self._loss.get_modes(head_output)  # (B, L).

    def generate_sequences(self, x, positions, n_steps):
        """Generate future events.

        Args:
            x: Features with shape (B, L, D).
            positions: Indices with positions to start generation from with shape (B, I).
            n_steps: The number of steps to generate.

        Returns:
            A list of batches with generated sequences for each input sequence. Each batch has shape (I, N, D).
        """
        if self._autoreg_adapter is None:
            raise RuntimeError("Need adapter for prediction.")
        predictor = RNNSequencePredictor(self._autoreg_adapter, max_steps=n_steps)
        return predictor(x, positions)

    def training_step(self, batch, _):
        predictions, targets = self.shared_step(*batch)
        loss = self._loss(predictions, targets)

        # Log statistics.
        self.log("train/loss", loss, prog_bar=True)
        x = batch
        if type(x) is tuple:
            x = x[0]
        if isinstance(x, PaddedBatch):
            self.log("sequence_length", x.seq_lens.float().mean(), prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        predictions, targets = self.shared_step(*batch)
        loss = self._loss(predictions, targets)
        self.log("dev/loss", loss, batch_size=len(targets))
        if self._metric is not None:
            labels_parameters = self._loss.split_predictions(predictions)[self._labels_field]
            labels_probs = self._loss[self._labels_field].get_proba(labels_parameters)  # (B, L, C).
            self._metric.update(labels_probs=labels_probs,
                                targets=targets.payload[self._labels_field],
                                mask=targets.seq_len_mask.bool())

    def on_validation_epoch_end(self):
        if self._metric is not None:
            metrics = self._metric.compute()
            for k, v in metrics.items():
                self.log(f"dev/{k}", v, prog_bar=True)
            self._metric.reset()

    def test_step(self, batch, _):
        predictions, targets = self.shared_step(*batch)
        if self._metric is not None:
            labels_parameters = self._loss.split_predictions(predictions)[self._labels_field]
            labels_probs = self._loss[self._labels_field].get_proba(labels_parameters)  # (B, L, C).
            self._metric.update(labels_probs=labels_probs,
                                targets=targets.payload[self._labels_field],
                                mask=targets.seq_len_mask.bool())

    def on_test_epoch_end(self):
        if self._metric is not None:
            metrics = self._metric.compute()
            for k, v in metrics.items():
                self.log(f"test/{k}", v, prog_bar=True)
            self._metric.reset()

    def configure_optimizers(self):
        optimizer = self._optimizer_partial(self.parameters())
        scheduler = self._lr_scheduler_partial(optimizer)

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler = {
                "scheduler": scheduler,
                "monitor": "dev/loss",
            }
        return [optimizer], [scheduler]

    def shared_step(self, x, _):
        predictions = self.forward(x)  # (B, L, D).
        # Shift predictions w.r.t. targets.
        lengths = (x.seq_lens - 1).clip(min=0)
        predictions = PaddedBatch(predictions.payload[:, :-1], lengths)
        targets = PaddedBatch({k: x.payload[k][:, 1:] for k in self._loss.loss_names},
                              lengths, x.seq_names)
        return predictions, targets
