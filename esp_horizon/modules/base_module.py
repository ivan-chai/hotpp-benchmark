from abc import ABC, abstractmethod
import pytorch_lightning as pl
import torch

from esp_horizon.data import PaddedBatch


class BaseModule(pl.LightningModule):
    """Base module class.

    The model is composed of the following modules:
    1. input encoder, responsible for input-to-vector conversion,
    2. sequential encoder, which captures time dependencies,
    3. fc head for embeddings projection (optional),
    4. loss, which estimates likelihood and predictions.

    Input encoder and sequential encoder are combined within SeqEncoder from Pytorch Lifestream.

    Parameters
        seq_encoder: Backbone model, which includes input encoder and sequential encoder.
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
                 test_metric=None):

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

    @property
    def seq_encoder(self):
        return self._seq_encoder

    def embed(self, x):
        """Compute embedding at each position."""
        return self._seq_encoder(x)  # (B, L, D).

    def apply_head(self, embeddings):
        payload, seq_lens  = embeddings.payload, embeddings.seq_lens
        if self._head is not None:
            payload = self._head(payload)
        return PaddedBatch(payload, seq_lens)

    def forward(self, x):
        embeddings = self.embed(x)
        outputs = self.apply_head(embeddings)
        return outputs

    def predict_next(self, outputs, fields=None):
        """Predict events from head outputs."""
        return self._loss.predict_next(outputs, fields=fields)  # (B, L).

    def predict_next_category_logits(self, outputs, fields=None):
        """Predict categorical fields logits from head outputs."""
        return self._loss.predict_next_category_logits(outputs, fields=fields)  # (B, L, C).

    @abstractmethod
    def generate_sequences(self, x, indices):
        """Generate future events.

        Args:
            x: Features with shape (B, L, D).
            indices: Indices with positions to start generation from with shape (B, I).

        Returns:
            A list of batches with generated sequences for each input sequence. Each batch has shape (I, N, D).
        """
        pass

    @abstractmethod
    def compute_loss(self, x, predictions):
        """Compute loss for the batch.

        Args:
            x: Input batch.
            predictions: Head output.

        Returns:
            A dict of losses and a dict of metrics.
        """
        pass

    def training_step(self, batch, _):
        x, _ = batch
        predictions = self.forward(x)  # (B, L, D).
        losses, metrics = self.compute_loss(x, predictions)
        loss = sum(losses.values())

        # Log statistics.
        for k, v in losses.items():
            self.log(f"train/loss_{k}", v)
        for k, v in metrics.items():
            self.log(f"train/{k}", v)
        self.log("train/loss", loss, prog_bar=True)
        self.log("sequence_length", x.seq_lens.float().mean(), prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        x, _ = batch
        predictions = self.forward(x)  # (B, L, D).
        losses, metrics = self.compute_loss(x, predictions)
        loss = sum(losses.values())

        # Log statistics.
        for k, v in losses.items():
            self.log(f"dev/loss_{k}", v, batch_size=len(x))
        for k, v in metrics.items():
            self.log(f"dev/{k}", v, batch_size=len(x))
        self.log("dev/loss", loss, batch_size=len(x))
        if self._dev_metric is not None:
            self._update_metric(self._dev_metric, predictions, x)

    def test_step(self, batch, _):
        x, _ = batch
        predictions = self.forward(x)  # (B, L, D).
        losses, metrics = self.compute_loss(x, predictions)
        loss = sum(losses.values())

        # Log statistics.
        for k, v in losses.items():
            self.log(f"test/loss_{k}", v)
        for k, v in metrics.items():
            self.log(f"test/{k}", v)
        self.log("test/loss", loss)
        if self._test_metric is not None:
            self._update_metric(self._test_metric, predictions, x)

    def on_validation_epoch_end(self):
        if self._dev_metric is not None:
            metrics = self._dev_metric.compute()
            for k, v in metrics.items():
                self.log(f"dev/{k}", v, prog_bar=True)
            self._dev_metric.reset()

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

    def on_before_optimizer_step(self, optimizer=None, optimizer_idx=None):
        self.log("grad_norm", self._get_grad_norm(), prog_bar=True)

    def _update_metric(self, metric, outputs, features):
        lengths = torch.minimum(outputs.seq_lens, features.seq_lens)
        predicted_timestamps = self.predict_next(outputs, fields=[self._timestamps_field]).payload[self._timestamps_field]  # (B, L).
        predicted_logits = self.predict_next_category_logits(outputs, fields=[self._labels_field]).payload[self._labels_field]  # (B, L, C).

        metric.update_next_item(lengths,
                                features.payload[self._timestamps_field],
                                features.payload[self._labels_field],
                                predicted_timestamps,
                                predicted_logits)

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

    @torch.no_grad()
    def _get_grad_norm(self):
        total_norm = 0.0
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if p.grad is None:
                warnings.warn(f"No grad for {name}")
                continue
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        return total_norm
