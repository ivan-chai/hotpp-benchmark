from abc import ABC, abstractmethod
import pytorch_lightning as pl
import torch

from esp_horizon.data import PaddedBatch


class Interpolator:
    def __init__(self, encoder, head):
        self._encoder = encoder
        self._head = head

    def __call__(self, states, time_deltas):
        outputs = self._encoder.interpolate(states, time_deltas)  # (B, L, S, D).
        b, l, s, d = outputs.payload.shape
        return PaddedBatch(self._head(outputs.payload.reshape(b * l, s, d)).reshape(b, l, s, -1),
                           outputs.seq_lens)


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
        loss: Training loss.
        timestamps_field: The name of the timestamps field.
        labels_field: The name of the labels field.
        head_partial: FC head model class which accepts input and output dimensions.
        optimizer_partial:
            optimizer init partial. Network parameters are missed.
        lr_scheduler_partial:
            scheduler init partial. Optimizer are missed.
        dev_metric: Dev set metric.
        test_metric: Test set metric.
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

        self._head = head_partial(seq_encoder.hidden_size, loss.input_size) if head_partial is not None else torch.nn.Identity()

        self._loss.interpolator = Interpolator(self._seq_encoder, self._head)

    def encode(self, x):
        """Apply sequential model."""
        hiddens, states = self._seq_encoder(x, return_full_states=True)  # (B, L, D), (N, B, L, D).
        return hiddens, states

    def apply_head(self, hiddens):
        """Project hidden states to model outputs."""
        return PaddedBatch(self._head(hiddens.payload), hiddens.seq_lens)

    def predict_next(self, outputs, states, fields=None, logits_fields_mapping=None):
        """Predict events from head outputs.

        NOTE: Predicted time is relative to the last event.
        """
        return self._loss.predict_next(outputs, states, fields=fields, logits_fields_mapping=logits_fields_mapping)  # (B, L) or (B, L, C).

    def forward(self, x):
        hiddens, states = self.encode(x)
        outputs = self.apply_head(hiddens)
        predictions = self.predict_next(outputs, states)
        return predictions

    @abstractmethod
    def generate_sequences(self, x, indices):
        """Generate future events.

        Args:
            x: Features with shape (B, L).
            indices: Indices with positions to start generation from with shape (B, I).

        Returns:
            Predicted sequences with shape (B, I, N).
        """
        pass

    def compute_loss(self, x, outputs, states):
        """Compute loss for the batch.

        Args:
            x: Input batch.
            outputs: Head output.
            states: Sequential model hidden states.

        Returns:
            A dict of losses and a dict of metrics.
        """
        losses, metrics = self._loss(x, outputs, states)
        return losses, metrics

    def training_step(self, batch, _):
        x, _ = batch
        hiddens, states = self.encode(x)
        outputs = self.apply_head(hiddens)  # (B, L, D).
        losses, metrics = self.compute_loss(x, outputs, states)
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
        hiddens, states = self.encode(x)
        outputs = self.apply_head(hiddens)  # (B, L, D).
        losses, metrics = self.compute_loss(x, outputs, states)
        loss = sum(losses.values())

        # Log statistics.
        for k, v in losses.items():
            self.log(f"dev/loss_{k}", v, batch_size=len(x))
        for k, v in metrics.items():
            self.log(f"dev/{k}", v, batch_size=len(x))
        self.log("dev/loss", loss, batch_size=len(x))
        if self._dev_metric is not None:
            self._update_metric(self._dev_metric, outputs, states, x)

    def test_step(self, batch, _):
        x, _ = batch
        hiddens, states = self.encode(x)
        outputs = self.apply_head(hiddens)  # (B, L, D).
        losses, metrics = self.compute_loss(x, outputs, states)
        loss = sum(losses.values())

        # Log statistics.
        for k, v in losses.items():
            self.log(f"test/loss_{k}", v, batch_size=len(x))
        for k, v in metrics.items():
            self.log(f"test/{k}", v, batch_size=len(x))
        self.log("test/loss", loss, batch_size=len(x))
        if self._test_metric is not None:
            self._update_metric(self._test_metric, outputs, states, x)

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

    def _update_metric(self, metric, outputs, states, features):
        lengths = torch.minimum(outputs.seq_lens, features.seq_lens)
        next_items = self.predict_next(outputs, states,
                                       fields=[self._timestamps_field],
                                       logits_fields_mapping={self._labels_field: self._labels_logits_field})
        predicted_timestamps = next_items.payload[self._timestamps_field]  # (B, L).
        predicted_logits = next_items.payload[self._labels_logits_field]  # (B, L, C).
        # Time is predicted as delta. Convert it to real time.
        predicted_timestamps += features.payload[self._timestamps_field]  # (B, L).

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
        names, parameters = zip(*[pair for pair in self.named_parameters() if pair[1].requires_grad])
        norms = torch.zeros(len(parameters), device=parameters[0].device)
        for i, (name, p) in enumerate(zip(names, parameters)):
            if p.grad is None:
                warnings.warn(f"No grad for {name}")
                continue
            norms[i] = p.grad.data.norm(2)
        return norms.square().sum().item() ** 0.5
