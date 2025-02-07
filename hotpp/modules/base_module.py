import warnings
from abc import ABC, abstractmethod
import pytorch_lightning as pl
import torch

from ..fields import PRESENCE, PRESENCE_PROB, LABELS_LOGITS
from ..data import PaddedBatch


class Interpolator:
    def __init__(self, encoder, head):
        self._encoder = encoder
        self._head = head

    def eval(self):
        self._encoder.eval()
        self._head.eval()

    def train(self):
        self._encoder.train()
        self._head.train()

    def modules(self):
        yield self._encoder
        yield self._head

    def __call__(self, states, time_deltas):
        outputs = self._encoder.interpolate(states, time_deltas)  # (B, L, S, D).
        return self._head(outputs)


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
        amounts_field (optional): The name of the amount field for some metrics.
        head_partial: FC head model class which accepts input and output dimensions.
        aggregator: Embeddings aggregator.
        optimizer_partial:
            optimizer init partial. Network parameters are missed.
        lr_scheduler_partial:
            scheduler init partial. Optimizer are missed.
        val_metric: Validation set metric.
        test_metric: Test set metric.
    """
    def __init__(self, seq_encoder, loss,
                 timestamps_field="timestamps",
                 labels_field="labels",
                 amounts_field=None,
                 head_partial=None,
                 aggregator=None,
                 optimizer_partial=None,
                 lr_scheduler_partial=None,
                 val_metric=None,
                 test_metric=None):

        super().__init__()
        self._timestamps_field = timestamps_field
        self._labels_field = labels_field
        self._amounts_field = amounts_field

        self._loss = loss
        self._seq_encoder = seq_encoder
        self._seq_encoder.is_reduce_sequence = False
        self._val_metric = val_metric
        self._test_metric = test_metric
        self._optimizer_partial = optimizer_partial
        self._lr_scheduler_partial = lr_scheduler_partial

        self._head = head_partial(seq_encoder.hidden_size, loss.input_size) if head_partial is not None else torch.nn.Identity()
        self._aggregator = aggregator

        self._need_states = self._loss.need_interpolator or self._seq_encoder.need_states
        if self._loss.need_interpolator:
            self._loss.interpolator = Interpolator(self._seq_encoder, self._head)

    def predict_next(self, inputs, outputs, states, fields=None, logits_fields_mapping=None, predict_delta=False):
        """Predict events from head outputs.

        Args:
            inputs: Input features with shape (B, L).
            outputs: Output of the head module with shape (B, L, D).
            states: Sequence model states with shape (N, B, L, D), where N is the number of layers.
            predict_delta: If True, return delta times. Generate absolute timestamps otherwise.
        """
        results = self._loss.predict_next(outputs, states, fields=fields, logits_fields_mapping=logits_fields_mapping)  # (B, L) or (B, L, C).
        if not predict_delta:
            # Convert delta time to time.
            results.payload[self._timestamps_field] += inputs.payload[self._timestamps_field]
        return results

    def forward(self, x, return_states=False):
        """Extract hidden activations and states."""
        hiddens, states = self._seq_encoder(x, return_states=return_states)  # (B, L, D), (N, B, L, D).
        outputs = self._head(hiddens)  # (B, L, D).
        return outputs, states

    def embed(self, x):
        """Extract embeddings from the output of the sequential encoder."""
        if self._aggregator is None:
            raise ValueError("Need an aggregator for embeddings extraction.")
        hiddens, _ = self._seq_encoder(x)  # (B, L, D).
        embeddings = self._aggregator(hiddens)
        return embeddings

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

    def training_step(self, batch, batch_idx):
        x, _ = batch
        outputs, states = self(x, return_states="full" if self._need_states else False)  # (B, L, D), (N, B, L, D).
        losses, metrics = self.compute_loss(x, outputs, states)
        loss = sum(losses.values())

        # Log statistics.
        for k, v in losses.items():
            self.log(f"train/loss_{k}", v)
        for k, v in metrics.items():
            self.log(f"train/{k}", v)
        self.log("train/loss", loss, prog_bar=True)
        self.log("sequence_length", x.seq_lens.float().mean(), prog_bar=True)
        if batch_idx == 0:
            with torch.no_grad():
                for k, v in self._compute_single_batch_metrics(x, outputs, states).items():
                    self.log(f"train/{k}", v, batch_size=len(x))
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        outputs, states = self(x, return_states="full" if self._need_states else False)  # (B, L, D), (N, B, L, D).
        losses, metrics = self.compute_loss(x, outputs, states)
        loss = sum(losses.values())

        # Log statistics.
        for k, v in losses.items():
            self.log(f"val/loss_{k}", v, batch_size=len(x))
        for k, v in metrics.items():
            self.log(f"val/{k}", v, batch_size=len(x))
        self.log("val/loss", loss, batch_size=len(x), prog_bar=True)
        if self._val_metric is not None:
            self._update_metric(self._val_metric, x, outputs, states)
        if batch_idx == 0:
            for k, v in self._compute_single_batch_metrics(x, outputs, states).items():
                self.log(f"val/{k}", v, batch_size=len(x))

    def test_step(self, batch, batch_idx):
        x, _ = batch
        outputs, states = self(x, return_states="full" if self._need_states else False)  # (B, L, D), (N, B, L, D).
        losses, metrics = self.compute_loss(x, outputs, states)
        loss = sum(losses.values())

        # Log statistics.
        for k, v in losses.items():
            self.log(f"test/loss_{k}", v, batch_size=len(x))
        for k, v in metrics.items():
            self.log(f"test/{k}", v, batch_size=len(x))
        self.log("test/loss", loss, batch_size=len(x), prog_bar=True)
        if self._test_metric is not None:
            self._update_metric(self._test_metric, x, outputs, states)
        if batch_idx == 0:
            for k, v in self._compute_single_batch_metrics(x, outputs, states).items():
                self.log(f"test/{k}", v, batch_size=len(x))

    def on_validation_epoch_end(self):
        if self._val_metric is not None:
            metrics = self._val_metric.compute()
            for k, v in metrics.items():
                self.log(f"val/{k}", v, prog_bar=True)
            self._val_metric.reset()

    def on_test_epoch_end(self):
        if self._test_metric is not None:
            metrics = self._test_metric.compute()
            for k, v in metrics.items():
                self.log(f"test/{k}", v, prog_bar=True)
            self._test_metric.reset()

    def configure_optimizers(self):
        optimizer = self._optimizer_partial(self.parameters())
        if self._lr_scheduler_partial is None:
            return optimizer
        else:
            scheduler = self._lr_scheduler_partial(optimizer)
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler = {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                }
            return [optimizer], [scheduler]

    def on_before_optimizer_step(self, optimizer=None, optimizer_idx=None):
        self.log("grad_norm", self._get_grad_norm(), prog_bar=True)

    @torch.autocast("cuda", enabled=False)
    def _update_metric(self, metric, features, outputs, states):
        outputs = outputs.to(torch.float)
        states = states.to(torch.float) if states is not None else None
        lengths = torch.minimum(outputs.seq_lens, features.seq_lens)
        next_items = self.predict_next(features, outputs, states,
                                       fields=[self._timestamps_field, self._labels_field],
                                       logits_fields_mapping={self._labels_field: LABELS_LOGITS})
        predicted_timestamps = next_items.payload[self._timestamps_field]  # (B, L).
        predicted_labels = next_items.payload[self._labels_field]  # (B, L).
        predicted_logits = next_items.payload[LABELS_LOGITS]  # (B, L, C).

        metric.update_next_item(lengths,
                                features.payload[self._timestamps_field],
                                features.payload[self._labels_field],
                                predicted_timestamps,
                                predicted_labels,
                                predicted_logits)

        if metric.horizon_prediction:
            indices = metric.select_horizon_indices(features.seq_lens)
            sequences = self.generate_sequences(features, indices)  # (B, I, N).
            predicted_mask = sequences.payload.get(PRESENCE, None)  # (B, I, N).
            predicted_probabilities = sequences.payload.get(PRESENCE_PROB, None)  # (B, I, N).
            kwargs = {}
            if metric.need_amount:
                if self._amounts_field is None:
                    raise ValueError("Need amount field name to compute the horizon stats metric")
                kwargs["amounts"] = features.payload[self._amounts_field]
                kwargs["seq_predicted_amounts"] = sequences.payload[self._amounts_field]
            metric.update_horizon(features.seq_lens,
                                  features.payload[self._timestamps_field],
                                  features.payload[self._labels_field],
                                  indices.payload,
                                  indices.seq_lens,
                                  sequences.payload[self._timestamps_field],
                                  sequences.payload[self._labels_field],
                                  sequences.payload[LABELS_LOGITS],
                                  seq_predicted_mask=predicted_mask,
                                  seq_predicted_probabilities=predicted_probabilities,
                                  **kwargs)

    @torch.autocast("cuda", enabled=False)
    def _compute_single_batch_metrics(self, inputs, outputs, states):
        """Slow debug metrics."""
        metrics = {}
        if hasattr(self._loss, "compute_metrics"):
            metrics.update(self._loss.compute_metrics(inputs, outputs, states))
        return metrics

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
