import pytorch_lightning as pl
import torch

from ptls.data_load import PaddedBatch


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
        labels_name: The name of the labels field.
        metric_partial: Metric for logging.
    """
    def __init__(self, seq_encoder, loss,
                 head_partial=None,
                 optimizer_partial=None,
                 lr_scheduler_partial=None,
                 labels_name="labels",
                 metric_partial=None):

        super().__init__()
        # self.save_hyperparameters()

        self._loss = loss
        self._seq_encoder = seq_encoder
        self._seq_encoder.is_reduce_sequence = False
        self._labels_name = labels_name
        if metric_partial is not None:
            self._metric = metric_partial(
                num_classes=self._loss[labels_name].num_classes
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

    def apply_head(self, encoder_output):
        payload, seq_lens  = encoder_output.payload, encoder_output.seq_lens
        if self._head is not None:
            payload = self._head(payload)
        return PaddedBatch(payload, seq_lens)

    def forward(self, x):
        encoder_output = self._seq_encoder(x)
        prediction = self.apply_head(encoder_output)
        return prediction

    def training_step(self, batch, _):
        prediction, target = self.shared_step(*batch)
        loss = self._loss(prediction, target)

        # Log statistics.
        self.log("train/loss", loss, prog_bar=True)
        x = batch
        if type(x) is tuple:
            x = x[0]
        if isinstance(x, PaddedBatch):
            self.log("sequence_length", x.seq_lens.float().mean(), prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        prediction, target = self.shared_step(*batch)
        loss = self._loss(prediction, target)
        self.log("dev/loss", loss, batch_size=len(target))
        if self._metric is not None:
            labels_parameters = self._loss.split_predictions(prediction)[self._labels_name]
            labels_probs = self._loss[self._labels_name].get_proba(labels_parameters)  # (B, L, C).
            self._metric.update(labels_probs=labels_probs,
                                targets=target.payload[self._labels_name],
                                mask=target.seq_len_mask.bool())

    def on_validation_epoch_end(self):
        if self._metric is not None:
            metrics = self._metric.compute()
            for k, v in metrics.items():
                self.log(f"dev/{k}", v, prog_bar=True)
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
        prediction = self.forward(x)  # (B, L, D).
        # Shift predictions w.r.t. targets.
        prediction = PaddedBatch(prediction.payload[:, :-1],
                                 prediction.seq_lens.clip(max=max(0, prediction.payload.shape[1] - 1)))
        target = PaddedBatch({k: x.payload[k][:, 1:] for k in self._loss.loss_names},
                             (x.seq_lens - 1).clip(min=0))
        return prediction, target
