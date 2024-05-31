import warnings
import torch
from esp_horizon.data import PaddedBatch
from .next_item import NextItemModule


class HyproModule(NextItemModule):
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
        hypro_encoder: Sequential encoder for the HYPRO model.
        hypro_head_partial: A model head for energy computation.
        hypro_loss: Energy training loss.
        hypro_loss_step: The loss computation step.
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
                 base_checkpoint,
                 hypro_encoder, hypro_head_partial,
                 hypro_loss, hypro_loss_step,
                 timestamps_field="timestamps",
                 labels_field="labels",
                 head_partial=None,
                 optimizer_partial=None,
                 lr_scheduler_partial=None,
                 val_metric=None,
                 test_metric=None,
                 autoreg_max_steps=None,
                 hypro_context=20,
                 hypro_sample_size=20):
        if hypro_context < autoreg_max_steps:
            raise ValueError("HYPRO context must be not less than autoreg steps.")
        super().__init__(
            seq_encoder=seq_encoder,
            loss=loss,
            timestamps_field=timestamps_field,
            labels_field=labels_field,
            head_partial=head_partial,
            optimizer_partial=optimizer_partial,
            lr_scheduler_partial=lr_scheduler_partial,
            val_metric=val_metric,
            test_metric=test_metric,
            autoreg_max_steps=autoreg_max_steps
        )
        self._base_checkpoint = base_checkpoint
        self._hypro_encoder = hypro_encoder
        self._hypro_head = hypro_head_partial(hypro_encoder.hidden_size, 1)
        self._hypro_loss = hypro_loss
        self._hypro_loss_step = hypro_loss_step
        self._hypro_context = hypro_context
        self._hypro_sample_size = hypro_sample_size

    def on_fit_start(self):
        checkpoint = torch.load(self._base_checkpoint)
        missing_keys, unexpected_keys = self.load_state_dict(checkpoint, strict=False)
        if unexpected_keys:
            raise RuntimeError(f"Unexpected base checkpoint keys: {unexpected_keys}")
        missing_keys = [k for k in missing_keys if not k.startswith("_hypro")]
        if missing_keys:
            raise RuntimeError(f"Missing base checkpoint keys: {missing_keys}")

    def training_step(self, batch, _):
        is_training = self.training
        # Don't hurn batchnorm statistics in the autoreg model.
        self.eval()
        if is_training:
            self._hypro_encoder.train()
            self._hypro_head.train()
            self._hypro_loss.train()
        result = super(HyproModule, self).training_step(batch, None)
        if is_training:
            self.train()
        return result

    def _select_indices_targets(self, x):
        step = self._hypro_loss_step
        l = x.seq_lens.max().item()
        # Skip first `step` events.
        indices = torch.arange(step, l - self._hypro_context - 1, step, device=x.device)  # (I).
        indices_lens = (indices[None] < x.seq_lens[:, None]).sum(1)  # (B).
        indices = PaddedBatch(indices[None].repeat(len(x), 1), indices_lens)  # (B, I).

        target_indices = 1 + indices.payload.unsqueeze(2) + torch.arange(self._hypro_context, device=x.device)  # (B, I, N).
        targets = {k: x.payload[k].unsqueeze(1).take_along_dim(target_indices, 2)  # (B, I, N).
                   for k in x.seq_names}
        targets = PaddedBatch(targets, indices_lens)  # (B, I, N).
        return indices, targets

    def _compute_energies(self, sequences):
        b, i, n = sequences.payload[next(iter(sequences.seq_names))].shape[:3]
        lengths = sequences.seq_lens
        mask = sequences.seq_len_mask  # (B, I).
        payload = {k: v[mask].reshape(-1, n, *v.shape[3:]) for k, v in sequences.payload.items()
                   if k in sequences.seq_names}  # (V, N, *).
        sequences = PaddedBatch(payload, torch.full([len(next(iter(payload.values())))], n, device=sequences.device, dtype=torch.long))
        hiddens, _ = self._hypro_encoder(sequences)  # (V, N, D).
        energies = self._hypro_head(hiddens)  # (V, N, 1) or (V, 1).
        # Take the output from the last step.
        energies = energies.payload[:, -1]  # (V, 1) or (V).
        if energies.numel() != len(energies):
            raise ValueError("Unexpected HYPRO head output shape")
        payload = torch.zeros(b, i, device=energies.device, dtype=energies.dtype)  # (B, I).
        payload.masked_scatter_(mask, energies.flatten())  # (B, I).
        return PaddedBatch(payload, lengths)  # (B, I).

    def compute_loss(self, x, outputs, states):
        """Compute loss for the batch.

        Args:
            x: Input batch.
            outputs: Head output.
            states: Sequential model hidden states.

        Returns:
            A dict of losses and a dict of metrics.
        """
        with torch.no_grad():
            indices, targets = self._select_indices_targets(x)  # (B, I), (B, I, N).
            sequences = [super(HyproModule, self).generate_sequences(x, indices, n_steps=self._hypro_context)
                         for _ in range(self._hypro_sample_size)]  # S * (B, I, N).
        target_energies = self._compute_energies(targets)  # (B, I).
        noise_energies = [self._compute_energies(s).payload for s in sequences]
        noise_energies = PaddedBatch(torch.stack(noise_energies, 2), indices.seq_lens)  # (B, I, S).
        losses, metrics = self._hypro_loss(target_energies, noise_energies)
        return losses, metrics

    def generate_sequences(self, x, indices):
        """Generate future events.

        Args:
            x: Features with shape (B, L).
            indices: Indices with positions to start generation from with shape (B, I).

        Returns:
            Predicted sequences with shape (B, I, N).
        """
        sequences = [super(HyproModule, self).generate_sequences(x, indices, n_steps=self._hypro_context) for _ in range(self._hypro_sample_size)]  # S * (B, I, N, *).
        energies = [self._compute_energies(s).payload for s in sequences]
        energies = PaddedBatch(torch.stack(energies, 2), indices.seq_lens)  # (B, I, S).
        weights = self._hypro_loss.get_weights(energies).payload  # (B, I, S).
        best_indices = weights.argmax(2).unsqueeze(2)  # (B, I, 1).

        # Select all but logits.
        sequences = PaddedBatch({k: torch.stack([s.payload[k] for s in sequences], 2) for k in sequences[0].seq_names},
                                indices.seq_lens)  # (B, I, S, N, *).
        result = PaddedBatch({k: v.take_along_dim(best_indices.unsqueeze(3), 2).squeeze(2)
                              for k, v in sequences.payload.items()
                              if (k in sequences.seq_names) and (k != self._labels_logits_field)},
                             indices.seq_lens)  # (B, I, N).
        # Gather logits.
        logits = sequences.payload[self._labels_logits_field]  # (B, I, S, N, C).
        probs = torch.nn.functional.softmax(logits, -1)  # (B, I, S, N, C).
        result.payload[self._labels_logits_field] = (probs * weights.unsqueeze(3).unsqueeze(4)).sum(2).clip(min=1e-6).log()  # (B, I, N, C).

        # Truncate to max steps.
        result = PaddedBatch({k: v[:, :, :self._autoreg_max_steps] for k, v in result.payload.items()},
                             indices.seq_lens)  # (B, I, N', *).
        return result  # (B, I, N) or (B, I, N, C).

    @torch.no_grad()
    def _get_grad_norm(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return super()._get_grad_norm()
