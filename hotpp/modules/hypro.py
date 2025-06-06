import warnings
import torch
from hotpp.data import PaddedBatch
from hotpp.utils.torch import module_mode, BATCHNORM_TYPES
from ..fields import LABELS_LOGITS
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
        hypro_context: The size of a prefix attached to the hypothesis before energy computation.
            By default equal to the autoreg_max_steps.
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
        hypro_logits_prediction: Either `best` or `mean-prob`.
    """
    def __init__(self, seq_encoder, loss,
                 base_checkpoint,
                 hypro_encoder, hypro_head_partial,
                 hypro_loss, hypro_loss_step,
                 autoreg_max_steps=None,
                 hypro_context=None,
                 hypro_sample_size=20,
                 hypro_logits_prediction="best",
                 **kwargs):
        if hypro_context is None:
            hypro_context = autoreg_max_steps
        super().__init__(
            seq_encoder=seq_encoder,
            loss=loss,
            autoreg_max_steps=autoreg_max_steps,
            **kwargs
        )
        self._base_checkpoint = base_checkpoint
        self._hypro_encoder = hypro_encoder
        self._hypro_head = hypro_head_partial(hypro_encoder.hidden_size, 1)
        self._hypro_loss = hypro_loss
        self._hypro_loss_step = hypro_loss_step
        self._hypro_context = hypro_context
        self._hypro_sample_size = hypro_sample_size
        self._hypro_logits_prediction = hypro_logits_prediction

    def on_fit_start(self):
        checkpoint = torch.load(self._base_checkpoint)
        missing_keys, unexpected_keys = self.load_state_dict(checkpoint, strict=False)
        if unexpected_keys:
            raise RuntimeError(f"Unexpected base checkpoint keys: {unexpected_keys}")
        missing_keys = [k for k in missing_keys if not k.startswith("_hypro")]
        if missing_keys:
            raise RuntimeError(f"Missing base checkpoint keys: {missing_keys}")

    def training_step(self, batch, _):
        # Don't hurn batchnorm statistics in the autoreg model.
        with module_mode(self, training=False, layer_types=BATCHNORM_TYPES):
            with module_mode(self._hypro_encoder, self._hypro_head, training=True):
                result = super(HyproModule, self).training_step(batch, None)
        return result

    def _select_indices_targets(self, x):
        step = self._hypro_loss_step
        l = x.seq_lens.max().item()
        # Skip first `step` events.
        indices = torch.arange(step, l - self._autoreg_max_steps - 1, step, device=x.device)  # (I).
        indices_lens = (indices[None] < x.seq_lens[:, None]).sum(1)  # (B).
        indices = PaddedBatch(indices[None].repeat(len(x), 1), indices_lens)  # (B, I).

        target_indices = 1 + indices.payload.unsqueeze(2) + torch.arange(self._autoreg_max_steps, device=x.device)  # (B, I, N).
        targets = {k: x.payload[k].unsqueeze(1).take_along_dim(target_indices, 2)  # (B, I, N).
                   for k in x.seq_names}
        targets = PaddedBatch(targets, indices_lens)  # (B, I, N).
        return indices, targets

    def _attach_prefixes(self, x, indices, sequences):
        """Attach input prefix to each sequence.

        NOTE. For some indices there is no prefix with the required size.
        We therefore pad missing events with zeros.

        Args:
            x: An input sequence with shape (B, L).
            indices: Positions from which sequences were generated with shape (B, I).
            sequences: Generated sequences with shape (B, I, N).

        Returns:
            Joined sequences with shape (B, I, P + N).
        """
        if self._hypro_context == 0:
            return sequences
        # Join fields before windowing.
        b, l = x.shape
        device = x.device
        fields = [field for field in sequences.seq_names if field != LABELS_LOGITS]
        joined = torch.stack([x.payload[field] for field in fields], -1)  # (B, L, D).

        # Extract prefixes.
        prefixes = [joined.roll(i, 1) for i in reversed(range(self._hypro_context))]  # P x (B, L, D).
        prefixes = torch.stack(prefixes, 2)  # (B, L, P, D).
        invalid = (torch.arange(self._hypro_context, device=device)[None] <
                   self._hypro_context - 1 - torch.arange(l, device=device)[:, None])  # (L, P).
        prefixes.masked_fill_(invalid[None, :, :, None], 0)
        prefixes = prefixes.take_along_dim(indices.payload[:, :, None, None], 1)  # (B, I, P, D).

        # Split and cat fields.
        payload = {}
        for i, field in enumerate(fields):
            payload[field] = torch.cat([prefixes[..., i].to(sequences.payload[field].dtype), sequences.payload[field]], 2)  # (B, I, P + N).
        return PaddedBatch(payload, indices.seq_lens)  # (B, I, P + N).

    def _compute_energies(self, x, indices, sequences):
        sequences = self._attach_prefixes(x, indices, sequences)  # (B, I, P + N).
        b, i, n = sequences.payload[sequences.seq_names[0]].shape[:3]
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
            sequences = [super(HyproModule, self).generate_sequences(x, indices)
                         for _ in range(self._hypro_sample_size)]  # S * (B, I, N).

        # Don't hurt batchnorm statistics with GT events.
        with module_mode(self._hypro_encoder, self._hypro_head, training=False, layer_types=BATCHNORM_TYPES):
            target_energies = self._compute_energies(x, indices, targets)  # (B, I).
        noise_energies = [self._compute_energies(x, indices, s).payload for s in sequences]
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
        sequences = [super(HyproModule, self).generate_sequences(x, indices) for _ in range(self._hypro_sample_size)]  # S * (B, I, N, *).
        energies = [self._compute_energies(x, indices, s).payload for s in sequences]
        energies = PaddedBatch(torch.stack(energies, 2), indices.seq_lens)  # (B, I, S).
        weights = self._hypro_loss.get_weights(energies).payload  # (B, I, S).
        best_indices = weights.argmax(2).unsqueeze(2)  # (B, I, 1).

        # Select all but logits.
        sequences = PaddedBatch({k: torch.stack([s.payload[k] for s in sequences], 2) for k in sequences[0].seq_names},
                                indices.seq_lens)  # (B, I, S, N, *).
        result = PaddedBatch({k: v.take_along_dim(best_indices.unsqueeze(3), 2).squeeze(2)
                              for k, v in sequences.payload.items()
                              if (k in sequences.seq_names) and (k != LABELS_LOGITS)},
                             indices.seq_lens)  # (B, I, N).

        # Gather logits.
        if self._hypro_logits_prediction == "best":
            logits_indices = best_indices.unsqueeze(3).unsqueeze(4)  # (B, I, 1, 1, 1).
            result.payload[LABELS_LOGITS] = sequences.payload[LABELS_LOGITS].take_along_dim(logits_indices, 2).squeeze(2)  # (B, I, N, C).
        elif self._hypro_logits_prediction == "mean-prob":
            logits = sequences.payload[LABELS_LOGITS]  # (B, I, S, N, C).
            probs = torch.nn.functional.softmax(logits, -1)  # (B, I, S, N, C).
            result.payload[LABELS_LOGITS] = (probs * weights.unsqueeze(3).unsqueeze(4)).sum(2).clip(min=1e-6).log()  # (B, I, N, C).
        else:
            raise ValueError(f"Unknown logits prediction mode: {self._hypro_logits_prediction}")

        # Truncate to max steps.
        result = PaddedBatch({k: v[:, :, :self._autoreg_max_steps] for k, v in result.payload.items()},
                             indices.seq_lens)  # (B, I, N', *).
        return result  # (B, I, N) or (B, I, N, C).

    @torch.no_grad()
    def _get_grad_norm(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return super()._get_grad_norm()
