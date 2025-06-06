import math
import random
import torch

from hotpp.data import PaddedBatch
from hotpp.utils.torch import deterministic
from .common import ScaleGradient
from .next_k import NextKLoss


class DiffusionLoss(NextKLoss):
    """Diffusion loss for next-k prediction.

    See the original paper:
    Zhou, Wang-Tao, et al. "Non-autoregressive diffusion-based temporal point processes
    for continuous-time long-term event prediction." Expert Systems with Applications, 2025.

    Submodules:
    - Embedder: converts events to embeddings.
    - Denoiser: converts a sequence of embeddings and a condition vector to denoised sequences.
    - Decoder: converts embeddings to the input of the loss function used for prediction.

    Args:
        next_item_loss: An instance of the next event prediction loss used for decoder training.
        denoiser_partial: Denoiser model constructor which accepts input/output dimension.
        decoder_partial: Decoder model constructor which accepts input and output dimensions.
        k: The number of future events to predict.
        loss_step: The period of loss evaluation.
        generation_steps: The number of denosing steps.
        alpha: One minus corruption noise level at each step in the range [0, 1].
        detach_embeddings_from_step: Don't propagate gradients to embeddings from latter reconstruction stages (int or False).
        detach_decoder: Don't propage gradients to other modules when training the decoder model.
        clamp_from_step: Apply clamping trick starting from the selected step. By default, disable clamping.
        prediction: One of "mode" and "sample".
    """
    def __init__(self, next_item_loss, k, embedder, denoiser_partial, decoder_partial,
                 timestamps_field="timestamps", max_time_delta=None, loss_step=1,
                 generation_steps=10, alpha=0.1, detach_embeddings_from_step=1, detach_decoder=True,
                 diffusion_loss_weight=1, decoder_loss_weight=1, embedder_regularizer=1,
                 clamp_from_step=False, prediction="sample"):
        super().__init__(next_item_loss, k,
                         timestamps_field=timestamps_field,
                         loss_step=loss_step)
        self._max_time_delta = max_time_delta
        self._generation_steps = generation_steps
        self._diffusion_loss_weight = diffusion_loss_weight
        self._decoder_loss_weight = decoder_loss_weight
        self._embedder_regularizer = embedder_regularizer
        self._embedder = embedder
        self._denoiser = denoiser_partial(embedder.output_size, k)
        self._decoder = decoder_partial(embedder.output_size, next_item_loss.input_size)
        self._detach_embeddings_from_step = detach_embeddings_from_step
        self._detach_decoder = detach_decoder
        self._clamp_from_step = clamp_from_step
        self.register_buffer("_alpha", torch.full([1 + generation_steps], alpha, dtype=torch.float))
        self._alpha[0] = 1
        self.register_buffer("_alpha_prods", self._alpha.cumprod(dim=0))
        self._prediction = prediction

    @property
    def input_size(self):
        return self._denoiser.condition_size

    def _noise(self, batch_size, device, dtype):
        # Generate noise with shape (B, N, D).
        if (self._prediction == "sample") or self.training:
            x = torch.randn(batch_size, self._k, self._embedder.output_size,
                            device=device, dtype=dtype)
        else:
            if self._prediction != "mode":
                raise ValueError(f"Unknown prediction mode: {self._prediction}")
            x = torch.zeros(batch_size, self._k, self._embedder.output_size,
                            device=device, dtype=dtype)
        return PaddedBatch(x, torch.full([batch_size], self._k, device=device, dtype=torch.long))

    def _corrupt(self, embeddings, steps):
        # Embeddings: (B, L, D).
        # Steps: (B).
        # Get corrupted variant of input embeddings at the given step [0, N].
        w0 = self._alpha_prods[steps].sqrt()[:, None, None]  # (B, 1, 1).
        wn = (1 - self._alpha_prods)[steps].sqrt()[:, None, None]  # (B, 1, 1).
        return PaddedBatch(w0 * embeddings.payload + wn * torch.randn_like(embeddings.payload),
                           embeddings.seq_lens)

    def _denoising_step(self, embeddings, condition, step):
        # Embeddings: (B, L, D).
        # Condition: (B, D).
        # Make a single denoising step from the given step [1, N] to the previous one.
        # Step is the current step for input embeddings.
        # Returns step - 1 embeddings.
        assert step >= 1
        dtype = embeddings.payload.dtype
        alpha = self._alpha.to(dtype)
        alpha_prods = self._alpha_prods.to(dtype)

        steps = torch.full([len(embeddings)], step, device=embeddings.device, dtype=torch.long)
        reconstruction = self._denoiser(embeddings, condition, steps)  # (B, L, D).
        if self._clamp_from_step and (step <= self._clamp_from_step):
            reconstruction = self._embedder.clamp(reconstruction)
        denum = 1 - alpha_prods[steps].to(dtype)  # (B).
        wx = alpha_prods[steps].sqrt() * (1 - alpha_prods[steps - 1]) / denum  # (B).
        wr = alpha_prods[steps - 1].sqrt() * (1 - alpha[steps]) / denum  # (B).
        x = wx[:, None, None] * embeddings.payload + wr[:, None, None] * reconstruction.payload  # (B, L, D).
        if (self._prediction == "sample") or self.training:
            sigma2 = (1 - alpha[steps]) * (1 - alpha_prods[steps - 1]) / denum  # (B).
            x = x + torch.randn_like(x) * sigma2[:, None, None].sqrt()
        elif self._prediction != "mode":
            raise ValueError(f"Unknown prediction mode: {self._prediction}")
        return PaddedBatch(x, embeddings.seq_lens)

    def _compute_time_deltas(self, x):
        """Replace timestamps with time deltas."""
        field = self._timestamps_field
        deltas = x.payload[field].clone()
        deltas[:, 1:] -= x.payload[field][:, :-1]
        deltas[:, 0] = 0
        deltas.clip_(min=0, max=self._max_time_delta)
        x = x.clone()
        x.payload[field] = deltas
        return x

    def _diffusion_loss(self, conditions, targets):
        # Conditions: (B, D).
        # Targets: (B, 1 + K).
        losses = {}
        metrics = {}

        b = len(targets)
        # Embed.
        embeddings = self._embedder(self._compute_time_deltas(targets))  # (B, 1 + K, D).
        embeddings = PaddedBatch(embeddings.payload[:, 1:], (embeddings.seq_lens - 1).clip(min=0))  # (B, K, D).

        # Corrupt.
        steps = torch.randint(1, self._generation_steps + 1, [b], device=embeddings.device)  # (B).
        steps[random.randint(0, b - 1)] = 1  # Need at least one 1.
        corrupted = self._corrupt(embeddings, steps)  # (B, K, D).

        # Reconstruct.
        reconstructed = self._denoiser(corrupted, conditions, steps)  # (B, K, D).

        # Compute losses.
        if self._detach_embeddings_from_step is False:
            reconstruction_target = embeddings.payload
        else:
            reconstruction_target = torch.where(steps[:, None, None] > self._detach_embeddings_from_step, embeddings.payload.detach(), embeddings.payload)
        losses["diffusion"] = ScaleGradient.apply((reconstructed.payload - reconstruction_target).square().mean(), self._diffusion_loss_weight)

        decoded = self._decoder(PaddedBatch(reconstructed.payload.detach() if self._detach_decoder else reconstructed.payload,
                                            reconstructed.seq_lens))  # (B, K, D).
        decoded = PaddedBatch(torch.cat([decoded.payload, torch.empty_like(decoded.payload[:, :1])], 1), decoded.seq_lens)  # (B, K + 1, D).
        decoder_losses, metrics = self._next_item(targets, decoded, None)
        metrics.update({"decoder_" + k: v for k, v in metrics.items()})
        losses.update({"decoder_" + k: ScaleGradient.apply(v, self._decoder_loss_weight)
                       for k, v in decoder_losses.items()})

        losses["embedder_regularizer"] = ScaleGradient.apply(self._alpha_prods[self._generation_steps] * embeddings.payload.square().mean(),
                                                             self._embedder_regularizer)
        return losses, metrics

    def forward(self, inputs, outputs, states):
        """Extract targets and compute loss between predictions and targets.

        Args:
            inputs: Input features with shape (B, L).
            outputs: Predicted values with shape (B, L, P).
            states (unused): Hidden model states with shape (N, B, L, D), where N is the number of layers.

        Returns:
            Losses dict and metrics dict.
        """
        # Join targets before windowing.
        b, l = inputs.shape
        targets = torch.stack([inputs.payload[name] for name in self.fields], -1)  # (B, L, D).

        # Extract windows.
        targets = self.extract_windows(targets, self._k + 1)   # (B, L - k, k + 1, D).
        assert targets.shape[:3] == (b, max(l - self._k, 0), self._k + 1)
        lengths = (inputs.seq_lens - self._k).clip(min=0)

        # Truncate lengths to match targets.
        outputs = PaddedBatch(outputs.payload[:, :targets.shape[1]], lengths)

        # Apply step.
        if self._loss_step > 1:
            lengths = (lengths - self._loss_step - 1).div(self._loss_step, rounding_mode="floor").clip(min=-1) + 1
            targets = targets[:, self._loss_step::self._loss_step]  # (B, L', k + 1, D).
            outputs = PaddedBatch(outputs.payload[:, self._loss_step::self._loss_step], lengths)  # (B, L', P).

        # Split targets.
        assert len(self.fields) == targets.shape[-1]
        windows = {name: targets[..., i].to(inputs.payload[name].dtype) for i, name in enumerate(self.fields)}  # (B, L', k + 1).
        targets = PaddedBatch(windows, lengths, inputs.seq_names)  # (B, L', k + 1).

        # Select by mask.
        mask = targets.seq_len_mask.bool()  # (B, L').
        lengths = torch.full([mask.sum().item()], self._k + 1, device=mask.device)  # (V).
        targets = PaddedBatch({k: v[mask] for k, v in targets.payload.items()},
                              lengths, targets.seq_names)  # (V, k + 1).
        outputs = outputs.payload[:, :mask.shape[1]][mask]  # (V, P).

        losses, metrics = self._diffusion_loss(conditions=outputs, targets=targets)
        return losses, metrics

    def predict_next(self, outputs, states, fields=None, logits_fields_mapping=None):
        """Predict next events.

        Args:
            outputs: Model outputs.
            states: Hidden model states with shape (N, B, L, D), where N is the number of layers.
            fields: The fields to predict next values for. By default, predict all fields.
            logits_fields_mapping: A mapping from field to the output logits field to predict logits for.

        Returns:
            PaddedBatch with predictions with shape (B, L) or (B, L, C) for logits.
        """
        sequences = self.predict_next_k(outputs, states,
                                        fields=fields,
                                        logits_fields_mapping=logits_fields_mapping)  # (B, L, K) or (B, L, K, C).
        result = {}
        for k, v in sequences.payload.items():
            result[k] = v[:, :, 0]  # (B, L) or (B, L, C).
        return PaddedBatch(result, sequences.seq_lens)

    def predict_next_k(self, outputs, states, fields=None, logits_fields_mapping=None):
        """Predict K future events.

        Args:
            outputs: Model outputs with shape (B, L, D).
            states (unused): Hidden model states with shape (N, B, L, D), where N is the number of layers.
            fields: The fields to predict next values for. By default, predict all fields.
            logits_fields: A mapping from field to the output logits field to predict logits for.

        Returns:
            PaddedBatch with predictions with shape (B, L, K) or (B, L, K, C) for logits.
        """
        mask = outputs.seq_len_mask.bool()  # (B, L).
        conditions = outputs.payload[mask]  # (V, P).

        # Generate embeddings.
        b = len(conditions)
        x = self._noise(b, device=conditions.device, dtype=outputs.payload.dtype)  # (V, N, D).
        for i in range(self._generation_steps, 0, -1):  # N -> 1.
            x = self._denoising_step(x, conditions, i)
        x = self._decoder(x)  # (V, N, D).

        # Predict.
        predictions = self._next_item.predict_next(x, None,
                                                   fields=fields,
                                                   logits_fields_mapping=logits_fields_mapping)  # (V, N) or (V, N, C).

        # Gather results.
        sequences = {k: torch.zeros(outputs.shape[0], outputs.shape[1], self._k,
                                    dtype=v.dtype, device=v.device).masked_scatter_(mask.unsqueeze(-1), v)
                     for k, v in predictions.payload.items() if v.ndim == 2}  # (B, L, N).
        sequences |= {k: torch.zeros(outputs.shape[0], outputs.shape[1], self._k, v.shape[2],
                                     dtype=v.dtype, device=v.device).masked_scatter_(mask.unsqueeze(-1).unsqueeze(-1), v)
                      for k, v in predictions.payload.items() if v.ndim == 3}  # (B, L, N, C).
        sequences = PaddedBatch(sequences, torch.full([len(outputs)], self._k, dtype=torch.long, device=x.device))

        # Revert time delta.
        self.revert_delta_and_sort_time_inplace(sequences)
        return sequences
