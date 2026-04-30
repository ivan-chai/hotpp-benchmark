"""V44: V43 + RNN context as additional decoder input.

Architecture:
- Diffusion supervision: positional, no horizon mask (inherited from V42->V43).
- Decoder input per slot k: concat([denoised_latent_k, query_k, context])
  -> shared MLP [128, 256] -> (presence, time, label).

Unifies all available signals:
- denoised_latent_k: diffusion's K-way representation
- query_k: DeTPP-style learnable per-slot identity
- context: direct RNN hidden state (what pure DeTPP relies on)

Should be >= V14 (which uses ConditionalHead on context only) because the
decoder additionally has access to the diffusion latent.
"""
import torch

from hotpp.data import PaddedBatch
from .detection import DetectionLoss
from .horizon_diffusion_v42 import HorizonDiffusionLossV42


class ContextQueryAugmentedHead(torch.nn.Module):
    """Decoder that fuses (latent_k, query_k, context) per slot.

    Forward expects ``x`` to be a PaddedBatch of latents shape (V, K, D_latent)
    and ``context`` (V, D_context) passed via ``forward(x, context=...)``.
    The shared MLP processes concatenated (latent, query, context) per slot.
    """

    def __init__(
        self,
        latent_dim,
        context_dim,
        output_size,
        k,
        query_size=64,
        hidden_dims=(128, 256),
        use_batch_norm=True,
        query_init_scale=0.02,
    ):
        super().__init__()
        self.queries = torch.nn.Parameter(
            torch.randn(k, query_size) * float(query_init_scale)
        )
        self.k = k
        self.output_size = output_size
        self.context_dim = context_dim

        layers = []
        last_dim = latent_dim + query_size + context_dim
        if use_batch_norm:
            layers.append(torch.nn.BatchNorm1d(last_dim))
        for dim in hidden_dims or []:
            layers.append(torch.nn.Linear(last_dim, dim, bias=not use_batch_norm))
            if use_batch_norm:
                layers.append(torch.nn.BatchNorm1d(dim))
            layers.append(torch.nn.ReLU())
            last_dim = dim
        layers.append(torch.nn.Linear(last_dim, output_size))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x, context=None):
        latents = x.payload  # (V, K, D_latent)
        v = latents.shape[0]
        if latents.shape[1] != self.k:
            raise ValueError(f"Expected K={self.k}, got {latents.shape[1]}")
        if context is None:
            raise ValueError("ContextQueryAugmentedHead requires `context` arg.")
        if context.shape != (v, self.context_dim):
            raise ValueError(
                f"context shape {tuple(context.shape)} != expected {(v, self.context_dim)}"
            )

        queries = self.queries.unsqueeze(0).expand(v, -1, -1)  # (V, K, D_query)
        context_per_slot = context.unsqueeze(1).expand(-1, self.k, -1)  # (V, K, D_context)
        combined = torch.cat([latents, queries, context_per_slot], dim=-1)
        flat = combined.flatten(0, 1)
        out = self.mlp(flat)
        out = out.view(v, self.k, self.output_size)
        return PaddedBatch(out, x.seq_lens)


class HorizonDiffusionLossDetppAllCompsInDecoder(HorizonDiffusionLossV42):
    """V42 with a context-and-query-augmented decoder."""

    def __init__(
        self,
        *args,
        query_size=64,
        query_hidden_dims=(128, 256),
        query_use_batch_norm=True,
        query_init_scale=0.02,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if isinstance(self._next_item, DetectionLoss):
            inner_input_size = self._next_item._next_item.input_size
            context_dim = self._denoiser.condition_size
            self._decoder = ContextQueryAugmentedHead(
                latent_dim=self._embedder.output_size,
                context_dim=context_dim,
                output_size=inner_input_size,
                k=self._k,
                query_size=query_size,
                hidden_dims=tuple(query_hidden_dims) if query_hidden_dims else None,
                use_batch_norm=query_use_batch_norm,
                query_init_scale=query_init_scale,
            )
            self._needs_context_decoder = True
        else:
            self._needs_context_decoder = False

    # We need the decoder to receive `context` (per-slot conditioning vector).
    # The cleanest way is to override `_diffusion_loss` and `predict_next_k`
    # to call self._decoder(reconstructed, context=conditions).
    def _diffusion_loss(self, conditions, targets, det_inputs=None, det_mask=None, det_lengths=None):
        if not isinstance(self._next_item, DetectionLoss):
            return super()._diffusion_loss(
                conditions=conditions, targets=targets,
                det_inputs=det_inputs, det_mask=det_mask, det_lengths=det_lengths,
            )

        import random
        import time
        from .common import ScaleGradient

        losses = {}
        metrics = {}
        batch_size = len(targets)
        t0 = time.perf_counter()

        embeddings = self._embed_targets(targets)

        steps = torch.randint(1, self._generation_steps + 1, [batch_size], device=embeddings.device)
        steps[random.randint(0, batch_size - 1)] = 1
        corrupted = self._corrupt(embeddings, steps)
        reconstructed = self._denoiser(corrupted, conditions, steps)
        t_denoise = time.perf_counter()

        if self._detach_embeddings_from_step is False:
            reconstruction_target = embeddings.payload
        else:
            reconstruction_target = torch.where(
                steps[:, None, None] > self._detach_embeddings_from_step,
                embeddings.payload.detach(),
                embeddings.payload,
            )
        losses["diffusion"] = ScaleGradient.apply(
            (reconstructed.payload - reconstruction_target).square().mean(),
            self._diffusion_loss_weight,
        )

        # Decoder takes (latents, context) and returns (V, K, P_base).
        decoder_input_payload = (
            reconstructed.payload.detach() if self._detach_decoder else reconstructed.payload
        )
        decoded = self._decoder(
            PaddedBatch(decoder_input_payload, reconstructed.seq_lens),
            context=conditions,
        )
        decoded_flat = decoded.payload.flatten(1)

        b_det, l_det = det_mask.shape
        det_payload = torch.zeros(
            b_det, l_det, decoded_flat.shape[-1],
            dtype=decoded_flat.dtype, device=decoded_flat.device,
        )
        det_payload.masked_scatter_(det_mask.unsqueeze(-1), decoded_flat)
        det_outputs = PaddedBatch(det_payload, det_lengths)
        decoder_losses, det_metrics = self._next_item(det_inputs, det_outputs, None)

        t_decoder = time.perf_counter()
        metrics.update({"decoder_" + k: v for k, v in det_metrics.items()})
        losses.update(
            {"decoder_" + k: ScaleGradient.apply(v, self._decoder_loss_weight) for k, v in decoder_losses.items()}
        )
        losses["embedder_regularizer"] = ScaleGradient.apply(
            self._alpha_prods[self._generation_steps] * embeddings.payload.square().mean(),
            self._embedder_regularizer,
        )
        metrics["perf_embed_denoise_s"] = t_denoise - t0
        metrics["perf_decoder_loss_s"] = t_decoder - t_denoise
        metrics["perf_total_loss_s"] = t_decoder - t0
        return losses, metrics

    def predict_next_k(self, outputs, states, fields=None, logits_fields_mapping=None):
        if not isinstance(self._next_item, DetectionLoss):
            return super().predict_next_k(outputs, states, fields=fields, logits_fields_mapping=logits_fields_mapping)

        from ..fields import PRESENCE, PRESENCE_PROB

        mask = outputs.seq_len_mask.bool()
        conditions = outputs.payload[mask]

        batch_size = len(conditions)
        x = self._noise(batch_size, device=conditions.device, dtype=outputs.payload.dtype)
        for step in range(self._generation_steps, 0, -1):
            x = self._denoising_step(x, conditions, step)

        decoded = self._decoder(x, context=conditions)
        decoded_flat = decoded.payload.flatten(1)
        bsz, seq_len = outputs.shape
        det_payload = torch.zeros(
            bsz, seq_len, decoded_flat.shape[-1],
            dtype=decoded_flat.dtype, device=decoded_flat.device,
        )
        det_payload.masked_scatter_(mask.unsqueeze(-1), decoded_flat)
        det_outputs = PaddedBatch(det_payload, outputs.seq_lens)

        if fields is None:
            fields = set(self._next_item.data_fields)
        sequences = self._next_item.predict_next_k(
            det_outputs, states, fields=fields, logits_fields_mapping=logits_fields_mapping,
        )

        # Reuse V3's presence_selection logic.
        presence_logits_field = PRESENCE + "_logits"
        if presence_logits_field in sequences.payload:
            presence_logits = sequences.payload[presence_logits_field].squeeze(-1)
            calibrated_logits = presence_logits - self._next_item._matching_thresholds - self._presence_threshold_bias
            if self._presence_selection == "topk":
                target_k = self._topk_target_length
                if target_k is None:
                    matching_priors = getattr(self._next_item, "_matching_priors", None)
                    if matching_priors is not None:
                        target_k = max(int(round(float(matching_priors.sum().item()))), 1)
                    else:
                        target_k = max(int(round(self._k / 4)), 1)
                target_k = min(max(int(target_k), 1), presence_logits.shape[-1])
                topk_indices = calibrated_logits.topk(target_k, dim=-1).indices
                presence_mask = torch.zeros_like(presence_logits, dtype=torch.bool)
                presence_mask.scatter_(dim=-1, index=topk_indices, value=True)
                sequences.payload[PRESENCE] = presence_mask
                sequences.payload[PRESENCE_PROB] = torch.sigmoid(calibrated_logits)
            elif self._presence_selection == "calibrated_floor":
                if self._min_presence_count is not None:
                    target_k = int(self._min_presence_count)
                else:
                    target_k = max(int(round(float(self._next_item._matching_priors.sum().item()))), 1)
                target_k = min(target_k, presence_logits.shape[-1])
                presence_mask = calibrated_logits > 0
                current_counts = presence_mask.sum(-1)
                need_more = current_counts < target_k
                if need_more.any():
                    topk_indices = calibrated_logits.topk(target_k, dim=-1).indices
                    floor_mask = torch.zeros_like(presence_mask)
                    floor_mask.scatter_(dim=-1, index=topk_indices, value=True)
                    presence_mask = torch.where(need_more.unsqueeze(-1), floor_mask, presence_mask)
                sequences.payload[PRESENCE] = presence_mask
                sequences.payload[PRESENCE_PROB] = torch.sigmoid(calibrated_logits)
            elif self._presence_selection == "zero":
                sequences.payload[PRESENCE] = presence_logits > 0
                sequences.payload[PRESENCE_PROB] = torch.sigmoid(presence_logits)
            elif self._use_detection_calibration_threshold:
                sequences.payload[PRESENCE] = calibrated_logits > 0
                sequences.payload[PRESENCE_PROB] = torch.sigmoid(calibrated_logits)
            else:
                sequences.payload[PRESENCE] = presence_logits > 0
                sequences.payload[PRESENCE_PROB] = torch.sigmoid(presence_logits)
        return sequences
