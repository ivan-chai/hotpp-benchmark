import torch

from hotpp.data import PaddedBatch
from .detection import DetectionLoss
from .horizon_diffusion_v3 import HorizonDiffusionLossV3


class HorizonDiffusionLossV4(HorizonDiffusionLossV3):
    """V3 + cross-slot self-attention before per-slot decoder.

    Motivation:
    The monolithic v1 decoder ``Linear(K*D -> K*P_base)`` accidentally
    provided full O(K^2) cross-slot mixing on top of the bidirectional
    GRU denoiser. Per-slot decoding (V3) preserves slot identity but
    loses that mixing capacity, which empirically hurts label prediction
    (decoder labels loss ~0.85 vs ~0.36 in v1_balance_b).

    V4 inserts a TransformerEncoder over the K denoised slots between
    the GRU denoiser and the per-slot Head. Slots can re-distribute
    information among themselves with structured attention while each
    final output still comes from a slot-specific projection.

    Args (in addition to V3):
        attention_d_model: Attention working dimension. Latents are projected
            from ``embedder.output_size`` up to this value, then back down to
            the per-slot decoder input. ``None`` means use embedder.output_size
            without projection.
        attention_layers: Number of TransformerEncoderLayer blocks.
        attention_heads: Number of MHA heads.
        attention_ff_dim: Feed-forward hidden dim inside each block.
        attention_dropout: Dropout probability inside the encoder.
    """

    def __init__(
        self,
        *args,
        attention_d_model=64,
        attention_layers=2,
        attention_heads=4,
        attention_ff_dim=128,
        attention_dropout=0.1,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self._attention_d_model = attention_d_model
        self._slot_in_proj = None
        self._slot_out_proj = None
        self._slot_attention = None

        if isinstance(self._next_item, DetectionLoss):
            d_emb = self._embedder.output_size
            d_model = attention_d_model if attention_d_model is not None else d_emb
            if attention_d_model is not None and attention_d_model != d_emb:
                self._slot_in_proj = torch.nn.Linear(d_emb, d_model)
                self._slot_out_proj = torch.nn.Linear(d_model, d_emb)
            encoder_layer = torch.nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=attention_heads,
                dim_feedforward=attention_ff_dim,
                dropout=attention_dropout,
                batch_first=True,
                activation="relu",
            )
            self._slot_attention = torch.nn.TransformerEncoder(
                encoder_layer,
                num_layers=attention_layers,
            )

    def _mix_slots(self, latents):
        """Apply cross-slot attention. ``latents`` is a PaddedBatch (V, K, D)."""
        if self._slot_attention is None:
            return latents
        x = latents.payload
        if self._slot_in_proj is not None:
            x = self._slot_in_proj(x)
        x = self._slot_attention(x)
        if self._slot_out_proj is not None:
            x = self._slot_out_proj(x)
        return PaddedBatch(x, latents.seq_lens)

    def _diffusion_loss(self, conditions, targets, det_inputs=None, det_mask=None, det_lengths=None):
        # Override only the DetectionLoss branch to inject slot mixing.
        if not isinstance(self._next_item, DetectionLoss):
            return super()._diffusion_loss(
                conditions=conditions,
                targets=targets,
                det_inputs=det_inputs,
                det_mask=det_mask,
                det_lengths=det_lengths,
            )

        # Replicate the V3 forward path manually so we can hook in the mixer.
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

        # Cross-slot mixing applied to the (possibly detached) latents fed to decoder.
        decoder_input_payload = (
            reconstructed.payload.detach() if self._detach_decoder else reconstructed.payload
        )
        mixed = self._mix_slots(PaddedBatch(decoder_input_payload, reconstructed.seq_lens))

        decoded = self._decoder(mixed)  # (V, K, P_base)
        decoded_flat = decoded.payload.flatten(1)
        b_det, l_det = det_mask.shape
        det_payload = torch.zeros(
            b_det, l_det, decoded_flat.shape[-1],
            dtype=decoded_flat.dtype, device=decoded_flat.device,
        )
        det_payload.masked_scatter_(det_mask.unsqueeze(-1), decoded_flat)
        det_outputs = PaddedBatch(det_payload, det_lengths)
        decoder_losses, metrics = self._next_item(det_inputs, det_outputs, None)

        t_decoder = time.perf_counter()
        metrics.update({"decoder_" + k: v for k, v in metrics.items()})
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

        # Cross-slot mixing before per-slot decoding.
        mixed = self._mix_slots(x)
        decoded = self._decoder(mixed)
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
            det_outputs,
            states,
            fields=fields,
            logits_fields_mapping=logits_fields_mapping,
        )

        # Same presence_selection logic as V3.
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
