"""V47: V44 + multi-sample inference (variance reduction).

Идея: V44 уже хорош, но diffusion стохастичен. На инференсе делаем N
независимых runs денойзинга со СВОИМИ noise seeds, усредняем decoded
predictions ПЕРЕД Hungarian matching. Это превращает stochasticity
диффузии в variance reduction → должно помочь детерминистской T-mAP.

Тренировка идентична V44 — можно просто загрузить V44 чекпоинт через
init_from_checkpoint + test_only и поменять только predict_next_k.
"""
import torch

from hotpp.data import PaddedBatch
from .detection import DetectionLoss
from ..fields import PRESENCE, PRESENCE_PROB
from .horizon_diffusion_v44 import HorizonDiffusionLossV44


class HorizonDiffusionLossV47(HorizonDiffusionLossV44):
    def __init__(self, *args, n_inference_samples=5, **kwargs):
        super().__init__(*args, **kwargs)
        self._n_inference_samples = int(n_inference_samples)

    def predict_next_k(self, outputs, states, fields=None, logits_fields_mapping=None):
        if self._n_inference_samples <= 1 or not isinstance(self._next_item, DetectionLoss):
            return super().predict_next_k(outputs, states, fields=fields, logits_fields_mapping=logits_fields_mapping)

        mask = outputs.seq_len_mask.bool()
        conditions = outputs.payload[mask]
        batch_size = len(conditions)

        # Run N independent denoising chains with different noise; average decoded outputs.
        decoded_samples = []
        for _ in range(self._n_inference_samples):
            x = self._noise(batch_size, device=conditions.device, dtype=outputs.payload.dtype)
            for step in range(self._generation_steps, 0, -1):
                x = self._denoising_step(x, conditions, step)
            decoded = self._decoder(x, context=conditions)  # (V, K, P_base)
            decoded_samples.append(decoded.payload)

        # Average raw decoded vector (presence_logit, time, label_logits) across samples.
        # Averaging logits is equivalent to geometric mean of probabilities — sensible.
        mean_decoded = torch.stack(decoded_samples, dim=0).mean(dim=0)  # (V, K, P_base)
        decoded_flat = mean_decoded.flatten(1)  # (V, K*P_base)

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

        # Same presence_selection block as V44 / parent.
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
