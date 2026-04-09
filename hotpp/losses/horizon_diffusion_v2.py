import torch

from hotpp.data import PaddedBatch
from ..fields import PRESENCE, PRESENCE_PROB
from .detection import DetectionLoss
from .horizon_diffusion_v1 import HorizonDiffusionLossV1


class HorizonDiffusionLossV2(HorizonDiffusionLossV1):
    """Horizon diffusion with a cleaner DeTPP inference interface.

    Differences from V1:
    - keeps the same training objective;
    - allows choosing how PRESENCE_PROB is produced for downstream metrics;
    - defaults to raw detection probabilities, matching DetectionLoss semantics.
    """

    def __init__(self, *args, presence_probability_mode="raw", **kwargs):
        super().__init__(*args, **kwargs)
        if presence_probability_mode not in {"raw", "calibrated_sigmoid", "calibrated_exp"}:
            raise ValueError(f"Unknown presence_probability_mode: {presence_probability_mode}.")
        self._presence_probability_mode = presence_probability_mode

    def _compute_presence_probabilities(self, presence_logprobs, calibrated_logprobs):
        if self._presence_probability_mode == "raw":
            return torch.exp(presence_logprobs).clip(max=1)
        if self._presence_probability_mode == "calibrated_exp":
            return torch.exp(calibrated_logprobs).clip(max=1)
        return torch.sigmoid(calibrated_logprobs)

    def predict_next_k(self, outputs, states, fields=None, logits_fields_mapping=None):
        if not isinstance(self._next_item, DetectionLoss):
            return super().predict_next_k(outputs, states, fields=fields, logits_fields_mapping=logits_fields_mapping)

        mask = outputs.seq_len_mask.bool()
        conditions = outputs.payload[mask]

        batch_size = len(conditions)
        x = self._noise(batch_size, device=conditions.device, dtype=outputs.payload.dtype)
        for step in range(self._generation_steps, 0, -1):
            x = self._denoising_step(x, conditions, step)

        decoded_values = self._decode_detection_sequence(x)
        bsz, seq_len = outputs.shape
        decoded_dim = decoded_values.shape[1]
        det_payload = torch.zeros(bsz, seq_len, decoded_dim, dtype=decoded_values.dtype, device=decoded_values.device)
        det_payload.masked_scatter_(mask.unsqueeze(-1), decoded_values)
        det_outputs = PaddedBatch(det_payload, outputs.seq_lens)

        if fields is None:
            fields = set(self._next_item.data_fields)
        sequences = self._next_item.predict_next_k(
            det_outputs,
            states,
            fields=fields,
            logits_fields_mapping=logits_fields_mapping,
        )

        presence_logits_field = PRESENCE + "_logits"
        if presence_logits_field not in sequences.payload:
            return sequences

        presence_logprobs = sequences.payload[presence_logits_field].squeeze(-1)
        calibrated_logprobs = presence_logprobs - self._next_item._matching_thresholds - self._presence_threshold_bias

        if self._presence_selection == "topk":
            target_k = self._topk_target_length
            if target_k is None:
                matching_priors = getattr(self._next_item, "_matching_priors", None)
                if matching_priors is not None:
                    target_k = max(int(round(float(matching_priors.sum().item()))), 1)
                else:
                    target_k = max(int(round(self._k / 4)), 1)
            target_k = min(max(int(target_k), 1), presence_logprobs.shape[-1])
            topk_indices = calibrated_logprobs.topk(target_k, dim=-1).indices
            presence_mask = torch.zeros_like(presence_logprobs, dtype=torch.bool)
            presence_mask.scatter_(dim=-1, index=topk_indices, value=True)
        elif self._presence_selection == "calibrated_floor":
            if self._min_presence_count is not None:
                target_k = int(self._min_presence_count)
            else:
                target_k = max(int(round(float(self._next_item._matching_priors.sum().item()))), 1)
            target_k = min(target_k, presence_logprobs.shape[-1])
            presence_mask = calibrated_logprobs > 0
            current_counts = presence_mask.sum(-1)
            need_more = current_counts < target_k
            if need_more.any():
                topk_indices = calibrated_logprobs.topk(target_k, dim=-1).indices
                floor_mask = torch.zeros_like(presence_mask)
                floor_mask.scatter_(dim=-1, index=topk_indices, value=True)
                presence_mask = torch.where(need_more.unsqueeze(-1), floor_mask, presence_mask)
        elif self._presence_selection == "zero":
            presence_mask = presence_logprobs > 0
        elif self._use_detection_calibration_threshold:
            presence_mask = calibrated_logprobs > 0
        else:
            presence_mask = presence_logprobs > 0

        sequences.payload[PRESENCE] = presence_mask
        sequences.payload[PRESENCE_PROB] = self._compute_presence_probabilities(
            presence_logprobs, calibrated_logprobs
        )
        return sequences
