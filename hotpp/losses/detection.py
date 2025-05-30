import torch

from torch_linear_assignment import batch_linear_assignment
from hotpp.data import PaddedBatch
from hotpp.utils.torch import deterministic, module_mode
from ..fields import PRESENCE, PRESENCE_PROB
from .common import ScaleGradient
from .next_item import NextItemLoss
from .next_k import NextKLoss


class DetectionLoss(NextKLoss):
    """The loss similar to Next-K, but with the matching loss.

    Args:
        next_item_loss: A base NextItemLoss for pairwise comparisons.
        k: The maximum number of future events to predict
            (must be larger than the average horizon sequence length).
        horizon: Predicted time interval.
        timestamps_field: The name of timestamps field used for ordering.
        caterogical_fields: The list of categorical features.
        loss_subset: The fraction of indices to compute the loss for
            (controls trade-off between the training speed and quality).
        drop_partial_windows: Compute the loss only for full-horizon ground truth windows.
            Turn off for datasets with short sequences. Possible values: True, False, and `calibration`,
            where `calibration` means drop partial windows only during calibration.
        prefetch_factor: Extract times more targets than predicted events (equal to `k`) for matching.
        match_weights: Weights of particular fields in matching cost computation.
        momentum: Activation statistics momentum value.
        next_item_adapter: String or dictionary with adapter names used for next-item predictions
            (either `head`, `first`, `mean`, `mode` or `label_mode`).
        next_item_loss_weight: The weight of the adapter-based next-item loss.
            Can be a dictionary with weights for each field.
    """
    def __init__(self, next_item_loss, k, horizon,
                 timestamps_field="timestamps",
                 categorical_fields=("labels",),
                 loss_subset=1, drop_partial_windows="calibration", prefetch_factor=1,
                 match_weights=None, momentum=0.1,
                 next_item_adapter="mean",
                 next_item_loss_weight=0,
                 next_item_trainable_scales=False
                 ):
        super().__init__(
            next_item_loss=next_item_loss,
            k=k,
            timestamps_field=timestamps_field
        )
        self._categorical_fields = categorical_fields
        self._horizon = horizon
        self._loss_subset = loss_subset
        if drop_partial_windows not in {True, False, "calibration"}:
            raise ValueError(f"Unknown drop_partial_windows value: {drop_partial_windows}")
        self._drop_partial_windows = drop_partial_windows
        self._match_weights = match_weights
        if self.get_delta_type(timestamps_field) != "start":
            raise ValueError("Need `start` delta time for the detection loss.")
        self._momentum = momentum
        try:
            self._next_item_adapter = dict(next_item_adapter)
        except ValueError:
            self._next_item_adapter = {k: next_item_adapter for k in self.data_fields}
        try:
            self._next_item_loss_weight = dict(next_item_loss_weight)
        except TypeError:
            self._next_item_loss_weight = {k: next_item_loss_weight for k in self.data_fields}
        self._prefetch_k = int(round(self._k * prefetch_factor))

        # Calibration statistics used for prediction.
        self.register_buffer("_matching_priors", torch.ones(k))
        self.register_buffer("_matching_thresholds", torch.zeros(k))

        if next_item_trainable_scales:
            self.next_offsets = torch.nn.ModuleDict({k: torch.nn.Parameter(torch.zeros([]))
                                                     for k in self.data_fields})
            self.next_scales = torch.nn.ModuleDict({k: torch.nn.Parameter(torch.ones([]))
                                                    for k in self.data_fields})
        else:
            self.next_offsets = {k: 0 for k in self.data_fields}
            self.next_scales = {k: 1 for k in self.data_fields}

    def update_calibration_statistics(self, matching, presence_logits):
        """Update calibration statistics.

        The method uses exponential smoothing to track head matching frequencies.
        These frequencies are used to choose the optimal presence threshold during inference.

        Args:
            matching: Loss matching with shape (B, L1, K).
            presence_logits: Predicted presence logits with shape (B, L2, K).
        """
        # (B, L1, K), (B, L2, K).
        matching = matching.payload[matching.seq_len_mask]  # (V, K).
        if len(matching) > 0:
            means = (matching >= 0).float().mean(0)  # (K).
            matching_priors = self._matching_priors * (1 - self._momentum) + means * self._momentum

        presence_logits = presence_logits.payload[presence_logits.seq_len_mask]  # (V, K).
        if len(presence_logits) > 0:
            presence_logits = torch.sort(presence_logits, dim=0)[0]  # (V, K).
            indices = (1 - self._matching_priors) * len(presence_logits)
            bottom_indices = indices.floor().long().clip(max=len(presence_logits) - 1)  # (K).
            up_indices = indices.ceil().long().clip(max=len(presence_logits) - 1)  # (K).
            bottom_quantiles = presence_logits.take_along_dim(bottom_indices[None], 0).squeeze(0)  # (K).
            up_quantiles = presence_logits.take_along_dim(up_indices[None], 0).squeeze(0)  # (K).
            quantiles = 0.5 * (bottom_quantiles + up_quantiles)
            matching_thresholds = self._matching_thresholds * (1 - self._momentum) + quantiles * self._momentum

        if torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            torch.distributed.all_reduce(matching_priors, torch.distributed.ReduceOp.SUM)
            matching_priors /= world_size
            assert (matching_priors <= 1).all(), "Distributed reduction failed."
            torch.distributed.all_reduce(matching_thresholds, torch.distributed.ReduceOp.SUM)
            matching_thresholds /= world_size

        self._matching_priors.copy_(matching_priors)
        self._matching_thresholds.copy_(matching_thresholds)

    @property
    def num_events(self):
        return self._k

    @property
    def data_fields(self):
        return [field for field in self.fields if field != PRESENCE]

    @property
    def input_size(self):
        return self._k * self._next_item.input_size  # One for the presence score.

    def forward(self, inputs, outputs, states):
        """Extract targets and compute loss between predictions and targets.

        Args:
            inputs: Input features with shape (B, L).
            outputs: Predicted values with shape (B, L, P).
            states: Hidden model states with shape (N, B, L, D), where N is the number of layers.

        Returns:
            Losses dict and metrics dict.
        """
        indices, matching, losses, matching_metrics = self.get_subset_matching(inputs, outputs)
        # (B, I), (B, I, K), (B, I, K, T), dict.

        # Update statistics.
        if self.training:
            with torch.no_grad():
                b, l = outputs.shape
                reshaped_outputs = PaddedBatch(outputs.payload.reshape(-1, self._k, self._next_item.input_size),
                                               torch.full([b * l], self._k, dtype=torch.long, device=outputs.device))  # (BL, K, P).
                reshaped_states = states.flatten(1, 2).unsqueeze(2) if states is not None else None  # (N, BL, 1, D).
                with module_mode(self, training=False):
                    presence_logits = self._next_item.predict_next(
                        reshaped_outputs, reshaped_states,
                        fields=set(),
                        logits_fields_mapping={PRESENCE: "_presence_logits"}
                    ).payload["_presence_logits"]  # (BL, K, 1).
                presence_logits = presence_logits.reshape(b, l, self._k)  # (B, L, K).
                if self._drop_partial_windows in {True, "calibration"}:
                    full_matching = PaddedBatch(matching.payload, indices.payload["full_mask"].sum(1))
                    presence_logits = PaddedBatch(presence_logits, (outputs.seq_lens - self._prefetch_k).clip(min=0))
                else:
                    full_matching = matching
                    presence_logits = PaddedBatch(presence_logits, outputs.seq_lens)
                self.update_calibration_statistics(full_matching, presence_logits)

        # Compute matching losses.
        if (matching.payload < 0).all():
            losses = {name: outputs.payload.mean() * 0 for name in self.fields}
            return losses, matching_metrics

        index_mask = indices.seq_len_mask  # (B, I).
        matching_mask = matching.payload >= 0  # (B, I, K).
        matching = matching.payload.clip(min=0)  # (B, I, K).

        losses = {k: v.take_along_dim(matching.unsqueeze(3), 3).squeeze(3)
                  for k, v in losses.items()}  # (B, I, K).

        pos_presence = losses.pop("_presence")
        neg_presence = -losses.pop("_presence_neg")
        presence_losses = torch.where(matching_mask, pos_presence, neg_presence)

        losses = {k: v[matching_mask].mean() for k, v in losses.items()}
        losses["_presence"] = presence_losses[index_mask].mean()

        # Compute next-item loss.
        if any(weight > 0 for weight in self._next_item_loss_weight.values()):
            predictions = self.predict_next(outputs, states,
                                            fields=self.data_fields,
                                            logits_fields_mapping={k: f"_{k}_logits" for k in self._categorical_fields})  # (B, L).
            # A workaround for "start" time delta scheme in the next-item loss function.
            fixed_predictions = {}
            for field in self.data_fields:
                if field in self._categorical_fields:
                    fixed_predictions[field] = predictions.payload[f"_{field}_logits"][:, :-1].flatten(0, 1).unsqueeze(1)  # (BL, 1, C).
                else:
                    fixed_predictions[field] = predictions.payload[field][:, :-1].flatten()[:, None, None]  # (BL, 1, 1).
            b, l = inputs.shape
            fixed_times = inputs.payload[self._timestamps_field]  # (B, L).
            fixed_times = torch.stack([fixed_times[:, :-1], fixed_times[:, 1:]], 2).flatten(0, 1)  # (BL, 2).
            fixed_inputs = {}
            for field in self.data_fields:
                if field == self._timestamps_field:
                    fixed_inputs[field] = fixed_times
                else:
                    fixed_inputs[field] = inputs.payload[field][:, 1:].flatten(0, 1).unsqueeze(1).repeat(1, 2)
            fixed_inputs = PaddedBatch(fixed_inputs,
                                       torch.full([b * (l - 1)], 2, device=inputs.device))  # (BL, 2).
            fixed_states = states[:, :, :-1].flatten(1, 2).unsqueeze(2) if states is not None else None  # (N, BL, 1, D).

            next_item_losses, _ = self._next_item(fixed_inputs, fixed_predictions, fixed_states, reduction="none")  # (BL, 1).
            mask = inputs.seq_len_mask[:, 1:].flatten()  # (BL).
            next_item_losses = {k: v[mask].mean() for k, v in next_item_losses.items()}
            for field in self.data_fields:
                losses[f"next_item_{field}"] = ScaleGradient.apply(next_item_losses[field], self._next_item_loss_weight[field])
        return losses, matching_metrics

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
        # Add logits to the prediction fields.
        logits_fields_mapping = dict(logits_fields_mapping or {})
        for field in [PRESENCE] + list(self._categorical_fields):
            if field not in logits_fields_mapping:
                logits_fields_mapping[field] = field + "_logits"
        presence_logits_field = logits_fields_mapping[PRESENCE]

        adapters = set(self._next_item_adapter.values())
        unknown_adapters = adapters - {"head", "first", "mean", "mode", "label_mode"}
        if unknown_adapters:
            raise ValueError(f"Unknown adapters: {unknown_adapters}.")

        next_values = {}

        if "head" in adapters:
            b, l, _ = outputs.payload.shape
            head_outputs = outputs.payload.reshape(b, l, self._k, -1)
            head_outputs = PaddedBatch(head_outputs[:, :, 0], outputs.seq_lens)
            head_predictions = self._next_item.predict_next(head_outputs, states, fields, logits_fields_mapping)
            for field, adapter in self._next_item_adapter.items():
                if adapter == "head":
                    if field in fields:
                        next_values[field] = head_predictions.payload[field]
                    logits_field = logits_fields_mapping.get(field, None)
                    if logits_field:
                        next_values[logits_field] = head_predictions.payload[logits_field]

        if adapters - {"head"}:
            # Reshape and apply the base predictor.
            b, l = outputs.shape
            lengths = outputs.seq_lens
            reshaped_outputs = PaddedBatch(outputs.payload.reshape(b * l, self._k, self._next_item.input_size),
                                           torch.full([b * l], self._k, device=outputs.device))  # (BL, K, P).
            states = states.reshape(len(states), b * l, 1, -1) if states is not None else None  # (N, BL, 1, D).
            predictions = self._next_item.predict_next(reshaped_outputs, states,
                                                       fields=fields,
                                                       logits_fields_mapping=logits_fields_mapping)  # (BL, K) or (BL, K, C).

            # Reshape, predict presence, and sort.
            sequences = PaddedBatch({k: v.reshape(b, l, self._k, *v.shape[2:]) for k, v in predictions.payload.items()},
                                    lengths)  # (B, L, K) or (B, L, K, C).
            presence = sequences.payload[presence_logits_field].squeeze(-1) > self._matching_thresholds  # (B, L, K).
            sequences = PaddedBatch(sequences.payload | {PRESENCE: presence}, sequences.seq_lens)
            self.revert_delta_and_sort_time_inplace(sequences)
            # Sequences contain time shift from the last seen timestamps.
            # Events are sorted by timestamp.

            # Prepare data.
            presence = sequences.payload[PRESENCE]
            presence_logits = sequences.payload[presence_logits_field].squeeze(-1)  # (B, L, K).
            presence_logits = presence_logits.detach()  # Don't pass gradient to presence during next-item loss computation.
            log_presence = torch.nn.functional.logsigmoid(presence_logits)  # (B, L, K).
            log_not_presence = torch.nn.functional.logsigmoid(-presence_logits)  # (B, L, K).
            assert log_presence.ndim == 3
            log_probs = {
                field: torch.nn.functional.log_softmax(sequences.payload[logits_fields_mapping[field]], dim=-1)
                for field in self._categorical_fields if field in fields
            } # (B, L, K, C).

            # Compute probability of each event being the first.
            # log_cum_prod is equal to log of 1, (1 - p1), (1 - p1)(1 - p2), ...
            # log_weights are equal to log normalized p1, (1 - p1)p2, (1 - p1)(1 - p2)p3, ...
            roll_log_not_presence = torch.cat([torch.zeros_like(log_not_presence[..., :1]), log_not_presence[..., :-1]], -1)   # (B, L, K).
            with deterministic(False):
                log_cum_prod = roll_log_not_presence.cumsum(-1)  # (B, L, K).
            log_weights = torch.nn.functional.log_softmax(log_cum_prod + log_presence, -1)  # (B, L, K).
            weighted_logits = {k: v + log_weights.unsqueeze(-1) for k, v in log_probs.items()}  # (B, L, K, C).

            # Find the first and the most probable event.
            indices = {}
            if "first" in adapters:
                # Get the first event with presence logit greater than the threshold.
                indices["first"] = (torch.arange(presence.shape[-1] - 1, -1, -1, device=presence.device) * presence).argmax(-1)  # (B, L).
            if "mode" in adapters:
                # Get the event with maximum probability of being the first.
                indices["mode"] = log_weights.argmax(-1)  # (B, L).

            for field, adapter in self._next_item_adapter.items():
                if (adapter == "head") or (field not in set(fields) | set(logits_fields_mapping)):
                    continue
                base_field = field
                if field in self._categorical_fields:
                    field = logits_fields_mapping[field]
                if adapter == "mean":
                    if base_field in self._categorical_fields:
                        next_values[field] = weighted_logits[base_field].logsumexp(2)  # (B, L, C).
                    else:
                        next_values[field] = (sequences.payload[field] * log_weights.exp()).sum(-1)  # (B, L).
                elif adapter == "label_mode":
                    seq_values = sequences.payload[field]
                    shaped_indices = weighted_logits[field].max(-1)[0].argmax(-1).unsqueeze(-1)  # (B, L, 1).
                    if seq_values.ndim == 4:
                        shaped_indices = shaped_indices.unsqueeze(-1)
                    next_values[field] = seq_values.take_along_dim(shaped_indices, 2).squeeze(2)
                elif adapter in indices:
                    seq_values = sequences.payload[field]
                    shaped_indices = indices[adapter].unsqueeze(-1)
                    if seq_values.ndim == 4:
                        shaped_indices = shaped_indices.unsqueeze(-1)
                    next_values[field] = seq_values.take_along_dim(shaped_indices, 2).squeeze(2)
                else:
                    raise ValueError(f"Unknown adapter {adapter}.")
                if base_field in self._categorical_fields:
                    next_values[field] = self.next_scales[base_field] * next_values[field] + self.next_offsets[base_field]
                    next_values[base_field] = next_values[field].argmax(-1)
                else:
                    next_values[field] = self.next_scales[base_field] * next_values[field] + self.next_offsets[base_field]
        return PaddedBatch(next_values, outputs.seq_lens)

    def predict_next_k(self, outputs, states, fields=None, logits_fields_mapping=None):
        """Predict K future events.

        Args:
            outputs: Model outputs.
            states: Hidden model states with shape (N, B, L, D), where N is the number of layers.
            fields: The fields to predict next values for. By default, predict all fields.
            logits_fields: A mapping from field to the output logits field to predict logits for.

        Returns:
            PaddedBatch with predictions with shape (B, L, K) or (B, L, K, C) for logits.
        """
        logits_fields_mapping = dict(logits_fields_mapping or {})
        for field in [PRESENCE]:
            if field not in logits_fields_mapping:
                logits_fields_mapping[field] = field + "_logits"
        presence_logits_field = logits_fields_mapping[PRESENCE]

        b, l = outputs.shape
        lengths = outputs.seq_lens
        outputs = PaddedBatch(outputs.payload.reshape(b * l, self._k, self._next_item.input_size),
                              torch.full([b * l], self._k, device=outputs.device))  # (BL, K, P).
        states = states.reshape(len(states), b * l, 1, -1) if states is not None else None  # (N, BL, 1, D).
        next_values = self._next_item.predict_next(outputs, states,
                                                   fields=fields,
                                                   logits_fields_mapping=logits_fields_mapping)  # (BL, K) or (BL, K, C).

        # Extract presence.
        presence_logit = next_values.payload[presence_logits_field]  # (BL, K, 1).
        next_values.payload[PRESENCE] = presence_logit.squeeze(2) > self._matching_thresholds  # (BL, K).
        next_values.payload[PRESENCE_PROB] = torch.exp(presence_logit.squeeze(2))  # (BL, K).

        # Reshape and return.
        sequences = PaddedBatch({k: v.reshape(b, l, self._k, *v.shape[2:]) for k, v in next_values.payload.items()},
                                lengths)  # (B, L, K) or (B, L, K, C).
        self.revert_delta_and_sort_time_inplace(sequences)
        return sequences

    def select_subset(self, batch, indices=None):
        """Select subset of features.

        Args:
            batch: Tensor or PaddedBatch with shape (B, L, *).
            indices: Sorted array of indices in the range [0, L - 1] to select with shape (B, I).

        Returns:
            Subset batch with shape (B, I, *).
        """
        if isinstance(batch, torch.Tensor):
            b, l = indices.shape
            return batch.take_along_dim(indices.payload["index"].reshape(b, l, *([1] * (batch.ndim - 2))), 1)
        payload, lengths = batch.payload, batch.seq_lens
        if isinstance(payload, torch.Tensor):
            payload = self.select_subset(payload, indices)
        else:
            payload = {k: (self.select_subset(v, indices) if k in batch.seq_names else v)
                       for k, v in payload.items()}
        if indices.payload["index"].numel() > 0:
            valid_mask = indices.payload["full_mask"] if self._drop_partial_windows in {True} else indices.seq_len_mask
            subset_lengths = valid_mask.sum(1)
        else:
            subset_lengths = torch.zeros_like(lengths)
        return PaddedBatch(payload, subset_lengths)

    def get_loss_indices(self, inputs):
        """Get positions to evaluate loss at.

        Args:
           inputs: Input features with shape (B, L).

        Returns:
           indices: Batch of indices with shape (B, I) or None if loss must be evaluated at each step.
        """
        b, l = inputs.shape
        k = self._prefetch_k
        n_indices = min(max(int(round(l * self._loss_subset)), 1), l)
        # Take full windows first.
        mask = torch.arange(l, device=inputs.device)[None] + k < inputs.seq_lens[:, None]  # (B, L).
        weights = torch.rand(b, l, device=inputs.device) * mask
        indices = weights.topk(n_indices, dim=1)[1].sort(dim=1)[0]  # (B, I).
        lengths = (indices < inputs.seq_lens[:, None]).sum(1)
        full_mask = indices + k < inputs.seq_lens[:, None]
        return PaddedBatch({"index": indices,
                            "full_mask": full_mask},
                           lengths)

    def extract_structured_windows(self, inputs):
        """Extract windows with shape (B, L, k + 1) from inputs with shape (B, L)."""
        # Join targets before windowing.
        b, l = inputs.shape
        device = inputs.device
        fields = list(sorted(set(self.fields) - {"_presence"}))
        inputs, lengths, length_mask = dict(inputs.payload), inputs.seq_lens, inputs.seq_len_mask

        # Pad events with out-of-horizon.
        inf_ts = inputs[self._timestamps_field][length_mask].max().item() + self._horizon + 1
        inputs[self._timestamps_field].masked_fill_(~length_mask, inf_ts)

        # Extract windows.
        k = self._prefetch_k
        joined = torch.stack([inputs[name] for name in fields], -1)  # (B, L, N).
        d = joined.shape[-1]
        parts = [joined.roll(-i, 1) for i in range(k + 1)]
        joined_windows = torch.stack(parts, 2)  # (B, L, k + 1, N).
        assert joined_windows.shape[:3] == (b, l, k + 1)

        # Split.
        windows = {}
        for i, name in enumerate(fields):
            windows[name] = joined_windows[..., i].to(inputs[name].dtype)  # (B, L, k + 1).

        # Pad partial windows with out-of-horizon.
        mask = torch.arange(l, device=device)[:, None] + torch.arange(k + 1, device=device) >= l  # (L, k + 1)
        windows[self._timestamps_field].masked_fill_(mask[None], inf_ts)

        return PaddedBatch(windows, lengths)

    def match_targets(self, outputs: PaddedBatch, targets: PaddedBatch):
        """Find closest prediction to each target.

        Args:
          outputs: Model outputs with shape (B, L, K, D).
          targets: Mapping from a field name to a tensor with target windows (B, L, 1 + T).
            The first value in each window is the current step, which is ignored during matching.

        Returns:
          - Relative matching with shape (B, L, K), with values in the range [-1, T - 1]
            with -1 meaning there is no matching.
          - Losses dictionary with shape (B, L, K, T).
          - Logging statistics dictionary.
        """
        metrics = {}

        assert outputs.shape == targets.shape
        assert (outputs.seq_lens == targets.seq_lens).all()
        device = outputs.device
        b, l = outputs.shape
        n_targets = targets.payload[targets.seq_names[0]].shape[2] - 1  # T.
        assert n_targets > 0
        lengths, lengths_mask = outputs.seq_lens, outputs.seq_len_mask
        targets, outputs = targets.payload, outputs.payload
        tails_mask = ~lengths_mask.bool()  # (B, L).

        # Compute positive pairwise scores.
        targets = {k: v.reshape(b * l, 1 + n_targets, 1) for k, v in targets.items()}  # (BL, 1 + T, 1).
        targets["_presence"] = torch.ones(b * l, 1 + n_targets, 1, dtype=torch.long, device=device)  # (BL, 1 + T, 1).
        outputs = outputs.reshape(b * l, 1, self._k, -1)  # (BL, 1, K, D).
        targets_batch = PaddedBatch(targets,
                                    torch.full([b * l], 1 + n_targets, dtype=torch.long, device=device))
        outputs_batch = PaddedBatch(outputs,
                                    torch.full([b * l], 1, dtype=torch.long, device=device))
        states = None
        losses, _ = self._next_item(targets_batch, outputs_batch, states, reduction="none")  # (BL, T, K).
        # Convert presence BCE likelihood to likelihood ratio.
        zero_presence_target = PaddedBatch({"_presence": torch.zeros_like(targets["_presence"])},
                                           targets_batch.seq_lens)
        losses["_presence_neg"] = -self._next_item(zero_presence_target, outputs_batch, states, reduction="none")[0]["_presence"]
        losses = {k: v.reshape(b, l, n_targets, self._k).permute(0, 1, 3, 2) for k, v in losses.items()}  # (B, L, K, T).
        with torch.no_grad():
            if self._match_weights is not None:
                losses_list = [losses[name] * weight for name, weight in self._match_weights.items()]
                if "_presence" in self._match_weights:
                    losses_list.append(losses["_presence_neg"] * self._match_weights["_presence"])
            else:
                losses_list = list(losses.values())
            costs = sum(losses_list[1:], start=losses_list[0])  # (B, L, K, T).

        # Fill out-of-horizon events costs with large cost to prevent them from matching.
        deltas = (targets[self._timestamps_field][:, 1:] - targets[self._timestamps_field][:, :1])  # (BL, T, 1).
        out_horizon_mask = (deltas.reshape(b, l, n_targets) >= self._horizon)  # (B, L, T).
        out_horizon_mask.logical_or_(tails_mask.unsqueeze(2))  # (B, L, T).
        valid_costs = costs.masked_select(~out_horizon_mask.unsqueeze(2))
        max_cost = valid_costs.max().item() if valid_costs.numel() > 0 else 1
        costs.masked_fill_(out_horizon_mask.unsqueeze(2), max_cost + 1)  # (B, L, K, T).

        # Compute matching.
        b, l, k, t = costs.shape
        matches = batch_linear_assignment(costs.reshape(b * l, k, t)).reshape(b, l, k)  # (B, L, K).

        # Replace out-of-horizon matches with -1.
        match_costs = costs.take_along_dim(matches.clip(min=0).unsqueeze(3), 3).squeeze(3)  # (B, L, K).
        assert match_costs.ndim == 3
        invalid_mask = match_costs >= max_cost
        if k > t:
            # Some matches are -1, mark them as invalid.
            invalid_mask.logical_or_(matches < 0)
        match_mask = ~invalid_mask  # (B, L, K).

        # Fill invalid matches.
        matches.masked_fill_(invalid_mask, -1)

        # Compute statistics and returns.
        n_matches = (matches >= 0).sum().item()
        n_predictions = lengths.sum().item() * self._k
        n_targets = out_horizon_mask.numel() - out_horizon_mask.sum().item()
        match_rate = n_matches / max(n_predictions, 1)
        target_match_rate = n_matches / max(n_targets, 1)
        matched_costs = match_costs[match_mask]
        metrics["prediction_match_rate"] = match_rate
        metrics["target_match_rate"] = target_match_rate
        metrics["max_cost"] = max_cost
        metrics["match_cost_mean"] = matched_costs.mean()
        metrics["match_cost_std"] = matched_costs.std()
        return PaddedBatch(matches, lengths), losses, metrics

    def get_subset_matching(self, inputs, outputs):
        """Apply stride and compute matching.

        Args:
            inputs: Model input features with shape (B, L).
            outputs: Model outputs model output features with shape (B, L, D).

        Returns:
            A tuple of:
                - indices of last seen event with shape (B, I).
                - relative matching with shape (B, I, K).
                - losses with shape (B, I, K, T).
                - metrics dictionary.
        """
        b, l = inputs.shape
        target_windows = self.extract_structured_windows(inputs)  # (B, L, k + 1), where first event is an input for the model.
        assert target_windows.shape == (b, l)

        # Reshape outputs.
        outputs = PaddedBatch(outputs.payload.reshape(b, l, self._k, self._next_item.input_size),
                              outputs.seq_lens)  # (B, L, K, P).
        assert (target_windows.seq_lens == outputs.seq_lens).all()

        # Subset outputs and targets.
        indices = self.get_loss_indices(inputs)
        target_windows = self.select_subset(target_windows, indices)  # (B, I, K + 1).
        outputs = self.select_subset(outputs, indices)  # (B, I, K, P).

        # Compute matching and return.
        l = outputs.shape[1]
        n_targets = target_windows.payload[target_windows.seq_names[0]].shape[2] - 1  # K.
        if (l == 0) or (n_targets == 0):
            matching = PaddedBatch(torch.full([b, l, self._k], -1, dtype=torch.long, device=inputs.device),
                                   target_windows.seq_lens)
            return indices, matching, {}, {}

        matching, losses, metrics = self.match_targets(
            outputs, target_windows
        ) # (B, I, K) with indices in the range [-1, L - 1].

        return indices, matching, losses, metrics
