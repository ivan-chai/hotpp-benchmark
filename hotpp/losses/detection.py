import torch

from torch_linear_assignment import batch_linear_assignment
from hotpp.data import PaddedBatch
from hotpp.utils.torch import deterministic
from .common import ScaleGradient
from .next_item import NextItemLoss
from .next_k import NextKLoss


class DetectionLoss(NextKLoss):
    """The loss similar to Next-K, but with the matching loss.

    Args:
        k: The maximum number of future events to predict.
        horizon: Predicted interval.
        losses: Mapping from the feature name to the loss function.
        loss_subset: The fraction of positions to compute loss for.
        loss_valid_only: Skip tails and evaluate only full windows. Turn off for datasets with short sequences.
        prefetch_factor: Extract times more targets than predicted events (equal to `k`) for scoring.
        match_weights: Weights of particular fields in matching.
        momentum: Activation statistics momentum value.
        next_item_adapter: String or dictionary for different fields (`first`, `mean`, or `mode`).
        next_item_timestamps_weight: The weight of the adapter-based next-item timestamps loss.
        next_item_labels_weight: The weight of the adapter-based next-item labels loss.
    """
    def __init__(self, next_item_loss, k, horizon,
                 timestamps_field="timestamps", labels_field="labels",
                 loss_subset=1, loss_valid_only=True, prefetch_factor=1,
                 match_weights=None, momentum=0.1,
                 next_item_adapter="mean",
                 next_item_timestamps_weight=0,
                 next_item_labels_weight=0):
        super().__init__(
            next_item_loss=next_item_loss,
            k=k,
            timestamps_field=timestamps_field
        )
        self._labels_field = labels_field
        self._horizon = horizon
        self._loss_subset = loss_subset
        self._loss_valid_only = loss_valid_only
        self._match_weights = match_weights
        if self.get_delta_type(timestamps_field) != "start":
            raise ValueError("Need `start` delta time for the detection loss.")
        self._momentum = momentum
        self._next_item_adapter = next_item_adapter
        self._next_item_timestamps_weight = next_item_timestamps_weight
        self._next_item_labels_weight = next_item_labels_weight
        self._prefetch_k = int(round(self._k * prefetch_factor))
        self.register_buffer("_matching_priors", torch.ones(k))
        self.register_buffer("_matching_thresholds", torch.zeros(k))

    def update_matching_statistics(self, matching, presence_logits):
        # (B, I, K), (B, I, K).
        matching = matching.payload[matching.seq_len_mask]  # (V, K).
        if len(matching) > 0:
            means = (matching >= 0).float().mean(0)  # (K).
            self._matching_priors *= (1 - self._momentum)
            self._matching_priors += self._momentum * means

        presence_logits = presence_logits.payload[presence_logits.seq_len_mask]  # (V, K).
        if len(presence_logits) > 0:
            presence_logits = torch.sort(presence_logits, dim=0)[0]  # (V, K).
            indices = ((1 - self._matching_priors) * len(presence_logits)).round().long().clip(max=len(presence_logits) - 1)  # (K).
            quantiles = presence_logits.take_along_dim(indices[None], 0).squeeze(0)  # (K).
            self._matching_thresholds *= (1 - self._momentum)
            self._matching_thresholds += self._momentum * quantiles

    @property
    def interpolator(self):
        return self._next_item.interpolator

    @interpolator.setter
    def interpolator(self, value):
        self._next_item.interpolator = value

    @property
    def num_events(self):
        return self._k

    @property
    def fields(self):
        return self._next_item.fields

    @property
    def input_size(self):
        return self._k * self._next_item.input_size  # One for the presence score.

    def get_delta_type(self, field):
        """Get time delta type."""
        return self._next_item.get_delta_type(field)

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
                reshaped_outputs = PaddedBatch(outputs.payload.flatten(0, 1).reshape(-1, self._k, self._next_item.input_size),
                                            torch.full([b * l], self._k, dtype=torch.long, device=outputs.device))  # (BL, K, P).
                reshaped_states = states.flatten(1, 2).unsqueeze(2) if states is not None else None  # (N, BL, 1, D).
                presence_logits = self._next_item.predict_next(
                    reshaped_outputs, reshaped_states,
                    fields=set(),
                    logits_fields_mapping={"_presence": "_presence_logit"}
                ).payload["_presence_logit"]  # (BL, K, 1).
                presence_logits = presence_logits.reshape(b, l, self._k)  # (B, L, K).
                presence_logits = PaddedBatch(presence_logits, outputs.seq_lens)
                self.update_matching_statistics(matching, presence_logits)

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
        if (self._next_item_timestamps_weight > 0) or (self._next_item_labels_weight > 0):
            predictions = self.predict_next(outputs, states,
                                            fields=(self._timestamps_field, self._labels_field),
                                            logits_fields_mapping={self._labels_field: "_labels_logits"})
            predictions = {
                self._timestamps_field: predictions.payload[self._timestamps_field].unsqueeze(-1),  # (B, L, 1).
                self._labels_field: predictions.payload["_labels_logits"]  # (B, L, C).
            }
            next_item_losses, _ = self._next_item(inputs, predictions, states)
            field = self._timestamps_field
            losses[f"next_item_{field}"] = ScaleGradient.apply(next_item_losses[field], self._next_item_timestamps_weight)
            field = self._labels_field
            losses[f"next_item_{field}"] = ScaleGradient.apply(next_item_losses[field], self._next_item_labels_weight)
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
        logits_fields_mapping = dict(logits_fields_mapping or {})
        logits_fields_mapping["_presence"] = "_presence_logit"
        if self._labels_field not in logits_fields_mapping:
            logits_fields_mapping[self._labels_field] = "_labels_logits"
        b, l = outputs.shape
        lengths = outputs.seq_lens
        outputs = PaddedBatch(outputs.payload.reshape(b * l, self._k, self._next_item.input_size),
                              outputs.seq_lens)  # (BL, K, P).
        states = states.reshape(len(states), b * l, 1, -1)  # (N, BL, 1, D).
        next_values = self._next_item.predict_next(outputs, states,
                                                   fields=fields,
                                                   logits_fields_mapping=logits_fields_mapping)  # (BL, K) or (BL, K, C).

        # Reshape and sort.
        sequences = PaddedBatch({k: v.reshape(b, l, self._k, *v.shape[2:]) for k, v in next_values.payload.items()},
                                lengths)  # (B, L, K) or (B, L, K, C).
        self.revert_delta_and_sort_time_inplace(sequences)
        # Sequences contain time shift from the last seen timestamps.
        # Events are sorted by timestamp.

        # Prepare data.
        presence_logits = sequences.payload[logits_fields_mapping["_presence"]].squeeze(-1)  # (B, L, K).
        presence = presence_logits > self._matching_thresholds  # (B, L, K).
        log_presence = torch.nn.functional.logsigmoid(presence_logits)  # (B, L, K).
        log_not_presence = torch.nn.functional.logsigmoid(-presence_logits)  # (B, L, K).
        assert log_presence.ndim == 3
        log_probs = torch.nn.functional.log_softmax(
            sequences.payload[logits_fields_mapping[self._labels_field]], dim=-1)  # (B, L, K, C).
        times = sequences.payload[self._timestamps_field]  # (B, L, K).

        # Compute probability of each event being the first.
        # log_cum_prod is equal to log of 1, (1 - p1), (1 - p1)(1 - p2), ...
        roll_log_not_presence = torch.cat([torch.zeros_like(log_not_presence[..., :1]), log_not_presence[..., :-1]], -1)   # (B, L, K).
        with deterministic(False):
            log_cum_prod = roll_log_not_presence.cumsum(-1)  # (B, L, K).
        # weights are equal to normalized p1, (1 - p1)p2, (1 - p1)(1 - p2)p3, ...
        log_weights = torch.nn.functional.log_softmax(log_cum_prod + log_presence, -1)  # (B, L, K).

        # Find most probable event.
        first_indices = (torch.arange(presence.shape[-1] - 1, -1, -1, device=presence.device) * presence).argmax(-1)  # (B, L).
        top_indices = (log_probs + log_weights.unsqueeze(-1)).max(-1)[0].argmax(-1)  # (B, L).

        # Predict next items.
        if isinstance(self._next_item_adapter, str):
            time_adapter = self._next_item_adapter
            label_adapter = self._next_item_adapter
        else:
            time_adapter = self._next_item_adapter[self._timestamps_field]
            label_adapter = self._next_item_adapter[self._labels_field]
        if time_adapter == "first":
            next_times = times.take_along_dim(first_indices.unsqueeze(-1), -1).squeeze(-1)  # (B, L).
        elif time_adapter == "mode":
            next_times = times.take_along_dim(top_indices.unsqueeze(-1), -1).squeeze(-1)  # (B, L).
        elif time_adapter == "mean":
            next_times = (times * log_weights.exp()).sum(-1)  # (B, L).
        else:
            raise ValueError(f"Unknown prediction type: {time_prediction}.")
        if label_adapter == "first":
            next_log_probs = log_probs.take_along_dim(first_indices.unsqueeze(-1).unsqueeze(-1), -2).squeeze(-2)  # (B, L, C).
        elif label_adapter == "mode":
            next_log_probs = log_probs.take_along_dim(top_indices.unsqueeze(-1).unsqueeze(-1), -2).squeeze(-2)  # (B, L, C).
        elif label_adapter == "mean":
            next_log_probs = (log_probs + log_weights.unsqueeze(-1)).logsumexp(2)  # (B, L, C).
        else:
            raise ValueError(f"Unknown prediction type: {label_prediction}.")

        next_values = PaddedBatch({
            self._timestamps_field: next_times,
            self._labels_field: next_log_probs.argmax(-1),  # (B, L).
            logits_fields_mapping[self._labels_field]: next_log_probs  # (B, L, C).
        }, lengths)

        return next_values

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
        logits_fields_mapping["_presence"] = "_presence_logit"
        b, l = outputs.shape
        lengths = outputs.seq_lens
        outputs = PaddedBatch(outputs.payload.reshape(b * l, self._k, self._next_item.input_size),
                              outputs.seq_lens)  # (BL, K, P).
        states = states.reshape(len(states), b * l, 1, -1)  # (N, BL, 1, D).
        next_values = self._next_item.predict_next(outputs, states,
                                                   fields=fields,
                                                   logits_fields_mapping=logits_fields_mapping)  # (BL, K) or (BL, K, C).

        # Extract presence.
        presence_logit = next_values.payload["_presence_logit"]  # (BL, K, 1).
        presence = presence_logit.squeeze(2) > self._matching_thresholds
        next_values.payload["_presence"] = presence

        # Update logits with presence value.
        for field, logit_field in logits_fields_mapping.items():
            if field != "_presence":
                next_values.payload[logit_field] += presence_logit  # (BL, K, C).

        # Replace disabled events with maximum time offset.
        if self._timestamps_field in next_values.payload:
            next_values.payload[self._timestamps_field].masked_fill_(~presence.bool(), self._horizon + 1)

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
            return batch.take_along_dim(indices.payload.reshape(b, l, *([1] * (batch.ndim - 2))), 1)
        payload, lengths = batch.payload, batch.seq_lens
        if isinstance(payload, torch.Tensor):
            payload = self.select_subset(payload, indices)
        else:
            payload = {k: (self.select_subset(v, indices) if k in batch.seq_names else v)
                       for k, v in payload.items()}
        if indices.payload.numel() > 0:
            valid, subset_lengths = torch.min((indices.payload < lengths[:, None]).long(), dim=1)  # (B), (B).
            subset_lengths[valid.bool()] = indices.shape[1]
        else:
            subset_lengths = torch.zeros_like(lengths)
        return PaddedBatch(payload, subset_lengths)

    def get_loss_indices(self, inputs):
        """Get positions to evaluate loss at.

        Args:
           inputs: Input features with shape (B, L).

        Returns:
           full: Flag indicating full set of indices
           indices: Batch of indices with shape (B, I) or None if loss must be evaluated at each step.
        """
        if self._loss_subset >= 1:
            b, l = inputs.shape
            indices = PaddedBatch(torch.arange(l, device=inputs.device)[None].repeat(b, 1),
                                  inputs.seq_lens)  # (B, L).
            return True, indices
        b, l = inputs.shape
        k = self._prefetch_k
        n_indices = min(max(int(round(l * self._loss_subset)), 1), l)
        # Take full windows first.
        mask = torch.arange(l, device=inputs.device)[None] + k < inputs.seq_lens[:, None]  # (B, L).
        weights = torch.rand(b, l, device=inputs.device) * mask
        subset_indices = weights.topk(n_indices, dim=1)[1].sort(dim=1)[0]  # (B, I).
        lengths = (subset_indices < inputs.seq_lens[:, None]).sum(1)
        return False, PaddedBatch(subset_indices, lengths)

    def extract_structured_windows(self, inputs):
        """Extract windows with shape (B, L - k, k + 1) from inputs with shape (B, L)."""
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

        if self._loss_valid_only:
            # Exclude partial windows.
            windows = {name: window[:, :l - k] for name, window in windows.items()}
            windows_lengths = (lengths - k).clip(min=0)
        else:
            # Pad partial windows with out of horizon.
            mask = torch.arange(l, device=device)[:, None] + torch.arange(k + 1, device=device) >= l  # (L, k + 1)
            windows[self._timestamps_field].masked_fill_(mask[None], inf_ts)
            windows_lengths = lengths

        return PaddedBatch(windows, windows_lengths)

    def match_targets(self, outputs: PaddedBatch, targets: PaddedBatch, subset_indices: PaddedBatch):
        """Find closest prediction to each target.

        Args:
          outputs: Model outputs with shape (B, L, K, D).
          targets: Mapping from a field name to a tensor with target windows (B, L, 1 + T).
            The first value in each window is the current step, which is ignored during matching.
          subset_indices: Align matching indices with original sequence (before subsetting).

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
        n_targets = targets.payload[next(iter(targets.seq_names))].shape[2] - 1  # T.
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
        target_windows = self.extract_structured_windows(inputs)  # (B, L', k + 1), where first event is an input for the model.
        b, l = target_windows.shape
        # Truncate outputs to the number of windows and extract k predictions.
        outputs = PaddedBatch(outputs.payload[:, :l].reshape(b, min(l, outputs.shape[1]), self._k, self._next_item.input_size),
                              torch.minimum(outputs.seq_lens, target_windows.seq_lens))  # (B, L', K, P).
        assert (target_windows.seq_lens == outputs.seq_lens).all()

        full, indices = self.get_loss_indices(inputs)
        indices = PaddedBatch(indices.payload[:, :l],
                              (indices.payload < target_windows.seq_lens[:, None]).sum(1))
        if not full:
            target_windows = self.select_subset(target_windows, indices)  # (B, I, k + 1).
            outputs = self.select_subset(outputs, indices)  # (B, I, K, P).

        l = outputs.shape[1]
        if l == 0:
            matching = PaddedBatch(torch.full([b, l, self._k], -1, dtype=torch.long, device=inputs.device),
                                   target_windows.seq_lens)
            return indices, matching, {}, {}

        #with torch.no_grad():
        matching, losses, metrics = self.match_targets(
            outputs, target_windows, indices
        ) # (B, I, K) with indices in the range [-1, L - 1].

        return indices, matching, losses, metrics
