import torch

from torch_linear_assignment import batch_linear_assignment
from hotpp.data import PaddedBatch
from .next_item import NextItemLoss
from .next_k import NextKLoss


class DetectionLoss(torch.nn.Module):
    """The loss similar to Next-K, but with the matching loss.

    Args:
        k: The maximum number of future events to predict.
        horizon: Predicted interval.
        losses: Mapping from the feature name to the loss function.
        prediction: The type of prediction (either `mean` or `mode`).
        loss_subset: The fraction of positions to compute loss for.
        prefetch_factor: Extract times more targets than predicted events (equal to `k`) for scoring.
    """
    def __init__(self, next_item_loss, k, horizon, timestamps_field="timestamps", loss_subset=1, prefetch_factor=1):
        super().__init__()
        self._next_item = next_item_loss
        self._k = k
        self._horizon = horizon
        self._timestamps_field = timestamps_field
        self._loss_subset = loss_subset
        self._prefetch_factor = prefetch_factor

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

        if (matching.payload < 0).all():
            losses = {name: outputs.payload[name].mean() * 0 for name in self.fields}
            return losses, matching_metrics

        index_mask = indices.seq_len_mask  # (B, I).
        matching_mask = matching.payload >= 0  # (B, I, k).
        index_matching = (matching.payload - indices.payload.unsqueeze(2) - 1).clip(min=0)  # (B, I, k).

        losses = {k: v.take_along_dim(index_matching.unsqueeze(3), 3).squeeze(3)
                  for k, v in losses.items()}  # (B, I, k).

        pos_presence = losses.pop("_presence")
        neg_presence = -losses.pop("_presence_neg")
        presence_losses = torch.where(matching_mask, pos_presence, neg_presence)

        losses = {k: v[matching_mask].mean() for k, v in losses.items()}
        losses["_presence"] = presence_losses[index_mask].mean()
        
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
        # Select parameters of the first predicted event.
        b, l = outputs.shape
        lengths = outputs.seq_lens
        outputs = PaddedBatch(outputs.payload.reshape(b * l, self._k, self._next_item.input_size)[:, :1, :],
                              torch.ones_like(outputs.seq_lens))  # (BL, 1, P).
        states = states.reshape(len(states), b * l, 1, -1)  # (N, BL, , D).
        next_values = self._next_item.predict_next(outputs, states,
                                                   fields=fields,
                                                   logits_fields_mapping=logits_fields_mapping)  # (BL, 1) or (BL, 1, C).
        return PaddedBatch({k: v.reshape(b, l, *v.shape[2:]) for k, v in next_values.payload.items()},
                           lengths)  # (B, L) or (B, L, C).

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
        b, l = outputs.shape
        lengths = outputs.seq_lens
        outputs = PaddedBatch(outputs.payload.reshape(b * l, self._k, self._next_item.input_size),
                              outputs.seq_lens)  # (BL, K, P).
        states = states.reshape(len(states), b * l, 1, -1)  # (N, BL, 1, D).
        next_values = self._next_item.predict_next(outputs, states,
                                                   fields=fields,
                                                   logits_fields_mapping=logits_fields_mapping)  # (BL, K) or (BL, K, C).
        return PaddedBatch({k: v.reshape(b, l, self._k, *v.shape[2:]) for k, v in next_values.payload.items()},
                           lengths)  # (B, L, K) or (B, L, K, C).

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
           Batch of indices with shape (B, I) or None if loss must be evaluated at each step.
        """
        if self._loss_subset >= 1:
            return None
        b, l = inputs.shape
        n_indices = min(max(int(round(l * self._loss_subset)), 1), l)
        weights = torch.rand(b, l, device=inputs.device) * inputs.seq_len_mask
        subset_indices = weights.topk(n_indices, dim=1)[1].sort(dim=1)[0]  # (B, I).
        lengths = (subset_indices < inputs.seq_lens[:, None]).sum(1)
        return PaddedBatch(subset_indices, lengths)

    def extract_structured_windows(self, inputs):
        """Extract windows with shape (B, L - k, k + 1) from inputs with shape (B, L)."""
        # Join targets before windowing.
        b, l = inputs.shape
        device = inputs.device
        fields = list(sorted(set(self.fields) - {"_presence"}))
        inputs, lengths = dict(inputs.payload), inputs.seq_lens
        joined = torch.stack([inputs[name] for name in fields], -1)  # (B, L, N).

        # Extract windows.
        k = int(round(self._k * self._prefetch_factor))
        joined_windows = NextKLoss.extract_windows(joined, k + 1)   # (B, L - k, k + 1, N).
        assert joined_windows.shape[:3] == (b, max(l - k, 0), k + 1)
        windows_lengths = (lengths - k).clip(min=0)
        if windows_lengths.numel() == 0:
            return PaddedBatch(torch.full((b, 0, self._k), -1, device=device),
                               torch.zeros(b, dtype=torch.long, device=device))

        # Split.
        windows = {}
        for i, name in enumerate(fields):
            windows[name] = joined_windows[..., i].to(inputs[name].dtype)
        return PaddedBatch(windows, windows_lengths)

    def match_targets(self, outputs: PaddedBatch, targets: PaddedBatch, subset_indices=None):
        """Find closest prediction to each target.

        Args:
          outputs: Model outputs with shape (B, L, T, D).
          targets: Mapping from a field name to a tensor with target windows (B, L, 1 + T).
            The first value in each window is the current step, which is ignored during matching.
          subset_indices: Align matching indices with original sequence (before subsetting).

        Returns:
          - Matching with shape (B, L, K), with values in the range [0, L - 1]
            or -1 if matching was found and -1 otherwise.
          - Losses dictionary with shape (B, L, k, T).
          - Logging statistics dictionary.
        """
        metrics = {}

        assert outputs.shape == targets.shape
        assert (outputs.seq_lens == targets.seq_lens).all()
        device = outputs.device
        b, l = outputs.shape
        n_targets = targets.payload[next(iter(targets.seq_names))].shape[2]
        lengths, lengths_mask = outputs.seq_lens, outputs.seq_len_mask
        targets, outputs = targets.payload, outputs.payload
        tails_mask = ~lengths_mask.bool()  # (B, L).

        # Compute positive pairwise scores.
        def prepare_target(x):
            # x: (B, L, 1 + T).
            k = x.shape[2] - 1
            x = x.flatten(0, 1).unsqueeze(1)  # (BL, 1, 1 + T).
            x = torch.cat((x[:, :, :1].repeat(1, 1, k), x[:, :, 1:]), 1)  # (BL, 2, T).
            return x.unsqueeze(2)  # (BL, 2, 1, T).

        reshaped_lengths = torch.full([b * l], 2, dtype=torch.long, device=device)
        targets = {k: prepare_target(v) for k, v in targets.items()}  # (BL, 2, 1, T).
        targets["_presence"] = torch.ones(b * l, 2, 1, n_targets - 1, dtype=torch.long, device=device)
        outputs = outputs.flatten(0, 1).unsqueeze(1).repeat(1, 2, 1, 1).unsqueeze(3)  # (BL, 2, k, 1, D).
        targets_batch = PaddedBatch(targets, reshaped_lengths)
        outputs_batch = PaddedBatch(outputs, reshaped_lengths)
        losses, _ = self._next_item(targets_batch, outputs_batch, None, reduction="none")  # (BL, 1, k, T).
        # Convert presence BCE likelihood to likelihood ratio.
        zero_presence_target = PaddedBatch({"_presence": torch.zeros_like(targets["_presence"])},
                                           reshaped_lengths)
        losses["_presence_neg"] = -self._next_item(zero_presence_target, outputs_batch, None, reduction="none")[0]["_presence"]
        losses = {k: v.reshape(b, l, self._k, n_targets - 1) for k, v in losses.items()}  # (B, L, k, T).
        losses_list = list(losses.values())
        costs = sum(losses_list[1:], start=losses_list[0])  # (B, L, k, T).

        # Fill out-of-horizon events costs with large cost to prevent them from matching.
        horizon_mask = (targets[self._timestamps_field].squeeze(2)[:, 1].reshape(b, l, n_targets - 1) >= self._horizon)  # (B, L, T).
        horizon_mask.logical_or_(tails_mask.unsqueeze(2))  # (B, L, T).
        valid_costs = costs.masked_select(horizon_mask.unsqueeze(2))
        max_cost = valid_costs.max().item() if valid_costs.numel() > 0 else 1
        costs.masked_fill_(horizon_mask.unsqueeze(2), max_cost + 1)  # (B, L, k, T).

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

        # Convert offsets to absolute indices.
        offsets = torch.arange(l, device=device)
        if subset_indices is None:
            subset_indices = torch.arange(l, device=device)[None]  # (1, L).
        else:
            subset_indices = subset_indices.payload
        matches += subset_indices.unsqueeze(2) + 1

        # Fill invalid matches.
        matches.masked_fill_(invalid_mask, -1)

        # Compute statistics and returns.
        n_matches = (matches >= 0).sum().item()
        n_predictions = lengths.sum().item() * self._k
        n_targets = horizon_mask.numel() - horizon_mask.sum().item()
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
                - indices with shape (B, I).
                - matching with shape (B, I, K).
                - losses with shape (B, I, K, T).
                - metrics dictionary.
        """
        target_windows = self.extract_structured_windows(inputs)  # (B, L', k + 1), where first event is an input for the model.
        b, l = target_windows.shape
        # Truncate outputs to the number of windows and extract k predictions.
        outputs = PaddedBatch(outputs.payload[:, :l].reshape(b, min(l, outputs.shape[1]), self._k, self._next_item.input_size),
                              torch.minimum(outputs.seq_lens, target_windows.seq_lens))  # (B, L', K, P).
        assert (target_windows.seq_lens == outputs.seq_lens).all()

        indices = self.get_loss_indices(target_windows)

        target_windows = self.select_subset(target_windows, indices)  # (B, I, k + 1).
        outputs = self.select_subset(outputs, indices)  # (B, I, K, P).

        l = outputs.shape[1]
        if l == 0:
            matching = PaddedBatch(torch.full([b, l, self._k], -1, dtype=torch.long, device=inputs.device),
                                   target_windows.seq_lens)
            return indices, matching, {}

        #with torch.no_grad():
        matching, losses, metrics = self.match_targets(
            outputs, target_windows, subset_indices=indices
        ) # (B, I, K) with indices in the range [-1, L - 1].

        return indices, matching, losses, metrics
