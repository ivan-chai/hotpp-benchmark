import torch

from esp_horizon.data import PaddedBatch


def autoreg_prepare_features(batch, timestamps_field="timestamps"):
    """Prepare features before autoregression.

    The method handles time in a special way:
    1. saves initial time for delta inverse,
    2. computes deltas.

    Args:
        batch: PaddedBatch with shape (B, T).

    Returns:
        PaddedBatch with prepared features and shape (B, T).
    """
    batch = batch.clone()
    times = batch.payload[timestamps_field].to(torch.float, copy=True)

    # Save time for delta inversion.
    batch.payload["_times"] = times.clone()
    batch.seq_names.add("_times")

    # Compute deltas.
    times[:, 1:] -= batch.payload["_times"][:, :-1]
    if batch.left:
        starts = (times.shape[1] - batch.seq_lens).clip(max=times.shape[1] - 1).unsqueeze(1)  # (B, 1).
        times.scatter_(1, starts, 0)
    else:
        times[:, 0] = 0

    batch.payload[timestamps_field] = times
    return batch


def autoreg_revert_features(batch, timestamps_field="timestamps"):
    """Inverse of prepare features.

    The method reverts deltas if necessary.

    Args:
        batch: PaddedBatch with shape (B, T).
    """
    batch = batch.clone()
    batch.payload[timestamps_field] = batch.payload.pop("_times")
    return batch


def autoreg_output_to_next_input_inplace(batch, outputs, timestamps_field="timestamps"):
    """Update features between generation iterations.

    Args:
        batch: Current input batch of features with shape (B, T).
        outputs: Model output dict with feature shapes (B) (single token without time dimension).

    Returns:
        Outputs, updated inplace.
    """
    # Force positive deltas.
    outputs[timestamps_field].clip_(min=0)  # (B).

    # Update absolute times.
    if batch.left:
        current_times = batch.payload["_times"][:, -1]  # (B).
    else:
        last = (batch.seq_lens - 1).unsqueeze(1)  # (B).
        current_times = batch.payload["_times"].take_along_dim(last, 1).squeeze(1)  # (B).
    outputs["_times"] = current_times + outputs[timestamps_field]  # (B).
    return outputs
