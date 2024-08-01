import torch


DEFAULT_BOUND_SAMPLES = 16
DEFAULT_BOUND_FACTOR = 2


def thinning(b, l, intensity_fn, max_steps, max_delta,
             bound_samples=DEFAULT_BOUND_SAMPLES, bound_factor=DEFAULT_BOUND_FACTOR,
             device=None, dtype=None):
    """Apply thinning algorithm.

    Args:
        b: Batch size.
        l: Length.
        intensity_fn: The mapping from time offsets with shape (B, L, S) to intensity values with shape (B, L, S).
        max_steps: The maximum number of steps in thinning algorithm.
        max_delta: The maximum time step size.

    Returns:
        Samples tensor with shape (B, L, N) and acceptance tensor with shape (B, L, N).
    """
    samples = torch.zeros(1 + max_steps, b, l, dtype=dtype, device=device)  # (1 + N, B, L).
    rejected = torch.ones(1 + max_steps, b, l, dtype=torch.bool, device=device)  # (1 + N, B, L).
    delta_sample = max_delta * torch.linspace(0, 1, bound_samples, dtype=dtype, device=device)  # (S).
    upper_range = intensity_fn(delta_sample.flip(0)[None, None].repeat(b, l, 1)).cummax(-1)[0].flip(2) * bound_factor  # (B, L, S).
    upper_step = max_delta / max(1, bound_samples - 1)
    tau_sample = -torch.rand_like(samples).log()  # (1 + N, B, L).
    trials = torch.rand_like(samples)  # (1 + N, B, L).
    for i in range(1, max_steps + 1):
        indices = (samples[i - 1] / upper_step).long().unsqueeze(2)  # (B, L).
        upper = upper_range.take_along_dim(indices, 2).squeeze(2)  # (B, L).
        samples[i] = samples[i - 1] * rejected[i - 1] + tau_sample[i] / upper
        rejected[i] = trials[i] * upper >= intensity_fn(samples[i].unsqueeze(2)).squeeze(2)
        mask = samples[i] > max_delta  # (B, L).
        samples[i].masked_fill_(mask, 0)  # Reset time for future sampling.
        rejected[i].masked_fill_(mask, 1)  # Reject zero time.
    return samples[1:].permute(1, 2, 0), (~rejected[1:]).permute(1, 2, 0)  # (B, L, N).


def thinning_expectation(b, l, intensity_fn, max_steps, max_delta,
                         bound_samples=DEFAULT_BOUND_SAMPLES, bound_factor=DEFAULT_BOUND_FACTOR,
                         device=None, dtype=None):
    """Estimate expectation with thinning algorithm."""
    sample, mask = thinning(b, l,
                            intensity_fn=intensity_fn,
                            max_steps=max_steps,
                            max_delta=max_delta,
                            bound_samples=bound_samples,
                            bound_factor=bound_factor,
                            dtype=dtype, device=device)  # (B, L, N), (B, L, N).
    sample, mask = sample.flatten(0, 1), mask.flatten(0, 1)  # (BL, N), (BL, N).
    accepted = mask.sum(1)
    empty = ~mask.any(-1)  # (BL).
    if empty.any():
        sample[empty, 0] = max_delta
        mask[empty, 0] = True
    expectations = (sample * mask).sum(1) / mask.sum(1)  # (BL).
    # Delta is always positive.
    return expectations.reshape(b, l).clip(min=0), accepted.reshape(b, l)  # (B, L), (B, L).


def thinning_sample(b, l, intensity_fn, max_steps, max_delta,
                    bound_samples=DEFAULT_BOUND_SAMPLES, bound_factor=DEFAULT_BOUND_FACTOR,
                    device=None, dtype=None):
    """Sample with thinning algorithm."""
    sample, mask = thinning(b, l,
                            intensity_fn=intensity_fn,
                            max_steps=max_steps,
                            max_delta=max_delta,
                            bound_samples=bound_samples,
                            bound_factor=bound_factor,
                            dtype=dtype, device=device)  # (B, L, N), (B, L, N).
    sample, mask = sample.flatten(0, 1), mask.flatten(0, 1)  # (BL, N), (BL, N).
    accepted = mask.sum(1)
    empty = ~mask.any(-1)  # (BL).
    if empty.any():
        sample[empty, 0] = max_delta
        mask[empty, 0] = True
    # Take the first value.
    mask = torch.logical_and(mask, mask.cumsum(1) == 1)  # (BL, N).
    return sample.masked_select(mask).reshape(b, l).clip(min=0), accepted.reshape(b, l)  # (B, L), (B, L).
