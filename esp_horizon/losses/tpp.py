import torch


def thinning(b, l, intensity_fn, max_steps, max_delta, device=None, dtype=None):
    """Apply thinning algorithm.

    Args:
        b: Batch size.
        l: Length.
        intensity_fn: The mapping from timestamps with shape (B, L) to intensity values with shape (B, L).
        max_steps: The maximum number of steps in thinning algorithm.
        max_delta: The maximum time step size.

    Returns:
        Samples tensor with shape (B, L, N) and acceptance tensor with shape (B, L, N).
    """
    samples = torch.zeros(1 + max_steps, b, l, dtype=dtype, device=device)  # (1 + N, B, L).
    rejected = torch.ones(1 + max_steps, b, l, dtype=torch.bool, device=device)  # (1 + N, B, L).
    for i in range(1, max_steps + 1):
        upper = intensity_fn(samples[i - 1])  # (B, L).
        tau = -torch.rand_like(upper).log() / upper  # (B, L).
        samples[i] = samples[i - 1] * rejected[i - 1] + tau
        rejected[i] = torch.rand_like(upper) * upper >= intensity_fn(samples[i])
        mask = samples[i] > max_delta  # (B, L).
        samples[i].masked_fill_(mask, 0)  # Reset time for future sampling.
        rejected[i].masked_fill_(mask, 1)  # Reject zero time.
    return samples[1:].permute(1, 2, 0), (~rejected[1:]).permute(1, 2, 0)  # (B, L, N).


def thinning_expectation(b, l, intensity_fn, max_steps, max_delta, device=None, dtype=None):
    """Estimate expectation with thinning algorithm."""
    sample, mask = thinning(b, l,
                            intensity_fn=intensity_fn,
                            max_steps=max_steps,
                            max_delta=max_delta,
                            dtype=dtype, device=device)  # (B, L, N), (B, L, N).
    sample, mask = sample.flatten(0, 1), mask.flatten(0, 1)  # (BL, N), (BL, N).
    empty = ~mask.any(-1)  # (BL).
    if empty.any():
        sample[empty, 0] = max_delta
        mask[empty, 0] = True
    expectations = (sample * mask).sum(1) / mask.sum(1)  # (BL).
    # Delta is always positive.
    return expectations.reshape(b, l).clip(min=0)  # (B, L).
