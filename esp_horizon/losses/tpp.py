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
