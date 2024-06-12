import torch
from collections import defaultdict

from hotpp.data import PaddedBatch


def apply_windows(args, forward, max_length=None, step=1):
    """Apply the model with limited input length to the input of arbitrary size.

    Args:
      args: A sequence of PaddedBatches, each with shape (B, L, D).
      forward: Maps batch (B, L, D) to the new batch with shape (B, L, P).
      max_length: Maximum length for inference. Use None to disable windowing.
      step: Window step. Small values will lead to slow inference, but accurate result.

    Returns:
      Model output with shape (B, L, P).
    """
    b, l = args[0].shape
    for x in args[1:]:
        if x.shape != (b, l):
            raise ValueError("Input shapes mismatch.")
    if (max_length is None) or (l <= max_length):
        return forward(*args)
    if step > max_length:
        raise ValueError("Some values will not be processed.")
    lengths = args[0].seq_lens
    args = [x.payload for x in args]
    outputs = []
    processed = 0
    for start in range(0, l, step):
        window = [PaddedBatch(x[:, start:start + max_length], (lengths - start).clip(min=0))
                  for x in args]
        output = forward(window)
        assert output.shape == window[0].shape, "Batch or length mismatch."
        assert processed >= start
        offset = processed - start
        if isinstance(output.payload, torch.Tensor):
            outputs.append(output.payload[:, offset:])
        else:
            outputs.append({k: v[:, offset:] for k, v in output.payload.items()})
        processed += max(output.payload.shape[1] - offset, 0)
    if isinstance(outputs[0], torch.Tensor):
        result = torch.cat(outputs, 1)  # (B, L, P).
    else:
        by_key = defaultdict(list)
        for output in outputs:
            for k, v in output.items():
                by_key[k].append(v)
        result = {k: torch.cat(v, 1) for k, v in by_key.items()}
        b, l = next(iter(result.values())).shape[:2]
        for v in result.values():
            assert v.shape[:2] == (b, l)
    result = PaddedBatch(result, lengths)
    assert result.shape == x.shape, (result.payload.shape, x.shape)
    return result
