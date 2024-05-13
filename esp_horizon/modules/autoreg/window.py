import torch
from collections import defaultdict


def apply_windows(x, forward, max_length=None, step=1):
    """Apply the model with limited input length to the input of arbitrary size.

    Args:
      x: PaddedBatch with shape (B, L, D).
      forward: Maps batch (B, L, D) to the new batch with shape (B, L, P).
      max_length: Maximum length for inference. Use None to disable windowing.
      step: Window step. Small values will lead to slow inference, but accurate result.

    Returns:
      Model output with shape (B, L, P).
    """
    b, l, d = x.payload.shape
    if (max_length is None) or (l <= max_length):
        return forward(x)
    if step > max_length:
        raise ValueError("Some values will not be processed.")
    cls = type(x)
    x, lengths = x.payload, x.seq_lens
    outputs = []
    processed = 0
    for start in range(0, l, step):
        window = cls(x[:, start:start + max_length], (lengths - start).clip(min=0))
        output = forward(window)
        assert output.seq_feature_shape[:2] == window.payload.shape[:2], "Batch or length mismatch."
        assert processed >= start
        offset = processed - start
        if isinstance(output.payload, torch.Tensor):
            outputs.append(output.payload[:, offset:])
        else:
            outputs.append({k: v[:, offset:] for k, v in output.payload.items()})
        processed += max(output.seq_feature_shape[1] - offset, 0)
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
    result = cls(result, lengths)
    assert result.seq_feature_shape[:2] == x.shape[:2], (result.seq_feature_shape, x.shape)
    return result
