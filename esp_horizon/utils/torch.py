import os
import torch
from contextlib import contextmanager


@contextmanager
def deterministic(is_active):
    if torch.cuda.is_available():
        was_active = os.environ.get("CUBLAS_WORKSPACE_CONFIG", "") == ":4096:8"  # Can be set by PyTorch Lightning.
        torch.use_deterministic_algorithms(is_active)
        try:
            yield None
        finally:
            torch.use_deterministic_algorithms(was_active)
    else:
        yield None
