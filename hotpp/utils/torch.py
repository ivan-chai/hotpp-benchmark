import os
import torch
from contextlib import contextmanager


BATCHNORM_TYPES = (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)


@contextmanager
def deterministic(is_active):
    """Use deterministic PyTorch computation."""
    if torch.cuda.is_available():
        was_active = os.environ.get("CUBLAS_WORKSPACE_CONFIG", "") == ":4096:8"  # Can be set by PyTorch Lightning.
        torch.use_deterministic_algorithms(is_active)
        try:
            yield None
        finally:
            torch.use_deterministic_algorithms(was_active)
    else:
        yield None


@contextmanager
def module_mode(*modules, training=True, layer_types=None):
    """Temporary alternate the PyTorch Module training mode.

    Args:
        module: Module to change mode for.
        training: Whether to set mode to training or eval.
        layer_types: Affect only specified layer types.
    """
    if layer_types is not None:
        layer_types = tuple(layer_types)
        layers = []
        for m in modules:
            for layer in m.modules():
                if isinstance(layer, layer_types):
                    layers.append(layer)
        modules = layers
    orig_training = [m.training for m in modules]
    try:
        for m in modules:
            m.train(training)
        yield modules
    finally:
        for m, mode in zip(modules, orig_training):
            m.train(mode)


def prefix_medians(x):
    """Compute median value for each prefix.

    Args:
        x: Input tensor with shape (B, L).

    Returns:
        Prefix medians with shape (B, L).
    """
    b, l = x.shape
    medians = []
    for i in range(l):
        medians.append(torch.median(x[:, :i + 1], dim=1)[0])  # (B).
    return torch.stack(medians, 1)  # (B, L).
