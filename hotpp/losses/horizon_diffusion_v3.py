"""Backwards-compat alias.

Renamed to ``horizon_diffusion_fixed_perslot_decoder``. The class
``HorizonDiffusionLossV3`` is kept as an alias for legacy configs,
mlflow runs, and downstream files (V4, V40, V41, V42, etc.) that still
import from this path.
"""
from .horizon_diffusion_fixed_perslot_decoder import (
    HorizonDiffusionLossFixedPerslotDecoder as HorizonDiffusionLossV3,
)

__all__ = ["HorizonDiffusionLossV3"]
