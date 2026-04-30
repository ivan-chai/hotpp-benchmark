"""Backwards-compat alias.

Renamed to ``horizon_diffusion_detpp_base_plus_residual``. The class
``HorizonDiffusionLossV5`` is kept as an alias for legacy configs and
downstream files (V6, V7, V9 chain via configs, etc.).
"""
from .horizon_diffusion_detpp_base_plus_residual import (
    HorizonDiffusionLossDetppBasePlusResidual as HorizonDiffusionLossV5,
)

__all__ = ["HorizonDiffusionLossV5"]
