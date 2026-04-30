"""Backwards-compat alias.

Renamed to ``horizon_diffusion_detpp_all_comps_in_decoder``. The class
``HorizonDiffusionLossV44`` and the helper ``ContextQueryAugmentedHead``
are kept as aliases for legacy configs and downstream files (V45, V46,
V48 via shim).
"""
from .horizon_diffusion_detpp_all_comps_in_decoder import (
    HorizonDiffusionLossDetppAllCompsInDecoder as HorizonDiffusionLossV44,
    ContextQueryAugmentedHead,
)

__all__ = ["HorizonDiffusionLossV44", "ContextQueryAugmentedHead"]
