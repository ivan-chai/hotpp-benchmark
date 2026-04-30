"""Backwards-compat alias.

Renamed to ``horizon_diffusion_detpp_all_comps_in_decoder_far_target``.
The class ``HorizonDiffusionLossV48`` is kept as an alias for legacy
configs and downstream files (V49, V50, V51).
"""
from .horizon_diffusion_detpp_all_comps_in_decoder_far_target import (
    HorizonDiffusionLossDetppAllCompsInDecoderFarTarget as HorizonDiffusionLossV48,
)

__all__ = ["HorizonDiffusionLossV48"]
