# Base losses.
from .common import BaseLoss, TimeMAELoss, TimeMSELoss, CrossEntropyLoss, BinaryCrossEntropyLoss, MAELoss, MSELoss
from .rmtpp import TimeRMTPPLoss

# High-level losses.
from .nhp import NHPLoss
from .next_item import NextItemLoss
from .next_k import NextKLoss
# from .detection import DetectionLoss
try:
    from .detection import DetectionLoss
except Exception:
    DetectionLoss = None
from .diffusion import DiffusionLoss
from .horizon_diffusion import HorizonDiffusionLoss
from .horizon_diffusion_v1 import HorizonDiffusionLossV1
from .horizon_diffusion_v2 import HorizonDiffusionLossV2
from .horizon_diffusion_fixed_perslot_decoder import HorizonDiffusionLossFixedPerslotDecoder
from .horizon_diffusion_v3 import HorizonDiffusionLossV3  # backwards-compat alias
from .horizon_diffusion_v4 import HorizonDiffusionLossV4
from .horizon_diffusion_detpp_base_plus_residual import HorizonDiffusionLossDetppBasePlusResidual
from .horizon_diffusion_v5 import HorizonDiffusionLossV5  # backwards-compat alias
from .horizon_diffusion_v6 import HorizonDiffusionLossV6
from .horizon_diffusion_v7 import HorizonDiffusionLossV7
from .horizon_diffusion_v40 import HorizonDiffusionLossV40
from .horizon_diffusion_v41 import HorizonDiffusionLossV41
from .horizon_diffusion_v42 import HorizonDiffusionLossV42
from .horizon_diffusion_v43 import HorizonDiffusionLossV43
from .horizon_diffusion_detpp_all_comps_in_decoder import HorizonDiffusionLossDetppAllCompsInDecoder
from .horizon_diffusion_v44 import HorizonDiffusionLossV44  # backwards-compat alias
from .horizon_diffusion_v46 import HorizonDiffusionLossV46
from .horizon_diffusion_v47 import HorizonDiffusionLossV47
from .horizon_diffusion_detpp_all_comps_in_decoder_far_target import HorizonDiffusionLossDetppAllCompsInDecoderFarTarget
from .horizon_diffusion_v48 import HorizonDiffusionLossV48  # backwards-compat alias
from .horizon_diffusion_v49 import HorizonDiffusionLossV49
from .horizon_diffusion_v50 import HorizonDiffusionLossV50
