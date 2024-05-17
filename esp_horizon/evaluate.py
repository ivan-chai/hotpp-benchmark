import logging

import hydra
import torch
from omegaconf import OmegaConf

from .train import test

logger = logging.getLogger(__name__)


@hydra.main(version_base="1.2", config_path=None)
def main(conf):
    OmegaConf.set_struct(conf, False)
    conf.pop("logger")

    model = hydra.utils.instantiate(conf.module)
    dm = hydra.utils.instantiate(conf.data_module)

    if "model_path" not in conf:
        raise ValueError("Need model_path for a model initialization")
    logger.info(f"Load weights from '{conf.model_path}'")
    model.load_state_dict(torch.load(conf.model_path))
    test(conf, model, dm)


if __name__ == "__main__":
    main()
