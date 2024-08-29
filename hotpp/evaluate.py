import logging

import hydra
import torch
from omegaconf import OmegaConf

from .train import train

logger = logging.getLogger(__name__)


@hydra.main(version_base="1.2", config_path=None)
def main(conf):
    OmegaConf.set_struct(conf, False)
    conf.pop("logger", None)
    conf["test_only"] = True
    train(conf)


if __name__ == "__main__":
    main()
