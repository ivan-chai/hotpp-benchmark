import logging

import hydra
import torch
from omegaconf import OmegaConf

from .train_multiseed import train_multiseed

logger = logging.getLogger(__name__)


@hydra.main(version_base="1.2", config_path=None)
def main(conf):
    OmegaConf.set_struct(conf, False)
    conf.pop("logger", None)
    conf["test_only"] = True
    train_multiseed(conf)


if __name__ == "__main__":
    main()
