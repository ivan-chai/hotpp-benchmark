import copy
import logging

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf

from .common import get_trainer
from .evaluate import test


logger = logging.getLogger(__name__)


def train(conf):
    if "seed_everything" in conf:
        pl.seed_everything(conf.seed_everything)

    conf = copy.deepcopy(conf)
    OmegaConf.set_struct(conf, False)
    resume_from_checkpoint = conf.trainer.pop("resume_from_checkpoint", None)

    model = hydra.utils.instantiate(conf.module)
    dm = hydra.utils.instantiate(conf.data_module)

    if conf.get("test_only", False):
        if "model_path" not in conf:
            raise ValueError("Need model_path for a model initialization")
        logger.info(f"Load weights from '{conf.model_path}'")
        model.load_state_dict(torch.load(conf.model_path))
        trainer = None
    else:
        trainer = get_trainer(conf)
        if conf.get("init_from_checkpoint", None):
            if resume_from_checkpoint:
                raise ValueError("Can't mix resume_from_checkpoint with init_from_checkpoint")
            model.load_state_dict(torch.load(conf["init_from_checkpoint"]))
        trainer.fit(model, dm, ckpt_path=resume_from_checkpoint)

        checkpoint_callback = trainer.checkpoint_callback
        if checkpoint_callback is not None:
            checkpoint_path = checkpoint_callback.best_model_path
            if not checkpoint_path:
                logging.warning("Empty checkpoint path")
            else:
                model.load_state_dict(torch.load(checkpoint_path)["state_dict"])
                logging.info(f"Loaded the best model from '{checkpoint_callback.best_model_path}'")

        if "model_path" in conf:
            torch.save(model.state_dict(), conf.model_path)
            logger.info(f"Model weights saved to '{conf.model_path}'")

    metrics = test(conf, model, dm, trainer=trainer)
    return model, metrics


@hydra.main(version_base="1.2", config_path=None)
def main(conf):
    train(conf)


if __name__ == "__main__":
    main()
