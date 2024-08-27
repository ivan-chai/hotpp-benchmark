import datetime
import logging
import re
import sys
import yaml

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint

from hotpp.utils.config import as_flat_config

logger = logging.getLogger(__name__)


def get_trainer(conf, **trainer_params_additional):
    trainer_params = conf.trainer
    model_selection = trainer_params.get("model_selection", None)

    if "callbacks" in trainer_params:
        logger.warning(f"Overwrite `trainer.callbacks`, was '{trainer_params.checkpoint_callback}'")
    trainer_params_callbacks = []

    if model_selection is not None:
        checkpoint_callback = ModelCheckpoint(monitor=model_selection["metric"], mode=model_selection["mode"])
        logger.info(f"Create ModelCheckpoint callback with monitor={model_selection['metric']}")
        trainer_params_callbacks.append(checkpoint_callback)
        del trainer_params.model_selection

    if trainer_params.get("checkpoints_every_n_val_epochs", None) is not None:
        every_n_val_epochs = trainer_params.checkpoints_every_n_val_epochs
        checkpoint_callback = ModelCheckpoint(every_n_epochs=every_n_val_epochs)
        logger.info(f"Create ModelCheckpoint callback every_n_epochs ='{every_n_val_epochs}'")
        trainer_params_callbacks.append(checkpoint_callback)
        del trainer_params.checkpoints_every_n_val_epochs

    if "logger" in conf:
        trainer_params_additional["logger"] = hydra.utils.instantiate(conf.logger)
        trainer_params_additional["logger"].log_hyperparams(as_flat_config(OmegaConf.to_container(conf, resolve=True)))

    if not isinstance(trainer_params.get("strategy", ""), str): # if strategy not exist or str do nothing,
        trainer_params_additional["strategy"] = hydra.utils.instantiate(trainer_params.strategy)
        del trainer_params.strategy

    lr_monitor = LearningRateMonitor(logging_interval="step")
    trainer_params_callbacks.append(lr_monitor)

    if len(trainer_params_callbacks) > 0:
        trainer_params_additional["callbacks"] = trainer_params_callbacks
    trainer_params = dict(trainer_params)
    trainer_params.update(trainer_params_additional)
    return pl.Trainer(**trainer_params)


def dump_report(metrics, fp):
    result = dict(metrics)
    result["date"] = f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}"
    yaml.safe_dump(result, fp)


def test(conf, model, dm):
    pl.seed_everything(42)

    trainer = get_trainer(conf, devices=1, precision=32)
    val_metrics = trainer.validate(model, dm)[0]
    test_metrics = trainer.test(model, dm)[0]
    metrics = dict(**val_metrics, **test_metrics)
    if "report" in conf:
        with open(conf["report"], "w") as fp:
            dump_report(metrics, fp)
    else:
        dump_report(metrics, sys.stdout)
    return metrics


def train(conf):
    if "seed_everything" in conf:
        pl.seed_everything(conf.seed_everything)

    OmegaConf.set_struct(conf, False)
    resume_from_checkpoint = conf.trainer.pop("resume_from_checkpoint", None)

    model = hydra.utils.instantiate(conf.module)
    dm = hydra.utils.instantiate(conf.data_module)

    if conf.get("test_only", False):
        if "model_path" not in conf:
            raise ValueError("Need model_path for a model initialization")
        logger.info(f"Load weights from '{conf.model_path}'")
        model.load_state_dict(torch.load(conf.model_path))
    else:
        trainer = get_trainer(conf)
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

    metrics = test(conf, model, dm)
    return model, metrics


@hydra.main(version_base="1.2", config_path=None)
def main(conf):
    train(conf)


if __name__ == "__main__":
    main()
