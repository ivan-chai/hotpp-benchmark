import copy
import datetime
import logging
import yaml

import hydra
import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint

from hotpp.utils.config import as_flat_config


logger = logging.getLogger(__name__)


def dump_report(metrics, fp):
    result = dict(metrics)
    result["date"] = f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}"
    yaml.safe_dump(result, fp)


def get_trainer(conf, **trainer_params_additional):
    trainer_params = copy.deepcopy(conf.trainer)
    OmegaConf.set_struct(trainer_params, False)
    if (trainer_params.get("accelerator", "default") != "cpu"
        and ("sync_batchnorm" not in trainer_params)):
        logging.info("Force batchnorm synchronization. Use explicit 'cpu' device to disable it.")
        trainer_params["sync_batchnorm"] = True
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
