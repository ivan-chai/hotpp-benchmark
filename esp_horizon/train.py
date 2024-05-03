import logging
import datetime
import yaml

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from esp_horizon.utils.config import as_flat_config

logger = logging.getLogger(__name__)


def get_trainer(conf):
    trainer_params = conf.trainer
    trainer_params_additional = {}
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
        checkpoint_callback = ModelCheckpoint(every_n_epochs=every_n_val_epochs, save_top_k=-1)
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
    return pl.Trainer(**trainer_params, **trainer_params_additional)


def dump_report(metrics, path):
    if len(metrics) != 1:
        raise NotImplementedError("Multiple test dataloaders")
    metrics = metrics[0]
    result = dict(metrics)
    result["date"] = f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}"
    with open(path, "w") as fp:
        yaml.safe_dump(result, fp)


@hydra.main(version_base="1.2", config_path=None)
def main(conf):
    if "seed_everything" in conf:
        pl.seed_everything(conf.seed_everything)

    model = hydra.utils.instantiate(conf.module)
    dm = hydra.utils.instantiate(conf.data_module)

    trainer = get_trainer(conf)
    trainer.fit(model, dm)

    checkpoint_callback = trainer.checkpoint_callback
    if checkpoint_callback is not None:
        model.load_state_dict(torch.load(checkpoint_callback.best_model_path)["state_dict"])
        logging.info(f"Loaded the best model from '{checkpoint_callback.best_model_path}'")

    if "model_path" in conf:
        torch.save(model.state_dict(), conf.model_path)
        logger.info(f"Model weights saved to '{conf.model_path}'")

    metrics = trainer.test(model, dm)
    if "report" in conf:
        dump_report(metrics, conf["report"])


if __name__ == "__main__":
    main()
