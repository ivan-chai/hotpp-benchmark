import copy
import logging
import os
import zipfile
from contextlib import contextmanager
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf

from .common import get_trainer
from .evaluate import test


logger = logging.getLogger(__name__)
logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


def _find_repo_hotpp(start_dir: Path):
    for candidate in [start_dir] + list(start_dir.parents):
        hotpp_dir = candidate / "hotpp"
        if hotpp_dir.exists() and hotpp_dir.is_dir():
            return hotpp_dir
    return None


@contextmanager
def torch_matmul_precision(precision="highest"):
    torch.set_float32_matmul_precision(precision)
    try:
        yield None
    finally:
        torch.set_float32_matmul_precision("highest")


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
        matmul_precision = conf.trainer.pop("matmul_precision", "highest")

        trainer = get_trainer(conf)
        _log_mlflow_artifacts(trainer, conf)
        if conf.get("init_from_checkpoint", None):
            if resume_from_checkpoint:
                raise ValueError("Can't mix resume_from_checkpoint with init_from_checkpoint")
            model.load_state_dict(torch.load(conf["init_from_checkpoint"]))
        with torch_matmul_precision(matmul_precision):
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


def _log_mlflow_artifacts(trainer, conf):
    if trainer is None:
        return
    loggers = trainer.loggers if hasattr(trainer, "loggers") else [trainer.logger]
    if not loggers:
        return
    mlflow_logger = next((l for l in loggers if l.__class__.__name__ == "MLFlowLogger"), None)
    if mlflow_logger is None:
        return

    client = mlflow_logger.experiment
    run_id = mlflow_logger.run_id

    # Log Hydra config files
    hydra_dir = Path(".hydra")
    if hydra_dir.exists():
        client.log_artifacts(run_id, str(hydra_dir))

    # Log resolved and unresolved configs
    resolved_cfg = Path("config_resolved.yaml")
    unresolved_cfg = Path("config_unresolved.yaml")
    resolved_cfg.write_text(OmegaConf.to_yaml(conf, resolve=True))
    unresolved_cfg.write_text(OmegaConf.to_yaml(conf, resolve=False))
    client.log_artifact(run_id, str(resolved_cfg))
    client.log_artifact(run_id, str(unresolved_cfg))

    # Log train.log + app.log
    for log_name in ("train.log", "app.log"):
        log_path = Path(log_name)
        if log_path.exists():
            client.log_artifact(run_id, str(log_path))

    # Log hotpp folder
    repo_root = Path(hydra.utils.get_original_cwd())
    hotpp_dir = _find_repo_hotpp(repo_root)
    if hotpp_dir is not None:
        client.log_artifacts(run_id, str(hotpp_dir), artifact_path="hotpp")



@hydra.main(version_base="1.2", config_path=None)
def main(conf):
    print(OmegaConf.to_yaml(conf, resolve=True))
    train(conf)


if __name__ == "__main__":
    main()
