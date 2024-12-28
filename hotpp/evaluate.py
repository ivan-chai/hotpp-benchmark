import copy
import datetime
import logging
import sys
import yaml

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf

from .common import get_trainer

logger = logging.getLogger(__name__)


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


def evaluate(conf):
    if "seed_everything" in conf:
        pl.seed_everything(conf.seed_everything)

    conf = copy.deepcopy(conf)
    OmegaConf.set_struct(conf, False)
    conf.pop("logger", None)
    resume_from_checkpoint = conf.trainer.pop("resume_from_checkpoint", None)

    model = hydra.utils.instantiate(conf.module)
    dm = hydra.utils.instantiate(conf.data_module)

    if "model_path" not in conf:
        raise ValueError("Need model_path for a model initialization")
    logger.info(f"Load weights from '{conf.model_path}'")
    model.load_state_dict(torch.load(conf.model_path))
    metrics = test(conf, model, dm)
    return model, metrics


@hydra.main(version_base="1.2", config_path=None)
def main(conf):
    evaluate(conf)


if __name__ == "__main__":
    main()
