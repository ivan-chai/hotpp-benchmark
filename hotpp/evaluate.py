import copy
import datetime
import logging
import sys

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf

from .common import get_trainer, dump_report

logger = logging.getLogger(__name__)


def test(conf, model, dm, trainer=None):
    pl.seed_everything(42)

    if trainer is None:
        trainer = get_trainer(conf, precision=32)
    if "val" in dm.splits:
        val_metrics = trainer.validate(model, dm)[0]
    else:
        val_metrics = {}
    if "test" in dm.splits:
        test_metrics = trainer.test(model, dm)[0]
    else:
        test_metrics = {}
    metrics = dict(**val_metrics, **test_metrics)
    if conf.get("test_downstream", False):
        from .eval_downstream import eval_downstream
        scores = eval_downstream(conf.downstream, trainer, dm, model)
        if scores is not None:
            # The main process.
            downstream_metrics = {}
            for split, (mean, std) in scores.items():
                downstream_metrics[f"{split}/downstream"] = mean
                downstream_metrics[f"{split}/downstream-std"] = std
            metrics.update(downstream_metrics)
            trainer.logger.log_metrics(downstream_metrics)

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
