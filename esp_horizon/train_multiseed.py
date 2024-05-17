import logging
from collections import defaultdict

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

from esp_horizon.utils.config import as_flat_config

from .train import train, dump_report

logger = logging.getLogger(__name__)


@hydra.main(version_base="1.2", config_path=None)
def main(conf):
    if "num_evaluation_seeds" not in conf:
        raise ValueError("Need the total number of evaluation seeds.")
    if "multiseed_report" not in conf:
        raise ValueError("Need the path to the multiseed evaluation report.")
    OmegaConf.set_struct(conf, False)
    conf.pop("logger")
    conf.pop("report")  # Don't overwrite single-seed test results.
    by_metric = defaultdict(list)
    for seed in range(conf.num_evaluation_seeds):
        conf["seed_everything"] = seed
        _, metrics = train(conf)
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            try:
                v = float(v)
            except TypeError:
                continue
            by_metric[k].append(v)
    multiseed_metrics = {"num_seeds": conf.num_evaluation_seeds}
    for k, vs in by_metric.items():
        if len(vs) != conf.num_evaluation_seeds:
            continue
        multiseed_metrics[k + " (mean)"] = float(np.mean(vs))
        multiseed_metrics[k + " (std)"] = float(np.std(vs))
    with open(conf.multiseed_report, "w") as fp:
        dump_report(multiseed_metrics, fp)


if __name__ == "__main__":
    main()
