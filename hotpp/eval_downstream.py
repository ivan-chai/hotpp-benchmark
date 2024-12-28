"""Compute embeddings and evaluate downstream prediction."""
import os
import pickle as pkl
import tempfile
from contextlib import contextmanager

import hydra
import luigi
import torch
from omegaconf import OmegaConf

from embeddings_validation import ReportCollect
from embeddings_validation.config import Config
from .embed import extract_embeddings


@contextmanager
def maybe_temporary_directory(root=None):
    if root is not None:
        if not os.path.exists(root):
            os.mkdir(root)
        yield root
    else:
        with tempfile.TemporaryDirectory() as root:
            yield root


def eval_embeddings(conf):
    OmegaConf.set_struct(conf, False)
    conf.workers = conf.get("workers", 1)
    conf.total_cpu_count = conf.get("total_cpu_count", conf.num_workers)

    task = ReportCollect(
        conf=Config.get_conf(conf),
        total_cpu_count=conf["total_cpu_count"],
    )
    luigi.build([task], workers=conf.workers,
                        local_scheduler=conf.get("local_scheduler", True),
                        log_level=conf.get("log_level", "INFO"))


def parse_result(path):
    scores = []
    with open(path, "r") as fp:
        for line in fp:
            if "embeddings" not in line:
                continue
            tokens = line.strip().split()
            scores.append(float(tokens[2]))
    print(scores)
    if len(scores) != 2:
        raise RuntimeError(f"Can't parse validation output.")
    return tuple(scores)  # (Train, Val).


@hydra.main(version_base=None)
def main(conf):
    with maybe_temporary_directory(conf.get("root", None)) as root:
        model_config = hydra.compose(config_name=conf.model_config)
        embeddings, targets = extract_embeddings(model_config)

        embeddings_path = os.path.join(root, "embeddings.pickle")
        with open(embeddings_path, "wb") as fp:
            pkl.dump(embeddings, fp)

        targets_path = os.path.join(root, "targets.csv")
        targets.dropna().to_csv(targets_path)

        conf.environment.work_dir = root
        conf.features.embeddings.read_params.file_name = embeddings_path
        conf.target.file_name = targets_path
        conf.split.train_id.file_name = targets_path
        conf.report_file = model_config.get("downstream_report", os.path.join(root, "downstream_report.txt"))
        if os.path.exists(conf.report_file):
            os.remove(conf.report_file)
        eval_embeddings(conf)

        scores = parse_result(conf.report_file)
        print(scores)


if __name__ == "__main__":
    main()
