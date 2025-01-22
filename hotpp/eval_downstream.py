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
from .common import dump_report
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
    scores = {}
    with open(path, "r") as fp:
        split = None
        for line in fp:
            if "split_name" in line:
                split = line.strip().split()[1].replace("scores_", "")
                if split == "valid":
                    split = "val"
            if "embeddings" not in line:
                continue
            if split is None:
                raise RuntimeError("Can't parse split name")
            tokens = line.strip().split()
            mean = float(tokens[2])
            std = float(tokens[6])
            scores[split] = (mean, std)
            split = None
    return scores


def eval_downstream(conf, model=None):
    with maybe_temporary_directory(conf.downstream.get("root", None)) as root:
        embeddings, targets = extract_embeddings(conf, model=model)

        index = embeddings.index.name

        embeddings_path = os.path.join(root, "embeddings.pickle")
        with open(embeddings_path, "wb") as fp:
            pkl.dump(embeddings.reset_index(), fp)

        targets_path = os.path.join(root, "targets.csv")
        targets.dropna().drop(columns=["split"]).to_csv(targets_path)

        conf.downstream.environment.work_dir = root
        conf.downstream.features.embeddings.read_params.file_name = embeddings_path
        conf.downstream.target.file_name = targets_path
        conf.downstream.split.train_id.file_name = targets_path
        conf.downstream.report_file = os.path.join(root, "downstream_report.txt")

        test_targets = targets[targets["split"] == "test"]
        if len(test_targets) > 0:
            test_ids_path = os.path.join(root, "test_ids.csv")
            test_targets[[]].to_csv(test_ids_path)  # Index only.
            conf.downstream.split.test_id.file_name = test_ids_path

        if os.path.exists(conf.downstream.report_file):
            os.remove(conf.downstream.report_file)
        eval_embeddings(conf.downstream)

        scores = parse_result(conf.downstream.report_file)
    return scores


@hydra.main(version_base=None)
def main(conf):
    downstream_report = conf.get("downstream_report", None)
    if downstream_report is None:
        raise RuntimeError("Need output dowstream report path.")

    scores = eval_downstream(conf)
    result = {}
    for split, (mean, std) in scores.items():
        result[f"{split}/{conf.downstream.target.col_target} (mean)"] = mean
        result[f"{split}/{conf.downstream.target.col_target} (std)"] = std
    with open(downstream_report, "w") as fp:
        dump_report(result, fp)


if __name__ == "__main__":
    main()
