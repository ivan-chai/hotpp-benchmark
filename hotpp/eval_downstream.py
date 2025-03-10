"""Compute embeddings and evaluate downstream prediction."""
import copy
import os
import pickle as pkl
import tempfile
from contextlib import contextmanager

import hydra
import luigi
import pandas as pd
import torch
from omegaconf import OmegaConf

try:
    from embeddings_validation import ReportCollect
    from embeddings_validation.config import Config
except ImportError:
    raise ImportError("Please, install embeddings_validation or hotpp-benchmark[downstream]")
from .common import get_trainer, dump_report
from .embed import InferenceDataModule, extract_embeddings, embeddings_to_pandas


@contextmanager
def maybe_temporary_directory(root=None):
    if root is not None:
        if not os.path.exists(root):
            os.mkdir(root)
        yield root
    else:
        with tempfile.TemporaryDirectory() as root:
            yield root


def extract_targets(datamodule, splits=None):
    target_names = datamodule.train_data.global_target_fields
    if splits is None:
        splits = datamodule.splits
    by_split = {}
    for split in splits:
        split_datamodule = InferenceDataModule(datamodule, split=split)
        ids = []
        targets = []
        for x, y in split_datamodule.predict_dataloader():
            x_ids = x.payload[datamodule.id_field]
            if isinstance(x_ids, torch.Tensor):
                x_ids = x_ids.cpu().tolist()
            elif isinstance(x_ids, np.ndarray):
                x_ids = x_ids.tolist()
            elif not isinstance(x_ids, list):
                raise ValueError(f"Unknown ids type: {type(x_ids)}")
            ids.extend(x_ids)
            targets.append(torch.stack([y.payload[name] for name in target_names], -1).cpu())  # (B, T).
        targets = torch.cat(targets)
        assert len(ids) == len(targets)
        by_split[split] = (ids, targets)
    return target_names, by_split


def targets_to_pandas(id_field, target_names, by_split):
    all_ids = []
    all_splits = []
    all_targets = []
    for split, (ids, targets) in by_split.items():
        assert len(ids) == len(targets)
        all_ids.extend(ids)
        all_splits.extend([split] * len(ids))
        all_targets.append(targets)
    all_targets = torch.cat(all_targets).numpy()

    columns = {id_field: all_ids,
               "split": all_splits}
    assert all_targets.shape[1] == len(target_names)
    for i, name in enumerate(target_names):
        columns[name] = all_targets[:, i]
    return pd.DataFrame(columns).set_index(id_field)


def eval_embeddings(conf):
    OmegaConf.set_struct(conf, False)
    conf.workers = conf.get("workers", 1)
    conf.total_cpu_count = conf.get("total_cpu_count", conf.workers)

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


def eval_downstream(downstream_config, trainer, datamodule, model):
    downstream_config = copy.deepcopy(downstream_config)
    OmegaConf.set_struct(downstream_config, False)
    with maybe_temporary_directory(downstream_config.get("root", None)) as root:
        splits = downstream_config.get("data_splits", datamodule.splits)
        embeddings = extract_embeddings(trainer, datamodule, model, splits=splits)
        embeddings = embeddings_to_pandas(datamodule.id_field, embeddings)
        if len(embeddings.index.unique()) != len(embeddings):
            raise ValueError("Duplicate ids")
        target_names, targets = extract_targets(datamodule, splits=splits)
        targets = targets_to_pandas(datamodule.id_field, target_names, targets)

        index = embeddings.index.name

        embeddings_path = os.path.join(root, "embeddings.pickle")
        with open(embeddings_path, "wb") as fp:
            pkl.dump(embeddings.reset_index().drop(columns=["split"]), fp)

        targets_path = os.path.join(root, "targets.csv")
        targets.dropna().drop(columns=["split"]).to_csv(targets_path)

        downstream_config.environment.work_dir = root
        downstream_config.features.embeddings.read_params.file_name = embeddings_path
        downstream_config.target.file_name = targets_path
        downstream_config.split.train_id.file_name = targets_path
        downstream_config.report_file = os.path.join(root, "downstream_report.txt")

        test_targets = targets[targets["split"] == "test"]
        if len(test_targets) > 0:
            test_ids_path = os.path.join(root, "test_ids.csv")
            test_targets[[]].to_csv(test_ids_path)  # Index only.
            downstream_config.split.test_id.file_name = test_ids_path

        if os.path.exists(downstream_config.report_file):
            os.remove(downstream_config.report_file)
        eval_embeddings(downstream_config)

        scores = parse_result(downstream_config.report_file)
    return scores


@hydra.main(version_base=None)
def main(conf):
    downstream_report = conf.get("downstream_report", None)
    if downstream_report is None:
        raise RuntimeError("Need output dowstream report path.")

    trainer = get_trainer(conf)
    dm = hydra.utils.instantiate(conf.data_module)
    model = hydra.utils.instantiate(conf.module)
    model.load_state_dict(torch.load(conf.model_path))

    scores = eval_downstream(conf.downstream, trainer, dm, model)
    result = {}
    for split, (mean, std) in scores.items():
        result[f"{split}/{conf.downstream.target.col_target} (mean)"] = mean
        result[f"{split}/{conf.downstream.target.col_target} (std)"] = std
    with open(downstream_report, "w") as fp:
        dump_report(result, fp)


if __name__ == "__main__":
    main()
