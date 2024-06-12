import numpy as np
import os
import pandas as pd
import pickle as pkl
import pytorch_lightning as pl
import sys
import tempfile
from contextlib import contextmanager

import hydra
import luigi
import torch
from omegaconf import OmegaConf

from embeddings_validation import ReportCollect
from embeddings_validation.config import Config
from .data import ShuffledDistributedDataset
from .train import get_trainer


@contextmanager
def maybe_temporary_directory(root=None):
    if root is not None:
        if not os.path.exists(root):
            os.mkdir(root)
        yield root
    else:
        with tempfile.TemporaryDirectory() as root:
            yield root


class InferenceModule(pl.LightningModule):
    def __init__(self, model, id_field, reducer):
        super().__init__()
        self.model = model
        self.id_field = id_field
        self.reducer = reducer

    def forward(self, batch):
        data, _ = batch  # Ignore labels.
        hiddens, _ = self.model.encode(data)  # (B, L, D).
        assert hiddens.payload.ndim == 3
        if self.reducer == "mean":
            embeddings = self.reduce_mean(hiddens)
        elif self.reducer == "last":
            embeddings = self.reduce_last(hiddens)
        else:
            raise ValueError(f"Unknown reducer: {self.reducer}.")
        ids = data.payload[self.id_field]  # (B).
        return embeddings, ids

    def reduce_mean(self, x):
        x, masks, lengths = x.payload, x.seq_len_mask.bool(), x.seq_lens  # (B, L, D), (B, L), (B).
        x = x.masked_fill(~masks.unsqueeze(2), 0)
        sums = x.sum(1)  # (B, D).
        embeddings = sums / lengths.unsqueeze(1)  # (B, D).
        return embeddings

    def reduce_last(self, x):
        invalid = x.seq_lens == 0  # (B).
        indices = (x.seq_lens - 1).clip(min=0)  # (B).
        embeddings = x.payload.take_along_dim(indices[:, None, None], 1).squeeze(1)  # (B, D).
        assert embeddings.ndim == 2
        embeddings.masked_fill_(invalid.unsqueeze(1), 0)  # (B, D).
        return embeddings


class InferenceDataModule(pl.LightningDataModule):
    def __init__(self, data, split):
        super().__init__()
        self.data = data
        self.split = split

    def predict_dataloader(self):
        dataset = getattr(self.data, f"{self.split}_data")
        loader_params = getattr(self.data, f"{self.split}_loader_params")

        num_workers = loader_params.get("num_workers", 0)
        dataset = ShuffledDistributedDataset(dataset,
                                             num_workers=num_workers)
        return torch.utils.data.DataLoader(
            dataset=dataset,
            collate_fn=dataset.dataset.collate_fn,
            shuffle=False,
            num_workers=num_workers,
            batch_size=loader_params.get("batch_size", 1)
        )


def extract_embeddings(conf):
    model = hydra.utils.instantiate(conf.module)
    dm = hydra.utils.instantiate(conf.data_module)
    model.load_state_dict(torch.load(conf.model_path))
    model = InferenceModule(model,
                            id_field=dm.id_field,
                            reducer=conf.get("reducer", "mean"))
    trainer = get_trainer(conf)
    embeddings = []
    ids = []
    for split in ["test", "val", "train"]:
        split_dm = InferenceDataModule(dm, split=split)
        split_embeddings, split_ids = zip(*trainer.predict(model, split_dm))  # (B, D), (B).
        embeddings.extend(split_embeddings)
        ids.extend(split_ids)
    embeddings = torch.cat(embeddings).cpu().numpy()
    ids = torch.cat(ids).cpu().numpy()
    if len(np.unique(ids)) != len(ids):
        raise RuntimeError("Duplicate ids")

    # Convert to embeddings_validation format.
    columns = {"id": ids}
    for i in range(embeddings.shape[1]):
        columns[f"emb_{i:04}"] = embeddings[:, i]
    return pd.DataFrame(columns)


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


@hydra.main(version_base=None)
def main(conf):
    with maybe_temporary_directory(conf.get("root", None)) as root:
        model_config = hydra.compose(config_name=conf.model_config)
        embeddings_path = os.path.join(root, "embeddings.pickle")
        embeddings = extract_embeddings(model_config)
        with open(embeddings_path, "wb") as fp:
            pkl.dump(embeddings, fp)

        conf.environment.work_dir = root
        conf.features.embeddings.read_params.file_name = embeddings_path
        conf.report_file = model_config.get("downstream_report", os.path.join(root, "downstream_report.txt"))
        if os.path.exists(conf.report_file):
            os.remove(conf.report_file)
        eval_embeddings(conf)


if __name__ == "__main__":
    main()
