"""Compute embeddings."""
import copy
import logging
from collections import defaultdict

import hydra
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from torchmetrics import Metric
from torchmetrics.utilities import dim_zero_cat

from .common import get_trainer
from .data import ShuffledDistributedDataset

logger = logging.getLogger(__name__)


class GatherMetric(Metric):
    """Gather predictions across all processes."""
    def __init__(self, n_values, compute_on_cpu=False):
        super().__init__(compute_on_cpu=compute_on_cpu)
        self.n_values = n_values
        for i in range(n_values):
            self.add_state(f"_out_{i}", default=[], dist_reduce_fx="cat")

    def update(self, *args):
        if len(args) != self.n_values:
            raise ValueError(f"Wrong number of inputs: {len(args)} != {self.n_values}")
        for i, v in enumerate(args):
            getattr(self, f"_out_{i}").append(v)

    def compute(self):
        results = []
        for i in range(self.n_values):
            try:
                values = dim_zero_cat(getattr(self, f"_out_{i}"))
                results.append(values)
            except ValueError:
                # Empty list.
                return None
        return results


class InferenceModule(pl.LightningModule):
    def __init__(self, model, n_outputs, compute_on_cpu=False):
        super().__init__()
        self.model = model
        self.n_outputs = n_outputs
        self.gather = GatherMetric(n_outputs, compute_on_cpu=compute_on_cpu)
        self.result = None

    def forward(self, batch):
        return self.model(batch)

    def test_step(self, batch):
        result = self(batch)
        assert len(result) == self.n_outputs
        self.gather.update(*result)

    def on_test_epoch_end(self):
        self.result = self.gather.compute()


class InferenceDataModule(pl.LightningDataModule):
    def __init__(self, data, split, rank, world_size):
        super().__init__()
        self.data = data.with_test_parameters()
        self.split = split
        self.rank = rank
        self.world_size = world_size

    def test_dataloader(self):
        dataset = getattr(self.data, f"{self.split}_data")
        loader_params = getattr(self.data, f"{self.split}_loader_params")

        num_workers = loader_params.get("num_workers", 0)
        dataset = ShuffledDistributedDataset(dataset, rank=self.rank, world_size=self.world_size)
        return torch.utils.data.DataLoader(
            dataset=dataset,
            collate_fn=dataset.dataset.collate_fn,
            shuffle=False,
            num_workers=num_workers,
            batch_size=loader_params.get("batch_size", 1)
        )


def distributed_predict(trainer, datamodule, model, n_outputs, splits=None):
    model = InferenceModule(model, n_outputs=n_outputs)
    if splits is None:
        splits = datamodule.splits
    by_split = {}
    for split in splits:
        model.gather.reset()
        split_datamodule = InferenceDataModule(datamodule, split=split,
                                               rank=trainer.local_rank,
                                               world_size=trainer.world_size)
        trainer.test(model, split_datamodule)
        by_split[split] = model.result
    return by_split


class EmbedderModule(pl.LightningModule):
    def __init__(self, model, id_field):
        super().__init__()
        self.model = model
        self.id_field = id_field

    def forward(self, batch):
        data, _ = batch
        embeddings = self.model.embed(data)  # (B, D).
        assert embeddings.ndim == 2
        # Embeddings: (B, D).
        ids = data.payload[self.id_field]  # (B).
        return ids, embeddings


def extract_embeddings(trainer, datamodule, model, splits=None):
    """Extract embeddings for dataloaders.

    Args:
      loaders: Mapping from a split name to a dataloader.

    Returns:
      Mapping from a split name to a tuple of ids and embeddings tensor (CPU).
    """
    model = EmbedderModule(model, id_field=datamodule.id_field)
    by_split = distributed_predict(trainer, datamodule, model, 2, splits=splits)
    by_split = {split: (ids.cpu().tolist(), embeddings) for split, (ids, embeddings) in by_split.items()}
    return by_split


def embeddings_to_pandas(id_field, by_split):
    all_ids = []
    all_splits = []
    all_embeddings = []
    for split, (ids, embeddings) in by_split.items():
        assert len(ids) == len(embeddings)
        assert embeddings.ndim == 2
        all_ids.extend(ids)
        all_splits.extend([split] * len(ids))
        all_embeddings.append(embeddings)
    all_embeddings = torch.cat(all_embeddings).float().cpu().numpy()

    columns = {id_field: all_ids,
               "split": all_splits}
    for i in range(embeddings.shape[1]):
        columns[f"emb_{i:06}"] = all_embeddings[:, i]
    return pd.DataFrame(columns).set_index(id_field)


@hydra.main(version_base=None)
def main(conf):
    embeddings_path = conf.get("embeddings_path", None)
    if embeddings_path is None:
        raise RuntimeError("Please, provide 'embeddings_path'.")
    if not embeddings_path.endswith(".parquet"):
        raise RuntimeError("Embeddings path must have the '.parquet' extension.")

    trainer = get_trainer(conf)
    datamodule = hydra.utils.instantiate(conf.data_module)
    model = hydra.utils.instantiate(conf.module)
    model.load_state_dict(torch.load(conf.model_path))

    embeddings = extract_embeddings(trainer, datamodule, model)
    embeddings_to_pandas(datamodule.id_field, embeddings).reset_index().to_parquet(embeddings_path)


if __name__ == "__main__":
    main()
