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

from .common import get_trainer
from .data import ShuffledDistributedDataset

logger = logging.getLogger(__name__)


class InferenceModule(pl.LightningModule):
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
        return embeddings, ids


class InferenceDataModule(pl.LightningDataModule):
    def __init__(self, data, split):
        super().__init__()
        self.data = data.with_test_parameters()
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


def extract_embeddings(trainer, datamodule, model, splits=None):
    """Extract embeddings for dataloaders.

    Args:
      loaders: Mapping from a split name to a dataloader.

    Returns:
      Mapping from a split name to a tuple of ids and embeddings tensor (CPU).
    """
    model = InferenceModule(model, id_field=datamodule.id_field)
    by_split = {}
    if splits is None:
        splits = datamodule.splits
    for split in splits:
        split_datamodule = InferenceDataModule(datamodule, split=split)
        split_embeddings, split_ids = zip(*trainer.predict(model, split_datamodule))  # (B, D), (B).
        split_embeddings = torch.cat(split_embeddings).cpu()
        split_ids = torch.cat(split_ids).cpu().tolist()
        by_split[split] = (split_ids, split_embeddings)
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
    all_embeddings = torch.cat(all_embeddings).numpy()

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
    embeddings_to_pandas(dm.id_field, embeddings).reset_index().to_parquet(embeddings_path)


if __name__ == "__main__":
    main()
