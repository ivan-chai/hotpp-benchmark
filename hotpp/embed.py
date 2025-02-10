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
        data, targets = batch
        embeddings = self.model.embed(data)  # (B, D).
        assert embeddings.ndim == 2
        # Embeddings: (B, D).
        ids = data.payload[self.id_field]  # (B).
        targets = {name: value for name, value in targets.payload.items()
                   if name not in targets.seq_names}  # Keep only global targets.
        return embeddings, ids, targets


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


def extract_embeddings(conf, model=None):
    # Use validation dataset parameters for all splits.
    conf = copy.deepcopy(conf)
    OmegaConf.set_struct(conf, False)
    dataset_params = conf.data_module.test_params if "test_params" in conf.data_module else None
    conf.data_module.train_params = dataset_params
    conf.data_module.val_params = dataset_params
    conf.data_module.test_params = dataset_params

    # Disable logging.
    conf.pop("logger", None)

    # Instantiate.
    if model is None:
        model = hydra.utils.instantiate(conf.module)
        model.load_state_dict(torch.load(conf.model_path))
    dm = hydra.utils.instantiate(conf.data_module)
    model = InferenceModule(model,
                            id_field=dm.id_field)
    trainer = get_trainer(conf)

    # Compute embeddings.
    embeddings = []
    ids = []
    splits = []
    targets = defaultdict(list)
    for split in dm.splits:
        split_dm = InferenceDataModule(dm, split=split)
        split_embeddings, split_ids, split_targets = zip(*trainer.predict(model, split_dm))  # (B, D), (B).
        embeddings.extend(split_embeddings)
        if isinstance(split_ids, torch.Tensor):
            split_ids = split_ids.cpu()
        ids.extend(split_ids)
        splits.extend([split] * (sum(map(len, split_embeddings))))
        for target in split_targets:
            for name, value in target.items():
                targets[name].append(value)
    embeddings = torch.cat(embeddings).cpu().numpy()
    ids = np.concatenate(ids)
    if len(np.unique(ids)) != len(ids):
        raise RuntimeError("Duplicate ids")
    targets = {name: torch.cat(values).cpu().numpy() for name, values in targets.items()}
    for name, values in targets.items():
        if len(values) != len(ids):
            raise RuntimeError(f"Some targets are missing for some IDs ({name}).")

    # Convert to Pandas DataFrame.
    columns = {dm.id_field: ids}
    for i in range(embeddings.shape[1]):
        columns[f"emb_{i:04}"] = embeddings[:, i]
    targets[dm.id_field] = ids
    targets["split"] = splits
    return pd.DataFrame(columns).set_index(dm.id_field), pd.DataFrame(targets).set_index(dm.id_field)


@hydra.main(version_base=None)
def main(conf):
    embeddings_path = conf.get("embeddings_path", None)
    if embeddings_path is None:
        raise RuntimeError("Please, provide 'embeddings_path'.")
    if not embeddings_path.endswith(".parquet"):
        raise RuntimeError("Embeddings path must have the '.parquet' extension.")
    embeddings, _ = extract_embeddings(conf)
    embeddings.reset_index().to_parquet(embeddings_path)


if __name__ == "__main__":
    main()
