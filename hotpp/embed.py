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
from torchmetrics.utilities import dim_zero_cat
from torchmetrics.utilities.distributed import gather_all_tensors

from .common import get_trainer
from .data import ShuffledDistributedDataset, DEFAULT_PARALLELIZM

logger = logging.getLogger(__name__)


MAX_STRING_LENGTH = 255


class TupleWithCPU(tuple):
    def cpu(self):
        return self

    def tolist(self):
        return list(self)

    def numel(self):
        return len(self)

    def ndim(self):
        return 1


class DistributedCollector:
    """Gather predictions across all processes."""
    def __init__(self, n_values):
        self.n_values = n_values
        self.reset()

    def reset(self):
        self.values = [list() for _ in range(self.n_values)]
        self.dtypes = [None for _ in range(self.n_values)]

    @staticmethod
    def _pack_strings(v):
        v = [s.encode("utf-8") for s in v]
        max_length = max(map(len, v))
        if max_length > MAX_STRING_LENGTH:
            raise ValueError(f"String is too long: {max_length}")
        v = [len(s).to_bytes() + s + b'\0' * (MAX_STRING_LENGTH - len(s)) for s in v]
        v = np.frombuffer(b"".join(v), dtype=np.uint8).reshape(len(v), 1 + MAX_STRING_LENGTH)  # (B, 1 + L).
        v = torch.from_numpy(v.copy())
        return v

    @staticmethod
    def _unpack_strings(v):
        v = v.cpu().numpy()  # (B, 1 + L).
        lengths, v = v[:, 0], v[:, 1:]
        v = [s.tobytes()[:l].decode("utf-8") for s, l in zip(v, lengths)]
        return v

    def update(self, *args):
        if len(args) != self.n_values:
            raise ValueError(f"Wrong number of inputs: {len(args)} != {self.n_values}")
        try:
            device = next(v.device for v in args if isinstance(v, torch.Tensor))
        except StopIteration:
            device = "cpu"
        for i, v in enumerate(args):
            if isinstance(v, (tuple, list)) and isinstance(v[0], str):
                dtype = "str"
                v = self._pack_strings(v).to(device)
            elif isinstance(v, torch.Tensor):
                dtype = "tensor"
            else:
                raise ValueError(f"Can't synchronize object: {v}")
            if self.dtypes[i] is None:
                self.dtypes[i] = dtype
            elif self.dtypes[i] != dtype:
                raise RuntimeError(f"Object type changed for output {i}: {v}")

            if torch.distributed.is_initialized() and (torch.distributed.get_world_size(torch.distributed.group.WORLD) > 1):
                v = dim_zero_cat(gather_all_tensors(v)).cpu()
            self.values[i].append(v)

    def compute(self):
        results = []
        for i in range(self.n_values):
            try:
                values = torch.cat(self.values[i], dim=0)
                dtype = self.dtypes[i]
                if dtype == "str":
                    values = TupleWithCPU(self._unpack_strings(values))
                else:
                    assert dtype == "tensor"
                results.append(values)
            except ValueError:
                # Empty list.
                return None
        return results


class InferenceModule(pl.LightningModule):
    def __init__(self, model, n_outputs):
        super().__init__()
        self.model = model
        self.n_outputs = n_outputs
        self.gather = DistributedCollector(n_outputs)
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
        dataset = ShuffledDistributedDataset(dataset, rank=self.rank, world_size=self.world_size,
                                             parallelize=loader_params.pop("parallelize", DEFAULT_PARALLELIZM))
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
        assert len(ids) == len(embeddings), f"{len(ids)} {len(embeddings)}"
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

    trainer = get_trainer(conf, precision=32)
    datamodule = hydra.utils.instantiate(conf.data_module)
    model = hydra.utils.instantiate(conf.module)
    model.load_state_dict(torch.load(conf.model_path))

    embeddings = extract_embeddings(trainer, datamodule, model)
    embeddings_to_pandas(datamodule.id_field, embeddings).reset_index().to_parquet(embeddings_path)


if __name__ == "__main__":
    main()
