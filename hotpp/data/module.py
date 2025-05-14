import torch
import pytorch_lightning as pl
from .dataset import HotppDataset, ShuffledDistributedDataset


def pop_loader_params(params):
    loader_params = {}
    for key in ["seed", "num_workers", "batch_size", "cache_size", "drop_last", "prefetch_factor"]:
        if key in params:
            loader_params[key] = params.pop(key)
    return loader_params


class HotppSampler(torch.utils.data.DistributedSampler):
    def __init__(self, dataset):
        # Skip super init.
        self.dataset = dataset

    #def __len__(self):
    #    return len(self.dataset)

    def __iter__(self):
        while True:
            yield None

    def set_epoch(self, epoch):
        assert hasattr(self.dataset, "epoch")
        self.dataset.epoch = epoch


class HotppDataModule(pl.LightningDataModule):
    def __init__(self,
                 train_path=None,
                 train_params=None,
                 val_path=None,
                 val_params=None,
                 test_path=None,
                 test_params=None,
                 **params
                 ):
        super().__init__()
        self._train_path = train_path
        self._val_path = val_path
        self._test_path = test_path
        self._test_params = test_params
        self._params = params

        if train_path is not None:
            train_params = dict(params, **(train_params or {}))
            self.train_loader_params = pop_loader_params(train_params)
            self.train_data = HotppDataset(train_path, **train_params)
        else:
            self.train_data = None
        if val_path is not None:
            val_params = dict(params, **(val_params or {}))
            self.val_loader_params = pop_loader_params(val_params)
            self.val_data = HotppDataset(val_path, **val_params)
        else:
            self.val_data = None
        if test_path is not None:
            test_params = dict(params, **(test_params or {}))
            self.test_loader_params = pop_loader_params(test_params)
            self.test_data = HotppDataset(test_path, **test_params)
        else:
            self.test_data = None

        train_id_field = self.train_data.id_field if self.train_data is not None else None
        val_id_field = self.val_data.id_field if self.val_data is not None else None
        test_id_field = self.test_data.id_field if self.test_data is not None else None
        id_field = train_id_field or val_id_field or test_id_field
        if ((train_id_field and (train_id_field != id_field)) or
            (val_id_field and (val_id_field != id_field)) or
            (test_id_field and (test_id_field != id_field))):
            raise ValueError("Different id fields in data splits.")
        if id_field is None:
            raise ValueError("No datasets provided.")
        self.id_field = id_field

    def with_test_parameters(self):
        """Return new datamodule with all datasets having test parameters."""
        return HotppDataModule(
            train_path=self._train_path,
            train_params=self._test_params,
            val_path=self._val_path,
            val_params=self._test_params,
            test_path=self._test_path,
            test_params=self._test_params,
            **self._params
        )

    @property
    def splits(self):
        splits = [split for split in ["test", "val", "train"]
                  if getattr(self, f"{split}_data") is not None]
        return splits

    def train_dataloader(self, rank=None, world_size=None):
        rank = self.trainer.local_rank if rank is None else rank
        world_size = self.trainer.world_size if world_size is None else world_size
        loader_params = {"drop_last": True,
                         "pin_memory": torch.cuda.is_available()}
        loader_params.update(self.train_loader_params)
        dataset = ShuffledDistributedDataset(self.train_data, rank=rank, world_size=world_size,
                                             cache_size=loader_params.pop("cache_size", 4096),
                                             seed=loader_params.pop("seed", 0))
        loader = torch.utils.data.DataLoader(
            dataset=dataset,
            collate_fn=dataset.dataset.collate_fn,
            **loader_params
        )
        super(torch.utils.data.DataLoader, loader).__setattr__("sampler", HotppSampler(dataset))  # Add set_epoch hook.
        return loader

    def val_dataloader(self, rank=None, world_size=None):
        rank = self.trainer.local_rank if rank is None else rank
        world_size = self.trainer.world_size if world_size is None else world_size
        loader_params = {"pin_memory": torch.cuda.is_available()}
        loader_params.update(self.val_loader_params)
        dataset = ShuffledDistributedDataset(self.val_data, rank=rank, world_size=world_size)  # Disable shuffle.
        loader = torch.utils.data.DataLoader(
            dataset=dataset,
            collate_fn=dataset.dataset.collate_fn,
            **loader_params
        )
        return loader

    def test_dataloader(self, rank=None, world_size=None):
        rank = self.trainer.local_rank if rank is None else rank
        world_size = self.trainer.world_size if world_size is None else world_size
        loader_params = {"pin_memory": torch.cuda.is_available()}
        loader_params.update(self.test_loader_params)
        dataset = ShuffledDistributedDataset(self.test_data, rank=rank, world_size=world_size)
        loader = torch.utils.data.DataLoader(
            dataset=dataset,
            collate_fn=dataset.dataset.collate_fn,
            **loader_params
        )
        return loader
