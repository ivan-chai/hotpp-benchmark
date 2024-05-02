import torch
import pytorch_lightning as pl
from .dataset import ESPDataset, ShuffledDistributedDataset


def pop_loader_params(params):
    loader_params = {}
    for key in ["seed", "num_workers", "batch_size", "cache_size", "drop_last", "prefetch_factor"]:
        if key in params:
            loader_params[key] = params.pop(key)
    return loader_params


class ESPSampler(torch.utils.data.DistributedSampler):
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


class ESPDataModule(pl.LightningDataModule):
    def __init__(self,
                 train_path=None,
                 train_params=None,
                 dev_path=None,
                 dev_params=None,
                 test_path=None,
                 test_params=None,
                 **params
                 ):
        super().__init__()
        if train_path is not None:
            train_params = dict(params, **(train_params or {}))
            self.train_loader_params = pop_loader_params(train_params)
            self.train_data = ESPDataset(train_path, **train_params)
        else:
            self.train_data = None
        if dev_path is not None:
            dev_params = dict(params, **(dev_params or {}))
            self.dev_loader_params = pop_loader_params(dev_params)
            self.dev_data = ESPDataset(dev_path, **dev_params)
        else:
            self.dev_data = None
        if test_path is not None:
            test_params = dict(params, **(test_params or {}))
            self.test_loader_params = pop_loader_params(test_params)
            self.test_data = ESPDataset(test_path, **test_params)
        else:
            self.test_data = None

        train_id_field = self.train_data.id_field if self.train_data is not None else None
        dev_id_field = self.dev_data.id_field if self.dev_data is not None else None
        test_id_field = self.test_data.id_field if self.test_data is not None else None
        id_field = train_id_field or dev_id_field or test_id_field
        if ((train_id_field and (train_id_field != id_field)) or
            (dev_id_field and (dev_id_field != id_field)) or
            (test_id_field and (test_id_field != id_field))):
            raise ValueError("Different id fields in data splits.")
        if id_field is None:
            raise ValueError("No datasets provided.")
        self.id_field = id_field

    def train_dataloader(self, rank=None, world_size=None):
        loader_params = {"drop_last": True,
                         "pin_memory": torch.cuda.is_available()}
        loader_params.update(self.train_loader_params)
        dataset = ShuffledDistributedDataset(self.train_data, rank=rank, world_size=world_size,
                                             num_workers=loader_params.get("num_workers", 0),
                                             cache_size=loader_params.pop("cache_size", 4096),
                                             seed=loader_params.pop("seed", 0))
        loader = torch.utils.data.DataLoader(
            dataset=dataset,
            collate_fn=dataset.dataset.collate_fn,
            **loader_params
        )
        super(torch.utils.data.DataLoader, loader).__setattr__("sampler", ESPSampler(dataset))  # Add set_epoch hook.
        return loader

    def val_dataloader(self, rank=None, world_size=None):
        loader_params = {"pin_memory": torch.cuda.is_available()}
        loader_params.update(self.dev_loader_params)
        dataset = ShuffledDistributedDataset(self.dev_data, rank=rank, world_size=world_size,
                                             num_workers=loader_params.get("num_workers", 0))  # Disable shuffle.
        loader = torch.utils.data.DataLoader(
            dataset=dataset,
            collate_fn=dataset.dataset.collate_fn,
            **loader_params
        )
        return loader

    def test_dataloader(self, rank=None, world_size=None):
        loader_params = {"pin_memory": torch.cuda.is_available()}
        loader_params.update(self.test_loader_params)
        dataset = ShuffledDistributedDataset(self.test_data, rank=rank, world_size=world_size,
                                             num_workers=loader_params.get("num_workers", 0))
        loader = torch.utils.data.DataLoader(
            dataset=dataset,
            collate_fn=dataset.dataset.collate_fn,
            **loader_params
        )
        return loader
