import os
import pyarrow as pa
import tempfile
import torch
from unittest import TestCase, main

from hotpp.data import HotppDataModule


def gather_distributed_dataset(data, split, world_size=1, epoch=1):
    get_loader_fn = getattr(data, f"{split}_dataloader")
    loaders = [get_loader_fn(rank=i, world_size=world_size)
               for i in range(world_size)]
    for loader in loaders:
        if hasattr(loader.sampler, "set_epoch"):
            loader.sampler.set_epoch(epoch)
    by_worker = list(map(list, loaders))
    return by_worker


class TestDDPDataLoader(TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name

        ids = pa.array(list(range(15)))
        timestamps = pa.array([list(range(i)) for i in range(15)])
        table = pa.Table.from_arrays([ids, timestamps], names=["id", "timestamps"])
        self.data15_path = os.path.join(self.root, "data15.parquet")
        pa.parquet.write_table(table, self.data15_path)

        ids = pa.array(list(range(16)))
        timestamps = pa.array([list(range(i)) for i in range(16)])
        table = pa.Table.from_arrays([ids, timestamps], names=["id", "timestamps"])
        self.data16_path = os.path.join(self.root, "data16.parquet")
        pa.parquet.write_table(table, self.data16_path)

    def tearDown(self):
        """ Called after every test. """
        self.tmp.cleanup()

    def test_workers(self):
        # 15 items, drop last, world 1, 0 / 1 worker.
        for num_workers in [0, 1]:
            data = HotppDataModule(train_path=self.data15_path,
                                   train_params={
                                       "batch_size": 4,
                                       "num_workers": num_workers,
                                       "cache_size": None  # Disable shuffle.
                                   })
            items = gather_distributed_dataset(data, "train")
            items = sum(items, [])
            ids = torch.cat([v.payload["id"] for v, _ in items]).tolist()
            self.assertEqual(len(ids), 12)
            self.assertEqual(set(ids), set(range(15)) - {12, 13, 14})

        # 15 items, drop last, world 1, 2 workers.
        data = HotppDataModule(train_path=self.data15_path,
                               train_params={
                                   "batch_size": 4,
                                   "num_workers": 2,
                                   "cache_size": None  # Disable shuffle.
                               })
        items = gather_distributed_dataset(data, "train")
        items = sum(items, [])
        ids = torch.cat([v.payload["id"] for v, _ in items]).tolist()
        self.assertEqual(len(ids), 12)
        self.assertEqual(set(ids), set(range(15)) - {9, 11, 13})

    def test_ddp(self):
        # 15 items, drop last, world 2.
        data = HotppDataModule(train_path=self.data15_path,
                               train_params={
                                   "batch_size": 4,
                                   "num_workers": 2,
                                   "cache_size": None  # Disable shuffle.
                               })
        items = gather_distributed_dataset(data, "train", world_size=2)
        items = sum(items, [])
        ids = torch.cat([v.payload["id"] for v, _ in items]).tolist()
        self.assertEqual(len(ids), 12)
        self.assertEqual(set(ids), set(range(15)) - {3, 7, 11})

        for world_size in [1, 2]:
            # 15 items, without drop last.
            data = HotppDataModule(test_path=self.data15_path,
                                   test_params={
                                       "batch_size": 4,
                                       "num_workers": 2
                                   })
            items = gather_distributed_dataset(data, "test", world_size)
            items = sum(items, [])
            ids = torch.cat([v.payload["id"] for v, _ in items]).tolist()
            self.assertEqual(len(ids), 15)
            self.assertEqual(set(ids), set(range(15)))

            # 16 items, last will not be dropped.
            data = HotppDataModule(train_path=self.data16_path,
                                   train_params={
                                       "batch_size": 4,
                                       "num_workers": 2,
                                       "cache_size": 4
                                   })
            items = gather_distributed_dataset(data, "train", world_size)
            items = sum(items, [])
            ids = torch.cat([v.payload["id"] for v, _ in items]).tolist()
            self.assertEqual(len(ids), 16)
            self.assertEqual(set(ids), set(range(16)))

    def test_seed(self):
        for world_size in [1, 2]:
            # 16 items, last will not be dropped.
            data = HotppDataModule(train_path=self.data16_path,
                                   train_params={
                                       "cache_size": 4,
                                       "batch_size": 4,
                                       "num_workers": 2
                                   })
            items = gather_distributed_dataset(data, "train", world_size)
            items = sum(items, [])
            ids1 = torch.cat([v.payload["id"] for v, _ in items]).tolist()

            items = gather_distributed_dataset(data, "train", world_size)
            items = sum(items, [])
            ids2 = torch.cat([v.payload["id"] for v, _ in items]).tolist()

            self.assertEqual(ids1, ids2)

    def test_shuffle(self):
        for world_size in [1, 2]:
            # 16 items, last will not be dropped.
            data = HotppDataModule(train_path=self.data16_path,
                                   train_params={
                                       "cache_size": 4,
                                       "batch_size": 4,
                                       "num_workers": 2
                                   })
            items = gather_distributed_dataset(data, "train", world_size, epoch=1)
            items = sum(items, [])
            ids1 = torch.cat([v.payload["id"] for v, _ in items]).tolist()

            items = gather_distributed_dataset(data, "train", world_size, epoch=2)
            items = sum(items, [])
            ids2 = torch.cat([v.payload["id"] for v, _ in items]).tolist()

            self.assertEqual(set(ids1), set(ids2))
            self.assertNotEqual(ids1, ids2)


if __name__ == "__main__":
    main()
