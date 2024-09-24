import os
import random
import torch
import numpy as np
from collections import defaultdict
from numbers import Number
from pyarrow.parquet import ParquetFile
from random import Random

from ptls.data_load import read_pyarrow_file
from ptls.data_load.datasets import parquet_file_scan

from .padded_batch import PaddedBatch


def get_nested_value(value):
    if isinstance(value, list):
        if len(value) == 0:
            return None
        return get_nested_value(value[0])
    return value


def get_parquet_length(path):
    with ParquetFile(path) as fp:
        return fp.metadata.num_rows


class HotppDataset(torch.utils.data.IterableDataset):
    """Generate subsequences from parquet file.

    Dataset can contain target labels. Global targets are assigned to each ID and
    local targets are assigned to particular events.

    Args:
        data: Path to a parquet dataset or a list of files.
        min_length: Minimum sequence length. Use 0 to disable subsampling.
        max_length: Maximum sequence length. Disable limit if `None`.
    """
    def __init__(self, data, min_length=0, max_length=None,
                 id_field="id",
                 timestamps_field="timestamps",
                 labels_field="labels",
                 global_target_field="global_target",
                 local_targets_field="local_targets",
                 local_targets_indices_field="local_targets_indices"):
        super().__init__()
        if isinstance(data, str):
            self.filenames = list(sorted(parquet_file_scan(data)))
        elif isinstance(data, list):
            self.filenames = data
        else:
            raise ValueError(f"Unknown data type: {type(data)}")
        if not self.filenames:
            raise RuntimeError("Empty dataset")
        self.total_length = sum(map(get_parquet_length, self.filenames))
        self.min_length = min_length
        self.max_length = max_length
        self.id_field = id_field
        self.timestamps_field = timestamps_field
        self.labels_field = labels_field
        self.global_target_field = global_target_field
        self.local_targets_field = local_targets_field
        self.local_targets_indices_field = local_targets_indices_field

    def shuffle_files(self, rnd=None):
        """Make a new dataset with shuffled partitions."""
        rnd = rnd if rnd is not None else random.Random()
        filenames = list(self.filenames)
        rnd.shuffle(filenames)
        return HotppDataset(filenames,
                            min_length=self.min_length, max_length=self.max_length,
                            id_field=self.id_field, timestamps_field=self.timestamps_field, global_target_field=self.global_target_field,
                            local_targets_field=self.local_targets_field, local_targets_indices_field=self.local_targets_indices_field)

    def is_seq_feature(self, name, value, batch=False):
        """Check whether feature is sequential using its name and value.

        Args:
            batch: Whether the value is a batch of features.
        """
        if name in {self.id_field, self.global_target_field}:
            return False
        if isinstance(value, list):
            ndim = 1
        elif isinstance(value, (np.ndarray, torch.Tensor)):
            ndim = value.ndim
        return ndim > int(batch)

    def process(self, features):
        if self.id_field not in features:
            raise ValueError("Need ID feature")
        if self.timestamps_field not in features:
            raise ValueError("Need timestamps feature")
        if (self.min_length > 0) or (self.max_length is not None):
            # Select subsequences.
            length = len(features[self.timestamps_field])
            max_length = min(length, self.max_length or length)
            min_length = min(length, self.min_length if self.min_length > 0 else max_length)
            out_length = random.randint(min_length, max_length)
            offset = random.randint(0, length - out_length)
            features = {k: (v[offset:offset + out_length] if self.is_seq_feature(k, v) else v)
                        for k, v in features.items()}
            assert len(features[self.timestamps_field]) == out_length
        features[self.timestamps_field] = features[self.timestamps_field].float()
        return features  # Tensors.

    def __len__(self):
        return self.total_length

    def __iter__(self):
        for filename in self.filenames:
            for rec in read_pyarrow_file(filename, use_threads=True):
                features = {k: (torch.from_numpy(v) if isinstance(v, np.ndarray) else torch.tensor(v))
                            for k, v in rec.items()}
                yield self.process(features)

    def collate_fn(self, batch):
        by_name = defaultdict(list)
        for features in batch:
            for name, value in features.items():
                by_name[name].append(value)
        lengths = torch.tensor(list(map(len, by_name[self.timestamps_field])))
        if self.local_targets_field in features:
            local_lengths = torch.tensor(list(map(len, by_name[self.local_targets_field])))
            if self.local_targets_indices_field not in features:
                raise ValueError("Need indices for local targets.")
            local_indices_lengths = torch.tensor(list(map(len, by_name[self.local_targets_indices_field])))
            assert (local_lengths == local_indices_lengths).all()

        # Check consistency.
        batch_sizes = list(map(len, by_name.values()))
        assert all([bs == batch_sizes[0] for bs in batch_sizes])

        # Pad sequences.
        features = {}
        for k, vs in by_name.items():
            if self.is_seq_feature(k, vs[0]):
                features[k] = torch.nn.utils.rnn.pad_sequence(vs, batch_first=True)  # (B, L, *).
            else:
                features[k] = torch.stack(vs)  # (B, *).

        # Extract targets and make PaddedBatch.
        targets = {}
        if self.local_targets_field in features:
            targets["local"] = PaddedBatch({"indices": features.pop(self.local_targets_indices_field),
                                            "targets": features.pop(self.local_targets_field)},
                                           local_lengths,
                                           seq_names={"indices", "targets"})
        if self.global_target_field in features:
            targets["global"] = features.pop(self.global_target_field)
        features = PaddedBatch(features, lengths,
                               seq_names={k for k, v in features.items()
                                          if self.is_seq_feature(k, v, batch=True)})
        return features, targets


class ShuffledDistributedDataset(torch.utils.data.IterableDataset):
    def __init__(self, dataset, num_workers=0, rank=None, world_size=None, cache_size=None, seed=0):
        super().__init__()
        self.dataset = dataset
        self.num_workers = max(num_workers, 1)
        self.rank = rank
        self.world_size = world_size
        self.cache_size = cache_size
        self.seed = seed
        self.epoch = 0

    def _get_context(self):
        dataset = self.dataset
        rank = os.environ.get("RANK", self.rank if self.rank is not None else 0)
        world_size = os.environ.get("WORLD_SIZE", self.world_size if self.world_size is not None else 1)
        total_workers = world_size * self.num_workers
        global_seed = self.seed + self.epoch

        worker_info = torch.utils.data.get_worker_info()
        if worker_info:
            dataset = worker_info.dataset.dataset
            if worker_info.num_workers != self.num_workers:
                raise ValueError("Inconsistent number of workers.")
            worker_id = rank * worker_info.num_workers + worker_info.id
            local_seed = global_seed * total_workers + worker_info.seed
        else:
            local_seed = global_seed * world_size + rank
            worker_id = None
        return dataset, worker_id, total_workers, rank, world_size, global_seed, local_seed

    def __iter__(self):
        dataset, worker_id, total_workers, rank, world_size, global_seed, local_seed = self._get_context()
        if (worker_id is None) and (total_workers == 1):
            worker_id = 0
        for i, item in enumerate(self._iter_shuffled(dataset, global_seed)):
            if (i - worker_id) % total_workers == 0:
                yield item

    def _iter_shuffled(self, dataset, seed):
        if self.cache_size is None:
            yield from dataset
        else:
            rnd = Random(seed)
            cache = []
            for item in dataset.shuffle_files(rnd):
                cache.append(item)
                if len(cache) >= self.cache_size:
                    rnd.shuffle(cache)
                    yield from cache
                    cache = []
            if len(cache) > 0:
                rnd.shuffle(cache)
                yield from cache
