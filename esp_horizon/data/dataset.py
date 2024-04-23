import random
import torch
import numpy as np
from collections import defaultdict
from numbers import Number

from ptls.data_load import PaddedBatch
from ptls.data_load.datasets import ParquetDataset


def get_nested_value(value):
    if isinstance(value, list):
        if len(value) == 0:
            return None
        return get_nested_value(value[0])
    return value


def cast_features(value):
    if isinstance(value, dict):
        return {k: cast_features(v) for k, v in value.items()}
    if isinstance(value, list):
        example = get_nested_value(value)
        if example is None:
            raise ValueError("Empty feature")
    else:
        example = value
    if isinstance(example, Number):
        return torch.tensor(value)
    raise NotImplementedError(f"Can't parse data type: {type(value)}.")


class ESPDataset(torch.utils.data.Dataset):
    """Generate subsequences from parquet file.

    Dataset can contain target labels. Global targets are assigned to each ID and
    local targets are assigned to particular events.

    Args:
        path: Path to a parquet dataset.
        min_length: Minimum sequence length. Use 0 to disable subsampling.
        max_length: Maximum sequence length. Disable limit if `None`.
    """
    def __init__(self, path, min_length=0, max_length=None,
                 id_name="id",
                 time_name="timestamps",
                 global_target_name="global_target",
                 local_targets_name="local_targets",
                 local_targets_indices_name="local_targets_indices"):
        super().__init__()
        self.dataset = PTLSParquetDataset([path])
        self.min_length = min_length
        self.max_length = max_length
        self.id_name = id_name
        self.time_name = time_name
        self.global_target_name = global_target_name
        self.local_targets_name = local_targets_name
        self.local_targets_indices_name = local_targets_indices_name

    def is_seq_feature(self, name, value, batch=False):
        """Check whether feature is sequential using its name and value.

        Args:
            batch: Whether the value is a batch of features.
        """
        if name in {self.id_name, self.global_target_name}:
            return False
        if isinstance(value, list):
            ndim = 1
        elif isinstance(value, (np.ndarray, torch.Tensor)):
            ndim = value.ndim
        return ndim > int(batch)

    def __len__(self):
        """Get the number of IDs in the dataset."""
        return len(self.dataset)

    def __getitem__(self, index):
        features = self.dataset[index]
        if self.min_length > 0:
            # Select subsequences.
            length = len(features[self.time_name])
            max_length = min(length, self.max_length) if self.max_length is not None else length
            out_length = random.randint(self.min_length, max_length)
            offset = random.randint(0, length - out_length)
            features = {k: (v[offset:offset + out_length] if self.is_seq_feature(k, v) else v)
                        for k, v in features.items()}
            assert len(features[self.time_name]) == out_length
        return cast_features(features)  # Tensors.

    def collate_fn(self, batch):
        by_name = defaultdict(list)
        for features in batch:
            for name, value in features.items():
                by_name[name].append(value)
        lengths = torch.tensor(list(map(len, by_name[self.time_name])))
        if self.local_targets_name in features:
            local_lengths = torch.tensor(list(map(len, by_name[self.local_targets_name])))
            if self.local_targets_indices_name not in features:
                raise ValueError("Need indices for local targets.")
            local_indices_lengths = torch.tensor(list(map(len, by_name[self.local_targets_indices_name])))
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
        if self.local_target_name in features:
            targets["local"] = PaddedBatch({"indices": features.pop(self.local_targets_indices_name),
                                                            "targets": features.pop(self.local_targets_name)},
                                                           local_lengths)
        if self.global_target_name in features:
            targets["global"] = features.pop(self.global_target_name)
        features = PaddedBatch(features, lengths)
        return features, targets
