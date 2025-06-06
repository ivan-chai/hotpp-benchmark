import torch
from hotpp.data import PaddedBatch


class TransformerState:
    """Store full transformer activations history and track indexing."""
    def __init__(self, times, states, seq_lens, index=None, index_lens=None):
        n, b, l, _ = states.shape
        if times.shape != (b, l):
            raise ValueError("Inconsistent times shape.")
        if seq_lens.shape != (b,):
            raise ValueError("Inconsistent lengths shape.")
        if index == "last":
            index = seq_lens - 1  # (B).
            index_lens = None
        elif index is None:
            index = torch.arange(l, device=states.device)[None].repeat(b, 1)  # (B, L).
            index_lens = seq_lens
        elif ((index.ndim not in {1, 2}) or (index.shape[0] != b) or
              (index.ndim == 2 and index_lens is None) or
              (index.ndim == 1 and index_lens is not None)):
            raise ValueError("Required index with shape (B, I) or (B) with corresponding index_lens.")
        self.times = times  # (B, L).
        self.payload = states  # (N, B, L, D).
        self.seq_lens = seq_lens
        self.index = index
        self.index_lens = index_lens

    def to(self, type_or_device):
        self.payload = self.payload.to(type_or_device)
        device = self.payload.device
        if self.times.device != device:
            self.times = self.times.to(device)
            self.seq_lens = self.seq_lens.to(device)
            if self.index is not None:
                self.index = self.index.to(device)
            if self.index_lens is not None:
                self.index_lens = self.index_lens.to(device)
        return self

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, key):
        all_slice = slice(None, None, None)
        if (not isinstance(key, tuple) or
            len(key) > 4 or
            key[0] != all_slice):
            raise NotImplementedError(f"Unexpected slice: {key}")
        new_times = self.times
        new_payload = self.payload
        new_seq_lens = self.seq_lens
        new_index = self.index
        new_index_lens = self.index_lens
        if len(key) > 1 and key[1] != all_slice:  # Batch.
            new_times = new_times[key[1]]
            new_payload = new_payload[:, key[1]]
            new_seq_lens = new_seq_lens[key[1]]
            new_index = new_index[key[1]]
            new_index_lens = new_index_lens[key[1]]
        if self.has_time_dimension and len(key) > 2 and key[2] != all_slice:  # Length.
            zero_mask = new_index_lens == 0  # (B).
            last_index = new_index.take_along_dim((new_index_lens[:, None] - 1).clip(0), 1).squeeze(1)  # (B).
            new_index = new_index[:, key[2]]
            new_index_lens = (new_index <= last_index[:, None]).sum(1)  # (B).
            new_index_lens[zero_mask] = 0
        return TransformerState(new_times, new_payload, new_seq_lens,
                                index=new_index, index_lens=new_index_lens)

    def take_along_dim(self, index, dim):
        if not self.has_time_dimension or dim != 2:
            raise NotImplementedError("Indexing over a non-time dimension.")
        if (index.ndim != 4) or (index.shape[0] != 1) or (index.shape[3] != 1):
            raise ValueError("Index can be applied only to batch and time dimensions.")
        zero_mask = self.index_lens == 0  # (B).
        last_index = self.index.take_along_dim((self.index_lens[:, None] - 1).clip(0), 1).squeeze(1)  # (B).
        new_index = self.index.take_along_dim(index[0, :, :, 0], dim - 1)
        new_index_lens = (new_index <= last_index[:, None]).sum(1)  # (B).
        new_index_lens[zero_mask] = 0
        return TransformerState(self.times, self.payload, self.seq_lens,
                                index=new_index, index_lens=new_index_lens)

    def squeeze(self, dim):
        if not self.has_time_dimension or dim != 2:
            raise NotImplementedError("Only time dimension squeezing is supported.")
        if (self.index_lens == 0).any():
            raise ValueError("Can't squeeze zero-length index.")
        if self.shape[dim] != 1:
            return self
        return TransformerState(self.times, self.payload, self.seq_lens,
                                index=self.index.squeeze(dim - 1), index_lens=None)

    @property
    def has_time_dimension(self):
        return self.index.ndim == 2

    @property
    def shape(self):
        n, b, _, d = self.payload.shape
        if self.has_time_dimension:
            return (n, b, self.index.shape[1], d)
        else:
            return (n, b, d)

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def dtype(self):
        return self.payload.dtype

    @property
    def device(self):
        return self.payload.device

    @property
    def seq_len_mask(self):
        return PaddedBatch(self.payload[0], self.seq_lens).seq_len_mask

    @property
    def index_times(self):
        if not self.has_time_dimension:
            raise NotImplementedError("Indexing single state")
        return PaddedBatch(self.times.take_along_dim(self.index, 1), self.index_lens)
