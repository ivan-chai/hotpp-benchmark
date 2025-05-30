import torch


def _gather_and_pad(x, indices, mask):
    if x.ndim > 2:
        b, l = indices.shape
        shape = [b, l] + [1] * (x.ndim - 2)
        indices = indices.reshape(*shape)
        mask = mask.reshape(*shape)
    return x.take_along_dim(indices, 1) * mask


class PaddedBatch:
    """Contains a padded batch of sequences with different lengths.

    Args:
        payload: Tensor or dictionary of tensors.
        length: Tensor of shape (B,) with lengths of sequences.
        seq_names: A set of sequential features names when payload is a dictionary. By default all values are sequential.
        left: Use left-side padding.
        flip_padding: Change padding from left to right or vice versa.
    """
    def __init__(self, payload, lengths, seq_names=None, left=False, flip_padding=False):
        if isinstance(payload, dict):
            if seq_names is None:
                seq_names = set(payload)
            else:
                seq_names = set(seq_names).intersection(payload)
            for name in seq_names:
                if payload[name].ndim < 2:
                    raise ValueError(f"The field {name} doesn't have a time dimension.")
            seq_names = tuple(sorted(seq_names))
        else:
            if seq_names is not None:
                raise ValueError("Tensor batch can't have seq_names.")
            if payload.ndim < 2:
                raise ValueError(f"Expected a tensor with shape (B, L, *), got {payload.shape}.")
        if flip_padding and seq_names:
            seq_feature = payload if isinstance(payload, torch.Tensor) else payload[seq_names[0]]
            l = seq_feature.shape[1]
            if left:
                indices = torch.arange(l, device=lengths.device)[None] + lengths[:, None] - l  # (B, L).
                mask = indices >= 0
                indices = indices.clip(min=0)
            else:
                indices = torch.arange(l, device=lengths.device)[None] + l - lengths[:, None]  # (B, L).
                mask = indices < l
                indices = indices.clip(max=l - 1)
            if isinstance(payload, dict):
                payload = {k: (_gather_and_pad(v, indices, mask) if FeatureDict.is_seq_feature(k, v, batch=True) else v)
                           for k, v in payload.items()}
            else:
                payload = _gather_and_pad(payload, indices, mask)
        self._payload = payload
        self._lengths = lengths
        self._seq_names = seq_names
        self._left = left

        # Check.
        if self._lengths.shape != (self.shape[0],):
            raise ValueError("Inconsistent lengths shape.")

    def __getitem__(self, key):
        if isinstance(self._payload, torch.Tensor):
            raise TypeError("Items are supported for dictionary batches only.")
        return PaddedBatch(self._payload[key], self._lengths, left=self._left)

    def clone(self):
        if isinstance(self._payload, torch.Tensor):
            payload = self._payload
            seq_names = None
        else:
            payload = dict(self._payload)
            seq_names = self.seq_names
        return PaddedBatch(payload, self._lengths, seq_names, left=self._left)

    @property
    def left(self):
        return self._left

    @property
    def payload(self):
        return self._payload

    @payload.setter
    def payload(self, value):
        if not isinstance(value, type(self._payload)):
            raise ValueError("Incompatible types.")
        self._payload = value

    @property
    def seq_lens(self):
        return self._lengths

    @property
    def seq_names(self):
        return self._seq_names

    @property
    def device(self):
        return self._lengths.device

    def __len__(self):
        return len(self._lengths)

    @property
    def shape(self):
        """Returns first two dimensions of the sequential features."""
        if isinstance(self.payload, torch.Tensor):
            return self.payload.shape[:2]
        else:
            if self.seq_names:
                return self.payload[self.seq_names[0]].shape[:2]
            else:
                batch_size = len(next(iter(self.payload.values())))
                return (batch_size, 0)

    def to(self, *args, **kwargs):
        lengths = self._lengths.to(*args, **kwargs)
        if isinstance(self._payload, dict):
            payload = {
                k: v.to(*args, **kwargs) if type(v) is torch.Tensor else v
                for k, v in self._payload.items()
            }
        else:
            payload = self._payload.to(*args, **kwargs)
        return PaddedBatch(payload, lengths, self._seq_names)

    @property
    def seq_len_mask(self):
        """mask with B*T size for valid tokens in `payload`
        """
        if type(self._payload) is dict:
            name = self.seq_names[0]
            l = self._payload[name].shape[1]
        else:
            l = self._payload.shape[1]
        indices = torch.arange(l, device=self._lengths.device)
        if self._left:
            indices = indices.flip(0)
        return indices[None] < self._lengths[:, None]
