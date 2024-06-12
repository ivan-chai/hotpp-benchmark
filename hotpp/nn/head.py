import torch
from hotpp.data import PaddedBatch


class Head(torch.nn.Sequential):
    """FC head for the sequence encoder

    Args:
        input_size: Embedding size.
        output_size: Output dimension.
        hidden_dims: Sizes of linear layers. If None, disable additional linear layers.
        use_batch_norm: Whether to use BatchNorm.
    """
    def __init__(self, input_size, output_size,
                 hidden_dims=None,
                 use_batch_norm=False):
        layers = []

        if use_batch_norm:
            layers.append(torch.nn.BatchNorm1d(input_size))

        last_dim = input_size
        for dim in hidden_dims or []:
            layers.append(torch.nn.Linear(last_dim, dim, bias=not use_batch_norm))
            if use_batch_norm:
                layers.append(torch.nn.BatchNorm1d(dim))
            layers.append(torch.nn.ReLU())
            last_dim = dim

        layers.append(torch.nn.Linear(last_dim, output_size))
        super().__init__(*layers)
        self.use_batch_norm = use_batch_norm
        self.output_size = output_size

    def forward(self, x):
        x, lengths, mask = x.payload, x.seq_lens, x.seq_len_mask.bool()
        assert x.ndim > 2  # (B, L, *, D).
        shape = list(x.shape)
        x_masked = x[mask]  # (V, *, D).
        v = len(x_masked)
        x_mapped = super().forward(x_masked.flatten(0, -2)).reshape(*([v] + shape[2:-1] + [self.output_size]))  # (V, *, D).
        x_new = torch.zeros(*[shape[:-1] + [self.output_size]], dtype=x_mapped.dtype, device=x_mapped.device)  # (B, L, *, D).
        x_new[mask] = x_mapped
        return PaddedBatch(x_new, lengths)
