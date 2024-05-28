import torch
from esp_horizon.data import PaddedBatch


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
        if isinstance(x, torch.Tensor):
            if x.ndim != 2:
                raise ValueError("Head input must be either PaddedBatch or a matrix.")
            return super().forward(x)
        x, lengths, mask = x.payload, x.seq_lens, x.seq_len_mask.bool()
        assert x.ndim == 3  # (B, L, D).
        b, l, d = x.shape
        x_new = torch.empty(b, l, self.output_size, dtype=x.dtype, device=x.device)
        x_new[mask] = super().forward(x[mask])
        return PaddedBatch(x_new, lengths)
