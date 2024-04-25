import torch


class Head(torch.nn.Sequential):
    """FC head for the sequence encoder

    Args:
        input_dim: Embedding size.
        output_dim: Output dimension.
        hidden_dims: Sizes of linear layers. If None, disable additional linear layers.
        use_batch_norm: Whether to use BatchNorm.
    """
    def __init__(self, input_dim, output_dim,
                 hidden_dims=None,
                 use_batch_norm=False):
        layers = []

        if use_batch_norm:
            layers.append(torch.nn.BatchNorm1d(input_dim))

        last_dim = input_dim
        for dim in hidden_dims or []:
            layers.append(torch.nn.Linear(last_dim, dim, bias=not use_batch_norm))
            if use_batch_norm:
                layers.append(torch.nn.BatchNorm1d(dim))
            layers.append(torch.nn.ReLU())
            last_dim = dim

        layers.append(torch.nn.Linear(last_dim, output_dim))
        super().__init__(*layers)
        self.use_batch_norm = use_batch_norm

    def forward(self, x):
        if self.use_batch_norm:
            b, t, d = x.shape
            x = x.reshape(b * t, d)
        x = super().forward(x)
        if self.use_batch_norm:
            x = x.reshape(b, t, -1)
        return x
