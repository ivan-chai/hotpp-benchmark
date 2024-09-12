import torch
from hotpp.data import PaddedBatch


class ConditionalHead(torch.nn.Sequential):
    """FC head for the sequence encoder

    Args:
        input_size: Embedding size.
        output_size: Output dimension (K x P, where K is the number of output tokens).
        k: The number of output tokens.
        num_layers: The number of transformer blocks.
        fc_dim: The hidden size of FC transformer layer. Default is equal to input_size.
        use_batch_norm: Whether to use BatchNorm before final projection.
    """
    def __init__(self, input_size, output_size, k,
                 query_size=None, hidden_dims=None, use_batch_norm=False):
        if output_size % k != 0:
            raise ValueError("Output size must be divisible by K.")
        if query_size is None:
            query_size = input_size

        layers = []
        last_dim = input_size + query_size
        if use_batch_norm:
            layers.append(torch.nn.BatchNorm1d(last_dim))
        for dim in hidden_dims or []:
            layers.append(torch.nn.Linear(last_dim, dim, bias=not use_batch_norm))
            if use_batch_norm:
                layers.append(torch.nn.BatchNorm1d(dim))
            layers.append(torch.nn.ReLU())
            last_dim = dim

        layers.append(torch.nn.Linear(last_dim, output_size // k))
        super().__init__(*layers)
        self.queries = torch.nn.Parameter(torch.randn(k, query_size))  # (K, D).
        self.use_batch_norm = use_batch_norm
        self.output_size = output_size
        self.k = k

    def forward_impl(self, ctx):
        b, d = ctx.shape
        x = self.queries[None].repeat(b, 1, 1)  # (B, K, D_x).
        x = torch.cat([ctx.unsqueeze(1).repeat(1, self.k, 1), x], -1).flatten(0, 1)  # (BK, D)
        x = super().forward(x)  # (BK, O).
        return x.reshape(b, self.output_size)  # (B, KO).

    def forward(self, x):
        x, lengths, mask = x.payload, x.seq_lens, x.seq_len_mask.bool()
        assert x.ndim > 2  # (B, L, *, D).
        shape = list(x.shape)
        x_masked = x[mask]  # (V, *, D).
        v = len(x_masked)
        x_mapped = self.forward_impl(x_masked.flatten(0, -2)).reshape(*([v] + shape[2:-1] + [self.output_size]))  # (V, *, D).
        x_new = torch.zeros(*[shape[:-1] + [self.output_size]], dtype=x_mapped.dtype, device=x_mapped.device)  # (B, L, *, D).
        x_new[mask] = x_mapped
        return PaddedBatch(x_new, lengths)
