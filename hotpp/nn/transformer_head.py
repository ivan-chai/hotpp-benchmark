import torch
from hotpp.data import PaddedBatch


class TransformerHead(torch.nn.Module):
    """FC head for the sequence encoder

    Args:
        input_size: Embedding size.
        output_size: Output dimension (K x P, where K is the number of output tokens).
        k: The number of output tokens.
        num_layers: The number of transformer blocks.
        fc_dim: The hidden size of FC transformer layer. Default is equal to input_size.
    """
    def __init__(self, input_size, output_size, k, num_heads,
                 num_layers=1, dropout=0, fc_dim=None):
        if output_size % k != 0:
            raise ValueError("Output size must be divisible by K.")
        super().__init__()
        self.output_size = output_size

        self.queries = torch.nn.Parameter(torch.randn(k, input_size))  # (K, D).
        layers = []
        for i in range(num_layers):
            layer = torch.nn.TransformerDecoderLayer(input_size, num_heads,
                                                     dim_feedforward=fc_dim or input_size,
                                                     dropout=dropout,
                                                     batch_first=True)
            layers.append(layer)
        self.layers = torch.nn.ModuleList(layers)
        self.out_proj = torch.nn.Linear(input_size, output_size // k)

    def forward_impl(self, ctx):
        b, d = ctx.shape
        ctx = ctx.unsqueeze(1)  # (B, 1, D).
        x = self.queries[None].repeat(b, 1, 1)  # (B, K, D).
        for layer in self.layers:
            x = layer(x, ctx)  # (B, K, D).
        x = self.out_proj(x)  # (B, K, O).
        return x.flatten(1, 2)  # (B, KO).

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
