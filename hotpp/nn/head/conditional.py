import torch
from hotpp.data import PaddedBatch


class ConditionalHead(torch.nn.Sequential):
    """FC head for the sequence encoder

    Args:
        input_size: Embedding size.
        output_size: Output dimension (K x P, where K is the number of output tokens).
        k: The number of output tokens.
        query_size: The dimension of the query. By default it is equal to input size.
        hidden_dims: Sizes of linear layers. If None, disable additional linear layers.
        activation_partial: A function used to construct an activation module.
        use_batch_norm: Whether to use BatchNorm before final projection.
    """
    def __init__(self, input_size, output_size, k,
                 query_size=None, hidden_dims=None,
                 activation_partial=torch.nn.ReLU, use_batch_norm=False):
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
            layers.append(activation_partial())
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

    def forward(self, x, indices=None):
        if indices is None:
            # val/test
            x, lengths, mask = x.payload, x.seq_lens, x.seq_len_mask.bool()
            assert x.ndim > 2  # (B, L, D)
            shape = list(x.shape)
            x_masked = x[mask]
            v = len(x_masked)
            x_mapped = self.forward_impl(x_masked.flatten(0, -2)).reshape(*([v] + shape[2:-1] + [self.output_size]))  # (V, K*P)
            x_new = torch.zeros(*[shape[:-1] + [self.output_size]], dtype=x_mapped.dtype, device=x_mapped.device)      #(B, L, K*P)
            x_new[mask] = x_mapped
            return PaddedBatch(x_new, lengths)

        else: 
            # train
            # indices - (B, I) , x.payload - (B, L, D)
            # x - (B, L, D), lengths - (B), mask - (B, L)
            x, lengths, mask = x.payload, x.seq_lens, x.seq_len_mask.bool()
            assert x.ndim > 2  # (B, L, *, D).
            # indices.payload["index"] - (B, I)
            idx = indices.payload["index"]      # (B, I) positions in [0, L-1]
            out_lengths = indices.seq_lens      # (B,)  valid ones from I
            B, I = idx.shape
            D = x.shape[-1] #D

            #(B, L, D) -> (B, I, D)
            selected = x.take_along_dim(idx.unsqueeze(-1).expand(-1, -1, D), dim=1)  # (B, I, D)

            #masking padding inside I positions
            #seq_len_mask for subset
            idx_mask = indices.seq_len_mask.bool()  # (B, I)
            selected_masked = selected[idx_mask]    # (V_i, D), V_i = sum(out_lengths)

            # predict only for V_i positions
            x_mapped = self.forward_impl(selected_masked)  # (V_i, K*P)

            # convert back to (B, I, K*P)
            x_new = torch.zeros(B, I, self.output_size,dtype=x_mapped.dtype, device=x_mapped.device)
            x_new[idx_mask] = x_mapped
            return PaddedBatch(x_new, out_lengths)
