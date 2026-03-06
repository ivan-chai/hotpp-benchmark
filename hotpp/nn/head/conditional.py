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

    def forward(self, x, indices=None):
        # x, lengths, mask = x.payload, x.seq_lens, x.seq_len_mask.bool()
        # assert x.ndim > 2  # (B, L, *, D).
        # shape = list(x.shape)
        # x_masked = x[mask]  # (V, *, D).
        # v = len(x_masked)
        # x_mapped = self.forward_impl(x_masked.flatten(0, -2)).reshape(*([v] + shape[2:-1] + [self.output_size]))  # (V, *, D).
        # x_new = torch.zeros(*[shape[:-1] + [self.output_size]], dtype=x_mapped.dtype, device=x_mapped.device)  # (B, L, *, D).
        # x_new[mask] = x_mapped
        # return PaddedBatch(x_new, lengths)

        x, lengths = x.payload, x.seq_lens
        assert x.ndim > 2  # (B, L, D)

        if indices is None:
            # при inference старое поведение без изменений
            mask = x.seq_len_mask.bool()
            shape = list(x.shape)
            x_masked = x[mask]
            v = len(x_masked)
            x_mapped = self.forward_impl(x_masked.flatten(0, -2)).reshape(*([v] + shape[2:-1] + [self.output_size]))  # (V, K*P)
            x_new = torch.zeros(*[shape[:-1] + [self.output_size]], dtype=x_mapped.dtype, device=x_mapped.device)      #(B, L, K*P)
            x_new[mask] = x_mapped
            return PaddedBatch(x_new, lengths)

        else:
            #при training только I позиций из indices
            idx = indices.x["index"]      # (B, I) позиции в [0, L-1]
            out_lengths = indices.seq_lens      # (B,)  сколько валидных из I
            B, I = idx.shape
            D = x.shape[-1]

            #(B, L, D) -> (B, I, D)
            selected = x.take_along_dim(idx.unsqueeze(-1).expand(-1, -1, D), dim=1)  # (B, I, D)

            #маскируем паддинг внутри I позиций
            #seq_len_mask для subset
            idx_mask = indices.seq_len_mask.bool()  # (B, I)
            selected_masked = selected[idx_mask]    # (V_i, D), V_i = sum(out_lengths)

            #прогоняем только V_i позиций через forward_impl
            x_mapped = self.forward_impl(selected_masked)  # (V_i, K*P)

            #собираем обратно в (B, I, K*P)
            x_new = torch.zeros(B, I, self.output_size,dtype=x_mapped.dtype, device=x_mapped.device)
            x_new[idx_mask] = x_mapped
            return PaddedBatch(x_new, out_lengths)
