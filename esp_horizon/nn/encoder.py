import torch

from ptls.nn.trx_encoder.encoders import BaseEncoder, IdentityEncoder


class DeltaEncoder(BaseEncoder):
    def forward(self, x):
        assert x.ndim == 1
        x = x.to(torch.float, copy=True)
        x[1:] = x[1:] - x[:-1]
        x[0] *= 0
        return x.unsqueeze(1)  # (T, 1).

    @property
    def output_size(self):
        return 1
