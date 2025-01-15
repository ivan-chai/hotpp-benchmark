import torch
from hotpp.data import PaddedBatch

from ptls.nn.trx_encoder.batch_norm import RBatchNormWithLens


class EmbeddingEncoder(torch.nn.Embedding):
    def forward(self, x):
        if x.payload.ndim != 2:
            raise ValueError(f"Expected tensor with shape (B, L), got {x.payload.shape}.")
        payload = x.payload
        payload = super().forward(payload.clip(min=0, max=self.num_embeddings - 1))
        return PaddedBatch(payload, x.seq_lens)  # (B, L, D).

    @property
    def output_size(self):
        return self.embedding_dim


class IdentityEncoder(torch.nn.Module):
    def forward(self, x):
        if x.payload.ndim != 2:
            raise ValueError(f"Expected tensor with shape (B, L), got {x.payload.shape}.")
        payload = x.payload.unsqueeze(-1)
        return PaddedBatch(payload, x.seq_lens)  # (B, L, 1).

    @property
    def output_size(self):
        return 1


class DeltaEncoder(torch.nn.Module):
    def forward(self, x):
        if x.payload.ndim != 2:
            raise ValueError(f"Expected tensor with shape (B, L), got {x.payload.shape}.")
        payload = x.payload.clone()
        payload[:, 1:] -= x.payload[:, :-1]
        payload[:, 0] = 0
        return PaddedBatch(payload.unsqueeze(-1), x.seq_lens)  # (B, L, 1).

    @property
    def output_size(self) -> int:
        return 1


class Embedder(torch.nn.Module):
    def __init__(self, embeddings=None, numeric_values=None,
                 use_batch_norm=True):
        super().__init__()
        encoders = {}
        for name, spec in (embeddings or {}).items():
            encoders[name] = EmbeddingEncoder(spec["in"], spec["out"])
        for name, spec in (numeric_values or {}).items():
            if spec == "identity":
                encoders[name] = IdentityEncoder()
            elif spec == "delta":
                encoders[name] = DeltaEncoder()
            else:
                raise ValueError(f"Unknown encoder: {spec}.")
        if not encoders:
            raise ValueError("Empty embedder")
        self.embeddings = torch.nn.ModuleDict(encoders)

        self.use_batch_norm = use_batch_norm
        self.embeddings_order = list(embeddings or {})
        self.numeric_order = list(numeric_values or {})

        if use_batch_norm:
            custom_embedding_size = sum([encoders[name].output_size for name in self.numeric_order])
            self.custom_embedding_batch_norm = RBatchNormWithLens(custom_embedding_size)

    @property
    def output_size(self):
        return sum([encoder.output_size for encoder in self.embeddings.values()])

    def forward(self, batch):
        embeddings = []
        for name in self.embeddings_order:
            embeddings.append(self.embeddings[name](batch[name]).payload)

        custom_embeddings = []
        for name in self.numeric_order:
            custom_embeddings.append(self.embeddings[name](batch[name]).payload)
        if custom_embeddings:
            custom_embedding = PaddedBatch(torch.cat(custom_embeddings, -1), batch.seq_lens)
            if self.use_batch_norm:
                custom_embedding = self.custom_embedding_batch_norm(custom_embedding).payload
            embeddings.append(custom_embedding)
        payload = torch.cat(embeddings, -1)  # (B, L, D).
        return PaddedBatch(payload, batch.seq_lens)
