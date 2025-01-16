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


class LogEncoder(torch.nn.Module):
    def __init__(self, clip=1e-6):
        super().__init__()
        self.clip = clip

    def forward(self, x):
        if x.payload.ndim != 2:
            raise ValueError(f"Expected tensor with shape (B, L), got {x.payload.shape}.")
        payload = torch.log(x.payload.clip(min=self.clip)).unsqueeze(-1)
        return PaddedBatch(payload, x.seq_lens)  # (B, L, 1).

    @property
    def output_size(self):
        return 1


class EncoderList(torch.nn.ModuleList):
    def __init__(self, modules):
        super().__init__(modules)

    @property
    def output_size(self):
        return sum([m.output_size for m in self])


class Embedder(torch.nn.Module):
    """Event embedder, that converts structured event into vector.

    Args:
        embeddings: Mapping from the field name to a dictionary of the form `{"in": <num-classes>, "out": <embedding-dim>}`.
        numeric_values: Mapping from the field name to an embedder module
          or one of "identity" and "log"
          or dictionary with "type" string and extra parameters
          or a list of multiple encoders.
        use_batch_norm: Whether to apply batch norm to numeric features embedding or not.
    """
    def __init__(self, embeddings=None, numeric_values=None,
                 use_batch_norm=True):
        super().__init__()
        encoders = {}
        for name, spec in (embeddings or {}).items():
            encoders[name] = EmbeddingEncoder(spec["in"], spec["out"])
        for name, spec in (numeric_values or {}).items():
            if isinstance(spec, (str, torch.nn.Module)):
                encoders[name] = self._make_encoder(spec)
            else:
                # spec is a list of multiple encoders.
                encoders[name] = EncoderList(list(map(self._make_encoder, spec)))
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
            encoder = self.embeddings[name]
            if isinstance(encoder, torch.nn.ModuleList):
                custom_embeddings.extend([e(batch[name]).payload for e in encoder])
            else:
                custom_embeddings.append(encoder(batch[name]).payload)
        if custom_embeddings:
            custom_embedding = PaddedBatch(torch.cat(custom_embeddings, -1), batch.seq_lens)
            if self.use_batch_norm:
                custom_embedding = self.custom_embedding_batch_norm(custom_embedding).payload
            embeddings.append(custom_embedding)
        payload = torch.cat(embeddings, -1)  # (B, L, D).
        return PaddedBatch(payload, batch.seq_lens)

    def _make_encoder(self, spec, **kwargs):
        if isinstance(spec, torch.nn.Module):
            encoder = spec
        elif isinstance(spec, str):
            if spec == "identity":
                encoder = IdentityEncoder(**kwargs)
            elif spec == "log":
                encoder = LogEncoder(**kwargs)
            else:
                raise ValueError(f"Unknown encoder: {spec}.")
        else:
            # spec is dictionary.
            kwargs = dict(spec)
            kwargs.pop("type")
            encoder = self._make_encoder(spec["type"], **kwargs)
        return encoder
