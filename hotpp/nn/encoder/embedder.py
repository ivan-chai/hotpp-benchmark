import torch
from hotpp.data import PaddedBatch

from ptls.nn.trx_encoder.batch_norm import RBatchNormWithLens


class EmbeddingEncoder(torch.nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim, normalize=False):
        super().__init__(num_embeddings, embedding_dim)
        self.normalize = normalize

    def forward(self, x):
        if x.payload.ndim != 2:
            raise ValueError(f"Expected tensor with shape (B, L), got {x.payload.shape}.")
        payload = x.payload
        payload = super().forward(payload.clip(min=0, max=self.num_embeddings - 1))
        if self.normalize:
            payload = torch.nn.functional.normalize(payload, dim=-1)
            if isinstance(self.normalize, (int, float)):
                payload = payload * self.normalize
            elif not isinstance(self.normalize, bool):
                raise ValueError(f"Unexpected normalization parameter: {self.normalize}")
        return PaddedBatch(payload, x.seq_lens)  # (B, L, D).

    def clamp(self, x):
        """Map categorical embeddings to nearest centroids.

        The method is primarily designed for diffusion models. Please, refer to the original paper:
        Zhou, Wang-Tao, et al. "Non-autoregressive diffusion-based temporal point processes
        for continuous-time long-term event prediction." Expert Systems with Applications, 2025.
        """
        centroids = self.weight[None]
        if self.normalize:
            centroids = torch.nn.functional.normalize(centroids, dim=-1)
        distances = torch.cdist(x.payload, centroids)  # (B, L, N).
        indices = distances.argmin(dim=-1)  # (B, L, 1).
        return PaddedBatch(super().forward(indices), x.seq_lens)  # (B, L, D).

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
        categorical_noise: The level of additive normal noise added to categorical embeddings, added to embeddings.
    """
    def __init__(self, embeddings=None, numeric_values=None,
                 use_batch_norm=True, categorical_noise=0):
        super().__init__()
        encoders = {}
        for name, spec in (embeddings or {}).items():
            encoders[name] = EmbeddingEncoder(spec["in"], spec["out"], normalize=spec.get("normalize", False))
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
        self.categorical_noise = categorical_noise

        if use_batch_norm:
            custom_embedding_size = sum([encoders[name].output_size for name in self.numeric_order])
            self.custom_embedding_batch_norm = RBatchNormWithLens(custom_embedding_size)

    @property
    def output_size(self):
        return sum([encoder.output_size for encoder in self.embeddings.values()])

    def forward(self, batch):
        embeddings = []
        for name in self.embeddings_order:
            x = self.embeddings[name](batch[name]).payload
            if self.categorical_noise > 0:
                x = x + torch.randn_like(x) * self.categorical_noise
            embeddings.append(x)

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
                custom_embedding = self.custom_embedding_batch_norm(custom_embedding)
            embeddings.append(custom_embedding.payload)
        payload = torch.cat(embeddings, -1)  # (B, L, D).
        return PaddedBatch(payload, batch.seq_lens)

    def clamp(self, embeddings):
        """Map categorical embeddings to nearest centroids.

        The method is primarily designed for diffusion models. Please, refer to the original paper:
        Zhou, Wang-Tao, et al. "Non-autoregressive diffusion-based temporal point processes
        for continuous-time long-term event prediction." Expert Systems with Applications, 2025.
        """
        result = []
        offset = 0
        for name in self.embeddings_order:
            layer = self.embeddings[name]
            dim = layer.output_size
            x = PaddedBatch(embeddings.payload[..., offset:offset + dim],  # (B, L, D).
                            embeddings.seq_lens)
            result.append(layer.clamp(x).payload)
            offset += dim
        result.append(embeddings.payload[..., offset:])
        result = PaddedBatch(torch.cat(result, dim=-1), embeddings.seq_lens)
        assert result.payload.shape == embeddings.payload.shape
        return result

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
