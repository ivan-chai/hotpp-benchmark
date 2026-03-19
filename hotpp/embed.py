"""Compute embeddings."""
import logging

import hydra
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch

from .common import get_trainer, model_eval, DistributedPredictor

logger = logging.getLogger(__name__)


class EmbedderModule(pl.LightningModule):
    def __init__(self, model, id_field):
        super().__init__()
        self.model = model
        self.id_field = id_field

    def forward(self, batch):
        data, _ = batch
        embeddings = self.model.embed(data)  # (B, D).
        assert embeddings.ndim == 2
        # Embeddings: (B, D).
        ids = data.payload[self.id_field]  # (B).
        return ids, embeddings


class EmbeddingsExtractor(DistributedPredictor):
    """Extract embeddings for dataloaders.

    Returns a mapping from a split name to a tuple of ids and embeddings tensor (CPU).
    """
    def __call__(self, model, pandas=False):
        model = EmbedderModule(model, id_field=self.id_field)
        by_split = super().__call__(model, 2)
        by_split = {split: (ids.cpu().tolist(), embeddings) for split, (ids, embeddings) in by_split.items()}
        if pandas:
            return self._embeddings_to_pandas(self.id_field, by_split)
        else:
            return by_split

    @staticmethod
    def _embeddings_to_pandas(id_field, by_split):
        all_ids = []
        all_splits = []
        all_embeddings = []
        for split, (ids, embeddings) in by_split.items():
            assert len(ids) == len(embeddings), f"{len(ids)} {len(embeddings)}"
            assert embeddings.ndim == 2
            all_ids.extend(ids)
            all_splits.extend([split] * len(ids))
            all_embeddings.append(embeddings)
        all_embeddings = torch.cat(all_embeddings).float().cpu().numpy()

        columns = {id_field: all_ids,
                "split": all_splits}
        for i in range(embeddings.shape[1]):
            columns[f"emb_{i:06}"] = all_embeddings[:, i]
        return pd.DataFrame(columns).set_index(id_field)


@hydra.main(version_base=None)
def main(conf):
    embeddings_path = conf.get("embeddings_path", None)
    if embeddings_path is None:
        raise RuntimeError("Please, provide 'embeddings_path'.")
    if not embeddings_path.endswith(".parquet"):
        raise RuntimeError("Embeddings path must have the '.parquet' extension.")

    trainer = get_trainer(conf, precision=32)
    datamodule = hydra.utils.instantiate(conf.data_module)
    model = hydra.utils.instantiate(conf.module)
    model.load_state_dict(torch.load(conf.model_path))

    extractor = EmbeddingsExtractor(trainer, datamodule)
    extractor(model, pandas=True).reset_index().to_parquet(embeddings_path)


if __name__ == "__main__":
    main()
