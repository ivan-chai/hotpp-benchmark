"""Compute embeddings."""
import logging
import pickle as pkl

import hydra
import torch
from omegaconf import OmegaConf

from .eval_downstream import extract_embeddings

logger = logging.getLogger(__name__)


@hydra.main(version_base=None)
def main(conf):
    embeddings_path = conf.get("embeddings_path", None)
    if embeddings_path is None:
        raise RuntimeError("Please, provide 'embeddings_path'.")
    embeddings = extract_embeddings(conf)
    with open(embeddings_path, "wb") as fp:
        pkl.dump(embeddings, fp)


if __name__ == "__main__":
    main()
