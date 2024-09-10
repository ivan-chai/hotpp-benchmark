import logging

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf

from hotpp.data import ShuffledDistributedDataset
from hotpp.data.module import HotppSampler
from tqdm import tqdm

logger = logging.getLogger(__name__)


def get_loader(dm):
    loader_params = {"drop_last": False,
                     "pin_memory": torch.cuda.is_available()}
    loader_params.update(dm.train_loader_params)
    dataset = ShuffledDistributedDataset(dm.val_data, rank=None, world_size=None,
                                         num_workers=loader_params.get("num_workers", 0),
                                         cache_size=loader_params.pop("cache_size", 4096),
                                         seed=loader_params.pop("seed", 0))
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        collate_fn=dataset.dataset.collate_fn,
        **loader_params
    )
    super(torch.utils.data.DataLoader, loader).__setattr__("sampler", HotppSampler(dataset))  # Add set_epoch hook.
    return loader


def calibrate(model, loader):
    model.eval()
    model._loss.train()
    for i, batch in enumerate(tqdm(loader)):
        model.training_step(batch, i)


@hydra.main(version_base="1.2", config_path=None)
def main(conf):
    OmegaConf.set_struct(conf, False)
    conf.pop("logger", None)

    model = hydra.utils.instantiate(conf.module)
    dm = hydra.utils.instantiate(conf.data_module)

    if "model_path" not in conf:
        raise ValueError("Need model_path for a model initialization")
    logger.info(f"Load weights from '{conf.model_path}'")

    model.load_state_dict(torch.load(conf.model_path))

    pl.seed_everything(conf.seed_everything)
    calibrate(model, get_loader(dm))

    torch.save(model.state_dict(), conf.model_path)
    logger.info(f"New weights are dumped to '{conf.model_path}'")


if __name__ == "__main__":
    main()
