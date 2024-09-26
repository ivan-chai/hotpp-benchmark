import logging
import time

import hydra
import torch
from tqdm import tqdm
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


def synchronize(device):
    if device == "cpu":
        return
    torch.cuda.synchronize(device)


@hydra.main(version_base="1.2", config_path=None)
def main(conf):
    OmegaConf.set_struct(conf, False)
    conf.pop("logger", None)
    model = hydra.utils.instantiate(conf.module)
    model.load_state_dict(torch.load(conf.model_path))
    dm = hydra.utils.instantiate(conf.data_module)

    n_steps = conf.get("rps_batches", 10)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    model.train()
    loader = dm.train_dataloader()
    it = iter(loader)
    synchronize(device)
    start = time.time()
    for k in tqdm(range(n_steps)):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        batch = [batch[0].to(device)] + list(batch[1:])
        model.training_step(batch, k)
    synchronize(device)
    print("Training RPS:", n_steps / (time.time() - start))

    model.eval()
    loader = dm.val_dataloader()
    it = iter(loader)
    synchronize(device)
    start = time.time()
    for k in tqdm(range(n_steps)):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        batch = [batch[0].to(device)] + list(batch[1:])
        with torch.no_grad():
            model.validation_step(batch, k)
    synchronize(device)
    print("Validation RPS:", n_steps / (time.time() - start))


if __name__ == "__main__":
    main()
