import logging
import time

import hydra
import torch
from tqdm import tqdm
from omegaconf import OmegaConf

from .common import get_trainer, initialize_trainer

logger = logging.getLogger(__name__)


def synchronize(device):
    if torch.device(device).type == "cpu":
        return
    torch.cuda.synchronize(device)


@hydra.main(version_base="1.2", config_path=None)
def main(conf):
    OmegaConf.set_struct(conf, False)
    conf.pop("logger", None)
    model = hydra.utils.instantiate(conf.module)
    dm = hydra.utils.instantiate(conf.data_module)
    trainer = get_trainer(conf)

    n_steps = conf.get("rps_batches", 100)
    n_warmup = conf.get("rps_warmup", 3)
    if not conf.get("rps_no_checkpoint", False):
        model.load_state_dict(torch.load(conf.model_path))

    with initialize_trainer(trainer, model):
        device = trainer.strategy.root_device

        # Ensure optimizers are available for both auto and manual optimization modules.
        # strategy.setup() may not call setup_optimizers outside of a fit() context.
        if not trainer.optimizers:
            opt_config = model.configure_optimizers()
            if isinstance(opt_config, tuple):
                trainer.optimizers = list(opt_config[0])
            elif isinstance(opt_config, list):
                trainer.optimizers = opt_config
            else:
                trainer.optimizers = [opt_config]

        if model.automatic_optimization:
            optimizer = trainer.optimizers[0]

        # Suppress Lightning's result-collection logging — training_step calls self.log()
        # which requires fit-loop infrastructure that isn't set up here.
        model.log = lambda *a, **kw: None
        model.log_dict = lambda *a, **kw: None

        def get_batch(it, loader):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(loader)
                batch = next(it)
            return model.transfer_batch_to_device(batch, device, 0), it

        def warmup_and_time(it, loader, step_fn, n_warmup, n_steps):
            # Pass batch_idx > 0 to avoid triggering slow debug metrics on batch_idx == 0.
            for k in range(1, 1 + n_warmup):
                batch, it = get_batch(it, loader)
                step_fn(batch, k)
            synchronize(device)
            start = time.time()
            size = 0
            for k in tqdm(range(1 + n_warmup, 1 + n_warmup + n_steps)):
                batch, it = get_batch(it, loader)
                size += len(batch[0])
                step_fn(batch, k)
            synchronize(device)
            return size / (time.time() - start), it

        model.train()
        loader = dm.train_dataloader(0, 1)
        it = iter(loader)

        def train_step(batch, k):
            if model.automatic_optimization:
                optimizer.zero_grad()
            loss = model.training_step(batch, k)
            if model.automatic_optimization:
                loss.backward()
                optimizer.step()

        rps, _ = warmup_and_time(it, loader, train_step, n_warmup, n_steps)
        print("Training RPS:", rps)

        model.eval()
        loader = dm.val_dataloader(0, 1)
        it = iter(loader)

        def val_step(batch, k):
            with torch.no_grad():
                model.validation_step(batch, k)

        rps, _ = warmup_and_time(it, loader, val_step, n_warmup, n_steps)
        print("Validation RPS:", rps)

        if hasattr(model, "generate_sequences"):
            model.eval()
            loader = dm.val_dataloader(0, 1)
            it = iter(loader)

            def gen_step(batch, k):
                with torch.no_grad():
                    indices = model._val_metric.select_horizon_indices(batch[0].seq_lens)
                    model.generate_sequences(batch[0], indices)

            rps, _ = warmup_and_time(it, loader, gen_step, n_warmup, n_steps)
            print("Sequence Generation RPS:", rps)


if __name__ == "__main__":
    main()
