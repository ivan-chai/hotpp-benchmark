import copy
import datetime
import functools
import logging
import time
import yaml
from contextlib import contextmanager

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
from torchmetrics.utilities.data import dim_zero_cat
from torchmetrics.utilities.distributed import gather_all_tensors

from hotpp.data import PaddedBatch, ShuffledDistributedDataset, DEFAULT_PARALLELIZM, update_loader_params_with_defaults
from hotpp.utils.config import as_flat_config


logger = logging.getLogger(__name__)


MAX_STRING_LENGTH = 255


def initialize(conf):
    """Set random seed and torch multiprocessing sharing strategy from config."""
    if "seed_everything" in conf:
        pl.seed_everything(conf.seed_everything)
    sharing_strategy = conf.get("multiprocessing_sharing_strategy", None)
    if sharing_strategy is not None:
        torch.multiprocessing.set_sharing_strategy(sharing_strategy)


def dump_report(metrics, fp):
    result = dict(metrics)
    result["date"] = f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}"
    yaml.safe_dump(result, fp)


def patch_precision_plugin(trainer):
    plugin = trainer.precision_plugin
    base_method = plugin.convert_input.__func__
    @functools.wraps(base_method)
    def convert_input(self, data):
        if isinstance(data, PaddedBatch):
            data = copy.copy(data)
            data.payload = base_method(self, data.payload)
            return data
        if isinstance(data, tuple):
            return tuple([convert_input(self, v) for v in data])
        return base_method(self, data)
    plugin.convert_input = convert_input.__get__(plugin, plugin.__class__)
    return trainer


class TrainingTimeCallback(pl.Callback):
    """Tracks cumulative wall-clock training time across restarts.

    The accumulated time is saved in checkpoints via state_dict / load_state_dict
    so that resuming from a checkpoint picks up where it left off.
    The metric ``train_time_hours`` is logged at the end of every training epoch.
    """

    def __init__(self):
        self._elapsed_seconds = 0.0  # time accumulated from all previous runs
        self._start_time = None      # wall-clock time of the current run's start

    # ------------------------------------------------------------------
    # Checkpoint persistence
    # ------------------------------------------------------------------

    def state_dict(self):
        return {"elapsed_seconds": self._total_elapsed()}

    def load_state_dict(self, state_dict):
        self._elapsed_seconds = state_dict["elapsed_seconds"]

    # ------------------------------------------------------------------
    # Training hooks
    # ------------------------------------------------------------------

    def on_train_start(self, trainer, pl_module):
        self._start_time = time.monotonic()

    def on_train_end(self, trainer, pl_module):
        if self._start_time is not None:
            self._elapsed_seconds += time.monotonic() - self._start_time
            self._start_time = None

    def on_train_epoch_end(self, trainer, pl_module):
        pl_module.log("train_time_hours", self._total_elapsed() / 3600.0,
                      on_step=False, on_epoch=True, prog_bar=True)

    # ------------------------------------------------------------------

    def _total_elapsed(self):
        elapsed = self._elapsed_seconds
        if self._start_time is not None:
            elapsed += time.monotonic() - self._start_time
        return elapsed


@contextmanager
def model_eval(model):
    """Context manager that switches a model to eval mode and restores its original training state on exit."""
    training_mode = model.training
    model.eval()
    try:
        yield model
    finally:
        model.train(training_mode)


class TupleWithCPU(tuple):
    def cpu(self):
        return self

    def tolist(self):
        return list(self)

    def numel(self):
        return len(self)

    def ndim(self):
        return 1


class DistributedCollector:
    """Gather predictions across all processes."""
    def __init__(self, n_values):
        self.n_values = n_values
        self.reset()

    def reset(self):
        self.values = [list() for _ in range(self.n_values)]
        self.dtypes = [None for _ in range(self.n_values)]

    @staticmethod
    def _pack_strings(v):
        v = [s.encode("utf-8") for s in v]
        max_length = max(map(len, v))
        if max_length > MAX_STRING_LENGTH:
            raise ValueError(f"String is too long: {max_length}")
        v = [len(s).to_bytes() + s + b'\0' * (MAX_STRING_LENGTH - len(s)) for s in v]
        v = np.frombuffer(b"".join(v), dtype=np.uint8).reshape(len(v), 1 + MAX_STRING_LENGTH)  # (B, 1 + L).
        v = torch.from_numpy(v.copy())
        return v

    @staticmethod
    def _unpack_strings(v):
        v = v.cpu().numpy()  # (B, 1 + L).
        lengths, v = v[:, 0], v[:, 1:]
        v = [s.tobytes()[:l].decode("utf-8") for s, l in zip(v, lengths)]
        return v

    def update(self, *args):
        if len(args) != self.n_values:
            raise ValueError(f"Wrong number of inputs: {len(args)} != {self.n_values}")
        try:
            device = next(v.device for v in args if isinstance(v, torch.Tensor))
        except StopIteration:
            device = "cpu"
        for i, v in enumerate(args):
            if isinstance(v, (tuple, list)) and isinstance(v[0], str):
                dtype = "str"
                v = self._pack_strings(v).to(device)
            elif isinstance(v, torch.Tensor):
                dtype = "tensor"
            else:
                raise ValueError(f"Can't synchronize object: {v}")
            if self.dtypes[i] is None:
                self.dtypes[i] = dtype
            elif self.dtypes[i] != dtype:
                raise RuntimeError(f"Object type changed for output {i}: {v}")

            if torch.distributed.is_initialized() and (torch.distributed.get_world_size(torch.distributed.group.WORLD) > 1):
                v = dim_zero_cat(gather_all_tensors(v)).cpu()
            else:
                v = v.cpu()
            self.values[i].append(v)

    def compute(self):
        results = []
        for i in range(self.n_values):
            if len(self.values[i]) == 0:
                # Empty list.
                return None
            values = torch.cat(self.values[i], dim=0)
            dtype = self.dtypes[i]
            if dtype == "str":
                values = TupleWithCPU(self._unpack_strings(values))
            else:
                assert dtype == "tensor"
            results.append(values)
        return results


@contextmanager
def initialize_trainer(trainer, model):
    """Context manager that initializes a trainer if it is in INITIALIZING state and tears it down on exit."""
    need_teardown = False
    if trainer.state.status == pl.trainer.states.TrainerStatus.INITIALIZING:
        logger.info("Initialize Trainer")
        trainer.strategy.connect(model)
        if not torch.distributed.is_initialized():
            # Only init the process group when it hasn't been set up yet (e.g. standalone
            # inference). When called from within a training run, the group is already
            # initialized; calling setup_environment() again would raise
            # "trying to initialize the default process group twice!" on all ranks
            # except the one that happens to win the race, causing a rendezvous timeout.
            trainer.strategy.setup_environment()
            need_teardown = True
        trainer.strategy.setup(trainer)
    if need_teardown and torch.distributed.is_initialized():
        torch.distributed.barrier()
    try:
        yield
    finally:
        if need_teardown:
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            trainer.strategy.teardown()


class InferenceDataModule(pl.LightningDataModule):
    def __init__(self, data, split, rank, world_size):
        super().__init__()
        self.data = data.with_test_parameters()
        self.split = split
        self.rank = rank
        self.world_size = world_size

    def test_dataloader(self):
        dataset = getattr(self.data, f"{self.split}_data")
        loader_params = update_loader_params_with_defaults(getattr(self.data, f"{self.split}_loader_params"))
        parallelize = loader_params.pop("parallelize", DEFAULT_PARALLELIZM)
        loader_params = {k: loader_params[k] for k in ["batch_size", "num_workers", "pin_memory", "persistent_workers", "multiprocessing_context"]}

        dataset = ShuffledDistributedDataset(dataset, rank=self.rank, world_size=self.world_size,
                                             parallelize=parallelize)
        return torch.utils.data.DataLoader(
            dataset=dataset,
            collate_fn=dataset.dataset.collate_fn,
            shuffle=False,
            **loader_params
        )


class DistributedPredictor:
    def __init__(self, trainer, datamodule, splits=None):
        self.trainer = trainer
        self.datamodule = datamodule
        self.splits = datamodule.splits if splits is None else splits
        self._dataloaders = {}

    @property
    def id_field(self):
        return self.datamodule.id_field

    def __call__(self, model, n_outputs):
        # DONE: rewrite this code to use two new context managers: initialize_trainer (put it in this file) and model_eval (put in common.py), if there is no default Torch manager for it.
        with initialize_trainer(self.trainer, model), \
             model_eval(model):
            # DONE: cache dataloaders and inherit "persistent_workers" flag from a datamodule.
            by_split = {}
            for split in self.splits:
                if split not in self._dataloaders:
                    split_datamodule = InferenceDataModule(self.datamodule, split=split,
                                                           rank=self.trainer.global_rank,
                                                           world_size=self.trainer.world_size)
                    self._dataloaders[split] = split_datamodule.test_dataloader()
                collector = DistributedCollector(n_outputs)
                with torch.no_grad():
                    for batch in self._dataloaders[split]:
                        batch = model.transfer_batch_to_device(batch, self.trainer.strategy.root_device, 0)
                        collector.update(*model(batch))
                by_split[split] = collector.compute()
        return by_split


def get_trainer(conf, **trainer_params_additional):
    trainer_params = copy.deepcopy(conf.trainer)
    OmegaConf.set_struct(trainer_params, False)
    if (trainer_params.get("accelerator", "default") != "cpu"
        and ("sync_batchnorm" not in trainer_params)):
        logging.info("Force batchnorm synchronization. Use explicit 'cpu' device to disable it.")
        trainer_params["sync_batchnorm"] = True
    model_selection = trainer_params.get("model_selection", None)

    if "callbacks" in trainer_params:
        logger.warning(f"Overwrite `trainer.callbacks`, was '{trainer_params.checkpoint_callback}'")
    trainer_params_callbacks = []

    if model_selection is not None:
        checkpoint_callback = ModelCheckpoint(monitor=model_selection["metric"], mode=model_selection["mode"])
        logger.info(f"Create ModelCheckpoint callback with monitor={model_selection['metric']}")
        trainer_params_callbacks.append(checkpoint_callback)
        del trainer_params.model_selection

    if trainer_params.get("checkpoints_every_n_val_epochs", None) is not None:
        every_n_val_epochs = trainer_params.checkpoints_every_n_val_epochs
        checkpoint_callback = ModelCheckpoint(every_n_epochs=every_n_val_epochs)
        logger.info(f"Create ModelCheckpoint callback every_n_epochs ='{every_n_val_epochs}'")
        trainer_params_callbacks.append(checkpoint_callback)
        del trainer_params.checkpoints_every_n_val_epochs

    if "logger" in conf:
        trainer_params_additional["logger"] = hydra.utils.instantiate(conf.logger)
        trainer_params_additional["logger"].log_hyperparams(as_flat_config(OmegaConf.to_container(conf, resolve=True)))

    if "profiler" in trainer_params:
        trainer_params_additional["profiler"] = hydra.utils.instantiate(trainer_params.profiler)
        del trainer_params.profiler

    if not isinstance(trainer_params.get("strategy", ""), str): # if strategy not exist or str do nothing,
        trainer_params_additional["strategy"] = hydra.utils.instantiate(trainer_params.strategy)
        del trainer_params.strategy

    lr_monitor = LearningRateMonitor(logging_interval="step")
    trainer_params_callbacks.append(lr_monitor)
    trainer_params_callbacks.append(TrainingTimeCallback())

    if len(trainer_params_callbacks) > 0:
        trainer_params_additional["callbacks"] = trainer_params_callbacks
    trainer_params = dict(trainer_params)
    trainer_params.update(trainer_params_additional)
    trainer = pl.Trainer(**trainer_params)
    trainer = patch_precision_plugin(trainer)
    return trainer
