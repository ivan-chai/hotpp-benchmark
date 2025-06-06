import atexit
import queue
import multiprocessing as mp
import os
import pytest
import tempfile
from functools import partial
from unittest import TestCase, main

import pandas as pd
import pytorch_lightning as pl
import torch

from hotpp.data import PaddedBatch, HotppDataModule
from hotpp.metrics import HorizonMetric
from hotpp.nn import RnnEncoder, Embedder, GRU
from hotpp.losses import NextItemLoss, TimeMAELoss, CrossEntropyLoss
from hotpp.modules import NextItemModule


def make_simple_dataset(path):
    timestamps = torch.stack([
        torch.arange(8),
        torch.arange(8) * 2 + 4,
        torch.arange(8) * 3 + 8,
        torch.arange(8) * 4 + 16,
    ]).tolist()  # (4, 8).
    labels = [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1],
        [0, 1, 0, 1, 0, 1, 0, 1],
        [0, 1, 0, 0, 1, 1, 0, 0],
    ]
    ids = list(range(4))
    dataset = pd.DataFrame({
        "id": ids,
        "timestamps": timestamps,
        "labels": labels
    })
    dataset.to_parquet(path)


def train_worker(output_queue, root, data_path, model, batch_size, num_workers, num_devices):
    pl.seed_everything(0)
    data = HotppDataModule(batch_size=batch_size, num_workers=num_workers, train_path=data_path, val_path=data_path)
    trainer = pl.Trainer(
        accelerator="cpu",
        strategy="ddp_spawn",
        devices=num_devices,
        max_epochs=10,
        check_val_every_n_epoch=5,
        num_sanity_val_steps=0,
        sync_batchnorm=True,
        logger=pl.loggers.CSVLogger(root)
    )
    trainer.fit(model, data)
    metrics = dict(trainer.callback_metrics)
    metrics |= trainer.validate(model, data)[0]
    metrics = {k: (v.item() if isinstance(v, torch.Tensor) else v) for k, v in metrics.items()}
    output_queue.put(metrics)


class TestDistributed(TestCase):
    @classmethod
    def setUpClass(self):
        torch.cuda.is_available = lambda: False  # A workaround to fix lightning initialization in parallel testing.
        print("CUDA IS HERE", torch.cuda.is_available())
        self.root = tempfile.TemporaryDirectory()
        print(f"Use temp folder {self.root.name}")
        self.data_path = os.path.join(self.root.name, "data.parquet")
        make_simple_dataset(self.data_path)

    @classmethod
    def tearDownClass(self):
        print(f"Clean temp folder")
        self.root.cleanup()

    def test_next_item(self):
        def make_model():
            pl.seed_everything(0)
            loss = NextItemLoss(losses={"timestamps": TimeMAELoss(), "labels": CrossEntropyLoss(2)})
            return NextItemModule(
                seq_encoder=RnnEncoder(
                    embedder=Embedder(embeddings={"labels": {"in": 2, "out": 7}}, numeric_values={"timestamps": "identity"},
                                      use_batch_norm=False),
                    rnn_partial=partial(GRU, hidden_size=loss.input_size)
                ),
                loss=loss,
                optimizer_partial=partial(torch.optim.Adam, lr=0.01),
                val_metric=HorizonMetric(horizon=4,
                                         otd_steps=3, otd_insert_cost=1, otd_delete_cost=1,
                                         map_deltas=[1], map_target_length=4),
                autoreg_max_steps=5
            )

        # Train on 1 device.
        model = make_model()
        metrics1 = self._train(model,
                               batch_size=4,
                               num_workers=0,
                               num_devices=1)

        # Train on 2 devices.
        model = make_model()
        metrics2 = self._train(model,
                               batch_size=2,
                               num_workers=0,
                               num_devices=2)
        self._cmp_metrics(metrics1, metrics2)

    def _train(self, model, batch_size, num_workers, num_devices):
        output_queue = mp.Queue()
        process = mp.Process(target=train_worker, args=(output_queue, self.root.name, self.data_path,
                                                        model, batch_size, num_workers, num_devices))
        process.start()
        atexit.register(lambda: process.kill() if process.is_alive() else None)
        while True:
            try:
                if not process.is_alive():
                    raise RuntimeError("The worker is dead")
                metrics = output_queue.get(timeout=0.1)
                break
            except queue.Empty:
                continue
        process.join()
        output_queue.close()
        return metrics

    def _cmp_metrics(self, metrics1, metrics2):
        self.assertEqual(set(metrics1), set(metrics2))
        all_equal = True
        for k, v1 in metrics1.items():
            v2 = metrics2[k]
            if abs(v1 - v2) < 1e-6:
                print(f"Metrics {k} are equal.")
            else:
                all_equal = False
                print(f"Metric {k} differs: {v1} != {v2}")
        self.assertTrue(all_equal)


if __name__ == "__main__":
    main()
