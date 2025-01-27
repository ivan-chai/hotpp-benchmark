#!/usr/bin/env python3
import math
from unittest import TestCase, main

import torch

from hotpp.data import PaddedBatch
from hotpp.losses import TimeMAELoss, CrossEntropyLoss
from hotpp.losses import NextItemLoss, DiffusionLoss
from hotpp.nn import Embedder, Head


class Model(torch.nn.Module):
    def __init__(self, length, dim):
        super().__init__()
        self.length = length
        self.weight = torch.nn.Parameter(torch.randn(1, length, dim))  # (B, L, D).

    def forward(self):
        return PaddedBatch(self.weight, torch.tensor([self.length]))


class Denoiser(torch.nn.Module):
    def __init__(self, input_size, length, batch_size, steps):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(batch_size, steps + 1, length, input_size))
        self.lengths = torch.full([batch_size], length, dtype=torch.long)

    @property
    def condition_size(self):
        return self.weight.shape[2]

    def forward(self, embeddings, conditions, steps):
        result = self.weight.take_along_dim(steps[:, None, None, None], 1).squeeze(1)
        return PaddedBatch(result, self.lengths)


class TestDiffusionLoss(TestCase):
    def test_corruption(self):
        steps = 10
        torch.manual_seed(0)

        losses = {
            "timestamps": TimeMAELoss(),
            "labels": CrossEntropyLoss(10)
        }

        embedder = Embedder(embeddings={"labels": {"in": 10, "out": 10}},
                            numeric_values={"timestamps": "identity"},
                            use_batch_norm=False)

        denoiser_partial = lambda dim, length: Denoiser(dim, length, batch_size=6, steps=steps)
        decoder_partial = lambda dim_in, dim_out: Head(dim_in, dim_out, hidden_dims=[32])

        loss = DiffusionLoss(NextItemLoss(losses), 4, embedder,
                             denoiser_partial, decoder_partial,
                             generation_steps=steps)

        b = 1000
        batch = PaddedBatch({
            "timestamps": torch.tensor([5.0, 5.3, 6.0, 6.1, 7.2, 8.0, 9.0, 20, 21, 21.5]).reshape(1, 10).repeat(b, 1),
            "labels":     torch.tensor([5  , 9  , 0  , 1  , 0  , 1  , 5  , 7 , 8 , 1]).reshape(1, 10).repeat(b, 1),
        }, torch.tensor([10] * b))  # (b, 10).

        embeddings = loss._embedder(loss._compute_time_deltas(batch))  # (B, L, D).
        embeddings = PaddedBatch(embeddings.payload[:, 1:], (embeddings.seq_lens - 1).clip(min=0))  # (B, L - 1, D).
        steps = torch.full([b], steps, device=embeddings.device)  # (B).
        corrupted = loss._corrupt(embeddings, steps)  # (B, L - 1, D).
        atol = 0.2
        self.assertTrue((corrupted.payload.mean(0).abs() < atol).all())
        self.assertTrue(((corrupted.payload.std(0) - 1).abs() < atol).all())

    def test_convergence(self):
        steps = 5
        torch.manual_seed(0)

        losses = {
            "timestamps": TimeMAELoss(),
            "labels": CrossEntropyLoss(10)
        }

        embedder = Embedder(embeddings={"labels": {"in": 10, "out": 10}},
                            numeric_values={"timestamps": "identity"},
                            use_batch_norm=False)

        denoiser_partial = lambda dim, length: Denoiser(dim, length, batch_size=6, steps=steps)
        decoder_partial = lambda dim_in, dim_out: Head(dim_in, dim_out, hidden_dims=[32])

        loss = DiffusionLoss(NextItemLoss(losses), 4, embedder,
                             denoiser_partial, decoder_partial,
                             generation_steps=steps)

        batch = PaddedBatch({
            "timestamps": torch.tensor([5.0, 5.3, 6.0, 6.1, 7.2, 8.0, 9.0, 20, 21, 21.5]).reshape(1, 10),
            "labels":     torch.tensor([5  , 9  , 0  , 1  , 0  , 1  , 5  , 7 , 8 , 1]).reshape(1, 10),
        }, torch.tensor([10]))

        n_valid = 10 - 4
        model = Model(n_valid, loss.input_size)

        optimizer = torch.optim.Adam(list(model.parameters()) + list(loss.parameters()), lr=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 3, 0.5)
        model.train()
        loss.train()
        for step in range(3000):
            prediction = model()
            optimizer.zero_grad()
            losses, _ = loss(batch, prediction, None)
            loss_value = sum(losses.values())
            if step % 300 == 0:
                print(f"Loss: {loss_value.item():.4f}")
                scheduler.step()
            loss_value.backward()
            optimizer.step()
        print(f"Final loss: {loss_value.item():.4f}")
        print()

        model.eval()
        loss.eval()
        with torch.no_grad():
            outputs = model()
            predictions = loss.predict_next_k(outputs, None).payload  # (B, L, K)
            predictions["timestamps"] += batch.payload["timestamps"][:, :n_valid].unsqueeze(2)

        for i in range(n_valid):
            gt_times = batch.payload["timestamps"][0, i + 1:i + 1 + 4]
            gt_labels = batch.payload["labels"][0, i + 1:i + 1 + 4]
            self.assertEqual(predictions["labels"][0, i].tolist(), gt_labels.tolist())
            self.assertTrue(predictions["timestamps"][0, i].allclose(gt_times, atol=0.2))


if __name__ == "__main__":
    main()
