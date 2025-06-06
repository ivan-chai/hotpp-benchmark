import math
from unittest import TestCase, main

import torch

from hotpp.data import PaddedBatch
from hotpp.nn.encoder import AttNHPTransformer
from hotpp.nn.encoder.transformer.state import TransformerState


class TestAttNHPTransformer(TestCase):
    def test_output_is_equal_to_decode(self):
        b, l = 8, 10
        input_size = 16
        hidden_size = 32
        n_heads = 4
        n_layers = 3
        for kwargs in [{}, {"dim_feedforward": 0}, {"dim_value": hidden_size // 2}]:
            model = AttNHPTransformer(input_size, hidden_size, n_heads, n_layers, **kwargs).eval()

            lengths = torch.cat([
                torch.tensor([0, 1, l], dtype=torch.long),
                torch.randint(0, l, (b - 3,))
            ])  # (B).
            embeddings = PaddedBatch(torch.randn(b, l, input_size), lengths)  # (B, L, D).
            times = PaddedBatch(torch.rand(b, l), lengths)  # (B, L).

            fw_outputs, fw_states = model(embeddings, times, return_states="full")  # (B, L, D), (N, B, L, D).
            self.assertEqual(fw_outputs.payload.shape, (b, l, hidden_size))
            self.assertEqual(fw_states.payload.shape, (n_layers, b, l, hidden_size))

            history_states = TransformerState(fw_states.times[:, :-1], fw_states.payload[:, :, :-1],
                                            (fw_states.seq_lens - 1).clip(min=0))
            last_indices = (lengths - 1).clip(min=0)
            last_mask = lengths > 0  # Need at least one history event.
            length1 = torch.ones_like(embeddings.seq_lens)
            last_embeddings = embeddings.payload.take_along_dim(last_indices[:, None, None], 1)  # (B, 1, D).
            last_embeddings = PaddedBatch(last_embeddings, length1)
            last_times = times.payload.take_along_dim(last_indices[:, None], 1)  # (B, 1).
            last_times = PaddedBatch(last_times, length1)
            dec_outputs, dec_states = model.decode(last_embeddings, last_times, history_states)  # (B, 1, D), (N, B, 1, D).
            gt_outputs = fw_outputs.payload.take_along_dim(last_indices[:, None, None], 1)  # (B, 1, D).
            gt_states = fw_states.payload.take_along_dim(last_indices[None, :, None, None], 2)  # (N, B, 1, D).
            self.assertTrue(dec_outputs.payload[last_mask].allclose(gt_outputs[last_mask], rtol=1e-3, atol=1e-6))
            self.assertTrue(dec_states.payload[:, last_mask].allclose(gt_states[:, last_mask], rtol=1e-3, atol=1e-6))

    def test_interpolate(self):
        b, l, s = 8, 10, 3
        input_size = 16
        hidden_size = 16
        n_heads = 4
        n_layers = 3
        for kwargs in [{}, {"dim_feedforward": 0}, {"dim_value": hidden_size // 2}]:
            model = AttNHPTransformer(input_size, hidden_size, n_heads, n_layers, **kwargs).eval()
            model.proj = torch.nn.Identity()
            token = model.inter_token

            lengths = torch.cat([
                torch.tensor([0, 1, l], dtype=torch.long),
                torch.randint(0, l, (b - 3,))
            ])  # (B).
            embeddings = PaddedBatch(torch.randn(b, l, input_size), lengths)  # (B, L, D).
            # Replace last embedding with interpolation token.
            for i, j in enumerate(lengths.numpy()):
                embeddings.payload[i, j - 1] = token
            times = PaddedBatch(torch.rand(b, l), lengths)  # (B, L).

            fw_outputs, fw_states = model(embeddings, times, return_states="full")  # (B, L, D), (N, B, L, D).
            self.assertEqual(fw_outputs.payload.shape, (b, l, hidden_size))
            self.assertEqual(fw_states.payload.shape, (n_layers, b, l, hidden_size))

            history_states = TransformerState(fw_states.times[:, :-1], fw_states.payload[:, :, :-1],
                                            (fw_states.seq_lens - 1).clip(min=0))
            last_indices = (lengths - 1).clip(min=0)
            last_mask = lengths > 0  # Need at least one history event.
            length1 = torch.ones_like(embeddings.seq_lens)
            last_times = times.payload.take_along_dim(last_indices[:, None], 1)[:, :, None].repeat(1, 1, s)  # (B, 1, S).
            last_times = PaddedBatch(last_times, length1)
            int_outputs = model.interpolate(last_times, history_states)  # (B, 1, S, D).
            gt_outputs = fw_outputs.payload.take_along_dim(last_indices[:, None, None], 1)[:, :, None, :].repeat(1, 1, s, 1)  # (B, 1, S, D).
            self.assertTrue(int_outputs.payload[last_mask].allclose(gt_outputs[last_mask], rtol=1e-3, atol=1e-6))

    def test_padding(self):
        b, l = 8, 10
        input_size = 16
        hidden_size = 32
        n_heads = 4
        n_layers = 3
        for kwargs in [{}, {"dim_feedforward": 0}, {"dim_value": hidden_size // 2}]:
            model = AttNHPTransformer(input_size, hidden_size, n_heads, n_layers, **kwargs).eval()

            lengths = torch.cat([
                torch.tensor([0, 1, l], dtype=torch.long),
                torch.randint(0, l, (b - 3,))
            ])  # (B).
            embeddings = PaddedBatch(torch.randn(b, l, input_size), lengths)  # (B, L, D).
            times = PaddedBatch(torch.rand(b, l), lengths)  # (B, L).
            mask = embeddings.seq_len_mask.bool()

            fw1_outputs, fw1_states = model(embeddings, times, return_states="full")  # (B, L, D), (N, B, L, D).

            embeddings.payload += ~mask.unsqueeze(-1)  # (B, L, D).
            fw2_outputs, fw2_states = model(embeddings, times, return_states="full")  # (B, L, D), (N, B, L, D).

            self.assertTrue(fw1_outputs.payload[mask].allclose(fw2_outputs.payload[mask], rtol=1e-3, atol=1e-6))
            self.assertTrue(fw1_states.payload[:, mask].allclose(fw2_states.payload[:, mask], rtol=1e-3, atol=1e-6))


if __name__ == "__main__":
    main()
