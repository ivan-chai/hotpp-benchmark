import math
from unittest import TestCase, main

import torch

from hotpp.data import PaddedBatch
from hotpp.nn.encoder.rnn import ContTimeLSTM


EPS = 1e-10


def sigm(x):
    return 1 / (1 + math.exp(-x))

def tanh(x):
    return math.tanh(x)

def softp(x):
    return math.log(1 + math.exp(x))


class TestContTimeLSTM(TestCase):
    def test_simple_parameters(self):
        rnn = ContTimeLSTM(1, 1)
        rnn.start.data.fill_(0)
        rnn.layer.weight.data.fill_(1)
        rnn.layer.bias.data.fill_(0.5)
        x = torch.tensor([
            1, -1
        ]).reshape(1, -1, 1)  # (B, L, D).
        dt = torch.tensor([
            2, 3
        ]).reshape(1, -1)  # (B, L).
        # Init.
        i__, o__ = [sigm(0.5)] * 2
        d__ = softp(0.5)
        z__ = tanh(0.5)
        cs__, ce__ = [i__ * z__] * 2

        # First step.
        c_0 = cs__ + (ce__ - cs__) * math.exp(-d__ * 2)
        h_0 = o__ * tanh(c_0)
        i_0, f_0, o_0 = [sigm(1.5 + h_0)] * 3
        d_0 = softp(1.5 + h_0)
        z_0 = tanh(1.5 + h_0)
        cs_0 = f_0 * c_0 + i_0 * z_0
        ce_0 = f_0 * ce__ + i_0 * z_0

        # Second step
        c_1 = cs_0 + (ce_0 - cs_0) * math.exp(-d_0 * 3)
        h_1 = o_0 * tanh(c_1)
        i_1, f_1, o_1 = [sigm(-1 + 0.5 + h_1)] * 3
        d_1 = softp(-1 + 0.5 + h_1)
        z_1 = tanh(-1 + 0.5 + h_1)
        cs_1 = f_1 * c_1 + i_1 * z_1
        ce_1 = f_1 * ce_0 + i_1 * z_1
        outputs_gt = torch.tensor([
            h_0, h_1
        ]).reshape(1, -1, 1)
        output_states_gt = torch.tensor([
            [o_0, cs_0, ce_0, d_0],
            [o_1, cs_1, ce_1, d_1],
        ]).reshape(1, 1, -1, 4)
        lengths = torch.full([x.shape[0]], x.shape[1], dtype=torch.long)
        outputs, output_states = rnn(PaddedBatch(x, lengths), PaddedBatch(dt, lengths), return_full_states=True)
        self.assertTrue(outputs.payload.allclose(outputs_gt))
        self.assertTrue(output_states.allclose(output_states_gt))

    def test_gradients(self):
        rnn = ContTimeLSTM(3, 5)
        x = torch.randn(2, 4, 3, requires_grad=True)
        dt = torch.rand(2, 4, requires_grad=True)
        lengths = torch.full([x.shape[0]], x.shape[1], dtype=torch.long)
        outputs, output_states = rnn(PaddedBatch(x, lengths), PaddedBatch(dt, lengths), return_full_states=True)
        outputs.payload.mean().backward()
        self.assertEqual(outputs.payload.shape, (2, 4, 5))
        self.assertTrue((x.grad[:, :-1].abs() > EPS).all())
        # The last output doesn't depend on the last input, as it's computed as interpolation.
        self.assertTrue((x.grad[:, -1].abs() < EPS).all())
        self.assertTrue((dt.grad.abs() > EPS).all())
        self.assertTrue((rnn.start.grad.abs() > EPS).all())

        x.grad = None
        dt.grad = None
        outputs, output_states = rnn(PaddedBatch(x, lengths), PaddedBatch(dt, lengths), return_full_states=True)
        output_states.mean().backward()
        self.assertTrue((x.grad.abs() > EPS).all())
        self.assertTrue((dt.grad.abs() > EPS).all())
        self.assertTrue((rnn.start.grad.abs() > EPS).all())

    def test_interpolation(self):
        rnn = ContTimeLSTM(3, 5)
        x = torch.randn(2, 4, 3, requires_grad=True)
        dt = torch.rand(2, 4, requires_grad=True)
        lengths = torch.full([x.shape[0]], x.shape[1], dtype=torch.long)
        outputs, output_states = rnn(PaddedBatch(x, lengths), PaddedBatch(dt, lengths), return_full_states=True)
        int_outputs = rnn.interpolate(output_states[:, :, :-1], PaddedBatch(dt[:, 1:, None], lengths - 1)).payload.squeeze(2)  # (B, L - 1, D).
        self.assertTrue(outputs.payload[:, 1:].allclose(int_outputs))


if __name__ == "__main__":
    main()
