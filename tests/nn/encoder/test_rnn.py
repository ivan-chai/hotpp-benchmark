import math
from unittest import TestCase, main

import torch

from hotpp.data import PaddedBatch
from hotpp.nn.encoder.rnn import ContTimeLSTM, ODEGRU


EPS = 1e-10


def exp(x):
    return math.exp(x)


def sigm(x):
    return 1 / (1 + math.exp(-x))

def tanh(x):
    return math.tanh(x)

def softp(x):
    return math.log(1 + math.exp(x))


def lin_rk4(x0, a, b, dt):
    k1 = a * x0 + b
    k2 = a * (x0 + dt * k1 / 2) + b
    k3 = a * (x0 + dt * k2 / 2) + b
    k4 = a * (x0 + dt * k3) + b
    return x0 + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6


class TestContTimeLSTM(TestCase):
    def test_simple_parameters(self):
        rnn = ContTimeLSTM(1, 1)
        rnn.load_state_dict({
            "bos": torch.full([1], 0.0),
            "cell.projection.weight": torch.full([7, 1], 1.0),
            "cell.weight": torch.full([1, 7], 1.0),
            "cell.bias": torch.full([7], 0.5),
        })
        x = torch.tensor([
            1, -1
        ]).reshape(1, -1, 1).float()  # (B, L, D).
        dt = torch.tensor([
            2, 3
        ]).reshape(1, -1)  # (B, L).
        # Init.
        i__, o__ = [sigm(0.5)] * 2
        d__ = softp(0.5)
        z__ = tanh(0.5)
        cs__, ce__ = [i__ * z__] * 2

        # First step.
        c_0 = ce__ + (cs__ - ce__) * math.exp(-d__ * 2)
        h_0 = o__ * tanh(c_0)
        i_0, f_0, o_0 = [sigm(1.5 + h_0)] * 3
        d_0 = softp(1.5 + h_0)
        z_0 = tanh(1.5 + h_0)
        cs_0 = f_0 * c_0 + i_0 * z_0
        ce_0 = f_0 * ce__ + i_0 * z_0

        # Second step
        c_1 = ce_0 + (cs_0 - ce_0) * math.exp(-d_0 * 3)
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
            [cs_0, ce_0, d_0, o_0],
            [cs_1, ce_1, d_1, o_1],
        ]).reshape(1, 1, -1, 4)
        lengths = torch.full([x.shape[0]], x.shape[1], dtype=torch.long)
        outputs, output_states = rnn(PaddedBatch(x, lengths), PaddedBatch(dt, lengths), return_states="full")
        self.assertTrue(outputs.payload.allclose(outputs_gt))
        self.assertTrue(output_states.allclose(output_states_gt))

    def test_gradients(self):
        rnn = ContTimeLSTM(3, 5)
        x = torch.randn(2, 4, 3, requires_grad=True)
        dt = torch.rand(2, 4, requires_grad=True)
        lengths = torch.full([x.shape[0]], x.shape[1], dtype=torch.long)
        outputs, output_states = rnn(PaddedBatch(x, lengths), PaddedBatch(dt, lengths), return_states="full")
        outputs.payload.mean().backward()
        self.assertEqual(outputs.payload.shape, (2, 4, 5))
        self.assertTrue((x.grad[:, :-1].abs() > EPS).all())
        # The last output doesn't depend on the last input, as it's computed as interpolation.
        self.assertTrue((x.grad[:, -1].abs() < EPS).all())
        self.assertTrue((dt.grad.abs() > EPS).all())
        self.assertTrue((rnn.bos.grad.abs() > EPS).all())

        x.grad = None
        dt.grad = None
        outputs, output_states = rnn(PaddedBatch(x, lengths), PaddedBatch(dt, lengths), return_states="full")
        output_states.mean().backward()
        self.assertTrue((x.grad.abs() > EPS).all())
        self.assertTrue((dt.grad.abs() > EPS).all())
        self.assertTrue((rnn.bos.grad.abs() > EPS).all())

    def test_interpolation(self):
        rnn = ContTimeLSTM(3, 5)
        x = torch.randn(2, 4, 3, requires_grad=True)
        dt = torch.rand(2, 4, requires_grad=True)
        lengths = torch.full([x.shape[0]], x.shape[1], dtype=torch.long)
        outputs, output_states = rnn(PaddedBatch(x, lengths), PaddedBatch(dt, lengths), return_states="full")
        int_outputs = rnn.interpolate(output_states[:, :, :-1], PaddedBatch(dt[:, 1:, None], lengths - 1)).payload.squeeze(2)  # (B, L - 1, D).
        self.assertTrue(outputs.payload[:, 1:].allclose(int_outputs))


class TestODEGRU(TestCase):
    def test_simple_parameters(self):
        rnn = ODEGRU(1, 1, num_diff_layers=1, n_steps=1, lipschitz=None)
        state_dict = {
            "h0": torch.full([1, 1], 0.0),
            "cell.cell.weight_ih": torch.full([3, 1], 1.0),
            "cell.cell.weight_hh": torch.full([3, 1], 1.0),
            "cell.cell.bias_ih": torch.full([3], 0.5),
            "cell.cell.bias_hh": torch.full([3], 0.5),
            "cell.diff_func.nn.0.weight": torch.full([1, 1], 2.0),
            "cell.diff_func.nn.0.bias": torch.full([1], 1.0)
        }
        rnn.load_state_dict(state_dict)
        x = torch.tensor([
            1, -1
        ]).reshape(1, -1, 1).float()  # (B, L, D).
        dt = torch.tensor([
            2, 3
        ]).reshape(1, -1)  # (B, L).
        # Init.
        h__ = 0

        # First step.
        hode_0 = lin_rk4(h__, 2, 1, 2)
        z_0 = sigm(hode_0 + 0.5 + 1 + 0.5)
        r_0 = sigm(hode_0 + 0.5 + 1 + 0.5)
        hbar_0 = tanh(r_0 * (hode_0 + 0.5) + 1 + 0.5)
        h_0 = (1 - z_0) * hbar_0 + z_0 * hode_0

        # Second step
        hode_1 = lin_rk4(h_0, 2, 1, 3)
        z_1 = sigm(hode_1 + 0.5 - 1 + 0.5)
        r_1 = sigm(hode_1 + 0.5 - 1 + 0.5)
        hbar_1 = tanh(r_1 * (hode_1 + 0.5) - 1 + 0.5)
        h_1 = (1 - z_1) * hbar_1 + z_1 * hode_1

        outputs_gt = torch.tensor([
            h_0, h_1
        ]).reshape(1, -1, 1)
        output_states_gt = torch.tensor([
            h_0, h_1
        ]).reshape(1, 1, -1, 1)
        lengths = torch.full([x.shape[0]], x.shape[1], dtype=torch.long)
        outputs, output_states = rnn(PaddedBatch(x, lengths), PaddedBatch(dt, lengths), return_states="full")
        self.assertTrue(outputs.payload.allclose(outputs_gt))
        self.assertTrue(output_states.allclose(output_states_gt))

    def test_gradients(self):
        rnn = ODEGRU(3, 5)
        torch.nn.init.normal_(rnn.h0)
        x = torch.randn(2, 4, 3, requires_grad=True)
        dt = torch.rand(2, 4, requires_grad=True)
        lengths = torch.full([x.shape[0]], x.shape[1], dtype=torch.long)
        outputs, output_states = rnn(PaddedBatch(x, lengths), PaddedBatch(dt, lengths), return_states="full")
        outputs.payload.mean().backward()
        self.assertEqual(outputs.payload.shape, (2, 4, 5))
        self.assertTrue((x.grad.abs() > EPS).all())
        self.assertTrue((dt.grad.abs() > EPS).all())
        self.assertTrue((rnn.h0.grad.abs() > EPS).all())

        x.grad = None
        dt.grad = None
        outputs, output_states = rnn(PaddedBatch(x, lengths), PaddedBatch(dt, lengths), return_states="full")
        output_states.mean().backward()
        self.assertTrue((x.grad.abs() > EPS).all())
        self.assertTrue((dt.grad.abs() > EPS).all())
        self.assertTrue((rnn.h0.grad.abs() > EPS).all())

    def test_interpolation(self):
        rnn = ODEGRU(3, 5)
        x = torch.randn(2, 4, 3, requires_grad=True)
        dt = torch.rand(2, 4, requires_grad=True)
        lengths = torch.full([x.shape[0]], x.shape[1], dtype=torch.long)
        outputs, output_states = rnn(PaddedBatch(x, lengths), PaddedBatch(dt, lengths), return_states="full")
        int_outputs = rnn.interpolate(output_states[:, :, :-1], PaddedBatch(dt[:, 1:, None], lengths - 1)).payload.squeeze(2)  # (B, L - 1, D).
        int_outputs = rnn.cell.cell(x[:, 1:].flatten(0, 1), int_outputs.flatten(0, 1)).reshape(*int_outputs.shape)
        self.assertTrue(outputs.payload[:, 1:].allclose(int_outputs))


if __name__ == "__main__":
    main()
