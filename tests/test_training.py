"""Tests for training components."""

import torch
import pytest

from chlu.core.chlu_unit import CHLUUnit
from chlu.core.hamiltonian import RelativisticHamiltonian
from chlu.training.losses import mse_loss, lyapunov_loss, chlu_loss
from chlu.training.replay_buffer import ReplayBuffer
from chlu.training.contrastive import HCDConfig, HCDTrainer


class TestLosses:
    def test_mse_loss(self):
        pred = torch.randn(8, 4)
        target = torch.randn(8, 4)
        loss = mse_loss(pred, target)
        assert loss.shape == ()
        assert loss.item() > 0

    def test_mse_loss_zero(self):
        x = torch.randn(8, 4)
        loss = mse_loss(x, x)
        assert loss.item() < 1e-6

    def test_lyapunov_loss(self):
        H = RelativisticHamiltonian(dim=4, hidden_dims=(16,))
        q_traj = torch.randn(10, 8, 4)
        p_traj = torch.randn(10, 8, 4)
        loss = lyapunov_loss(H, q_traj, p_traj)
        assert loss.shape == ()
        assert loss.item() >= 0

    def test_chlu_loss(self):
        H = RelativisticHamiltonian(dim=4, hidden_dims=(16,))
        pred = torch.randn(8, 4)
        target = torch.randn(8, 4)
        q_traj = torch.randn(10, 8, 4)
        p_traj = torch.randn(10, 8, 4)

        total, metrics = chlu_loss(pred, target, H, q_traj, p_traj)
        assert total.shape == ()
        assert "loss" in metrics
        assert "mse" in metrics
        assert "lyapunov" in metrics


class TestReplayBuffer:
    def test_push_and_size(self):
        buf = ReplayBuffer(capacity=100, dim=4)
        assert len(buf) == 0

        q = torch.randn(10, 4)
        p = torch.randn(10, 4)
        buf.push(q, p)
        assert len(buf) == 10

    def test_sample(self):
        buf = ReplayBuffer(capacity=100, dim=4)
        buf.push(torch.randn(50, 4), torch.randn(50, 4))

        q, p = buf.sample(16)
        assert q.shape == (16, 4)
        assert p.shape == (16, 4)

    def test_sample_with_device(self):
        buf = ReplayBuffer(capacity=100, dim=4)
        buf.push(torch.randn(50, 4), torch.randn(50, 4))

        q, p = buf.sample(8, device=torch.device("cpu"))
        assert q.device.type == "cpu"

    def test_circular_overflow(self):
        buf = ReplayBuffer(capacity=10, dim=2)
        # Push more than capacity
        buf.push(torch.randn(15, 2), torch.randn(15, 2))
        assert len(buf) == 10  # Capped at capacity

    def test_sample_too_large(self):
        buf = ReplayBuffer(capacity=10, dim=2)
        buf.push(torch.randn(5, 2), torch.randn(5, 2))
        with pytest.raises(ValueError):
            buf.sample(10)


class TestHCDTrainer:
    def test_train_step(self):
        model = CHLUUnit(
            input_dim=2, latent_dim=4, dt=0.01, n_steps=3, hidden_dims=(16,),
        )
        config = HCDConfig(lr=1e-3, epochs=1, batch_size=8)
        trainer = HCDTrainer(model, config)

        x = torch.randn(8, 2)
        target = torch.randn(8, 2)
        metrics = trainer.train_step(x, target)

        assert "total_loss" in metrics
        assert "mse" in metrics

    def test_warmup_fills_buffer(self):
        model = CHLUUnit(
            input_dim=2, latent_dim=4, dt=0.01, n_steps=3, hidden_dims=(16,),
        )
        config = HCDConfig(lr=1e-3, epochs=1, batch_size=8)
        trainer = HCDTrainer(model, config)

        # Before training, buffer is empty
        assert len(trainer.buffer) == 0

        x = torch.randn(8, 2)
        target = torch.randn(8, 2)
        trainer.train_step(x, target)

        # After one step, buffer should have states
        assert len(trainer.buffer) > 0
