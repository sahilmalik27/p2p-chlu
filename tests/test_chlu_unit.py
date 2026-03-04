"""Tests for the CHLUUnit module."""

import torch
import pytest

from chlu.core.chlu_unit import CHLUUnit


class TestCHLUUnit:
    @pytest.fixture
    def model(self):
        return CHLUUnit(
            input_dim=4,
            latent_dim=8,
            c=2.0,
            m0=1.0,
            alpha=0.01,
            dt=0.01,
            n_steps=5,
            hidden_dims=(32, 32),
        )

    def test_encode_shapes(self, model):
        x = torch.randn(16, 4)
        q, p = model.encode(x)
        assert q.shape == (16, 8)
        assert p.shape == (16, 8)

    def test_decode_shapes(self, model):
        q = torch.randn(16, 8)
        out = model.decode(q)
        assert out.shape == (16, 4)

    def test_forward_shapes(self, model):
        x = torch.randn(16, 4)
        out = model(x)
        assert out.shape == (16, 4)

    def test_forward_with_phase(self, model):
        x = torch.randn(16, 4)
        out, q, p = model(x, return_phase=True)
        assert out.shape == (16, 4)
        assert q.shape == (16, 8)
        assert p.shape == (16, 8)

    def test_evolve_sequence(self, model):
        x0 = torch.randn(4, 4)
        seq = model.evolve_sequence(x0, seq_len=10)
        assert seq.shape == (4, 10, 4)

    def test_set_gamma(self, model):
        model.set_gamma(0.5)
        assert model.integrator.gamma == 0.5
        model.set_gamma(0.0)
        assert model.integrator.gamma == 0.0

    def test_gradient_flow(self, model):
        """Gradients should flow through the full forward pass."""
        x = torch.randn(4, 4)
        out = model(x)
        loss = out.sum()
        loss.backward()

        # Encoder and decoder must have gradients
        assert model.encoder.weight.grad is not None
        assert model.decoder.weight.grad is not None

        # Potential hidden layers should have gradients
        # (The final bias of V_θ is a constant offset that doesn't affect ∇_q V,
        # so it correctly has no gradient.)
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.hamiltonian.potential.parameters()
        )
        assert has_grad, "No gradient in any potential parameter"

    def test_custom_n_steps(self, model):
        """Should accept override of n_steps."""
        x = torch.randn(4, 4)
        out = model(x, n_steps=2)
        assert out.shape == (4, 4)
