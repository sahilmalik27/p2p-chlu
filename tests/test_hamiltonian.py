"""Tests for the relativistic Hamiltonian."""

import torch
import pytest

from chlu.core.hamiltonian import RelativisticHamiltonian, PotentialMLP


class TestPotentialMLP:
    def test_output_shape(self):
        net = PotentialMLP(dim=4, hidden_dims=(32, 32))
        q = torch.randn(8, 4)
        out = net(q)
        assert out.shape == (8,)

    def test_single_sample(self):
        net = PotentialMLP(dim=2)
        q = torch.randn(2)
        out = net(q)
        assert out.shape == ()


class TestRelativisticHamiltonian:
    @pytest.fixture
    def hamiltonian(self):
        return RelativisticHamiltonian(dim=4, c=2.0, m0=1.0, alpha=0.01)

    def test_mass_positive(self, hamiltonian):
        """Mass inverse should always be positive (exp parameterization)."""
        assert (hamiltonian.mass_inv > 0).all()

    def test_kinetic_energy_shape(self, hamiltonian):
        p = torch.randn(8, 4)
        T = hamiltonian.kinetic_energy(p)
        assert T.shape == (8,)

    def test_kinetic_energy_positive(self, hamiltonian):
        """Relativistic KE is always positive (includes rest mass)."""
        p = torch.randn(8, 4)
        T = hamiltonian.kinetic_energy(p)
        assert (T > 0).all()

    def test_kinetic_energy_at_rest(self, hamiltonian):
        """At p=0, T = m₀c² (rest energy)."""
        p = torch.zeros(1, 4)
        T = hamiltonian.kinetic_energy(p)
        expected = hamiltonian.m0 * hamiltonian.c**2
        assert torch.allclose(T, torch.tensor([expected]), atol=1e-5)

    def test_hamiltonian_shape(self, hamiltonian):
        q = torch.randn(8, 4)
        p = torch.randn(8, 4)
        H = hamiltonian(q, p)
        assert H.shape == (8,)

    def test_velocity_bounded(self, hamiltonian):
        """Relativistic velocity should be bounded by c."""
        # Use very large momenta
        p = torch.randn(100, 4) * 1000
        v = hamiltonian.dT_dp(p)
        speed = v.norm(dim=-1)
        # Speed should not exceed c (with some numerical tolerance)
        assert (speed < hamiltonian.c + 0.1).all(), f"Max speed: {speed.max():.4f}, c={hamiltonian.c}"

    def test_velocity_at_zero(self, hamiltonian):
        """At p=0, velocity should be 0."""
        p = torch.zeros(1, 4)
        v = hamiltonian.dT_dp(p)
        assert torch.allclose(v, torch.zeros_like(v), atol=1e-6)

    def test_dV_dq_shape(self, hamiltonian):
        q = torch.randn(8, 4)
        grad = hamiltonian.dV_dq(q)
        assert grad.shape == (8, 4)

    def test_confinement_gradient(self):
        """Confinement gradient should be 2αq."""
        H = RelativisticHamiltonian(dim=2, alpha=0.5, hidden_dims=(8,))
        # Zero out potential to isolate confinement
        with torch.no_grad():
            for p in H.potential.parameters():
                p.zero_()
        q = torch.tensor([[1.0, 0.0]])
        grad = H.dV_dq(q)
        # ∇(α‖q‖²) = 2αq = 2*0.5*[1,0] = [1,0]
        expected = torch.tensor([[1.0, 0.0]])
        assert torch.allclose(grad, expected, atol=0.1)  # potential not exactly 0
