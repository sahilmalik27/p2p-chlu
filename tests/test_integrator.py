"""Tests for the Velocity Verlet integrator."""

import torch
import pytest

from chlu.core.hamiltonian import RelativisticHamiltonian
from chlu.core.integrator import VelocityVerletIntegrator


class TestVelocityVerletIntegrator:
    @pytest.fixture
    def setup(self):
        H = RelativisticHamiltonian(dim=2, c=2.0, m0=1.0, alpha=0.1, hidden_dims=(32,))
        integrator = VelocityVerletIntegrator(H, dt=0.01, gamma=0.0)
        return H, integrator

    def test_single_step_shapes(self, setup):
        H, integrator = setup
        q = torch.randn(4, 2)
        p = torch.randn(4, 2)
        q_new, p_new = integrator.step(q, p)
        assert q_new.shape == (4, 2)
        assert p_new.shape == (4, 2)

    def test_integrate_shapes(self, setup):
        _, integrator = setup
        q = torch.randn(4, 2)
        p = torch.randn(4, 2)
        q_f, p_f = integrator.integrate(q, p, n_steps=10)
        assert q_f.shape == (4, 2)
        assert p_f.shape == (4, 2)

    def test_trajectory_shapes(self, setup):
        _, integrator = setup
        q = torch.randn(4, 2)
        p = torch.randn(4, 2)
        q_f, p_f, q_traj, p_traj = integrator.integrate(
            q, p, n_steps=10, return_trajectory=True,
        )
        assert q_traj.shape == (11, 4, 2)  # n_steps+1
        assert p_traj.shape == (11, 4, 2)
        # Final state should match last trajectory entry
        assert torch.allclose(q_f, q_traj[-1])
        assert torch.allclose(p_f, p_traj[-1])

    def test_energy_conservation(self, setup):
        """With γ=0 (conservative), energy should be approximately conserved."""
        H, integrator = setup
        q = torch.randn(1, 2) * 0.1
        p = torch.randn(1, 2) * 0.1

        E_init = H(q, p).item()
        q_f, p_f = integrator.integrate(q, p, n_steps=100)
        E_final = H(q_f, p_f).item()

        # Energy drift should be small for symplectic integrator
        relative_drift = abs(E_final - E_init) / (abs(E_init) + 1e-10)
        assert relative_drift < 0.1, f"Energy drift: {relative_drift:.4f}"

    def test_dissipation_reduces_momentum(self):
        """With γ>0, momentum magnitude should decrease."""
        H = RelativisticHamiltonian(dim=2, c=2.0, m0=1.0, alpha=0.1, hidden_dims=(32,))
        integrator = VelocityVerletIntegrator(H, dt=0.01, gamma=0.5)

        q = torch.zeros(1, 2)
        p = torch.randn(1, 2)
        p_norm_init = p.norm().item()

        q_f, p_f = integrator.integrate(q, p, n_steps=50)
        p_norm_final = p_f.norm().item()

        assert p_norm_final < p_norm_init, "Dissipation should reduce momentum"
