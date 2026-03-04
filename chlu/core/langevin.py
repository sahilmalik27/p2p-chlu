"""Langevin dynamics for generative sampling.

Implements the stochastic dynamics:
    dp = −∇_q V(q) dt − γ p dt + √(2γkT) dW

Used for thermodynamic generation (Exp C: MNIST).
"""

from __future__ import annotations

import torch
from torch import Tensor

from chlu.core.hamiltonian import RelativisticHamiltonian


class LangevinSampler:
    """Langevin dynamics sampler for generative inference.

    Evolves phase-space state (q, p) under stochastic dynamics with
    temperature-controlled noise injection and friction.

    Args:
        hamiltonian: The Hamiltonian providing the force field.
        dt: Integration step size.
        gamma: Friction coefficient (must be > 0 for Langevin).
        temperature: Initial temperature kT.
    """

    def __init__(
        self,
        hamiltonian: RelativisticHamiltonian,
        dt: float = 0.01,
        gamma: float = 0.1,
        temperature: float = 1.0,
    ) -> None:
        self.hamiltonian = hamiltonian
        self.dt = dt
        self.gamma = gamma
        self.temperature = temperature

    def step(self, q: Tensor, p: Tensor) -> tuple[Tensor, Tensor]:
        """One Langevin step with Euler-Maruyama integration.

        dp = −∇V dt − γp dt + √(2γkT dt) ξ
        dq = ∇_p T dt

        Args:
            q: Positions, shape (..., dim).
            p: Momenta, shape (..., dim).

        Returns:
            (q_new, p_new).
        """
        H = self.hamiltonian
        dt = self.dt

        # Force from potential + confinement
        grad_V = H.dV_dq(q)

        # Stochastic noise
        noise_scale = (2.0 * self.gamma * self.temperature * dt) ** 0.5
        noise = torch.randn_like(p) * noise_scale

        # Momentum update: dp = -∇V dt - γp dt + noise
        p_new = p - grad_V * dt - self.gamma * p * dt + noise

        # Position update using relativistic velocity
        velocity = H.dT_dp(p_new)
        q_new = q + velocity * dt

        return q_new, p_new

    def sample(
        self,
        q: Tensor,
        p: Tensor,
        n_steps: int,
        temperature_schedule: Tensor | None = None,
        return_trajectory: bool = False,
    ) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor, Tensor]:
        """Run Langevin dynamics for n_steps.

        Args:
            q: Initial positions, shape (batch, dim).
            p: Initial momenta, shape (batch, dim).
            n_steps: Number of Langevin steps.
            temperature_schedule: Optional tensor of shape (n_steps,) specifying
                temperature at each step. If None, uses self.temperature.
            return_trajectory: If True, return full trajectory.

        Returns:
            If return_trajectory is False: (q_final, p_final).
            If True: (q_final, p_final, q_traj, p_traj).
        """
        if return_trajectory:
            q_traj = [q]
            p_traj = [p]

        for i in range(n_steps):
            if temperature_schedule is not None:
                self.temperature = temperature_schedule[i].item()
            q, p = self.step(q, p)
            if return_trajectory:
                q_traj.append(q)
                p_traj.append(p)

        if return_trajectory:
            return (
                q,
                p,
                torch.stack(q_traj, dim=0),
                torch.stack(p_traj, dim=0),
            )
        return q, p
