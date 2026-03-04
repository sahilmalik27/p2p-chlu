"""Dissipative Velocity Verlet integrator (symplectic).

Implements the leapfrog-style integration scheme from arXiv:2603.01768v1:
    1. p_{t+½}   = p_t      − (ε/2) ∇_q [V(q_t) + α‖q_t‖²]
    2. q_{t+1}   = q_t      + ε ∇_p T(p_{t+½})
    3. p*_{t+1}  = p_{t+½}  − (ε/2) ∇_q [V(q_{t+1}) + α‖q_{t+1}‖²]
    4. p_{t+1}   = (1 − γ) p*_{t+1}

When γ=0, this is the standard (energy-conserving) Velocity Verlet.
When γ>0, momentum is damped each step (dissipative).
"""

from __future__ import annotations

import torch
from torch import Tensor

from chlu.core.hamiltonian import RelativisticHamiltonian


class VelocityVerletIntegrator:
    """Symplectic Velocity Verlet integrator with optional dissipation.

    Args:
        hamiltonian: The Hamiltonian providing forces and velocities.
        dt: Integration step size ε.
        gamma: Dissipation coefficient γ ∈ [0, 1). 0 = conservative.
    """

    def __init__(
        self,
        hamiltonian: RelativisticHamiltonian,
        dt: float = 0.01,
        gamma: float = 0.0,
    ) -> None:
        self.hamiltonian = hamiltonian
        self.dt = dt
        self.gamma = gamma

    def step(self, q: Tensor, p: Tensor) -> tuple[Tensor, Tensor]:
        """Perform one Velocity Verlet step.

        Args:
            q: Positions, shape (..., dim).
            p: Momenta, shape (..., dim).

        Returns:
            (q_new, p_new) after one integration step.
        """
        H = self.hamiltonian
        half_dt = self.dt / 2.0

        # Step 1: half-step momentum update
        grad_V_q = H.dV_dq(q)
        p_half = p - half_dt * grad_V_q

        # Step 2: full-step position update using relativistic velocity
        velocity = H.dT_dp(p_half)
        q_new = q + self.dt * velocity

        # Step 3: second half-step momentum update
        grad_V_q_new = H.dV_dq(q_new)
        p_star = p_half - half_dt * grad_V_q_new

        # Step 4: dissipation
        p_new = (1.0 - self.gamma) * p_star

        return q_new, p_new

    def integrate(
        self,
        q: Tensor,
        p: Tensor,
        n_steps: int,
        return_trajectory: bool = False,
    ) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor, Tensor]:
        """Integrate for multiple steps.

        Args:
            q: Initial positions, shape (batch, dim).
            p: Initial momenta, shape (batch, dim).
            n_steps: Number of integration steps.
            return_trajectory: If True, return full trajectory.

        Returns:
            If return_trajectory is False: (q_final, p_final).
            If True: (q_final, p_final, q_traj, p_traj) where
                q_traj/p_traj have shape (n_steps+1, batch, dim).
        """
        if return_trajectory:
            q_traj = [q]
            p_traj = [p]

        for _ in range(n_steps):
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
