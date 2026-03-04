"""Relativistic Hamiltonian H(q, p) = T(p) + V_θ(q) + α‖q‖².

Implements the energy function from arXiv:2603.01768v1 with:
- Relativistic kinetic energy T(p) = √(c² pᵀ M⁻¹ p + m₀² c⁴)
- Learnable MLP potential V_θ(q)
- Quadratic confinement α‖q‖²
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class PotentialMLP(nn.Module):
    """Learnable potential energy V_θ(q) as a small MLP."""

    def __init__(self, dim: int, hidden_dims: tuple[int, ...] = (128, 128)) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.SiLU())
            in_dim = h
        # Scalar output — potential energy
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, q: Tensor) -> Tensor:
        """Compute V_θ(q). Returns shape (...,)."""
        return self.net(q).squeeze(-1)


class RelativisticHamiltonian(nn.Module):
    """Relativistic Hamiltonian with learnable potential and mass matrix.

    H(q, p) = T(p) + V_θ(q) + α‖q‖²

    where T(p) = √(c² pᵀ M⁻¹ p + m₀² c⁴) is relativistic kinetic energy,
    M is a learnable diagonal positive-definite mass matrix (exp-parameterized),
    and α is the confinement strength.
    """

    def __init__(
        self,
        dim: int,
        c: float = 1.0,
        m0: float = 1.0,
        alpha: float = 0.01,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__()
        self.dim = dim
        self.c = c
        self.m0 = m0
        self.alpha = alpha

        # Learnable potential
        self.potential = PotentialMLP(dim, hidden_dims)

        # Log-mass for exp parameterization → guaranteed positive-definite diagonal
        self.log_mass = nn.Parameter(torch.zeros(dim))

    @property
    def mass_inv(self) -> Tensor:
        """Inverse mass matrix diagonal: M⁻¹ = exp(-log_mass)."""
        return torch.exp(-self.log_mass)

    def kinetic_energy(self, p: Tensor) -> Tensor:
        """Relativistic kinetic energy T(p) = √(c² pᵀ M⁻¹ p + m₀² c⁴).

        Args:
            p: Momenta, shape (..., dim).

        Returns:
            Scalar kinetic energy per sample, shape (...,).
        """
        c, m0 = self.c, self.m0
        # pᵀ M⁻¹ p with diagonal M⁻¹
        p_Minv_p = (p**2 * self.mass_inv).sum(dim=-1)
        return torch.sqrt(c**2 * p_Minv_p + m0**2 * c**4)

    def confinement(self, q: Tensor) -> Tensor:
        """Confinement energy α‖q‖²."""
        return self.alpha * (q**2).sum(dim=-1)

    def forward(self, q: Tensor, p: Tensor) -> Tensor:
        """Compute total Hamiltonian H(q, p).

        Args:
            q: Positions, shape (..., dim).
            p: Momenta, shape (..., dim).

        Returns:
            Total energy per sample, shape (...,).
        """
        T = self.kinetic_energy(p)
        V = self.potential(q)
        C = self.confinement(q)
        return T + V + C

    def dT_dp(self, p: Tensor) -> Tensor:
        """Relativistic velocity ∇_p T = c² M⁻¹ p / √(c² pᵀ M⁻¹ p + m₀² c⁴).

        This is the velocity, bounded by c due to the relativistic form.

        Args:
            p: Momenta, shape (..., dim).

        Returns:
            Velocity, shape (..., dim).
        """
        c, m0 = self.c, self.m0
        p_Minv_p = (p**2 * self.mass_inv).sum(dim=-1, keepdim=True)
        denom = torch.sqrt(c**2 * p_Minv_p + m0**2 * c**4)
        return c**2 * self.mass_inv * p / denom

    def dV_dq(self, q: Tensor) -> Tensor:
        """Gradient of potential + confinement w.r.t. q.

        ∇_q [V_θ(q) + α‖q‖²]

        Args:
            q: Positions, shape (..., dim). Must have requires_grad or
               this function will enable it temporarily.

        Returns:
            Force (negative gradient), shape (..., dim).
        """
        q_in = q.detach().requires_grad_(True)
        V = self.potential(q_in) + self.confinement(q_in)
        grad = torch.autograd.grad(
            V.sum(), q_in, create_graph=self.training
        )[0]
        return grad
