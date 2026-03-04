"""Loss functions for CHLU training.

MSE reconstruction loss + Lyapunov regularization for orbital stability.
"""

from __future__ import annotations

import torch
from torch import Tensor

from chlu.core.hamiltonian import RelativisticHamiltonian


def mse_loss(pred: Tensor, target: Tensor) -> Tensor:
    """Mean squared error over all dimensions.

    Args:
        pred: Predicted values, shape (..., dim).
        target: Target values, shape (..., dim).

    Returns:
        Scalar MSE loss.
    """
    return torch.mean((pred - target) ** 2)


def lyapunov_loss(
    hamiltonian: RelativisticHamiltonian,
    q_traj: Tensor,
    p_traj: Tensor,
) -> Tensor:
    """Lyapunov regularization: penalize energy drift over trajectory.

    Encourages the Hamiltonian to remain approximately constant along
    the trajectory (energy conservation). Measures variance of H along time.

    Args:
        hamiltonian: The Hamiltonian to evaluate.
        q_traj: Position trajectory, shape (T, batch, dim).
        p_traj: Momentum trajectory, shape (T, batch, dim).

    Returns:
        Scalar Lyapunov loss (variance of energy over time).
    """
    # Compute energy at each time step: shape (T, batch)
    energies = hamiltonian(q_traj, p_traj)
    # Variance over time, mean over batch
    return energies.var(dim=0).mean()


def chlu_loss(
    pred: Tensor,
    target: Tensor,
    hamiltonian: RelativisticHamiltonian,
    q_traj: Tensor,
    p_traj: Tensor,
    lambda_lyap: float = 0.01,
) -> tuple[Tensor, dict[str, float]]:
    """Combined CHLU training loss.

    L = MSE(pred, target) + λ * L_lyapunov

    Args:
        pred: Model predictions.
        target: Ground truth targets.
        hamiltonian: For Lyapunov computation.
        q_traj: Position trajectory from integration.
        p_traj: Momentum trajectory from integration.
        lambda_lyap: Lyapunov regularization weight.

    Returns:
        (total_loss, metrics_dict) where metrics_dict contains individual terms.
    """
    l_mse = mse_loss(pred, target)
    l_lyap = lyapunov_loss(hamiltonian, q_traj, p_traj)
    total = l_mse + lambda_lyap * l_lyap

    metrics = {
        "loss": total.item(),
        "mse": l_mse.item(),
        "lyapunov": l_lyap.item(),
    }
    return total, metrics
