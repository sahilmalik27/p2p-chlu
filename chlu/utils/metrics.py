"""Evaluation metrics for trajectory analysis."""

from __future__ import annotations

import torch
from torch import Tensor


def trajectory_mse(pred: Tensor, target: Tensor) -> float:
    """Mean squared error between predicted and target trajectories.

    Args:
        pred: Predicted trajectory, shape (T, dim).
        target: Ground truth trajectory, shape (T, dim).

    Returns:
        Scalar MSE value.
    """
    min_len = min(pred.shape[0], target.shape[0])
    return torch.mean((pred[:min_len] - target[:min_len]) ** 2).item()


def energy_drift(energies: Tensor) -> float:
    """Measure energy drift as relative std deviation over a trajectory.

    Args:
        energies: Energy values over time, shape (T,).

    Returns:
        Relative energy drift: std(E) / |mean(E)|.
    """
    mean_e = energies.mean()
    if mean_e.abs() < 1e-10:
        return energies.std().item()
    return (energies.std() / mean_e.abs()).item()


def max_kinetic_energy(velocities: Tensor) -> float:
    """Maximum kinetic energy from velocity trajectory.

    Args:
        velocities: Velocity values, shape (T,) or (T, dim).

    Returns:
        Max KE = 0.5 * max(|v|²).
    """
    if velocities.dim() > 1:
        v_sq = (velocities**2).sum(dim=-1)
    else:
        v_sq = velocities**2
    return (0.5 * v_sq.max()).item()


def velocity_bound_violations(velocities: Tensor, c: float = 1.0) -> float:
    """Fraction of time steps where |v| > c.

    Args:
        velocities: Velocity values, shape (T,) or (T, dim).
        c: Speed limit.

    Returns:
        Fraction of violations in [0, 1].
    """
    if velocities.dim() > 1:
        speeds = velocities.norm(dim=-1)
    else:
        speeds = velocities.abs()
    return (speeds > c).float().mean().item()
