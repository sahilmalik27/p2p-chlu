"""Lemniscate (figure-8) trajectory generator.

Generates parametric lemniscate of Bernoulli trajectories:
    x(t) = a sin(t) / (1 + cos²(t))
    y(t) = a sin(t) cos(t) / (1 + cos²(t))
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.utils.data import Dataset


def lemniscate_trajectory(
    n_points: int,
    n_cycles: int = 3,
    a: float = 1.0,
    noise_std: float = 0.0,
) -> Tensor:
    """Generate a lemniscate (figure-8) trajectory.

    Args:
        n_points: Total number of points in the trajectory.
        n_cycles: Number of complete figure-8 cycles.
        a: Scale parameter.
        noise_std: Gaussian noise standard deviation.

    Returns:
        Trajectory of shape (n_points, 2) with (x, y) coordinates.
    """
    t = torch.linspace(0, 2 * torch.pi * n_cycles, n_points)
    cos_t = torch.cos(t)
    sin_t = torch.sin(t)
    denom = 1.0 + cos_t**2

    x = a * sin_t / denom
    y = a * sin_t * cos_t / denom

    traj = torch.stack([x, y], dim=-1)

    if noise_std > 0:
        traj = traj + torch.randn_like(traj) * noise_std

    return traj


class LemniscateDataset(Dataset):
    """Dataset of (input, target) pairs from a lemniscate trajectory.

    Each sample is (trajectory[t], trajectory[t+1]).

    Args:
        n_points: Points per trajectory.
        n_cycles: Number of cycles to generate.
        a: Scale parameter.
        noise_std: Noise level.
    """

    def __init__(
        self,
        n_points: int = 3000,
        n_cycles: int = 3,
        a: float = 1.0,
        noise_std: float = 0.0,
    ) -> None:
        self.trajectory = lemniscate_trajectory(n_points, n_cycles, a, noise_std)
        # (input, target) = consecutive pairs
        self.inputs = self.trajectory[:-1]
        self.targets = self.trajectory[1:]

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        return self.inputs[idx], self.targets[idx]
