"""Sine wave trajectory generator.

Generates sine wave trajectories with varying frequencies,
including perturbation support for safety experiments.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.utils.data import Dataset


def sine_trajectory(
    n_points: int = 1000,
    omega: float = 1.0,
    amplitude: float = 1.0,
    phase: float = 0.0,
    dt: float = 0.01,
) -> Tensor:
    """Generate a sine wave trajectory as (position, velocity) pairs.

    Args:
        n_points: Number of time steps.
        omega: Angular frequency.
        amplitude: Wave amplitude.
        phase: Phase offset.
        dt: Time step.

    Returns:
        Trajectory of shape (n_points, 2) with (x, v) pairs.
    """
    t = torch.arange(n_points, dtype=torch.float32) * dt
    x = amplitude * torch.sin(omega * t + phase)
    v = amplitude * omega * torch.cos(omega * t + phase)
    return torch.stack([x, v], dim=-1)


class SineWaveDataset(Dataset):
    """Dataset of sine wave trajectories with varying frequencies.

    Each trajectory has a random frequency ω ~ U(omega_min, omega_max).

    Args:
        n_trajectories: Number of trajectories to generate.
        n_points: Points per trajectory.
        omega_min: Minimum frequency.
        omega_max: Maximum frequency.
        dt: Time step.
    """

    def __init__(
        self,
        n_trajectories: int = 100,
        n_points: int = 1000,
        omega_min: float = 0.5,
        omega_max: float = 2.0,
        dt: float = 0.01,
    ) -> None:
        self.trajectories: list[Tensor] = []
        self.omegas: list[float] = []

        for _ in range(n_trajectories):
            omega = omega_min + (omega_max - omega_min) * torch.rand(1).item()
            traj = sine_trajectory(n_points, omega=omega, dt=dt)
            self.trajectories.append(traj)
            self.omegas.append(omega)

        # Flatten into (input, target) pairs across all trajectories
        inputs_list: list[Tensor] = []
        targets_list: list[Tensor] = []
        for traj in self.trajectories:
            inputs_list.append(traj[:-1])
            targets_list.append(traj[1:])

        self.inputs = torch.cat(inputs_list, dim=0)
        self.targets = torch.cat(targets_list, dim=0)

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        return self.inputs[idx], self.targets[idx]
