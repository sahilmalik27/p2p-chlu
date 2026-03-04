"""Visualization utilities for CHLU experiments."""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None  # type: ignore[assignment]


def _ensure_matplotlib() -> None:
    if plt is None:
        raise ImportError("matplotlib is required for plotting. pip install matplotlib")


def plot_trajectories(
    trajectories: dict[str, np.ndarray],
    title: str = "Trajectories",
    save_path: str | None = None,
) -> None:
    """Plot 2D trajectories for multiple models.

    Args:
        trajectories: Dict mapping model name → array of shape (T, 2).
        title: Plot title.
        save_path: If provided, save figure to this path.
    """
    _ensure_matplotlib()

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    colors = ["#2196F3", "#FF5722", "#4CAF50", "#9E9E9E"]

    for i, (name, traj) in enumerate(trajectories.items()):
        color = colors[i % len(colors)]
        alpha = 0.4 if name == "Ground Truth" else 0.8
        lw = 1.0 if name == "Ground Truth" else 0.6
        ax.plot(traj[:, 0], traj[:, 1], label=name, color=color, alpha=alpha, linewidth=lw)

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_phase_space(
    phase_data: dict[str, np.ndarray],
    title: str = "Phase Space",
    save_path: str | None = None,
) -> None:
    """Plot phase space (position vs velocity).

    Args:
        phase_data: Dict mapping name → array of shape (T, 2) where col 0=x, col 1=v.
        title: Plot title.
        save_path: Save path.
    """
    _ensure_matplotlib()

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    colors = ["#2196F3", "#FF5722", "#4CAF50"]

    for i, (name, data) in enumerate(phase_data.items()):
        color = colors[i % len(colors)]
        ax.plot(data[:, 0], data[:, 1], label=name, color=color, alpha=0.7, linewidth=0.5)

    ax.set_title(title)
    ax.set_xlabel("Position (q)")
    ax.set_ylabel("Velocity (p)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_kinetic_energy(
    ke_data: dict[str, np.ndarray],
    title: str = "Kinetic Energy",
    save_path: str | None = None,
) -> None:
    """Plot kinetic energy over time for multiple models.

    Args:
        ke_data: Dict mapping name → 1D array of KE values.
        title: Plot title.
        save_path: Save path.
    """
    _ensure_matplotlib()

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    colors = ["#2196F3", "#FF5722", "#4CAF50"]

    for i, (name, ke) in enumerate(ke_data.items()):
        color = colors[i % len(colors)]
        ax.plot(ke, label=name, color=color, alpha=0.8, linewidth=0.8)

    ax.set_title(title)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Kinetic Energy")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_energy(
    energies: dict[str, np.ndarray],
    title: str = "Energy Conservation",
    save_path: str | None = None,
) -> None:
    """Plot Hamiltonian energy over time.

    Args:
        energies: Dict mapping name → 1D array of energy values.
        title: Plot title.
        save_path: Save path.
    """
    _ensure_matplotlib()

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    for name, E in energies.items():
        ax.plot(E, label=name, alpha=0.8, linewidth=0.8)

    ax.set_title(title)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("H(q, p)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_generated_digits(
    images: np.ndarray,
    n_per_digit: int = 10,
    save_path: str | None = None,
) -> None:
    """Plot a grid of generated MNIST digits.

    Args:
        images: Array of shape (10 * n_per_digit, 784).
        n_per_digit: Samples per digit class.
        save_path: Save path.
    """
    _ensure_matplotlib()

    fig, axes = plt.subplots(10, n_per_digit, figsize=(n_per_digit * 1.2, 12))

    for digit in range(10):
        for j in range(n_per_digit):
            idx = digit * n_per_digit + j
            img = images[idx].reshape(28, 28)
            img = np.clip(img, 0, 1)
            ax = axes[digit, j] if n_per_digit > 1 else axes[digit]
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            if j == 0:
                ax.set_ylabel(str(digit), fontsize=12, rotation=0, labelpad=15)

    fig.suptitle("Generated MNIST Digits (Langevin Dynamics)", fontsize=14)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)
