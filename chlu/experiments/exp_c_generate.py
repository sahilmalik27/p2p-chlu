"""Experiment C: Thermodynamic Generation (MNIST).

Train CHLU on 10k MNIST images, then generate via Langevin dynamics:
1. Start from class centroid + noise
2. Run Langevin dynamics with temperature annealing
3. Apply tanh at step 1000
4. Success: recognizable digit modes emerge
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

from chlu.core.chlu_unit import CHLUUnit
from chlu.core.langevin import LangevinSampler
from chlu.training.contrastive import HCDConfig, HCDTrainer
from chlu.utils.plotting import plot_generated_digits


class MNISTFlatDataset(torch.utils.data.Dataset):
    """Flattened MNIST dataset for CHLU training.

    Each sample is (image_flat, image_flat) — autoencoding task.

    Args:
        n_samples: Max number of samples to use (None = all).
        train: Use training split.
    """

    def __init__(self, n_samples: int | None = 10000, train: bool = True) -> None:
        from torchvision import datasets, transforms

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(-1)),  # Flatten 28x28 → 784
        ])

        full_dataset = datasets.MNIST(
            root="./data", train=train, download=True, transform=transform,
        )

        if n_samples is not None and n_samples < len(full_dataset):
            indices = torch.randperm(len(full_dataset))[:n_samples]
            self.data = torch.stack([full_dataset[i][0] for i in indices])
            self.labels = torch.tensor([full_dataset[i][1] for i in indices])
        else:
            self.data = torch.stack([full_dataset[i][0] for i in range(len(full_dataset))])
            self.labels = torch.tensor([full_dataset[i][1] for i in range(len(full_dataset))])

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img = self.data[idx]
        return img, img  # Autoencoding


def compute_centroids(dataset: MNISTFlatDataset) -> dict[int, torch.Tensor]:
    """Compute mean image for each digit class.

    Returns:
        Dict mapping digit → centroid tensor of shape (784,).
    """
    centroids: dict[int, torch.Tensor] = {}
    for digit in range(10):
        mask = dataset.labels == digit
        centroids[digit] = dataset.data[mask].mean(dim=0)
    return centroids


def train_chlu_mnist(
    dataset: MNISTFlatDataset,
    epochs: int = 50,
    device: torch.device | None = None,
) -> CHLUUnit:
    """Train CHLU on flattened MNIST for autoencoding."""
    device = device or torch.device("cpu")
    model = CHLUUnit(
        input_dim=784,
        latent_dim=64,
        c=2.0,
        m0=1.0,
        alpha=0.001,
        dt=0.005,
        n_steps=10,
        gamma=0.0,
        hidden_dims=(256, 128),
    )

    config = HCDConfig(
        lr=5e-4,
        lambda_lyap=0.01,
        lambda_cd=0.005,
        epochs=epochs,
        batch_size=128,
        log_interval=10,
        buffer_capacity=5000,
    )

    trainer = HCDTrainer(model, config, device)
    trainer.train(dataset)
    return model


def generate_digits(
    model: CHLUUnit,
    centroids: dict[int, torch.Tensor],
    n_per_digit: int = 10,
    langevin_steps: int = 1000,
    temperature_start: float = 1.0,
    temperature_end: float = 0.01,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Generate digits via Langevin dynamics from centroids.

    Args:
        model: Trained CHLUUnit.
        centroids: Per-digit mean images.
        n_per_digit: Samples per digit class.
        langevin_steps: Number of Langevin steps.
        temperature_start: Initial temperature.
        temperature_end: Final temperature.
        device: Device.

    Returns:
        Generated images, shape (10 * n_per_digit, 784).
    """
    device = device or torch.device("cpu")
    model.to(device).eval()

    # Temperature annealing schedule (linear)
    temp_schedule = torch.linspace(temperature_start, temperature_end, langevin_steps)

    sampler = LangevinSampler(
        hamiltonian=model.hamiltonian,
        dt=0.005,
        gamma=0.1,
        temperature=temperature_start,
    )

    all_generated = []

    for digit in range(10):
        centroid = centroids[digit].to(device)

        # Start from centroid + noise
        x0 = centroid.unsqueeze(0).expand(n_per_digit, -1)
        noise = torch.randn_like(x0) * 0.1
        x_init = x0 + noise

        # Encode to phase space
        q, p = model.encode(x_init)

        # Run Langevin dynamics
        q_final, p_final = sampler.sample(
            q, p, n_steps=langevin_steps, temperature_schedule=temp_schedule,
        )

        # Decode and apply tanh for bounded output
        generated = torch.tanh(model.decode(q_final))
        # Rescale from [-1, 1] to [0, 1]
        generated = (generated + 1.0) / 2.0
        all_generated.append(generated)

    return torch.cat(all_generated, dim=0)


def run(
    output_dir: str = "results/exp_c",
    epochs: int = 50,
    device: str = "cpu",
    n_per_digit: int = 10,
) -> None:
    """Run Experiment C: MNIST Generation.

    Args:
        output_dir: Directory for results.
        epochs: Training epochs.
        device: Device string ("cpu" or "cuda").
        n_per_digit: Number of samples to generate per digit.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(device)

    print("=" * 60)
    print("Experiment C: Thermodynamic Generation (MNIST)")
    print("=" * 60)

    # Load data
    print("\n--- Loading MNIST (10k samples) ---")
    dataset = MNISTFlatDataset(n_samples=10000)
    centroids = compute_centroids(dataset)

    # Train
    print("\n--- Training CHLU ---")
    model = train_chlu_mnist(dataset, epochs=epochs, device=dev)

    # Save best CHLU checkpoint
    ckpt_dir = out / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "experiment": "exp_c_generate",
            "task": "mnist_generation",
        },
        ckpt_dir / "chlu_best.pt",
    )
    print(f"Saved best CHLU checkpoint to {ckpt_dir / 'chlu_best.pt'}")

    # Generate
    print(f"\n--- Generating {n_per_digit} digits per class ---")
    model.cpu()
    generated = generate_digits(
        model, centroids, n_per_digit=n_per_digit,
        langevin_steps=1000,
        temperature_start=1.0,
        temperature_end=0.01,
    )

    # Plot grid
    plot_generated_digits(
        generated.detach().numpy(),
        n_per_digit=n_per_digit,
        save_path=str(out / "generated_digits.png"),
    )

    # Save raw
    torch.save(generated.detach(), out / "generated_digits.pt")

    print(f"\nGenerated {generated.shape[0]} digits, saved to {out}")
