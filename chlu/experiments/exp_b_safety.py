"""Experiment B: Kinetic Safety (Perturbed Sine Wave).

Train on 100 sine trajectories with ω ~ U(0.5, 2.0), T=1000.
At inference, perturb initial states and verify:
- Velocity saturates at c (relativistic bound)
- Perturbation → phase shift only (not amplitude blowup)
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

from chlu.baselines.lstm_baseline import LSTMBaseline
from chlu.baselines.node_baseline import NeuralODEBaseline
from chlu.core.chlu_unit import CHLUUnit
from chlu.data.sine_wave import SineWaveDataset, sine_trajectory
from chlu.training.contrastive import HCDConfig, HCDTrainer
from chlu.utils.metrics import max_kinetic_energy, velocity_bound_violations
from chlu.utils.plotting import plot_phase_space, plot_kinetic_energy


def train_chlu(
    dataset: SineWaveDataset,
    epochs: int = 100,
    device: torch.device | None = None,
) -> CHLUUnit:
    """Train CHLU on sine wave data."""
    device = device or torch.device("cpu")
    model = CHLUUnit(
        input_dim=2,
        latent_dim=16,
        c=2.0,
        m0=1.0,
        alpha=0.01,
        dt=0.01,
        n_steps=5,
        gamma=0.0,
        hidden_dims=(64, 64),
    )

    config = HCDConfig(
        lr=1e-3,
        lambda_lyap=0.1,
        lambda_cd=0.01,
        epochs=epochs,
        batch_size=64,
        log_interval=20,
    )

    trainer = HCDTrainer(model, config, device)
    trainer.train(dataset)
    return model


def train_lstm(
    dataset: SineWaveDataset,
    epochs: int = 100,
    device: torch.device | None = None,
) -> LSTMBaseline:
    """Train LSTM baseline."""
    device = device or torch.device("cpu")
    model = LSTMBaseline(input_dim=2, hidden_dim=64, num_layers=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    for epoch in range(epochs):
        for x, target in loader:
            x, target = x.to(device), target.to(device)
            pred = model(x)
            loss = nn.functional.mse_loss(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model


def train_node(
    dataset: SineWaveDataset,
    epochs: int = 100,
    device: torch.device | None = None,
) -> NeuralODEBaseline:
    """Train Neural ODE baseline."""
    device = device or torch.device("cpu")
    model = NeuralODEBaseline(
        input_dim=2, latent_dim=16, hidden_dim=64, dt=0.01, n_steps=5,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    for epoch in range(epochs):
        for x, target in loader:
            x, target = x.to(device), target.to(device)
            pred = model(x)
            loss = nn.functional.mse_loss(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model


def evaluate_perturbation(
    model: nn.Module,
    omega: float = 1.0,
    perturb_scale: float = 5.0,
    n_steps: int = 1000,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Roll out from a perturbed initial condition.

    Returns:
        (predicted_trajectory, nominal_trajectory) each of shape (n_steps, 2).
    """
    model.eval()

    # Nominal initial state
    x0_nominal = sine_trajectory(1, omega=omega)  # (1, 2)

    # Perturbed: add large perturbation to velocity
    x0_perturbed = x0_nominal.clone()
    x0_perturbed[0, 1] += perturb_scale  # kick the velocity

    # Roll out both
    if hasattr(model, "evolve_sequence"):
        pred_perturbed = model.evolve_sequence(x0_perturbed, seq_len=n_steps).squeeze(0)
        pred_nominal = model.evolve_sequence(x0_nominal, seq_len=n_steps).squeeze(0)
    else:
        preds_p, preds_n = [], []
        curr_p, curr_n = x0_perturbed, x0_nominal
        for _ in range(n_steps):
            curr_p = model(curr_p)
            curr_n = model(curr_n)
            preds_p.append(curr_p.squeeze(0))
            preds_n.append(curr_n.squeeze(0))
        pred_perturbed = torch.stack(preds_p)
        pred_nominal = torch.stack(preds_n)

    return pred_perturbed, pred_nominal


def run(
    output_dir: str = "results/exp_b",
    epochs: int = 100,
    device: str = "cpu",
) -> dict[str, dict[str, float]]:
    """Run Experiment B: Kinetic Safety.

    Args:
        output_dir: Directory for results.
        epochs: Training epochs.
        device: Device string.

    Returns:
        Metrics for each model.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(device)

    print("=" * 60)
    print("Experiment B: Kinetic Safety (Perturbed Sine)")
    print("=" * 60)

    # Generate training data
    dataset = SineWaveDataset(
        n_trajectories=100, n_points=1000, omega_min=0.5, omega_max=2.0,
    )

    # Train
    print("\n--- Training CHLU ---")
    chlu_model = train_chlu(dataset, epochs=epochs, device=dev)

    print("\n--- Training LSTM ---")
    lstm_model = train_lstm(dataset, epochs=epochs, device=dev)

    print("\n--- Training Neural ODE ---")
    node_model = train_node(dataset, epochs=epochs, device=dev)

    # Save best CHLU checkpoint
    ckpt_dir = out / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": chlu_model.state_dict(),
            "experiment": "exp_b_safety",
            "task": "sine_perturbation",
        },
        ckpt_dir / "chlu_best.pt",
    )
    print(f"Saved best CHLU checkpoint to {ckpt_dir / 'chlu_best.pt'}")

    # Evaluate with perturbation
    print("\n--- Evaluating with perturbation (5x velocity kick) ---")
    results = {}
    phase_data = {}
    ke_data = {}

    for name, model in [("CHLU", chlu_model), ("LSTM", lstm_model), ("NeuralODE", node_model)]:
        model.cpu()
        try:
            pred_pert, pred_nom = evaluate_perturbation(model, perturb_scale=5.0)

            if torch.isnan(pred_pert).any() or torch.isinf(pred_pert).any():
                print(f"  {name}: DIVERGED (NaN/Inf in predictions)")
                results[name] = {"max_kinetic_energy": float("inf"), "max_velocity": float("inf")}
                continue

            # Kinetic energy: 0.5 * v²
            ke_pert = 0.5 * pred_pert[:, 1] ** 2
            max_ke = ke_pert.max().item()
            max_v = pred_pert[:, 1].abs().max().item()

            results[name] = {"max_kinetic_energy": max_ke, "max_velocity": max_v}
            phase_data[name] = pred_pert.detach().numpy()
            ke_data[name] = ke_pert.detach().numpy()

            print(f"  {name}: max_KE={max_ke:.4f}, max_v={max_v:.4f}")
        except (RuntimeError, AssertionError) as e:
            print(f"  {name}: DIVERGED ({type(e).__name__}: {e})")
            results[name] = {"max_kinetic_energy": float("inf"), "max_velocity": float("inf")}

    # Plot phase space
    plot_phase_space(
        phase_data,
        title="Exp B: Phase Space After Perturbation",
        save_path=str(out / "phase_space.png"),
    )

    # Plot kinetic energy
    plot_kinetic_energy(
        ke_data,
        title="Exp B: Kinetic Energy Over Time",
        save_path=str(out / "kinetic_energy.png"),
    )

    print(f"\nResults saved to {out}")
    return results
