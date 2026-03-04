"""Experiment A: Long-Horizon Stability (Lemniscate).

Train on 3 cycles of a lemniscate (figure-8), then infer 50 cycles.
Compare CHLU vs LSTM vs Neural ODE.
Success criterion: orbit stays closed and bounded after 50 cycles.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from chlu.baselines.lstm_baseline import LSTMBaseline
from chlu.baselines.node_baseline import NeuralODEBaseline
from chlu.core.chlu_unit import CHLUUnit
from chlu.data.lemniscate import LemniscateDataset, lemniscate_trajectory
from chlu.training.contrastive import HCDConfig, HCDTrainer
from chlu.utils.metrics import trajectory_mse, energy_drift
from chlu.utils.plotting import plot_trajectories, plot_energy


def train_chlu(
    dataset: LemniscateDataset,
    epochs: int = 100,
    device: torch.device | None = None,
) -> CHLUUnit:
    """Train CHLU on lemniscate data."""
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
        lambda_lyap=0.01,
        lambda_cd=0.1,
        epochs=epochs,
        batch_size=64,
        log_interval=20,
    )

    trainer = HCDTrainer(model, config, device)
    trainer.train(dataset)
    return model


def train_lstm(
    dataset: LemniscateDataset,
    epochs: int = 100,
    device: torch.device | None = None,
) -> LSTMBaseline:
    """Train LSTM baseline on lemniscate data."""
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
    dataset: LemniscateDataset,
    epochs: int = 100,
    device: torch.device | None = None,
) -> NeuralODEBaseline:
    """Train Neural ODE baseline on lemniscate data."""
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


@torch.no_grad()
def evaluate_long_horizon(
    model: nn.Module,
    n_cycles_infer: int = 50,
    points_per_cycle: int = 1000,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Roll out model for n_cycles_infer and return predicted + ground truth.

    Returns:
        (predicted, ground_truth) each of shape (n_points, 2).
    """
    model.eval()
    n_points = n_cycles_infer * points_per_cycle
    gt = lemniscate_trajectory(n_points + 1, n_cycles=n_cycles_infer)

    x0 = gt[0:1]  # (1, 2)
    if hasattr(model, "evolve_sequence"):
        pred = model.evolve_sequence(x0, seq_len=n_points)
        pred = pred.squeeze(0)  # (n_points, 2)
    else:
        # Fallback: step-by-step
        preds = []
        current = x0
        for _ in range(n_points):
            current = model(current)
            preds.append(current.squeeze(0))
        pred = torch.stack(preds)

    return pred, gt[1:]


def run(
    output_dir: str = "results/exp_a",
    epochs: int = 100,
    device: str = "cpu",
) -> dict[str, float]:
    """Run Experiment A: Long-Horizon Stability.

    Args:
        output_dir: Directory for saving results and plots.
        epochs: Training epochs.
        device: Device string.

    Returns:
        Dictionary of metrics for each model.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(device)

    print("=" * 60)
    print("Experiment A: Long-Horizon Stability (Lemniscate)")
    print("=" * 60)

    # Generate training data (3 cycles)
    dataset = LemniscateDataset(n_points=3000, n_cycles=3)

    # Train all models
    print("\n--- Training CHLU ---")
    chlu_model = train_chlu(dataset, epochs=epochs, device=dev)

    print("\n--- Training LSTM ---")
    lstm_model = train_lstm(dataset, epochs=epochs, device=dev)

    print("\n--- Training Neural ODE ---")
    node_model = train_node(dataset, epochs=epochs, device=dev)

    # Evaluate: 50-cycle rollout
    print("\n--- Evaluating 50-cycle rollout ---")
    results = {}

    for name, model in [("CHLU", chlu_model), ("LSTM", lstm_model), ("NeuralODE", node_model)]:
        model.cpu()
        pred, gt = evaluate_long_horizon(model, n_cycles_infer=50)
        mse = trajectory_mse(pred, gt)
        results[name] = {"mse": mse}
        print(f"  {name}: MSE = {mse:.6f}")

    # Plot
    trajectories = {}
    for name, model in [("CHLU", chlu_model), ("LSTM", lstm_model), ("NeuralODE", node_model)]:
        pred, gt = evaluate_long_horizon(model, n_cycles_infer=50)
        trajectories[name] = pred.numpy()
    trajectories["Ground Truth"] = gt.numpy()

    plot_trajectories(
        trajectories,
        title="Exp A: 50-Cycle Lemniscate Rollout",
        save_path=str(out / "trajectories.png"),
    )

    print(f"\nResults saved to {out}")
    return results
