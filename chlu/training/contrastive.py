"""Hamiltonian Contrastive Divergence trainer.

Wake-sleep training scheme from arXiv:2603.01768v1:
- Wake phase: evolve z₀ → z_wake (γ=0), compute MSE + Lyapunov
- Sleep phase: sample from replay buffer, evolve → z_sleep (γ=0)
- Contrastive update: Δθ ∝ −∇θ H(z_wake) + ∇θ H(z_sleep)
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm

from chlu.core.chlu_unit import CHLUUnit
from chlu.core.integrator import VelocityVerletIntegrator
from chlu.training.losses import lyapunov_loss, mse_loss
from chlu.training.replay_buffer import ReplayBuffer


@dataclass
class HCDConfig:
    """Configuration for Hamiltonian Contrastive Divergence training."""

    lr: float = 1e-3
    lambda_lyap: float = 0.01
    lambda_cd: float = 0.1
    sleep_steps: int = 20
    buffer_capacity: int = 10000
    warmup_epochs: int = 5
    epochs: int = 100
    batch_size: int = 64
    log_interval: int = 10


class HCDTrainer:
    """Hamiltonian Contrastive Divergence trainer for CHLUUnit.

    Implements the wake-sleep training procedure with replay buffer.
    CD loss uses softplus normalization on energies to prevent divergence,
    combined with spectral normalization on the potential network.

    Args:
        model: The CHLUUnit model to train.
        config: Training configuration.
        device: Device for training.
    """

    def __init__(
        self,
        model: CHLUUnit,
        config: HCDConfig | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.model = model
        self.config = config or HCDConfig()
        self.device = device or torch.device("cpu")
        self.model.to(self.device)
        self._current_epoch = 0

        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.config.lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.config.epochs, eta_min=self.config.lr * 0.01,
        )
        self.buffer = ReplayBuffer(
            capacity=self.config.buffer_capacity,
            dim=model.latent_dim,
        )

        # Sleep-phase integrator (conservative, γ=0)
        self.sleep_integrator = VelocityVerletIntegrator(
            hamiltonian=model.hamiltonian,
            dt=model.integrator.dt,
            gamma=0.0,
        )

    def wake_phase(
        self, x: Tensor, target: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, dict[str, float]]:
        """Wake phase: encode input, integrate, compute reconstruction loss.

        Args:
            x: Input batch, shape (batch, input_dim).
            target: Target output, shape (batch, input_dim).

        Returns:
            (wake_loss, q_wake, p_wake, metrics).
        """
        # Ensure conservative dynamics for training
        old_gamma = self.model.integrator.gamma
        self.model.integrator.gamma = 0.0

        q, p = self.model.encode(x)
        q_wake, p_wake, q_traj, p_traj = self.model.integrator.integrate(
            q, p, self.model.n_steps, return_trajectory=True,
        )
        pred = self.model.decode(q_wake)

        l_mse = mse_loss(pred, target)
        l_lyap = lyapunov_loss(self.model.hamiltonian, q_traj, p_traj)
        wake_loss = l_mse + self.config.lambda_lyap * l_lyap

        self.model.integrator.gamma = old_gamma

        metrics = {"mse": l_mse.item(), "lyapunov": l_lyap.item()}
        return wake_loss, q_wake, p_wake, metrics

    def sleep_phase(self, batch_size: int) -> tuple[Tensor, Tensor, Tensor] | None:
        """Sleep phase: sample from replay buffer, evolve, compute energy.

        Args:
            batch_size: Number of samples to draw from buffer.

        Returns:
            (H_sleep, q_sleep, p_sleep) or None if buffer is too small.
        """
        if len(self.buffer) < batch_size:
            return None

        q_sleep, p_sleep = self.buffer.sample(batch_size, device=self.device)

        # Evolve hallucinated states
        q_sleep, p_sleep = self.sleep_integrator.integrate(
            q_sleep, p_sleep, self.config.sleep_steps,
        )

        H_sleep = self.model.hamiltonian(q_sleep, p_sleep).mean()
        return H_sleep, q_sleep, p_sleep

    def train_step(
        self, x: Tensor, target: Tensor,
    ) -> dict[str, float]:
        """One training step of Hamiltonian Contrastive Divergence.

        CD loss uses softplus normalization: log(1+exp(H)) to prevent
        unbounded energy values from causing gradient explosion.
        CD is only enabled after warmup_epochs to let MSE converge first.

        Args:
            x: Input batch.
            target: Target batch.

        Returns:
            Dictionary of training metrics.
        """
        self.model.train()
        self.optimizer.zero_grad()

        # Wake phase
        wake_loss, q_wake, p_wake, metrics = self.wake_phase(x, target)

        # Contrastive divergence with softplus-normalized energy
        H_wake = self.model.hamiltonian(q_wake, p_wake).mean()
        cd_loss = torch.tensor(0.0, device=self.device)

        sleep_result = self.sleep_phase(x.shape[0])
        # Store wake states in replay buffer (detached to avoid graph retention)
        self.buffer.push(q_wake.detach(), p_wake.detach())

        if sleep_result is not None and self._current_epoch >= self.config.warmup_epochs:
            H_sleep, _, _ = sleep_result
            # Softplus normalization prevents unbounded energy from blowing up CD
            cd_loss = F.softplus(H_wake) - F.softplus(H_sleep)
            cd_loss = cd_loss.clamp(-2.0, 2.0)

        total_loss = wake_loss + self.config.lambda_cd * cd_loss
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        cd_val = cd_loss.item() if isinstance(cd_loss, Tensor) else cd_loss
        metrics.update({
            "total_loss": total_loss.item(),
            "cd_loss": cd_val,
            "H_wake": H_wake.item(),
        })
        return metrics

    def train(
        self,
        dataset: torch.utils.data.Dataset,
        val_dataset: torch.utils.data.Dataset | None = None,
    ) -> list[dict[str, float]]:
        """Full training loop.

        Args:
            dataset: Training dataset yielding (input, target) pairs.
            val_dataset: Optional validation dataset.

        Returns:
            List of per-epoch metrics.
        """
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=True,
        )

        history: list[dict[str, float]] = []
        best_mse = float("inf")
        best_state: dict | None = None
        best_epoch = 0

        for epoch in range(self.config.epochs):
            self._current_epoch = epoch
            epoch_metrics: dict[str, list[float]] = {}

            pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{self.config.epochs}", leave=False)
            for x, target in pbar:
                x = x.to(self.device)
                target = target.to(self.device)

                step_metrics = self.train_step(x, target)

                for k, v in step_metrics.items():
                    epoch_metrics.setdefault(k, []).append(v)

                pbar.set_postfix(
                    loss=f"{step_metrics['total_loss']:.8f}",
                    mse=f"{step_metrics['mse']:.8f}",
                )

            # Aggregate epoch metrics
            avg_metrics = {k: sum(v) / len(v) for k, v in epoch_metrics.items()}
            avg_metrics["epoch"] = epoch + 1
            history.append(avg_metrics)

            # Track best model by MSE
            if avg_metrics["mse"] < best_mse:
                best_mse = avg_metrics["mse"]
                best_epoch = epoch + 1
                best_state = copy.deepcopy(self.model.state_dict())

            # Step learning rate scheduler
            self.scheduler.step()

            # Early stopping: if MSE hasn't improved in 20 epochs, stop
            if epoch + 1 > 20 and best_epoch < epoch + 1 - 20:
                print(f"[Early stop] No MSE improvement for 20 epochs (best at epoch {best_epoch})")
                break

            if (epoch + 1) % self.config.log_interval == 0:
                print(
                    f"[Epoch {epoch+1}] "
                    f"loss={avg_metrics['total_loss']:.8f} "
                    f"mse={avg_metrics['mse']:.8f} "
                    f"lyap={avg_metrics['lyapunov']:.8f} "
                    f"cd={avg_metrics['cd_loss']:.8f}"
                    f" lr={self.scheduler.get_last_lr()[0]:.6f}"
                )

        # Restore best model weights
        if best_state is not None:
            self.model.load_state_dict(best_state)
            print(f"[Best model] epoch={best_epoch} mse={best_mse:.8f}")

        # Store best info for checkpoint saving
        self._best_epoch = best_epoch
        self._best_mse = best_mse

        return history
