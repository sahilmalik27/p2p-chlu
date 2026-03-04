"""Neural ODE baseline for temporal sequence prediction.

Uses torchdiffeq for ODE integration with a learned vector field.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

try:
    from torchdiffeq import odeint
except ImportError:
    odeint = None


class ODEFunc(nn.Module):
    """Learned vector field dz/dt = f_θ(z)."""

    def __init__(self, dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, t: Tensor, z: Tensor) -> Tensor:
        return self.net(z)


class NeuralODEBaseline(nn.Module):
    """Neural ODE baseline model.

    Encodes input → latent, evolves via ODE, decodes back.

    Args:
        input_dim: Feature dimension of input/output.
        latent_dim: Dimension of the ODE state space.
        hidden_dim: Hidden dimension of the ODE function.
        dt: Time step for integration endpoints.
        n_steps: Number of time steps for integration.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        hidden_dim: int = 128,
        dt: float = 0.01,
        n_steps: int = 10,
    ) -> None:
        super().__init__()
        if odeint is None:
            raise ImportError("torchdiffeq is required for NeuralODEBaseline. pip install torchdiffeq")

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.dt = dt
        self.n_steps = n_steps

        self.encoder = nn.Linear(input_dim, latent_dim)
        self.ode_func = ODEFunc(latent_dim, hidden_dim)
        self.decoder = nn.Linear(latent_dim, input_dim)

    def forward(self, x: Tensor, n_steps: int | None = None) -> Tensor:
        """Forward pass: encode → integrate ODE → decode.

        Args:
            x: Input, shape (batch, input_dim).
            n_steps: Override number of integration steps.

        Returns:
            Prediction, shape (batch, input_dim).
        """
        steps = n_steps if n_steps is not None else self.n_steps
        z0 = self.encoder(x)
        t_span = torch.linspace(0, self.dt * steps, 2, device=x.device)
        z_traj = odeint(self.ode_func, z0, t_span, method="dopri5")
        z_final = z_traj[-1]
        return self.decoder(z_final)

    def evolve_sequence(
        self,
        x0: Tensor,
        seq_len: int,
        steps_per_output: int | None = None,
    ) -> Tensor:
        """Generate a sequence by repeated ODE integration.

        Args:
            x0: Initial input, shape (batch, input_dim).
            seq_len: Number of output time steps.
            steps_per_output: Integration steps between outputs.

        Returns:
            Sequence, shape (batch, seq_len, input_dim).
        """
        spo = steps_per_output if steps_per_output is not None else self.n_steps
        z = self.encoder(x0)
        outputs: list[Tensor] = []

        for _ in range(seq_len):
            t_span = torch.linspace(0, self.dt * spo, 2, device=x0.device)
            z_traj = odeint(self.ode_func, z, t_span, method="dopri5")
            z = z_traj[-1]
            outputs.append(self.decoder(z))

        return torch.stack(outputs, dim=1)
