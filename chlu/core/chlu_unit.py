"""CHLUUnit — the main nn.Module wrapping Hamiltonian dynamics.

Drop-in replacement for LSTM / Neural ODE in temporal sequence tasks.
Maps input sequences to output sequences via symplectic integration
in a learned phase space.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from chlu.core.hamiltonian import RelativisticHamiltonian
from chlu.core.integrator import VelocityVerletIntegrator


class CHLUUnit(nn.Module):
    """Causal Hamiltonian Learning Unit.

    Encodes input → phase space (q, p), evolves via Velocity Verlet
    under a learnable relativistic Hamiltonian, then decodes back.

    Args:
        input_dim: Dimension of input features.
        latent_dim: Dimension of the phase space (q and p each have this dim).
        c: Speed limit for relativistic kinetic energy.
        m0: Rest mass parameter.
        alpha: Confinement strength.
        dt: Integration step size.
        n_steps: Number of Verlet steps per forward call.
        gamma: Dissipation coefficient (0 = conservative for training).
        hidden_dims: Hidden layer sizes for the potential MLP.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        c: float = 1.0,
        m0: float = 1.0,
        alpha: float = 0.01,
        dt: float = 0.01,
        n_steps: int = 10,
        gamma: float = 0.0,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.n_steps = n_steps

        # Encoder: input → (q₀, p₀)
        self.encoder = nn.Linear(input_dim, 2 * latent_dim)

        # Hamiltonian
        self.hamiltonian = RelativisticHamiltonian(
            dim=latent_dim, c=c, m0=m0, alpha=alpha, hidden_dims=hidden_dims,
        )

        # Integrator
        self.integrator = VelocityVerletIntegrator(
            hamiltonian=self.hamiltonian, dt=dt, gamma=gamma,
        )

        # Decoder: q → output
        self.decoder = nn.Linear(latent_dim, input_dim)

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Encode input to phase-space coordinates.

        Args:
            x: Input tensor, shape (..., input_dim).

        Returns:
            (q, p) each of shape (..., latent_dim).
        """
        z = self.encoder(x)
        q, p = z.split(self.latent_dim, dim=-1)
        return q, p

    def decode(self, q: Tensor) -> Tensor:
        """Decode positions back to output space.

        Args:
            q: Positions, shape (..., latent_dim).

        Returns:
            Output, shape (..., input_dim).
        """
        return self.decoder(q)

    def forward(
        self,
        x: Tensor,
        n_steps: int | None = None,
        return_phase: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor, Tensor]:
        """Forward pass: encode → integrate → decode.

        Args:
            x: Input, shape (batch, input_dim).
            n_steps: Override default number of integration steps.
            return_phase: If True, also return (q, p) after integration.

        Returns:
            Output prediction, shape (batch, input_dim).
            If return_phase: (output, q_final, p_final).
        """
        steps = n_steps if n_steps is not None else self.n_steps
        q, p = self.encode(x)
        q, p = self.integrator.integrate(q, p, steps)
        out = self.decode(q)

        if return_phase:
            return out, q, p
        return out

    def evolve_sequence(
        self,
        x0: Tensor,
        seq_len: int,
        steps_per_output: int | None = None,
    ) -> Tensor:
        """Generate a sequence by repeated integration.

        Encodes x0, then alternates: integrate → decode → append.

        Args:
            x0: Initial input, shape (batch, input_dim).
            seq_len: Number of output time steps to produce.
            steps_per_output: Verlet steps between each output.

        Returns:
            Sequence of shape (batch, seq_len, input_dim).
        """
        spo = steps_per_output if steps_per_output is not None else self.n_steps
        q, p = self.encode(x0)
        outputs: list[Tensor] = []

        for _ in range(seq_len):
            q, p = self.integrator.integrate(q, p, spo)
            outputs.append(self.decode(q))

        return torch.stack(outputs, dim=1)

    def set_gamma(self, gamma: float) -> None:
        """Set dissipation coefficient (e.g. 0 for training, >0 for inference)."""
        self.integrator.gamma = gamma
