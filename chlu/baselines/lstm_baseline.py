"""LSTM baseline for temporal sequence prediction.

Standard LSTM encoder-decoder for comparison with CHLU.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class LSTMBaseline(nn.Module):
    """LSTM baseline model for sequence-to-sequence prediction.

    Args:
        input_dim: Feature dimension of input/output.
        hidden_dim: LSTM hidden state dimension.
        num_layers: Number of stacked LSTM layers.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.decoder = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for single-step prediction.

        Args:
            x: Input, shape (batch, input_dim).

        Returns:
            Prediction, shape (batch, input_dim).
        """
        # Treat as sequence of length 1
        x_seq = x.unsqueeze(1)  # (batch, 1, input_dim)
        out, _ = self.lstm(x_seq)
        return self.decoder(out[:, -1, :])

    def evolve_sequence(
        self,
        x0: Tensor,
        seq_len: int,
        steps_per_output: int | None = None,
    ) -> Tensor:
        """Generate a sequence autoregressively.

        Args:
            x0: Initial input, shape (batch, input_dim).
            seq_len: Number of time steps to generate.
            steps_per_output: Ignored (present for API compatibility).

        Returns:
            Sequence, shape (batch, seq_len, input_dim).
        """
        batch = x0.shape[0]
        device = x0.device
        h = torch.zeros(self.num_layers, batch, self.hidden_dim, device=device)
        c = torch.zeros(self.num_layers, batch, self.hidden_dim, device=device)

        current = x0.unsqueeze(1)  # (batch, 1, input_dim)
        outputs: list[Tensor] = []

        for _ in range(seq_len):
            out, (h, c) = self.lstm(current, (h, c))
            pred = self.decoder(out[:, -1, :])
            outputs.append(pred)
            current = pred.unsqueeze(1)

        return torch.stack(outputs, dim=1)
