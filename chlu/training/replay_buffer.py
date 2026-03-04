"""Replay buffer for sleep-phase hallucinated states.

Stores (q, p) states generated during the sleep phase
of Hamiltonian Contrastive Divergence.
"""

from __future__ import annotations

import torch
from torch import Tensor


class ReplayBuffer:
    """Fixed-capacity circular replay buffer for phase-space states.

    Args:
        capacity: Maximum number of states to store.
        dim: Dimensionality of q and p vectors.
    """

    def __init__(self, capacity: int, dim: int) -> None:
        self.capacity = capacity
        self.dim = dim
        self.q_buf = torch.zeros(capacity, dim)
        self.p_buf = torch.zeros(capacity, dim)
        self.size = 0
        self.ptr = 0

    def push(self, q: Tensor, p: Tensor) -> None:
        """Add states to the buffer.

        Args:
            q: Positions, shape (batch, dim).
            p: Momenta, shape (batch, dim).
        """
        q = q.detach().cpu()
        p = p.detach().cpu()
        batch_size = q.shape[0]

        for i in range(batch_size):
            self.q_buf[self.ptr] = q[i]
            self.p_buf[self.ptr] = p[i]
            self.ptr = (self.ptr + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device | None = None) -> tuple[Tensor, Tensor]:
        """Sample random states from the buffer.

        Args:
            batch_size: Number of states to sample.
            device: Device to place the tensors on.

        Returns:
            (q, p) each of shape (batch_size, dim).

        Raises:
            ValueError: If buffer has fewer states than requested.
        """
        if self.size < batch_size:
            raise ValueError(
                f"Buffer has {self.size} states, but {batch_size} requested."
            )
        indices = torch.randint(0, self.size, (batch_size,))
        q = self.q_buf[indices]
        p = self.p_buf[indices]
        if device is not None:
            q = q.to(device)
            p = p.to(device)
        return q, p

    def __len__(self) -> int:
        return self.size
