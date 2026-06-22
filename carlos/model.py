"""Aggregate Deep Neural Network (ADNN) for CARLOS timing-value regression."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


def hidden_width(dim: int) -> int:
    """Paper Sec. 4.7: q = max(30*d, 60)."""
    return max(30 * dim, 60)


class ADNN(nn.Module):
    """Feedforward regressor R^Theta: R^{d+1} -> R (Eq. 17)."""

    def __init__(self, dim: int, hidden: int | None = None, activation: str = "relu"):
        super().__init__()
        self.dim = dim
        h = hidden if hidden is not None else hidden_width(dim)
        act: nn.Module
        if activation == "tanh":
            act = nn.Tanh()
        else:
            act = nn.ReLU()

        self.net = nn.Sequential(
            nn.Linear(dim + 1, h),
            act,
            nn.Linear(h, h),
            act,
            nn.Linear(h, h),
            act,
            nn.Linear(h, 1),
        )

    def forward(self, state: Tensor) -> Tensor:
        """state: (B, d+1) -> (B, 1) timing value."""
        return self.net(state)

    def stop(self, state: Tensor, payoff: Tensor, t: Tensor, maturity: float) -> Tensor:
        """Paper Eq. 11: exercise when R^Theta <= 0 and h(x) > 0, or t >= T."""
        timing = self.forward(state).squeeze(-1)
        at_maturity = t >= maturity - 1e-12
        return ((timing <= 0) & (payoff > 0)) | at_maturity
