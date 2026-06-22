"""Payoff functions for CARLOS benchmark contracts (Eq. 16)."""

from __future__ import annotations

import torch
from torch import Tensor


def basket_put(x: Tensor, strike: float, dim: int) -> Tensor:
    """Arithmetic basket put: (K - (1/d) sum X^i)_+."""
    if x.dim() == 1:
        avg = x.sum() / dim
    else:
        avg = x.sum(dim=-1) / dim
    return torch.clamp(strike - avg, min=0.0)


def max_call(x: Tensor, strike: float) -> Tensor:
    """Max call: (max_i X^i - K)_+."""
    if x.dim() == 1:
        mx = x.max()
    else:
        mx = x.max(dim=-1).values
    return torch.clamp(mx - strike, min=0.0)
