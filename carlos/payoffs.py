"""Payoff functions for CARLOS benchmark contracts (Eq. 16)."""

from __future__ import annotations

from enum import Enum

import numpy as np
import torch
from torch import Tensor


class PayoffKind(str, Enum):
    BASKET_PUT = "basket_put"
    MAX_CALL = "max_call"


def basket_put(x: Tensor, strike: float, dim: int) -> Tensor:
    """Arithmetic basket put: (K - (1/d) sum X^i)_+."""
    if x.dim() == 1:
        avg = x.sum() / dim
    else:
        avg = x.sum(dim=-1) / dim
    return torch.clamp(strike - avg, min=0.0)


def basket_put_np(x: np.ndarray, strike: float, dim: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 0:
        return np.maximum(strike - x, 0.0)
    if x.ndim == 1:
        if dim == 1:
            return np.maximum(strike - x, 0.0)
        if x.shape[0] == dim:
            avg = x.sum() / dim
            return np.maximum(strike - avg, 0.0)
    avg = x.sum(axis=-1) / dim
    return np.maximum(strike - avg, 0.0)


def max_call(x: Tensor, strike: float) -> Tensor:
    """Max call: (max_i X^i - K)_+."""
    if x.dim() == 1:
        mx = x.max()
    else:
        mx = x.max(dim=-1).values
    return torch.clamp(mx - strike, min=0.0)


def max_call_np(x: np.ndarray, strike: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 0:
        return np.maximum(x - strike, 0.0)
    if x.ndim == 1:
        if x.shape[0] == 1:
            return np.maximum(x[0] - strike, 0.0)
        mx = x.max()
        return np.maximum(mx - strike, 0.0)
    mx = x.max(axis=-1)
    return np.maximum(mx - strike, 0.0)
