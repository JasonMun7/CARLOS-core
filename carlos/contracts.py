"""Contract payoff dispatch for CARLOS benchmarks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor

from carlos.payoffs import PayoffKind, basket_put, basket_put_np, max_call, max_call_np

if TYPE_CHECKING:
    from carlos.config import CarlosConfig


def payoff_np(x: np.ndarray, cfg: CarlosConfig) -> np.ndarray:
    if cfg.payoff == PayoffKind.MAX_CALL:
        return max_call_np(x, cfg.strike)
    return basket_put_np(x, cfg.strike, cfg.dim)


def payoff_torch(x: Tensor, cfg: CarlosConfig) -> Tensor:
    if cfg.payoff == PayoffKind.MAX_CALL:
        return max_call(x, cfg.strike)
    return basket_put(x, cfg.strike, cfg.dim)


def default_sampling_box(cfg: CarlosConfig) -> tuple[list[float], list[float]]:
    if cfg.x_mins is not None and cfg.x_maxs is not None:
        return list(cfg.x_mins), list(cfg.x_maxs)
    return [cfg.x_min] * cfg.dim, [cfg.x_max] * cfg.dim


def sample_uniform_state(rng: np.random.Generator, cfg: CarlosConfig) -> tuple[float, np.ndarray]:
    mins, maxs = default_sampling_box(cfg)
    t = float(rng.uniform(0, cfg.T))
    x = np.array([float(rng.uniform(mins[i], maxs[i])) for i in range(cfg.dim)], dtype=np.float64)
    return t, x


def state_from_tx(t: float, x: np.ndarray, dim: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size == 1 and dim > 1:
        x = np.full(dim, x[0], dtype=np.float64)
    elif x.size != dim:
        raise ValueError(f"expected state dim {dim}, got {x.size}")
    return np.concatenate([[t], x])
