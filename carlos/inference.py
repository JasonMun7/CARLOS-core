"""Batched ADNN inference helpers."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from carlos.config import CarlosConfig
from carlos.contracts import payoff_torch, state_from_tx
from carlos.device import get_device
from carlos.model import ADNN


def _as_2d_x(x_arr: np.ndarray, dim: int) -> np.ndarray:
    x = np.asarray(x_arr, dtype=np.float64)
    if x.ndim == 0:
        return np.full((1, dim), float(x), dtype=np.float64)
    if x.ndim == 1:
        if x.shape[0] == dim:
            return x.reshape(1, dim)
        if dim == 1:
            return x.reshape(-1, 1)
        raise ValueError(f"expected vector length {dim}, got {x.shape[0]}")
    if x.shape[-1] != dim:
        raise ValueError(f"expected last dim {dim}, got {x.shape[-1]}")
    return x


def build_state_batch(
    t_arr: np.ndarray,
    x_arr: np.ndarray,
    cfg: CarlosConfig,
    device: torch.device | None = None,
) -> Tensor:
    """Build (B, d+1) state tensor from time and asset arrays."""
    device = device or get_device()
    t_arr = np.asarray(t_arr, dtype=np.float64).reshape(-1)
    x_2d = _as_2d_x(x_arr, cfg.dim)
    if t_arr.shape[0] == 1 and x_2d.shape[0] > 1:
        t_arr = np.full(x_2d.shape[0], t_arr[0], dtype=np.float64)
    elif t_arr.shape[0] != x_2d.shape[0]:
        raise ValueError("t_arr and x_arr batch size mismatch")
    rows = [state_from_tx(float(t), x_2d[i], cfg.dim) for i, t in enumerate(t_arr)]
    return torch.tensor(np.stack(rows), dtype=torch.float32, device=device)


@torch.no_grad()
def payoff_batch_torch(states: Tensor, cfg: CarlosConfig) -> Tensor:
    """Payoff h(x) for states (B, d+1)."""
    x = states[:, 1:]
    return payoff_torch(x, cfg)


def model_device(model: ADNN) -> torch.device:
    return next(model.parameters()).device


@torch.no_grad()
def stop_batch(
    model: ADNN,
    t_arr: np.ndarray,
    x_arr: np.ndarray,
    cfg: CarlosConfig,
    device: torch.device | None = None,
) -> np.ndarray:
    """Batch Eq. 11 stop flags; returns bool array (B,)."""
    device = device or model_device(model)
    model.eval()
    states = build_state_batch(t_arr, x_arr, cfg, device)
    payoffs = payoff_batch_torch(states, cfg)
    t_tensor = torch.tensor(np.asarray(t_arr, dtype=np.float32).reshape(-1), device=device)
    if t_tensor.numel() == 1 and states.shape[0] > 1:
        t_tensor = t_tensor.expand(states.shape[0])
    stops = model.stop(states, payoffs, t_tensor, cfg.T)
    return stops.cpu().numpy().astype(bool)


@torch.no_grad()
def forward_timing_batch(
    model: ADNN,
    t_arr: np.ndarray,
    x_arr: np.ndarray,
    cfg: CarlosConfig,
    device: torch.device | None = None,
) -> np.ndarray:
    device = device or model_device(model)
    model.eval()
    states = build_state_batch(t_arr, x_arr, cfg, device)
    return model(states).squeeze(-1).cpu().numpy()
