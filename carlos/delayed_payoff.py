"""Algorithm 3 delayed payoff evaluation (Sec. 4.3, Eq. 29-31)."""

from __future__ import annotations

import numpy as np
import torch

from carlos.config import B1Config
from carlos.model import ADNN
from carlos.payoffs import basket_put_np


def delta_wait(t_init: float, cfg: B1Config, grid_level: int) -> float:
    """Eq. 31: exploration window length."""
    dt_tr = cfg.dt_for_level(grid_level)
    return (cfg.c_dlst ** (grid_level + 1)) * dt_tr * (1.0 - t_init / cfg.T)


def _phi(model: ADNN, t: float, x: float, cfg: B1Config) -> int:
    """Return 1 if stop, 0 if continue (Eq. 11)."""
    device = torch.device("cpu")
    h = float(basket_put_np(np.array([x]), cfg.strike, cfg.dim)[0])
    if t >= cfg.T - 1e-12:
        return 1
    state = torch.tensor([[[t, x]]], dtype=torch.float32, device=device)
    payoff = torch.tensor([h], dtype=torch.float32, device=device)
    t_tensor = torch.tensor([t], dtype=torch.float32, device=device)
    with torch.no_grad():
        stop = model.stop(state, payoff, t_tensor, cfg.T).item()
    return 1 if stop else 0


def compute_delayed_payoff(
    model: ADNN,
    path: np.ndarray,
    t_init: float,
    cfg: B1Config,
    grid_level: int,
) -> float:
    """
    Algorithm 3: traffic-light delayed stopping along a single path.
    Returns y_dpf (discounted payoff), not timing value.
    """
    n_cols = path.shape[0]
    dt = cfg.dt_for_level(grid_level)
    times = t_init + np.arange(n_cols) * dt
    wait = delta_wait(t_init, cfg, grid_level)

    x0 = float(path[0])
    phi0 = _phi(model, t_init, x0, cfg)
    z = 0 if phi0 == 1 else 1  # yellow=0 in stopping region, green=1 in continuation

    for s in range(1, n_cols):
        t_s = times[s]
        x_s = float(path[s])
        phi_s = _phi(model, t_s, x_s, cfg)
        h_s = float(basket_put_np(np.array([x_s]), cfg.strike, cfg.dim)[0])

        z_new = z
        if z == 1 and phi_s == 1:
            z_new = 2  # red: stop now
        elif z == 0 and phi_s == 0:
            z_new = 1  # entered continuation

        if z_new == 2 or (z == 0 and (t_s - t_init) >= wait):
            return float(np.exp(-cfg.r * (t_s - t_init)) * h_s)

        z = z_new

    t_T = times[-1]
    h_T = float(basket_put_np(np.array([path[-1]]), cfg.strike, cfg.dim)[0])
    return float(np.exp(-cfg.r * (t_T - t_init)) * h_T)


def compute_timing_target(
    model: ADNN,
    path: np.ndarray,
    t_init: float,
    cfg: B1Config,
    grid_level: int,
) -> float:
    """Eq. 30: y = y_dpf - h(x_t)."""
    x0 = float(path[0])
    h0 = float(basket_put_np(np.array([x0]), cfg.strike, cfg.dim)[0])
    y_dpf = compute_delayed_payoff(model, path, t_init, cfg, grid_level)
    return y_dpf - h0
