"""Algorithm 3 delayed payoff evaluation (Sec. 4.3, Eq. 29-31)."""

from __future__ import annotations

import numpy as np

from carlos.config import CarlosConfig
from carlos.contracts import payoff_np
from carlos.inference import stop_batch
from carlos.model import ADNN


def delta_wait(t_init: float, cfg: CarlosConfig, grid_level: int) -> float:
    dt_tr = cfg.dt_for_level(grid_level)
    return (cfg.c_dlst ** (grid_level + 1)) * dt_tr * (1.0 - t_init / cfg.T)


def _path_x(path: np.ndarray, step: int) -> np.ndarray:
    if path.ndim == 1:
        return np.array([path[step]], dtype=np.float64)
    return np.asarray(path[step], dtype=np.float64)


def _scalar_x(x: np.ndarray) -> float | np.ndarray:
    if x.size == 1:
        return float(x[0])
    return x


def compute_delayed_payoff(
    model: ADNN,
    path: np.ndarray,
    t_init: float,
    cfg: CarlosConfig,
    grid_level: int,
) -> float:
    n_cols = path.shape[0]
    dt = cfg.dt_for_level(grid_level)
    times = t_init + np.arange(n_cols) * dt
    wait = delta_wait(t_init, cfg, grid_level)

    x0 = _path_x(path, 0)
    phi0 = bool(stop_batch(model, np.array([t_init]), x0.reshape(1, -1), cfg)[0])
    z = 0 if phi0 else 1

    for s in range(1, n_cols):
        t_s = times[s]
        x_s = _path_x(path, s)
        phi_s = bool(stop_batch(model, np.array([t_s]), x_s.reshape(1, -1), cfg)[0])
        h_s = float(payoff_np(_scalar_x(x_s), cfg))

        z_new = z
        if z == 1 and phi_s:
            z_new = 2
        elif z == 0 and not phi_s:
            z_new = 1

        if z_new == 2 or (z == 0 and (t_s - t_init) >= wait):
            return float(np.exp(-cfg.r * (t_s - t_init)) * h_s)
        z = z_new

    t_T = times[-1]
    h_T = float(payoff_np(_scalar_x(_path_x(path, -1)), cfg))
    return float(np.exp(-cfg.r * (t_T - t_init)) * h_T)


def compute_timing_target(
    model: ADNN,
    path: np.ndarray,
    t_init: float,
    cfg: CarlosConfig,
    grid_level: int,
) -> float:
    x0 = _path_x(path, 0)
    h0 = float(payoff_np(_scalar_x(x0), cfg))
    y_dpf = compute_delayed_payoff(model, path, t_init, cfg, grid_level)
    return y_dpf - h0


def compute_timing_targets_for_paths(
    model: ADNN,
    paths_list: list[np.ndarray],
    t_inits: np.ndarray,
    cfg: CarlosConfig,
    grid_level: int,
) -> np.ndarray:
    """
    Algorithm 3 targets for simulate_from paths (possibly different lengths).

    Groups paths by column count and runs the vectorized fast path per group.
    Never zero-pads; never shares the model across threads.
    """
    m = len(paths_list)
    if m == 0:
        return np.zeros(0, dtype=np.float64)

    out = np.zeros(m, dtype=np.float64)
    by_len: dict[int, list[int]] = {}
    for i, path in enumerate(paths_list):
        by_len.setdefault(path.shape[0], []).append(i)

    for indices in by_len.values():
        idx = np.asarray(indices, dtype=np.intp)
        stacked = np.stack([paths_list[i] for i in indices], axis=0)
        group_t = np.asarray(t_inits, dtype=np.float64)[idx]
        out[idx] = compute_timing_targets_batch_fast(
            model, stacked, group_t, cfg, grid_level
        )
    return out


def compute_timing_targets_batch(
    model: ADNN,
    paths: np.ndarray,
    t_inits: np.ndarray,
    cfg: CarlosConfig,
    grid_level: int,
) -> np.ndarray:
    """Batch Algorithm 3 for a uniform (M, S+1) or (M, S+1, d) path array."""
    try:
        return compute_timing_targets_batch_fast(model, paths, t_inits, cfg, grid_level)
    except (ValueError, IndexError):
        pass
    m = paths.shape[0]
    out = np.zeros(m, dtype=np.float64)
    for i in range(m):
        path_i = paths[i] if paths.ndim == 2 else paths[i]
        out[i] = compute_timing_target(model, path_i, float(t_inits[i]), cfg, grid_level)
    return out


def compute_timing_targets_batch_fast(
    model: ADNN,
    paths: np.ndarray,
    t_inits: np.ndarray,
    cfg: CarlosConfig,
    grid_level: int,
) -> np.ndarray:
    """
    Vectorized delayed payoff where paths share column count.
    Batches stop_batch calls per timestep across paths.
    """
    m, n_cols = paths.shape[0], paths.shape[1]
    dt = cfg.dt_for_level(grid_level)
    waits = np.array([delta_wait(float(t), cfg, grid_level) for t in t_inits])
    times = t_inits[:, None] + np.arange(n_cols)[None, :] * dt

    active = np.ones(m, dtype=bool)
    z = np.zeros(m, dtype=np.int32)
    y_dpf = np.zeros(m, dtype=np.float64)
    done = np.zeros(m, dtype=bool)

    x0 = paths[:, 0] if paths.ndim == 2 else paths[:, 0, :]
    phi0 = stop_batch(model, t_inits, x0, cfg)
    z[~phi0] = 1
    h0 = payoff_np(x0 if paths.ndim == 3 else paths[:, 0], cfg)
    if h0.ndim == 0:
        h0 = np.full(m, float(h0))

    for s in range(1, n_cols):
        if not np.any(active):
            break
        idx = np.where(active)[0]
        t_s = times[idx, s]
        x_s = paths[idx, s] if paths.ndim == 2 else paths[idx, s, :]
        phi_s = stop_batch(model, t_s, x_s, cfg)
        h_s = payoff_np(x_s, cfg)

        z_new = z[idx].copy()
        cont = z[idx] == 1
        stop = z[idx] == 0
        z_new[cont & phi_s] = 2
        z_new[stop & ~phi_s] = 1

        wait_hit = (z[idx] == 0) & ((times[idx, s] - t_inits[idx]) >= waits[idx])
        red = z_new == 2
        finish = red | wait_hit
        if np.any(finish):
            fin_idx = idx[finish]
            y_dpf[fin_idx] = np.exp(-cfg.r * (times[fin_idx, s] - t_inits[fin_idx])) * h_s[finish]
            done[fin_idx] = True
            active[fin_idx] = False
        z[idx] = z_new

    tail = ~done
    if np.any(tail):
        t_T = times[tail, -1]
        x_T = paths[tail, -1] if paths.ndim == 2 else paths[tail, -1, :]
        h_T = payoff_np(x_T, cfg)
        y_dpf[tail] = np.exp(-cfg.r * (t_T - t_inits[tail])) * h_T

    return y_dpf - h0
