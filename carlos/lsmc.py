"""Backward Longstaff-Schwartz Monte Carlo (Sec. 2.1, Eq. 8-10)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from carlos.config import CarlosConfig
from carlos.contracts import payoff_np


def polynomial_basis(x: np.ndarray, dim: int) -> np.ndarray:
    """Regression basis: (1, x_1..x_d, x_1^2..x_d^2) for multi-dim LSMC."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        if x.shape[0] == dim:
            x = x.reshape(1, dim)
        else:
            x = x.reshape(-1, 1)
    if dim == 1:
        col = x[:, 0]
        return np.column_stack([np.ones_like(col), col, col**2])
    cols = [np.ones(x.shape[0])]
    for i in range(dim):
        cols.append(x[:, i])
    for i in range(dim):
        cols.append(x[:, i] ** 2)
    return np.column_stack(cols)


def path_assets(paths: np.ndarray, step: int) -> np.ndarray:
    if paths.ndim == 2:
        return paths[:, step]
    return paths[:, step, :]


def payoff_paths(paths: np.ndarray, cfg: CarlosConfig) -> np.ndarray:
    if paths.ndim == 1:
        return payoff_np(paths, cfg)
    if paths.ndim == 2 and paths.shape[1] == cfg.dim:
        return payoff_np(paths, cfg)
    return payoff_np(path_assets(paths, -1 if paths.ndim == 2 else 0), cfg)


@dataclass
class LSMCResult:
    states: np.ndarray
    targets: np.ndarray


def build_training_set(paths: np.ndarray, cfg: CarlosConfig) -> LSMCResult:
    k_paths = paths.shape[0]
    n_plus = paths.shape[1]
    n_steps = n_plus - 1
    dt = cfg.T / n_steps

    cashflow = payoff_np(path_assets(paths, n_steps), cfg)
    exercise_step = np.full(k_paths, n_steps, dtype=np.int32)

    state_rows: list[list[float]] = []
    target_rows: list[float] = []

    for n in range(n_steps - 1, -1, -1):
        x_n = path_assets(paths, n)
        h_n = payoff_np(x_n, cfg)
        hold_steps = exercise_step - n
        discounted = cashflow * np.exp(-cfg.r * dt * hold_steps)

        itm = h_n > 1e-12
        min_samples = max(cfg.dim + 2, 4)
        if np.count_nonzero(itm) >= min_samples:
            x_itm = x_n[itm]
            y_itm = discounted[itm]
            X = polynomial_basis(x_itm, cfg.dim)
            coeffs, _, _, _ = np.linalg.lstsq(X, y_itm, rcond=None)
            continuation = X @ coeffs

            exercise = np.zeros(k_paths, dtype=bool)
            exercise[itm] = h_n[itm] >= continuation

            cashflow[exercise] = h_n[exercise]
            exercise_step[exercise] = n

            t_n = n * dt
            timing = continuation - h_n[itm]
            for xi, yi in zip(x_itm, timing):
                if cfg.dim == 1:
                    state_rows.append([t_n, float(np.asarray(xi).reshape(-1)[0])])
                else:
                    state_rows.append([t_n, *list(np.asarray(xi, dtype=np.float64).reshape(-1))])
                target_rows.append(float(yi))
        else:
            exercise = itm & (h_n >= discounted)
            cashflow[exercise] = h_n[exercise]
            exercise_step[exercise] = n

    return LSMCResult(states=np.asarray(state_rows, dtype=np.float64), targets=np.asarray(target_rows, dtype=np.float64))


def lsmc_price(paths: np.ndarray, cfg: CarlosConfig) -> float:
    k_paths = paths.shape[0]
    n_plus = paths.shape[1]
    n_steps = n_plus - 1
    dt = cfg.T / n_steps

    cashflow = payoff_np(path_assets(paths, n_steps), cfg)
    exercise_step = np.full(k_paths, n_steps, dtype=np.int32)

    for n in range(n_steps - 1, -1, -1):
        x_n = path_assets(paths, n)
        h_n = payoff_np(x_n, cfg)
        hold_steps = exercise_step - n
        discounted = cashflow * np.exp(-cfg.r * dt * hold_steps)

        itm = h_n > 1e-12
        min_samples = max(cfg.dim + 2, 4)
        if np.count_nonzero(itm) >= min_samples:
            X = polynomial_basis(x_n[itm], cfg.dim)
            coeffs, _, _, _ = np.linalg.lstsq(X, discounted[itm], rcond=None)
            continuation = X @ coeffs
            exercise = np.zeros(k_paths, dtype=bool)
            exercise[itm] = h_n[itm] >= continuation
        else:
            exercise = itm & (h_n >= discounted)

        cashflow[exercise] = h_n[exercise]
        exercise_step[exercise] = n

    t_ex = exercise_step * dt
    return float(np.mean(cashflow * np.exp(-cfg.r * t_ex)))
