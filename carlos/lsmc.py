"""Backward Longstaff-Schwartz Monte Carlo (Sec. 2.1, Eq. 8-10)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from carlos.config import CarlosConfig
from carlos.payoffs import basket_put_np


def polynomial_basis(x: np.ndarray) -> np.ndarray:
    """Basis (1, x, x^2) for d=1."""
    x = np.asarray(x, dtype=np.float64)
    return np.column_stack([np.ones_like(x), x, x**2])


def payoff_paths(paths: np.ndarray, cfg: CarlosConfig) -> np.ndarray:
    return basket_put_np(paths, cfg.strike, cfg.dim)


@dataclass
class LSMCResult:
    states: np.ndarray  # (M, d+1) columns [t, x...]
    targets: np.ndarray  # (M,) timing values y = w - h(x)


def build_training_set(paths: np.ndarray, cfg: CarlosConfig) -> LSMCResult:
    """
    Backward LSMC; stack (t, x) -> timing value for Stage 1b ADNN init.

    paths: (K, N+1) asset values at each grid point.
    """
    k_paths, n_plus = paths.shape
    n_steps = n_plus - 1
    dt = cfg.T / n_steps

    cashflow = payoff_paths(paths[:, n_steps], cfg)
    exercise_step = np.full(k_paths, n_steps, dtype=np.int32)

    state_rows: list[list[float]] = []
    target_rows: list[float] = []

    for n in range(n_steps - 1, -1, -1):
        x_n = paths[:, n]
        h_n = payoff_paths(x_n, cfg)
        hold_steps = exercise_step - n
        discounted = cashflow * np.exp(-cfg.r * dt * hold_steps)

        itm = h_n > 1e-12
        if np.count_nonzero(itm) >= cfg.dim + 2:
            x_itm = x_n[itm]
            y_itm = discounted[itm]
            X = polynomial_basis(x_itm)
            coeffs, _, _, _ = np.linalg.lstsq(X, y_itm, rcond=None)
            continuation = X @ coeffs

            exercise = np.zeros(k_paths, dtype=bool)
            exercise[itm] = h_n[itm] >= continuation

            cashflow[exercise] = h_n[exercise]
            exercise_step[exercise] = n

            t_n = n * dt
            timing = continuation - h_n[itm]
            for xi, yi in zip(x_itm, timing):
                state_rows.append([t_n, float(xi)])
                target_rows.append(float(yi))
        else:
            exercise = itm & (h_n >= discounted)
            cashflow[exercise] = h_n[exercise]
            exercise_step[exercise] = n

    states = np.asarray(state_rows, dtype=np.float64)
    targets = np.asarray(target_rows, dtype=np.float64)
    return LSMCResult(states=states, targets=targets)


def lsmc_price(paths: np.ndarray, cfg: CarlosConfig) -> float:
    """Bermudan LSMC price from path matrix."""
    k_paths, n_plus = paths.shape
    n_steps = n_plus - 1
    dt = cfg.T / n_steps

    cashflow = payoff_paths(paths[:, n_steps], cfg)
    exercise_step = np.full(k_paths, n_steps, dtype=np.int32)

    for n in range(n_steps - 1, -1, -1):
        x_n = paths[:, n]
        h_n = payoff_paths(x_n, cfg)
        hold_steps = exercise_step - n
        discounted = cashflow * np.exp(-cfg.r * dt * hold_steps)

        itm = h_n > 1e-12
        if np.count_nonzero(itm) >= cfg.dim + 2:
            X = polynomial_basis(x_n[itm])
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
