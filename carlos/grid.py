"""Grid schedule and adaptive transitions (Sec. 4.5-4.6, Eq. 32)."""

from __future__ import annotations

import math

import numpy as np

from carlos.config import B1Config


def should_advance_grid(
    prev_rewards: np.ndarray,
    curr_rewards: np.ndarray,
    alpha: float = 0.05,
) -> bool:
    """
    Eq. 32: advance when 0 is in 95% CI for mean reward difference.
    Uses Delta-method style split for zero vs non-zero differences (Eq. 35).
    """
    d = curr_rewards - prev_rewards
    v = len(d)
    if v == 0:
        return True

    nz = d[d != 0]
    p_hat = len(nz) / v
    mean_bar = float(np.mean(d))

    if len(nz) == 0:
        se = math.sqrt(p_hat * (1 - p_hat) * (0.0**2) / v) if v > 0 else 0.0
    else:
        mu_nz = float(np.mean(nz))
        var_nz = float(np.var(nz, ddof=1)) if len(nz) > 1 else 0.0
        se = math.sqrt(
            (p_hat / v) * var_nz + (p_hat * (1 - p_hat) / v) * (mu_nz**2)
        )

    z = 1.96 if alpha == 0.05 else 1.0
    lo = mean_bar - z * se
    hi = mean_bar + z * se
    return lo <= 0.0 <= hi


def build_validation_paths(cfg: B1Config, seed: int = 999) -> np.ndarray:
    from carlos.simulator import make_simulator

    max_level = cfg.max_grid_levels()
    steps = cfg.grid_steps_for_level(max_level)
    sim = make_simulator(cfg, num_paths=cfg.val_paths, seed=seed, num_steps=steps)
    sim.run()
    return np.array(sim.paths(0))
