"""Tests for Algorithm 3 delayed stopping."""

from __future__ import annotations

import numpy as np
import torch

from carlos.config import CarlosConfig
from carlos.delayed_payoff import (
    compute_delayed_payoff,
    compute_timing_target,
    compute_timing_targets_for_paths,
    delta_wait,
)
from carlos.model import ADNN
from carlos.simulator import make_simulator, paths_tensor, x_init_vector


def test_delta_wait_positive() -> None:
    cfg = CarlosConfig()
    w = delta_wait(0.0, cfg, grid_level=0)
    assert w > 0.0


def test_delayed_payoff_finite() -> None:
    cfg = CarlosConfig()
    model = ADNN(cfg.dim)
    path = np.linspace(36.0, 30.0, cfg.num_steps + 1)
    y = compute_delayed_payoff(model, path, t_init=0.0, cfg=cfg, grid_level=0)
    assert np.isfinite(y)
    assert y >= 0.0


def test_mixed_length_paths_match_per_path_targets() -> None:
    """simulate_from paths differ in length when t_init > 0; must not zero-pad."""
    cfg = CarlosConfig()
    model = ADNN(cfg.dim)
    torch.manual_seed(0)
    steps = cfg.grid_steps_for_level(0)

    paths_list: list[np.ndarray] = []
    t_inits: list[float] = []
    for t_i in (0.0, 0.25, 0.5, 0.75):
        sim = make_simulator(cfg, num_paths=1, seed=100 + int(t_i * 100), num_steps=steps)
        sim.simulate_from(t_i, x_init_vector(cfg, cfg.x0), 1, 200 + int(t_i * 100))
        paths_list.append(paths_tensor(sim)[0])
        t_inits.append(t_i)

    assert len({p.shape[0] for p in paths_list}) > 1

    expected = np.array(
        [
            compute_timing_target(model, path, t, cfg, grid_level=0)
            for path, t in zip(paths_list, t_inits)
        ]
    )
    actual = compute_timing_targets_for_paths(
        model, paths_list, np.asarray(t_inits), cfg, grid_level=0
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)
