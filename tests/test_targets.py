"""Tests for RL target generation."""

from __future__ import annotations

import numpy as np
import torch

from carlos.config import CarlosConfig
from carlos.contracts import state_from_tx
from carlos.model import ADNN
from carlos.sampling import SampleBatch
from carlos.simulator import make_simulator
from carlos.targets import compute_training_targets, compute_training_targets_sequential


def test_training_targets_match_sequential_mixed_t_init() -> None:
    """Length-grouped batch targets must agree with fully sequential sim."""
    cfg = CarlosConfig(dev_mode=True)
    model = ADNN(cfg.dim)
    torch.manual_seed(0)

    t_inits = np.array([0.0, 0.2, 0.5, 0.8, 1.0])
    x_inits = np.full(len(t_inits), cfg.x0, dtype=np.float64)
    states = np.stack([state_from_tx(float(t), x, cfg.dim) for t, x in zip(t_inits, x_inits)])
    batch = SampleBatch(states=states, t_inits=t_inits, x_inits=x_inits)

    steps = cfg.grid_steps_for_level(0)
    sim = make_simulator(cfg, num_paths=1, seed=0, num_steps=steps)

    batched = compute_training_targets(model, batch, cfg, level=0, steps=steps, seed=7, loop_count=0)
    sequential = compute_training_targets_sequential(
        model, batch, cfg, level=0, steps=steps, seed=7, loop_count=0, sim=sim
    )
    np.testing.assert_allclose(batched, sequential, rtol=1e-10, atol=1e-10)


def test_length_grouped_targets_match_per_path() -> None:
    """Grouped batch_fast must match per-path compute_timing_target."""
    from carlos.delayed_payoff import compute_timing_target, compute_timing_targets_for_paths
    from carlos.simulator import make_simulator, paths_tensor, x_init_vector

    cfg = CarlosConfig()
    model = ADNN(cfg.dim)
    torch.manual_seed(0)
    steps = cfg.grid_steps_for_level(0)

    paths_list: list[np.ndarray] = []
    t_inits: list[float] = []
    for t_i in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9):
        sim = make_simulator(cfg, num_paths=1, seed=50 + int(t_i * 100), num_steps=steps)
        sim.simulate_from(t_i, x_init_vector(cfg, cfg.x0), 1, 60 + int(t_i * 100))
        paths_list.append(paths_tensor(sim)[0])
        t_inits.append(t_i)

    t_arr = np.asarray(t_inits)
    expected = np.array(
        [compute_timing_target(model, p, float(t), cfg, 0) for p, t in zip(paths_list, t_arr)]
    )
    grouped = compute_timing_targets_for_paths(model, paths_list, t_arr, cfg, grid_level=0)
    np.testing.assert_allclose(grouped, expected, rtol=1e-10, atol=1e-10)
