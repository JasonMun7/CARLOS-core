"""Batch target generation for RL loop."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from carlos.config import CarlosConfig
from carlos.delayed_payoff import compute_timing_target, compute_timing_targets_for_paths
from carlos.model import ADNN
from carlos.sampling import SampleBatch
from carlos.simulator import make_simulator, paths_tensor, x_init_vector


def _simulate_one(
    t_i: float,
    x_i: np.ndarray,
    seed: int,
    steps: int,
    cfg: CarlosConfig,
) -> np.ndarray:
    sim = make_simulator(cfg, num_paths=1, seed=seed, num_steps=steps)
    sim.simulate_from(t_i, x_init_vector(cfg, x_i), 1, seed)
    return paths_tensor(sim)[0]


def _simulate_paths_parallel(
    batch: SampleBatch,
    cfg: CarlosConfig,
    steps: int,
    seed: int,
    loop_count: int,
) -> tuple[list[np.ndarray], np.ndarray]:
    m = len(batch.t_inits)
    workers = cfg.target_workers or min(8, max(1, (os.cpu_count() or 4)))
    paths: list[np.ndarray | None] = [None] * m
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = []
        for i in range(m):
            futs.append(
                (
                    i,
                    pool.submit(
                        _simulate_one,
                        float(batch.t_inits[i]),
                        np.asarray(batch.x_inits[i]),
                        seed + loop_count * 1000 + i,
                        steps,
                        cfg,
                    ),
                )
            )
        for i, fut in futs:
            paths[i] = fut.result()
    assert all(p is not None for p in paths)
    return paths, batch.t_inits.astype(np.float64)


def compute_training_targets(
    model: ADNN,
    batch: SampleBatch,
    cfg: CarlosConfig,
    level: int,
    steps: int,
    seed: int,
    loop_count: int,
    sim=None,  # noqa: ARG001 — kept for API compatibility
) -> np.ndarray:
    """Simulate paths in parallel, then length-grouped batched Algorithm 3 targets."""
    paths_list, t_inits = _simulate_paths_parallel(batch, cfg, steps, seed, loop_count)
    return compute_timing_targets_for_paths(model, paths_list, t_inits, cfg, level)


def compute_training_targets_sequential(
    model: ADNN,
    batch: SampleBatch,
    cfg: CarlosConfig,
    level: int,
    steps: int,
    seed: int,
    loop_count: int,
    sim,
) -> np.ndarray:
    """Legacy sequential path for tests."""
    m = len(batch.t_inits)
    targets = np.zeros(m, dtype=np.float64)
    for i in range(m):
        t_i = float(batch.t_inits[i])
        x_i = np.asarray(batch.x_inits[i])
        sim.simulate_from(t_i, x_init_vector(cfg, x_i), 1, seed + loop_count * 1000 + i)
        path = paths_tensor(sim)[0]
        targets[i] = compute_timing_target(model, path, t_i, cfg, level)
    return targets
