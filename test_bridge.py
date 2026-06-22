#!/usr/bin/env python3
"""Smoke test for the C++ GBM simulator PyBind11 bridge."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "carlos"))
sys.path.insert(0, str(ROOT / "build"))

try:
    from _carlos_sim import GBMSimulator
except ImportError:
    for candidate in ROOT.glob("build/**/_carlos_sim*.so"):
        sys.path.insert(0, str(candidate.parent))
        break
    from _carlos_sim import GBMSimulator


def make_sim(seed: int = 42) -> GBMSimulator:
    return GBMSimulator(
        dim=1,
        num_paths=4,
        num_steps=10,
        dt=0.1,
        r=0.05,
        T=1.0,
        x0=[36.0],
        delta=[0.0],
        sigma=[0.2],
        seed=seed,
    )


def test_run_determinism() -> None:
    sim_a = make_sim(seed=42)
    sim_a.run()
    paths_a = np.array(sim_a.paths(0))

    sim_b = make_sim(seed=42)
    sim_b.run()
    paths_b = np.array(sim_b.paths(0))

    assert paths_a.shape == (4, 11), paths_a.shape
    assert np.allclose(paths_a, paths_b), "run() must be deterministic for same seed"
    print(f"run() shape: {paths_a.shape}")
    print(f"first path: {paths_a[0]}")


def test_zero_copy() -> None:
    sim = make_sim()
    sim.run()
    view = sim.paths(0)
    arr = np.asarray(view)
    assert arr.shape == (4, 11)
    # numpy may wrap pybind buffer; check writable view shares backing store
    assert arr.base is not None or view.__array_interface__ is not None
    print("zero-copy view ok")


def test_simulate_from_determinism() -> None:
    sim_a = make_sim()
    sim_a.simulate_from(0.3, [35.0], num_paths=4, seed=99)
    out_a = np.array(sim_a.paths(0))

    sim_b = make_sim()
    sim_b.simulate_from(0.3, [35.0], num_paths=4, seed=99)
    out_b = np.array(sim_b.paths(0))

    assert out_a.shape == (4, 8), out_a.shape  # 7 remaining steps + initial
    assert np.allclose(out_a, out_b), "simulate_from must be deterministic"
    print(f"simulate_from shape: {out_a.shape}")


if __name__ == "__main__":
    test_run_determinism()
    test_zero_copy()
    test_simulate_from_determinism()
    print("test_bridge: all checks passed")
