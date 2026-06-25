"""C++ GBM simulator loader."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from carlos.config import CarlosConfig

ROOT = Path(__file__).resolve().parent.parent


def load_simulator_class():
    for path in [ROOT / "carlos", ROOT / "build"]:
        p = str(path)
        if p not in sys.path:
            sys.path.insert(0, p)
    for candidate in ROOT.glob("build/**/_carlos_sim*.so"):
        parent = str(candidate.parent)
        if parent not in sys.path:
            sys.path.insert(0, parent)
    from _carlos_sim import GBMSimulator

    return GBMSimulator


def make_simulator(cfg: CarlosConfig, num_paths: int, seed: int, num_steps: int | None = None):
    GBMSimulator = load_simulator_class()
    steps = num_steps if num_steps is not None else cfg.num_steps
    dt = cfg.T / steps
    return GBMSimulator(
        dim=cfg.dim,
        num_paths=num_paths,
        num_steps=steps,
        dt=dt,
        r=cfg.r,
        T=cfg.T,
        x0=cfg.x0s,
        delta=cfg.deltas,
        sigma=cfg.sigmas,
        seed=seed,
    )


def paths_tensor(sim, dim: int | None = None) -> np.ndarray:
    """
    Path array from simulator after run/simulate_from.

    d=1: (K, S+1)
    d>1: (K, S+1, d)
    """
    d = dim if dim is not None else int(sim.dim)
    if d == 1:
        return np.array(sim.paths(0))
    cols = [np.array(sim.paths(i)) for i in range(d)]
    return np.stack(cols, axis=-1)


def x_init_vector(cfg: CarlosConfig, x) -> list[float]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size == 1 and cfg.dim > 1:
        return [float(x[0])] * cfg.dim
    if x.size != cfg.dim:
        raise ValueError(f"expected dim {cfg.dim}, got {x.size}")
    return [float(v) for v in x]
