"""C++ GBM simulator loader."""

from __future__ import annotations

import sys
from pathlib import Path

from carlos.config import B1Config

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


def make_simulator(cfg: B1Config, num_paths: int, seed: int, num_steps: int | None = None):
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
