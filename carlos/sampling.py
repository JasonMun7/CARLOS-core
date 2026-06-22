"""Algorithm 2 anchor-set sampling (Sec. 4.2, Eq. 20)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from carlos.config import B1Config
from carlos.model import ADNN
from carlos.payoffs import basket_put_np
from carlos.simulator import make_simulator


@dataclass
class SampleBatch:
    states: np.ndarray  # (M, d+1)
    t_inits: np.ndarray
    x_inits: np.ndarray


def _state_from_tx(t: float, x: float, dim: int) -> np.ndarray:
    row = [t] + [x] * dim
    return np.asarray(row, dtype=np.float64)


def _phi(model: ADNN, t: float, x: float, cfg: B1Config) -> bool:
    device = torch.device("cpu")
    h = float(basket_put_np(np.array([x]), cfg.strike, cfg.dim)[0])
    state = torch.tensor([[[t, x]]], dtype=torch.float32, device=device)
    payoff = torch.tensor([h], dtype=torch.float32, device=device)
    t_tensor = torch.tensor([t], dtype=torch.float32, device=device)
    with torch.no_grad():
        return bool(model.stop(state, payoff, t_tensor, cfg.T).item())


def sample_training_inputs(
    model: ADNN,
    cfg: B1Config,
    grid_level: int,
    m: int,
    seed: int,
    num_pilot_paths: int = 256,
) -> SampleBatch:
    steps = cfg.grid_steps_for_level(grid_level)
    dt = cfg.dt_for_level(grid_level)
    weights = cfg.sampling_weights(grid_level)

    counts = {
        "exl": int(round(weights["exl"] * m)),
        "plus": int(round(weights["plus"] * m)),
        "minus": int(round(weights["minus"] * m)),
    }
    counts["ter"] = m - sum(counts.values())

    sim = make_simulator(cfg, num_paths=num_pilot_paths, seed=seed, num_steps=steps)
    sim.run()
    paths = np.array(sim.paths(0))

    exl_pts: list[tuple[float, float]] = []
    plus_pts: list[tuple[float, float]] = []
    minus_pts: list[tuple[float, float]] = []
    ter_pts: list[tuple[float, float]] = [(cfg.T, float(paths[p, -1])) for p in range(num_pilot_paths)]

    for p in range(num_pilot_paths):
        for s in range(steps + 1):
            t_s = s * dt
            x_s = float(paths[p, s])
            h_s = float(basket_put_np(np.array([x_s]), cfg.strike, cfg.dim)[0])
            if h_s > 0:
                exl_pts.append((t_s, x_s))

            if s >= 1:
                t_prev = (s - 1) * dt
                x_prev = float(paths[p, s - 1])
                phi_prev = _phi(model, t_prev, x_prev, cfg)
                phi_curr = _phi(model, t_s, x_s, cfg)
                if phi_prev != phi_curr:
                    if phi_curr:
                        minus_pts.append((t_s, x_s))
                    else:
                        plus_pts.append((t_prev, x_prev))

    rng = np.random.default_rng(seed)

    def pick(pool: list[tuple[float, float]], n: int) -> list[tuple[float, float]]:
        if not pool or n <= 0:
            return []
        idx = rng.integers(0, len(pool), size=n)
        return [pool[i] for i in idx]

    selected = (
        pick(exl_pts, counts["exl"])
        + pick(plus_pts, counts["plus"])
        + pick(minus_pts, counts["minus"])
        + pick(ter_pts, counts["ter"])
    )

    while len(selected) < m:
        selected.append((float(rng.uniform(0, cfg.T)), float(rng.uniform(cfg.x_min, cfg.x_max))))

    selected = selected[:m]
    states = np.stack([_state_from_tx(t, x, cfg.dim) for t, x in selected])
    t_inits = np.array([t for t, _ in selected])
    x_inits = np.array([x for _, x in selected])
    return SampleBatch(states=states, t_inits=t_inits, x_inits=x_inits)
