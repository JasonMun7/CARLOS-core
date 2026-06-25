"""Algorithm 2 anchor-set sampling (Sec. 4.2, Eq. 20)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from carlos.config import CarlosConfig
from carlos.contracts import payoff_np, sample_uniform_state, state_from_tx
from carlos.inference import stop_batch
from carlos.model import ADNN
from carlos.simulator import make_simulator, paths_tensor


@dataclass
class SampleBatch:
    states: np.ndarray  # (M, d+1)
    t_inits: np.ndarray
    x_inits: np.ndarray  # (M,) for d=1 or (M, d)


def sample_training_inputs(
    model: ADNN,
    cfg: CarlosConfig,
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
    paths = paths_tensor(sim)

    exl_pts: list[tuple[float, np.ndarray]] = []
    plus_pts: list[tuple[float, np.ndarray]] = []
    minus_pts: list[tuple[float, np.ndarray]] = []
    ter_pts: list[tuple[float, np.ndarray]] = [
        (cfg.T, paths[p, -1].copy()) for p in range(num_pilot_paths)
    ]

    for p in range(num_pilot_paths):
        t_all: list[float] = []
        x_all: list[np.ndarray] = []
        for s in range(steps + 1):
            t_s = s * dt
            x_s = paths[p, s] if paths.ndim == 2 else paths[p, s, :]
            t_all.append(t_s)
            x_all.append(np.asarray(x_s, dtype=np.float64))

        h_all = payoff_np(np.stack(x_all), cfg)
        for s in range(steps + 1):
            if h_all[s] > 0:
                exl_pts.append((t_all[s], x_all[s]))

        if steps >= 1:
            t_arr = np.array(t_all)
            x_arr = np.stack(x_all)
            stops = stop_batch(model, t_arr, x_arr, cfg)
            for s in range(1, steps + 1):
                if stops[s - 1] != stops[s]:
                    if stops[s]:
                        minus_pts.append((t_all[s], x_all[s]))
                    else:
                        plus_pts.append((t_all[s - 1], x_all[s - 1]))

    rng = np.random.default_rng(seed)

    def pick(pool: list[tuple[float, np.ndarray]], n: int) -> list[tuple[float, np.ndarray]]:
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
        selected.append(sample_uniform_state(rng, cfg))

    selected = selected[:m]
    states = np.stack([state_from_tx(t, x, cfg.dim) for t, x in selected])
    t_inits = np.array([t for t, _ in selected], dtype=np.float64)
    if cfg.dim == 1:
        x_inits = np.array([float(np.asarray(x).reshape(-1)[0]) for _, x in selected])
    else:
        x_inits = np.stack([np.asarray(x, dtype=np.float64).reshape(cfg.dim) for _, x in selected])
    return SampleBatch(states=states, t_inits=t_inits, x_inits=x_inits)
