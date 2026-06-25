"""Pricing and target utilities shared by agent and RL loop."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from carlos.config import CarlosConfig
from carlos.contracts import payoff_np, payoff_torch
from carlos.device import get_device
from carlos.inference import stop_batch
from carlos.model import ADNN
from carlos.simulator import make_simulator, paths_tensor
from carlos import ui


def compute_targets_fixed_k(
    model: ADNN,
    paths: np.ndarray,
    t_init: float,
    cfg: CarlosConfig,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """V1 fixed-k delay targets (legacy agent stub)."""
    num_paths = paths.shape[0]
    n_cols = paths.shape[1]
    dt = cfg.T / cfg.num_steps
    times = t_init + np.arange(n_cols) * dt

    states: list[Tensor] = []
    targets: list[float] = []

    model.eval()
    with torch.no_grad():
        for p in range(num_paths):
            path = paths[p]
            x_t = path[0] if path.ndim == 1 else path[0]
            h_t = float(payoff_np(x_t, cfg))

            state0 = torch.tensor([[t_init, float(x_t[0]) if np.ndim(x_t) else float(x_t)]], dtype=torch.float32, device=device)
            if cfg.dim > 1:
                row = [t_init] + list(np.asarray(x_t, dtype=np.float64).reshape(-1))
                state0 = torch.tensor([row], dtype=torch.float32, device=device)
            payoff0 = torch.tensor([h_t], dtype=torch.float32, device=device)
            t0 = torch.tensor([t_init], dtype=torch.float32, device=device)
            would_stop = model.stop(state0, payoff0, t0, cfg.T).item()

            if would_stop and h_t > 0:
                stop_idx = min(cfg.delay_k, n_cols - 1)
                t_stop = times[stop_idx]
                x_stop = path[stop_idx]
                h_stop = float(payoff_np(x_stop, cfg))
                y_dpf = np.exp(-cfg.r * (t_stop - t_init)) * h_stop
                y = y_dpf - h_t
            else:
                for s in range(1, n_cols):
                    t_s = times[s]
                    x_s = path[s]
                    h_s = float(payoff_np(x_s, cfg))
                    if cfg.dim > 1:
                        state_s = torch.tensor([[t_s, *list(np.asarray(x_s))]], dtype=torch.float32, device=device)
                    else:
                        state_s = torch.tensor([[t_s, float(x_s)]], dtype=torch.float32, device=device)
                    payoff_s = torch.tensor([h_s], dtype=torch.float32, device=device)
                    t_tensor = torch.tensor([t_s], dtype=torch.float32, device=device)
                    if model.stop(state_s, payoff_s, t_tensor, cfg.T).item():
                        t_stop = t_s
                        h_stop = h_s
                        break
                else:
                    t_stop = times[-1]
                    h_stop = float(payoff_np(path[-1], cfg))

                y_dpf = np.exp(-cfg.r * (t_stop - t_init)) * h_stop
                y = y_dpf - h_t

            if cfg.dim > 1:
                states.append(torch.tensor([t_init, *list(np.asarray(x_t).reshape(-1))], dtype=torch.float32))
            else:
                states.append(torch.tensor([t_init, float(x_t)], dtype=torch.float32))
            targets.append(y)

    state_batch = torch.stack(states).to(device)
    target_batch = torch.tensor(targets, dtype=torch.float32, device=device).unsqueeze(-1)
    return state_batch, target_batch


@torch.no_grad()
def validate_price(
    model: ADNN,
    cfg: CarlosConfig | None = None,
    *,
    paths: np.ndarray | None = None,
    num_steps: int | None = None,
    show_target: bool | None = None,
    seed: int = 123,
) -> float:
    cfg = cfg or CarlosConfig()
    if show_target is None:
        show_target = not cfg.dev_mode

    steps = num_steps if num_steps is not None else cfg.num_steps

    if paths is not None:
        price = forward_reward_on_paths(model, paths, cfg, steps)
    else:
        sim = make_simulator(cfg, num_paths=cfg.val_paths, seed=seed, num_steps=steps)
        sim.run()
        sim_paths = paths_tensor(sim)
        price = forward_reward_on_paths(model, sim_paths, cfg, steps)

    if show_target:
        ui.validation_price(price, cfg.target_price)
    else:
        ui.validation_price(price)
    return price


@torch.no_grad()
def forward_rewards_per_path(
    model: ADNN,
    paths: np.ndarray,
    cfg: CarlosConfig,
    num_steps: int | None = None,
) -> np.ndarray:
    """Per-path discounted payoffs using Eq. 11 (batched by timestep)."""
    model.eval()
    steps = num_steps if num_steps is not None else cfg.num_steps
    dt = cfg.T / steps
    k_paths = paths.shape[0]

    if paths.ndim == 2 and paths.shape[1] == steps + 1:
        path_steps = steps
    elif paths.ndim == 3 and paths.shape[1] == steps + 1:
        path_steps = steps
    else:
        path_steps = paths.shape[1] - 1

    alive = np.ones(k_paths, dtype=bool)
    t_stop = np.full(k_paths, cfg.T, dtype=np.float64)
    payoff = np.zeros(k_paths, dtype=np.float64)

    for s in range(path_steps + 1):
        if not np.any(alive):
            break
        idx = np.where(alive)[0]
        t_s = s * dt
        x_s = paths[idx, s] if paths.ndim == 2 else paths[idx, s, :]
        stops = stop_batch(model, np.full(len(idx), t_s), x_s, cfg)
        h_s = payoff_np(x_s, cfg)
        exercise = stops & (h_s > 0)
        if np.any(exercise):
            ex_idx = idx[exercise]
            alive[ex_idx] = False
            t_stop[ex_idx] = t_s
            payoff[ex_idx] = h_s[exercise]

    tail = alive
    if np.any(tail):
        x_T = paths[tail, -1] if paths.ndim == 2 else paths[tail, -1, :]
        payoff[tail] = payoff_np(x_T, cfg)
        t_stop[tail] = cfg.T

    return np.exp(-cfg.r * t_stop) * payoff


@torch.no_grad()
def forward_reward_on_paths(
    model: ADNN,
    paths: np.ndarray,
    cfg: CarlosConfig,
    num_steps: int | None = None,
) -> float:
    rewards = forward_rewards_per_path(model, paths, cfg, num_steps)
    return float(np.mean(rewards))
