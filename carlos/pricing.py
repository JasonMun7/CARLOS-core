"""Pricing and target utilities shared by agent and RL loop."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from carlos.config import B1Config
from carlos.model import ADNN
from carlos.payoffs import basket_put
from carlos.simulator import make_simulator


def compute_targets_fixed_k(
    model: ADNN,
    paths: np.ndarray,
    t_init: float,
    cfg: B1Config,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """V1 fixed-k delay targets (Phase 3 stub; replaced by delayed_payoff in Phase 4)."""
    num_paths, n_cols = paths.shape
    dt = cfg.T / cfg.num_steps
    times = t_init + np.arange(n_cols) * dt

    states: list[Tensor] = []
    targets: list[float] = []

    model.eval()
    with torch.no_grad():
        for p in range(num_paths):
            path = paths[p]
            x_t = float(path[0])
            h_t = float(basket_put(torch.tensor([x_t]), cfg.strike, cfg.dim).item())

            state0 = torch.tensor([[t_init, x_t]], dtype=torch.float32, device=device)
            payoff0 = torch.tensor([h_t], dtype=torch.float32, device=device)
            t0 = torch.tensor([t_init], dtype=torch.float32, device=device)
            would_stop = model.stop(state0, payoff0, t0, cfg.T).item()

            if would_stop and h_t > 0:
                stop_idx = min(cfg.delay_k, n_cols - 1)
                t_stop = times[stop_idx]
                x_stop = float(path[stop_idx])
                h_stop = float(
                    basket_put(torch.tensor([x_stop]), cfg.strike, cfg.dim).item()
                )
                y_dpf = np.exp(-cfg.r * (t_stop - t_init)) * h_stop
                y = y_dpf - h_t
            else:
                for s in range(1, n_cols):
                    t_s = times[s]
                    x_s = float(path[s])
                    h_s = float(
                        basket_put(torch.tensor([x_s]), cfg.strike, cfg.dim).item()
                    )
                    state_s = torch.tensor([[t_s, x_s]], dtype=torch.float32, device=device)
                    payoff_s = torch.tensor([h_s], dtype=torch.float32, device=device)
                    t_tensor = torch.tensor([t_s], dtype=torch.float32, device=device)
                    if model.stop(state_s, payoff_s, t_tensor, cfg.T).item():
                        t_stop = t_s
                        h_stop = h_s
                        break
                else:
                    t_stop = times[-1]
                    h_stop = float(
                        basket_put(torch.tensor([path[-1]]), cfg.strike, cfg.dim).item()
                    )

                y_dpf = np.exp(-cfg.r * (t_stop - t_init)) * h_stop
                y = y_dpf - h_t

            states.append(torch.tensor([t_init, x_t], dtype=torch.float32))
            targets.append(y)

    state_batch = torch.stack(states).to(device)
    target_batch = torch.tensor(targets, dtype=torch.float32, device=device).unsqueeze(-1)
    return state_batch, target_batch


@torch.no_grad()
def validate_price(
    model: ADNN,
    cfg: B1Config | None = None,
    seed: int = 123,
    num_steps: int | None = None,
) -> float:
    """Forward MC price from (t=0, X=X0) using Eq. 11 stopping rule."""
    cfg = cfg or B1Config()
    device = torch.device("cpu")
    model = model.to(device)
    model.eval()

    steps = num_steps if num_steps is not None else cfg.num_steps
    sim = make_simulator(cfg, num_paths=cfg.val_paths, seed=seed, num_steps=steps)
    sim.run()
    paths = np.array(sim.paths(0))
    dt = cfg.T / steps

    total = 0.0
    for p in range(cfg.val_paths):
        t_stop = cfg.T
        payoff = 0.0
        for s in range(steps + 1):
            t_s = s * dt
            x_s = float(paths[p, s])
            h_s = float(basket_put(torch.tensor([x_s]), cfg.strike, cfg.dim).item())
            state = torch.tensor([[t_s, x_s]], dtype=torch.float32, device=device)
            payoff_t = torch.tensor([h_s], dtype=torch.float32, device=device)
            t_tensor = torch.tensor([t_s], dtype=torch.float32, device=device)
            if model.stop(state, payoff_t, t_tensor, cfg.T).item() and h_s > 0:
                t_stop = t_s
                payoff = h_s
                break
        else:
            x_T = float(paths[p, -1])
            payoff = float(basket_put(torch.tensor([x_T]), cfg.strike, cfg.dim).item())
            t_stop = cfg.T

        total += np.exp(-cfg.r * t_stop) * payoff

    price = total / cfg.val_paths
    print(f"validate_price: {price:.4f}  (Table 3 CARLOS target: {cfg.target_price})")
    return price


@torch.no_grad()
def forward_rewards_per_path(
    model: ADNN,
    paths: np.ndarray,
    cfg: B1Config,
    num_steps: int | None = None,
) -> np.ndarray:
    """Per-path discounted payoffs using Eq. 11."""
    device = torch.device("cpu")
    model.eval()
    steps = num_steps if num_steps is not None else cfg.num_steps
    dt = cfg.T / steps
    k_paths = paths.shape[0]
    rewards = np.zeros(k_paths, dtype=np.float64)

    for p in range(k_paths):
        t_stop = cfg.T
        payoff = 0.0
        for s in range(steps + 1):
            t_s = s * dt
            x_s = float(paths[p, s])
            h_s = float(basket_put(torch.tensor([x_s]), cfg.strike, cfg.dim).item())
            state = torch.tensor([[t_s, x_s]], dtype=torch.float32, device=device)
            payoff_t = torch.tensor([h_s], dtype=torch.float32, device=device)
            t_tensor = torch.tensor([t_s], dtype=torch.float32, device=device)
            if model.stop(state, payoff_t, t_tensor, cfg.T).item() and h_s > 0:
                t_stop = t_s
                payoff = h_s
                break
        else:
            payoff = float(basket_put(torch.tensor([paths[p, -1]]), cfg.strike, cfg.dim).item())
            t_stop = cfg.T
        rewards[p] = np.exp(-cfg.r * t_stop) * payoff

    return rewards


@torch.no_grad()
def forward_reward_on_paths(
    model: ADNN,
    paths: np.ndarray,
    cfg: B1Config,
    num_steps: int | None = None,
) -> float:
    """Average discounted payoff on pre-simulated paths using Eq. 11."""
    rewards = forward_rewards_per_path(model, paths, cfg, num_steps)
    return float(np.mean(rewards))
