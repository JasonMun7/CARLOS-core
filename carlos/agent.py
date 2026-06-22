"""V1 CARLOS Stage-2 training stub with simplified exploratory delay."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Adam

from carlos.model import ADNN
from carlos.payoffs import basket_put

ROOT = Path(__file__).resolve().parent.parent


def _load_simulator():
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


@dataclass
class B1Config:
    """Table 2 B1 basket put."""

    dim: int = 1
    x0: float = 36.0
    strike: float = 40.0
    T: float = 1.0
    r: float = 0.05
    sigma: float = 0.2
    delta: float = 0.0
    x_min: float = 30.0
    x_max: float = 40.0
    num_steps: int = 20
    delay_k: int = 3
    batch_size: int = 64
    lr: float = 1e-4
    epochs: int = 5
    val_paths: int = 10_000
    target_price: float = 4.592


def _make_simulator(cfg: B1Config, num_paths: int, seed: int):
    GBMSimulator = _load_simulator()
    dt = cfg.T / cfg.num_steps
    return GBMSimulator(
        dim=cfg.dim,
        num_paths=num_paths,
        num_steps=cfg.num_steps,
        dt=dt,
        r=cfg.r,
        T=cfg.T,
        x0=[cfg.x0],
        delta=[cfg.delta],
        sigma=[cfg.sigma],
        seed=seed,
    )


def _compute_targets(
    model: ADNN,
    paths: np.ndarray,
    t_init: float,
    cfg: B1Config,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Compute timing-value targets y = y_dpf - h(x_t) along simulated paths."""
    num_paths, n_cols = paths.shape
    dt = cfg.T / cfg.num_steps
    start_step = int(round(t_init / dt))
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
                stop_idx = n_cols - 1
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
                        stop_idx = s
                        t_stop = t_s
                        x_stop = x_s
                        h_stop = h_s
                        break
                else:
                    t_stop = times[-1]
                    x_stop = float(path[-1])
                    h_stop = float(
                        basket_put(torch.tensor([x_stop]), cfg.strike, cfg.dim).item()
                    )

                y_dpf = np.exp(-cfg.r * (t_stop - t_init)) * h_stop
                y = y_dpf - h_t

            states.append(torch.tensor([t_init, x_t], dtype=torch.float32))
            targets.append(y)

    state_batch = torch.stack(states).to(device)
    target_batch = torch.tensor(targets, dtype=torch.float32, device=device).unsqueeze(-1)
    return state_batch, target_batch


def train_v1(cfg: B1Config | None = None, global_seed: int = 0) -> ADNN:
    cfg = cfg or B1Config()
    device = torch.device("cpu")
    model = ADNN(cfg.dim).to(device)
    optimizer = Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.MSELoss()
    dt = cfg.T / cfg.num_steps

    sim = _make_simulator(cfg, num_paths=cfg.batch_size, seed=global_seed)

    for epoch in range(cfg.epochs):
        model.train()
        t_init = float(np.random.uniform(0.0, cfg.T))
        x_init = float(np.random.uniform(cfg.x_min, cfg.x_max))
        batch_seed = global_seed + epoch * cfg.batch_size + 1

        sim.simulate_from(t_init, [x_init], cfg.batch_size, batch_seed)
        paths = np.array(sim.paths(0))

        states, targets = _compute_targets(model, paths, t_init, cfg, device)
        preds = model(states)
        loss = criterion(preds, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"epoch {epoch + 1}/{cfg.epochs}  loss={loss.item():.6f}  t_init={t_init:.3f}")

    return model


@torch.no_grad()
def validate_price(model: ADNN, cfg: B1Config | None = None, seed: int = 123) -> float:
    """Forward MC price from (t=0, X=X0) using Eq. 11 stopping rule."""
    cfg = cfg or B1Config()
    device = torch.device("cpu")
    model = model.to(device)
    model.eval()

    sim = _make_simulator(cfg, num_paths=cfg.val_paths, seed=seed)
    sim.run()
    paths = np.array(sim.paths(0))
    dt = cfg.T / cfg.num_steps

    total = 0.0
    for p in range(cfg.val_paths):
        t_stop = cfg.T
        payoff = 0.0
        for s in range(cfg.num_steps + 1):
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


def main() -> None:
    torch.manual_seed(0)
    np.random.seed(0)
    cfg = B1Config(epochs=3, val_paths=1000)
    model = train_v1(cfg)
    validate_price(model, cfg)


if __name__ == "__main__":
    main()
