"""V1 CARLOS Stage-2 training stub with simplified exploratory delay."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from carlos.config import B1Config
from carlos.model import ADNN
from carlos.pricing import compute_targets_fixed_k, validate_price
from carlos.simulator import make_simulator
from carlos.stage1 import run_stage1


def train_v1(
    cfg: B1Config | None = None,
    global_seed: int = 0,
    init_model: ADNN | None = None,
) -> ADNN:
    cfg = cfg or B1Config()
    device = torch.device("cpu")
    model = init_model or run_stage1(cfg, seed=global_seed)
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.MSELoss()

    sim = make_simulator(cfg, num_paths=cfg.batch_size, seed=global_seed)

    for epoch in range(cfg.epochs):
        model.train()
        t_init = float(np.random.uniform(0.0, cfg.T))
        x_init = float(np.random.uniform(cfg.x_min, cfg.x_max))
        batch_seed = global_seed + epoch * cfg.batch_size + 1

        sim.simulate_from(t_init, [x_init], cfg.batch_size, batch_seed)
        paths = np.array(sim.paths(0))

        states, targets = compute_targets_fixed_k(model, paths, t_init, cfg, device)
        preds = model(states)
        loss = criterion(preds, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"epoch {epoch + 1}/{cfg.epochs}  loss={loss.item():.6f}  t_init={t_init:.3f}")

    return model


def main() -> None:
    torch.manual_seed(0)
    np.random.seed(0)
    cfg = B1Config(dev_mode=True, rl_epochs=3)
    model = train_v1(cfg)
    validate_price(model, cfg)


if __name__ == "__main__":
    main()
