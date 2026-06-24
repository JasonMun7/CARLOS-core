"""Stage 1: LSMC initialization of ADNN R^[0] (Sec. 3.1)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from carlos.config import CarlosConfig
from carlos.lsmc import build_training_set, lsmc_price
from carlos.model import ADNN
from carlos.pricing import validate_price
from carlos.simulator import make_simulator
from carlos import ui


def train_adnn_on_dataset(
    cfg: CarlosConfig,
    states: np.ndarray,
    targets: np.ndarray,
    model: ADNN | None = None,
) -> ADNN:
    device = torch.device("cpu")
    net = model or ADNN(cfg.dim)
    net = net.to(device)
    optimizer = Adam(net.parameters(), lr=cfg.lr)
    criterion = nn.MSELoss()

    x = torch.tensor(states, dtype=torch.float32)
    y = torch.tensor(targets, dtype=torch.float32).unsqueeze(-1)
    loader = DataLoader(TensorDataset(x, y), batch_size=cfg.batch_size, shuffle=True)

    for epoch in range(cfg.stage1_epochs):
        net.train()
        total_loss = 0.0
        n_batches = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = net(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        ui.stage1_epoch(epoch + 1, cfg.stage1_epochs, total_loss / n_batches)

    return net


def run_stage1(cfg: CarlosConfig | None = None, seed: int = 42) -> ADNN:
    cfg = cfg or CarlosConfig()
    sim = make_simulator(cfg, num_paths=cfg.stage1_paths, seed=seed)
    sim.run()
    paths = np.array(sim.paths(0))

    bermudan = lsmc_price(paths, cfg)
    dataset = build_training_set(paths, cfg)
    ui.stage1_summary(bermudan, len(dataset.targets))

    model = train_adnn_on_dataset(cfg, dataset.states, dataset.targets)
    return model


def main() -> None:
    torch.manual_seed(0)
    np.random.seed(0)
    cfg = CarlosConfig(dev_mode=True)
    model = run_stage1(cfg)
    validate_price(model, cfg)


if __name__ == "__main__":
    main()
