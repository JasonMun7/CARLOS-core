"""Algorithm 1 CARLOS RL training loop (Stage 2)."""

from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from carlos.config import CarlosConfig, RLState, validation_bank_seed
from carlos.delayed_payoff import compute_timing_target
from carlos.grid import build_validation_paths, should_advance_grid
from carlos.model import ADNN
from carlos.pricing import forward_rewards_per_path, validate_price
from carlos.sampling import sample_training_inputs
from carlos.simulator import make_simulator
from carlos.stage1 import run_stage1


def _train_batch(
    model: ADNN,
    states: np.ndarray,
    targets: np.ndarray,
    cfg: CarlosConfig,
    lr: float,
) -> float:
    device = torch.device("cpu")
    model.train()
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    x = torch.tensor(states, dtype=torch.float32)
    y = torch.tensor(targets, dtype=torch.float32).unsqueeze(-1)
    loader = DataLoader(TensorDataset(x, y), batch_size=cfg.batch_size, shuffle=True)

    total_loss = 0.0
    n = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        loss = criterion(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n += 1
    return total_loss / max(n, 1)


def run_rl_loop(
    cfg: CarlosConfig | None = None,
    seed: int = 0,
    max_loops_per_level: int = 5,
    init_model: ADNN | None = None,
    val_paths: np.ndarray | None = None,
    score_at_end: bool = True,
) -> ADNN:
    cfg = cfg or CarlosConfig()
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = init_model or run_stage1(cfg, seed=seed)
    if val_paths is None:
        val_paths = build_validation_paths(cfg, seed=validation_bank_seed(seed))
    rl_state = RLState(lr=cfg.lr)

    max_level = cfg.max_grid_levels()
    print(f"Grid levels: 0..{max_level}")

    for level in range(max_level + 1):
        steps = cfg.grid_steps_for_level(level)
        dt = cfg.dt_for_level(level)
        print(f"\n=== Grid level {level}: dt={dt:.5f}, steps={steps} ===")

        sim = make_simulator(cfg, num_paths=1, seed=seed, num_steps=steps)
        prev_rewards: np.ndarray | None = None
        loops_at_level = 0

        while loops_at_level < max_loops_per_level:
            m = cfg.rl_training_inputs
            batch = sample_training_inputs(
                model, cfg, level, m, seed=seed + rl_state.loop_count * 17
            )

            targets = np.zeros(m, dtype=np.float64)
            for i in range(m):
                t_i = float(batch.t_inits[i])
                x_i = float(batch.x_inits[i])
                sim.simulate_from(t_i, [x_i], 1, seed + rl_state.loop_count * 1000 + i)
                path = np.array(sim.paths(0))[0]
                targets[i] = compute_timing_target(model, path, t_i, cfg, level)

            loss = _train_batch(model, batch.states, targets, cfg, rl_state.lr)
            loops_at_level += 1
            rl_state.loop_count += 1

            path_slice = val_paths[:, : steps + 1]
            curr_rewards = forward_rewards_per_path(model, path_slice, cfg, steps)
            curr_val = float(np.mean(curr_rewards))
            print(
                f"loop {rl_state.loop_count}  loss={loss:.6f}  "
                f"val_reward={curr_val:.4f}"
            )

            if prev_rewards is not None and should_advance_grid(
                prev_rewards, curr_rewards, cfg.grid_transition_alpha
            ):
                print("Grid saturation detected (Eq. 32)")
                break

            prev_rewards = curr_rewards

        rl_state.lr *= cfg.lr_decay
        price = validate_price(
            model,
            cfg,
            paths=val_paths,
            num_steps=steps,
            show_target=not cfg.dev_mode,
        )
        print(f"Level {level} validate_price: {price:.4f}")

        if level == max_level:
            break

    if score_at_end:
        finest_steps = cfg.finest_grid_steps()
        validate_price(
            model,
            cfg,
            paths=val_paths,
            num_steps=finest_steps,
            show_target=not cfg.dev_mode,
        )

    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="CARLOS Stage 2 RL loop")
    parser.add_argument("--loops", type=int, default=5, help="Max loops per grid level")
    parser.add_argument("--dev", action="store_true", help="Smoke test (reduced paths)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = CarlosConfig(dev_mode=args.dev)
    run_rl_loop(cfg, seed=args.seed, max_loops_per_level=args.loops)


if __name__ == "__main__":
    main()
