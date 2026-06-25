"""Algorithm 1 CARLOS RL training loop (Stage 2)."""

from __future__ import annotations

import copy
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from carlos.config import CarlosConfig, RLState, validation_bank_seed
from carlos.device import force_cpu, get_device
from carlos.grid import build_validation_paths, should_advance_grid
from carlos.model import ADNN
from carlos.pricing import forward_rewards_per_path, validate_price
from carlos.timing import is_enabled, nested, report, section
from carlos.sampling import sample_training_inputs
from carlos.stage1 import run_stage1
from carlos.targets import compute_training_targets
from carlos import ui


def _train_batch(
    model: ADNN,
    states: np.ndarray,
    targets: np.ndarray,
    cfg: CarlosConfig,
    rl_state: RLState,
) -> float:
    """One fresh Adam pass per RL loop (509f18f behavior; Table 6 epochs are Stage 1)."""
    device = get_device()
    model.train()
    optimizer = Adam(model.parameters(), lr=rl_state.lr)
    criterion = nn.MSELoss()

    x = torch.tensor(states, dtype=torch.float32)
    y = torch.tensor(targets, dtype=torch.float32).unsqueeze(-1)
    loader = DataLoader(TensorDataset(x, y), batch_size=cfg.batch_size, shuffle=True)

    total_loss = 0.0
    n = 0
    for _epoch in range(cfg.rl_epochs):
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


def _checkpoint_if_best(
    model: ADNN,
    cfg: CarlosConfig,
    val_paths: np.ndarray,
    rl_state: RLState,
) -> None:
    price = validate_price(
        model,
        cfg,
        paths=val_paths,
        num_steps=cfg.finest_grid_steps(),
        show_target=False,
    )
    if price > rl_state.best_finest_price:
        rl_state.best_finest_price = price
        rl_state.best_state_dict = copy.deepcopy(model.state_dict())


def _restore_best(model: ADNN, rl_state: RLState) -> None:
    if rl_state.best_state_dict is not None:
        model.load_state_dict(rl_state.best_state_dict)


def run_rl_loop(
    cfg: CarlosConfig | None = None,
    seed: int = 0,
    max_loops_per_level: int = 5,
    init_model: ADNN | None = None,
    val_paths: np.ndarray | None = None,
    score_at_end: bool = True,
) -> ADNN:
    cfg = cfg or CarlosConfig()
    if cfg.profile:
        from carlos.timing import set_enabled

        set_enabled(True)
    if not cfg.dev_mode and cfg.benchmark_id:
        force_cpu(True)

    torch.manual_seed(seed)
    np.random.seed(seed)

    with section("stage1.total"):
        model = init_model or run_stage1(cfg, seed=seed)
    if val_paths is None:
        with section("validation.build_paths"):
            val_paths = build_validation_paths(cfg, seed=validation_bank_seed(seed))
    rl_state = RLState(lr=cfg.lr)
    _checkpoint_if_best(model, cfg, val_paths, rl_state)

    max_level = cfg.max_grid_levels()
    ui.grid_overview(max_level)

    for level in range(max_level + 1):
        steps = cfg.grid_steps_for_level(level)
        dt = cfg.dt_for_level(level)
        ui.grid_level_header(level, dt, steps)

        prev_rewards: np.ndarray | None = None
        loops_at_level = 0

        while loops_at_level < max_loops_per_level:
            m = cfg.rl_training_inputs
            loop_label = rl_state.loop_count + 1
            with nested(f"rl.loop_{loop_label}.sampling"):
                batch = sample_training_inputs(
                    model, cfg, level, m, seed=seed + rl_state.loop_count * 17
                )

            with nested(f"rl.loop_{loop_label}.targets"):
                targets = compute_training_targets(
                    model,
                    batch,
                    cfg,
                    level,
                    steps,
                    seed,
                    rl_state.loop_count,
                    sim=None,
                )

            with nested(f"rl.loop_{loop_label}.train"):
                loss = _train_batch(model, batch.states, targets, cfg, rl_state)
            loops_at_level += 1
            rl_state.loop_count += 1

            path_slice = val_paths[:, : steps + 1] if val_paths.ndim == 2 else val_paths[:, : steps + 1, :]
            with nested(f"rl.loop_{loop_label}.validation"):
                curr_rewards = forward_rewards_per_path(model, path_slice, cfg, steps)
            curr_val = float(np.mean(curr_rewards))
            ui.loop_metrics(rl_state.loop_count, loss, curr_val)
            _checkpoint_if_best(model, cfg, val_paths, rl_state)

            if (
                prev_rewards is not None
                and loops_at_level >= cfg.min_loops_before_saturation
                and should_advance_grid(
                    prev_rewards, curr_rewards, cfg.grid_transition_alpha
                )
            ):
                ui.grid_saturated()
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
        ui.level_price(level, price)
        _checkpoint_if_best(model, cfg, val_paths, rl_state)

        if level == max_level:
            break

    _restore_best(model, rl_state)

    if score_at_end:
        finest_steps = cfg.finest_grid_steps()
        validate_price(
            model,
            cfg,
            paths=val_paths,
            num_steps=finest_steps,
            show_target=not cfg.dev_mode,
        )

    if is_enabled():
        report()
    return model


def main() -> None:
    from carlos.cli import dispatch

    raise SystemExit(dispatch(["train", *sys.argv[1:]]))


if __name__ == "__main__":
    main()
