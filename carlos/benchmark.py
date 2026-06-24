"""Benchmark runs with pass/fail scoring (ADR 0001)."""

from __future__ import annotations

from carlos.config import B1_TRAINING_SEED, CarlosConfig, b1_benchmark, validation_bank_seed
from carlos.grid import build_validation_paths
from carlos.model import ADNN
from carlos.pricing import validate_price
from carlos.rl_loop import run_rl_loop
from carlos import ui


def score_b1(
    model: ADNN,
    cfg: CarlosConfig,
    val_paths,
    training_seed: int = B1_TRAINING_SEED,
) -> float:
    """B1 validation price on finest grid using the fixed validation path bank."""
    steps = cfg.finest_grid_steps()
    price = validate_price(
        model,
        cfg,
        paths=val_paths,
        num_steps=steps,
        show_target=True,
    )
    lo = cfg.target_price - cfg.target_tolerance
    hi = cfg.target_price + cfg.target_tolerance
    ui.benchmark_verdict(
        passed=cfg.benchmark_passes(price),
        price=price,
        lo=lo,
        hi=hi,
        seed=training_seed,
        elapsed=ui.elapsed_s(),
    )
    return price


def run_b1_benchmark(
    seed: int = B1_TRAINING_SEED,
    max_loops_per_level: int = 5,
) -> int:
    """
    Official B1 benchmark run. Returns 0 on pass, 1 on fail.
    Uses Table 6 path counts; not valid for smoke tests.
    """
    cfg = b1_benchmark()
    val_paths = build_validation_paths(cfg, seed=validation_bank_seed(seed))
    model = run_rl_loop(
        cfg,
        seed=seed,
        max_loops_per_level=max_loops_per_level,
        val_paths=val_paths,
        score_at_end=False,
    )
    price = score_b1(model, cfg, val_paths, training_seed=seed)
    return 0 if cfg.benchmark_passes(price) else 1
