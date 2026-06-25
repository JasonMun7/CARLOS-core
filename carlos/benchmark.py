"""Benchmark runs with pass/fail scoring (ADR 0001, 0002)."""

from __future__ import annotations

from carlos.benchmarks import BENCHMARK_LABELS, BENCHMARKS, get_benchmark
from carlos.config import B1_TRAINING_SEED, CarlosConfig, validation_bank_seed
from carlos.device import force_cpu
from carlos.grid import build_validation_paths
from carlos.model import ADNN
from carlos.pricing import validate_price
from carlos.rl_loop import run_rl_loop
from carlos import ui


def score_benchmark(
    model: ADNN,
    cfg: CarlosConfig,
    val_paths,
    preset_id: str,
    training_seed: int = B1_TRAINING_SEED,
) -> float:
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
        preset_id=preset_id,
        label=BENCHMARK_LABELS.get(preset_id, preset_id),
    )
    return price


def run_benchmark(
    preset_id: str,
    seed: int = B1_TRAINING_SEED,
    max_loops_per_level: int = 5,
) -> int:
    """Run a scored benchmark preset. Returns 0 on pass, 1 on fail."""
    key = preset_id.lower().replace(".", "").replace("-", "")
    cfg = get_benchmark(key)
    cfg.dev_mode = False
    cfg.profile = False
    force_cpu(True)
    val_paths = build_validation_paths(cfg, seed=validation_bank_seed(seed))
    model = run_rl_loop(
        cfg,
        seed=seed,
        max_loops_per_level=max_loops_per_level,
        val_paths=val_paths,
        score_at_end=False,
    )
    price = score_benchmark(model, cfg, val_paths, key, training_seed=seed)
    return 0 if cfg.benchmark_passes(price) else 1


def run_b1_benchmark(
    seed: int = B1_TRAINING_SEED,
    max_loops_per_level: int = 5,
) -> int:
    return run_benchmark("b1", seed=seed, max_loops_per_level=max_loops_per_level)


def run_all_benchmarks(
    seed: int = B1_TRAINING_SEED,
    max_loops_per_level: int = 5,
) -> dict[str, bool]:
    results: dict[str, bool] = {}
    for key in BENCHMARKS:
        code = run_benchmark(key, seed=seed, max_loops_per_level=max_loops_per_level)
        results[key] = code == 0
    return results


def list_benchmarks() -> None:
    ui.benchmark_list(BENCHMARKS, BENCHMARK_LABELS)
