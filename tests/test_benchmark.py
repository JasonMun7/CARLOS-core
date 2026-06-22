"""Tests for B1 benchmark scoring protocol (ADR 0001)."""

from __future__ import annotations

import numpy as np

from carlos.config import (
    B1_TOLERANCE,
    CarlosConfig,
    b1_benchmark,
    validation_bank_seed,
)
from carlos.model import ADNN
from carlos.pricing import forward_reward_on_paths, validate_price


def test_b1_benchmark_preset_full_paths() -> None:
    cfg = b1_benchmark()
    assert cfg.dev_mode is False
    assert cfg.stage1_paths == 10_000
    assert cfg.target_price == 4.592
    assert cfg.target_tolerance == B1_TOLERANCE


def test_validation_bank_seed_offset() -> None:
    assert validation_bank_seed(0) == 1000
    assert validation_bank_seed(42) == 1042


def test_benchmark_passes_tolerance() -> None:
    cfg = b1_benchmark()
    assert cfg.benchmark_passes(4.592)
    assert cfg.benchmark_passes(4.542)
    assert cfg.benchmark_passes(4.641)
    assert not cfg.benchmark_passes(4.50)


def test_validate_price_uses_provided_paths(capsys) -> None:
    cfg = CarlosConfig(dev_mode=True, val_paths=8)
    model = ADNN(cfg.dim)
    rng = np.random.default_rng(0)
    paths = 36.0 * np.exp(rng.normal(0, 0.02, size=(8, cfg.num_steps + 1)))

    price_from_bank = forward_reward_on_paths(model, paths, cfg, cfg.num_steps)
    price_reported = validate_price(
        model, cfg, paths=paths, num_steps=cfg.num_steps, show_target=False
    )

    assert price_from_bank == price_reported
    out = capsys.readouterr().out
    assert "Table 3 CARLOS target" not in out
