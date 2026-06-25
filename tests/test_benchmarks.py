"""Tests for paper benchmark suite (ADR 0002)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from carlos.benchmarks import BENCHMARKS, BENCHMARK_LABELS, get_benchmark
from carlos.config import benchmark_tolerance
from carlos.payoffs import PayoffKind
from carlos.stage1 import run_stage1


TABLE3 = {
    "b1": (4.592, 0.005),
    "b2": (1.474, 0.001),
    "m2a": (14.171, 0.015),
    "m2b": (15.711, 0.022),
    "m3": (11.510, 0.008),
    "m5a": (26.55, 0.032),
    "m5b": (12.009, 0.010),
}

TABLE6 = {
    "b1": (10_000, 10_000, 5, 60),
    "b2": (10_000, 20_000, 10, 60),
    "m2a": (20_000, 20_000, 10, 60),
    "m2b": (20_000, 20_000, 10, 60),
    "m3": (30_000, 30_000, 10, 90),
    "m5a": (50_000, 50_000, 10, 150),
    "m5b": (50_000, 50_000, 10, 150),
}


@pytest.mark.parametrize("preset_id", list(BENCHMARKS.keys()))
def test_preset_table3_targets(preset_id: str) -> None:
    cfg = get_benchmark(preset_id)
    target, std = TABLE3[preset_id]
    assert cfg.target_price == target
    assert cfg.target_std == std
    assert cfg.target_tolerance == benchmark_tolerance(std)
    assert preset_id in BENCHMARK_LABELS


@pytest.mark.parametrize("preset_id", list(BENCHMARKS.keys()))
def test_preset_table6_hyperparams(preset_id: str) -> None:
    cfg = get_benchmark(preset_id)
    k, m, epochs, hidden = TABLE6[preset_id]
    assert cfg.stage1_paths == k
    assert cfg.rl_training_inputs == m
    assert cfg.stage1_epochs == epochs
    assert cfg.rl_epochs == 1
    assert cfg.adnn_hidden == hidden


def test_m2a_max_call_payoff() -> None:
    cfg = get_benchmark("m2a")
    assert cfg.payoff == PayoffKind.MAX_CALL
    assert cfg.dim == 2
    assert cfg.x0s == [100.0, 100.0]


def test_m5b_heterogeneous_vol() -> None:
    cfg = get_benchmark("m5b")
    assert cfg.sigmas == [0.08, 0.16, 0.24, 0.32, 0.4]


@pytest.mark.parametrize("preset_id", ["b1", "b2", "m2a"])
def test_dev_stage1_smoke(preset_id: str) -> None:
    """Dev-mode Stage 1 wiring smoke (not scored)."""
    cfg = get_benchmark(preset_id)
    cfg = dataclasses.replace(cfg, dev_mode=True)
    np.random.seed(0)
    model = run_stage1(cfg, seed=0)
    assert model.dim == cfg.dim
