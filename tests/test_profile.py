"""Tests for profiling and benchmark presets."""

from __future__ import annotations

import time

from carlos.benchmarks import BENCHMARKS, get_benchmark
from carlos.config import b1_benchmark, benchmark_tolerance
from carlos.timing import add, is_enabled, records, report, section, set_enabled


def test_profile_section_accumulates() -> None:
    set_enabled(True)
    with section("test.phase"):
        time.sleep(0.01)
    assert is_enabled()
    assert records()["test.phase"] >= 0.01
    set_enabled(False)


def test_profile_report_smoke(capsys) -> None:
    set_enabled(True)
    add("a", 1.0)
    add("b", 1.0)
    report()
    set_enabled(False)
    out = capsys.readouterr().out
    assert "Profile" in out or "a" in out


def test_all_benchmark_presets_table6() -> None:
    assert len(BENCHMARKS) == 7
    b2 = get_benchmark("b2")
    assert b2.dim == 2
    assert b2.rl_training_inputs == 20_000
    assert b2.payoff.value == "basket_put"
    m5a = get_benchmark("m5a")
    assert m5a.dim == 5
    assert m5a.adnn_hidden == 150


def test_benchmark_tolerance_floor() -> None:
    assert benchmark_tolerance(0.001) == 0.05
    assert benchmark_tolerance(0.02) == 0.06


def test_b1_preset_unchanged_target() -> None:
    cfg = b1_benchmark()
    assert cfg.target_price == 4.592
    assert cfg.benchmark_passes(4.5665)
