"""Paper benchmark presets (Tables 2, 3, 6)."""

from __future__ import annotations

from typing import Callable

from carlos.config import CarlosConfig, benchmark_tolerance
from carlos.payoffs import PayoffKind

# Table 3 CARLOS targets and reported std devs
_PAPER = {
    "b1": (4.592, 0.005, PayoffKind.BASKET_PUT, 1, 36.0, 40.0, 1.0, 0.05, 0.2, 0.0, 30.0, 40.0),
    "b2": (1.474, 0.001, PayoffKind.BASKET_PUT, 2, 40.0, 40.0, 1.0, 0.06, 0.2, 0.0, 35.0, 45.0),
    "m2a": (14.171, 0.015, PayoffKind.MAX_CALL, 2, 100.0, 100.0, 3.0, 0.05, 0.2, 0.1, 100.0, 300.0),
    "m2b": (15.711, 0.022, PayoffKind.MAX_CALL, 2, 100.0, 100.0, 3.0, 0.05, 0.2, 0.05, 100.0, 300.0),
    "m3": (11.510, 0.008, PayoffKind.MAX_CALL, 3, 90.0, 100.0, 3.0, 0.05, 0.2, 0.1, 90.0, 300.0),
    "m5a": (26.55, 0.032, PayoffKind.MAX_CALL, 5, 100.0, 100.0, 3.0, 0.05, 0.2, 0.1, 100.0, 300.0),
    "m5b": (12.009, 0.010, PayoffKind.MAX_CALL, 5, 70.0, 100.0, 3.0, 0.05, 0.08, 0.1, 70.0, 300.0),
}

# Table 6: stage1_paths, rl_training_inputs, stage1_epochs, adnn_hidden
_TABLE6 = {
    "b1": (10_000, 10_000, 5, 60),
    "b2": (10_000, 20_000, 10, 60),
    "m2a": (20_000, 20_000, 10, 60),
    "m2b": (20_000, 20_000, 10, 60),
    "m3": (30_000, 30_000, 10, 90),
    "m5a": (50_000, 50_000, 10, 150),
    "m5b": (50_000, 50_000, 10, 150),
}


def _preset(
    key: str,
    *,
    dev_mode: bool = False,
    delta_vec: list[float] | None = None,
    sigma_vec: list[float] | None = None,
    x0_vec: list[float] | None = None,
) -> CarlosConfig:
    (
        target,
        std,
        payoff,
        dim,
        x0,
        strike,
        T,
        r,
        sigma,
        delta,
        x_min,
        x_max,
    ) = _PAPER[key]
    k_paths, m_inputs, epochs, hidden = _TABLE6[key]
    mins = [x_min] * dim
    maxs = [x_max] * dim
    if x0_vec is None and dim > 1:
        x0_vec = [x0] * dim
    if sigma_vec is None and dim > 1:
        sigma_vec = [sigma] * dim
    if delta_vec is None and dim > 1:
        delta_vec = [delta] * dim
    if key == "m2b":
        delta_vec = [0.05, 0.15]
    if key == "m5b":
        sigma_vec = [0.08, 0.16, 0.24, 0.32, 0.4]
    return CarlosConfig(
        dim=dim,
        x0=x0,
        x0_vec=x0_vec,
        strike=strike,
        T=T,
        r=r,
        sigma=sigma,
        sigma_vec=sigma_vec,
        delta=delta,
        delta_vec=delta_vec,
        x_min=x_min,
        x_max=x_max,
        x_mins=mins,
        x_maxs=maxs,
        payoff=payoff,
        benchmark_id=key,
        adnn_hidden=hidden,
        stage1_paths=k_paths,
        stage1_epochs=epochs,
        rl_epochs=1,
        rl_training_inputs=m_inputs,
        target_price=target,
        target_std=std,
        target_tolerance=benchmark_tolerance(std),
        dev_mode=dev_mode,
    )


def b1_benchmark() -> CarlosConfig:
    return _preset("b1")


def b2_benchmark() -> CarlosConfig:
    return _preset("b2")


def m2a_benchmark() -> CarlosConfig:
    return _preset("m2a")


def m2b_benchmark() -> CarlosConfig:
    return _preset("m2b")


def m3_benchmark() -> CarlosConfig:
    return _preset("m3")


def m5a_benchmark() -> CarlosConfig:
    return _preset("m5a")


def m5b_benchmark() -> CarlosConfig:
    return _preset("m5b")


BENCHMARKS: dict[str, Callable[[], CarlosConfig]] = {
    "b1": b1_benchmark,
    "b2": b2_benchmark,
    "m2a": m2a_benchmark,
    "m2b": m2b_benchmark,
    "m3": m3_benchmark,
    "m5a": m5a_benchmark,
    "m5b": m5b_benchmark,
}

BENCHMARK_LABELS: dict[str, str] = {
    "b1": "B1 Basket Put (d=1)",
    "b2": "B2 Basket Put (d=2)",
    "m2a": "M2.A Max Call (d=2)",
    "m2b": "M2.B Max Call (d=2)",
    "m3": "M3 Max Call (d=3)",
    "m5a": "M5.A Max Call (d=5)",
    "m5b": "M5.B Max Call (d=5)",
}


def get_benchmark(preset_id: str) -> CarlosConfig:
    key = preset_id.lower().replace(".", "").replace("-", "")
    if key not in BENCHMARKS:
        raise KeyError(f"unknown benchmark preset: {preset_id}")
    return BENCHMARKS[key]()
