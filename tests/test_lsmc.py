"""Tests for backward LSMC."""

from __future__ import annotations

import numpy as np

from carlos.config import CarlosConfig
from carlos.lsmc import build_training_set, lsmc_price


def test_lsmc_finite_targets() -> None:
    cfg = CarlosConfig()
    rng = np.random.default_rng(0)
    paths = 36.0 * np.exp(
        np.cumsum(rng.normal(-0.01, 0.05, size=(100, cfg.num_steps + 1)), axis=1)
    )
    result = build_training_set(paths, cfg)
    assert result.states.shape[0] == result.targets.shape[0]
    assert result.states.shape[1] == 2
    assert np.all(np.isfinite(result.targets))


def test_lsmc_price_positive() -> None:
    cfg = CarlosConfig()
    rng = np.random.default_rng(1)
    paths = 36.0 * np.exp(
        np.cumsum(rng.normal(-0.005, 0.08, size=(500, cfg.num_steps + 1)), axis=1)
    )
    price = lsmc_price(paths, cfg)
    assert price > 0.0
    assert np.isfinite(price)


def test_sampling_weights_sum_to_one() -> None:
    cfg = CarlosConfig()
    w = cfg.sampling_weights(0)
    assert abs(sum(w.values()) - 1.0) < 1e-9
