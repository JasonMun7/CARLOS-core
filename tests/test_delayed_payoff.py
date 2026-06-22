"""Tests for Algorithm 3 delayed stopping."""

from __future__ import annotations

import numpy as np
import torch

from carlos.config import CarlosConfig
from carlos.delayed_payoff import compute_delayed_payoff, delta_wait
from carlos.model import ADNN


def test_delta_wait_positive() -> None:
    cfg = CarlosConfig()
    w = delta_wait(0.0, cfg, grid_level=0)
    assert w > 0.0


def test_delayed_payoff_finite() -> None:
    cfg = CarlosConfig()
    model = ADNN(cfg.dim)
    path = np.linspace(36.0, 30.0, cfg.num_steps + 1)
    y = compute_delayed_payoff(model, path, t_init=0.0, cfg=cfg, grid_level=0)
    assert np.isfinite(y)
    assert y >= 0.0
