"""Tests for Algorithm 2 sampling."""

from __future__ import annotations

import numpy as np

from carlos.config import CarlosConfig
from carlos.model import ADNN
from carlos.sampling import sample_training_inputs


def test_sample_count_and_weights() -> None:
    cfg = CarlosConfig(dev_mode=True)
    model = ADNN(cfg.dim)
    m = 64
    batch = sample_training_inputs(model, cfg, grid_level=0, m=m, seed=42)
    assert batch.states.shape == (m, cfg.dim + 1)
    assert batch.t_inits.shape == (m,)
    w = cfg.sampling_weights(0)
    assert abs(sum(w.values()) - 1.0) < 1e-9
