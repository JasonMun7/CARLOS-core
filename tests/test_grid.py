"""Tests for grid saturation (Eq. 32)."""

from __future__ import annotations

import numpy as np

from carlos.grid import should_advance_grid


def test_should_advance_grid_when_no_improvement() -> None:
    prev = np.linspace(3.5, 4.5, 100)
    curr = prev.copy()
    assert should_advance_grid(prev, curr)


def test_should_not_advance_when_clear_improvement() -> None:
    prev = np.full(100, 3.7)
    curr = np.full(100, 4.5)
    assert not should_advance_grid(prev, curr)
