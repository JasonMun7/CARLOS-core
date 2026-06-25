"""Runtime phase timing for CARLOS pipeline (avoid naming module `profile` — stdlib clash)."""

from __future__ import annotations

import os
import platform
import threading
import time
from contextlib import contextmanager
from typing import Iterator

import torch

from carlos import ui

_enabled = False
_records: dict[str, float] = {}
_stack: list[tuple[str, float]] = []


def set_enabled(enabled: bool) -> None:
    global _enabled
    _enabled = enabled
    if enabled:
        _records.clear()


def is_enabled() -> bool:
    return _enabled


def add(label: str, seconds: float) -> None:
    if not _enabled:
        return
    _records[label] = _records.get(label, 0.0) + seconds


@contextmanager
def section(label: str) -> Iterator[None]:
    if not _enabled:
        yield
        return
    t0 = time.perf_counter()
    try:
        yield
    finally:
        add(label, time.perf_counter() - t0)


@contextmanager
def nested(label: str) -> Iterator[None]:
    if not _enabled:
        yield
        return
    t0 = time.perf_counter()
    _stack.append((label, t0))
    try:
        yield
    finally:
        start_label, start_t = _stack.pop()
        add(start_label, time.perf_counter() - start_t)


def hardware_info() -> dict[str, str]:
    device = "cpu"
    if torch.cuda.is_available():
        device = f"cuda ({torch.cuda.get_device_name(0)})"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = "mps"
    return {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": device,
        "threads": str(os.cpu_count() or threading.active_count()),
    }


def report() -> None:
    if not _enabled or not _records:
        return
    total = sum(_records.values())
    hw = hardware_info()
    if ui.is_quiet():
        print("\n=== Profile ===")
        for k, v in sorted(_records.items(), key=lambda kv: -kv[1]):
            pct = 100.0 * v / total if total > 0 else 0.0
            print(f"  {k}: {v:.3f}s ({pct:.1f}%)")
        print(f"  total: {total:.3f}s")
        for k, v in hw.items():
            print(f"  {k}: {v}")
        return
    from rich.table import Table
    from rich.panel import Panel

    table = Table(show_header=True, header_style="bold")
    table.add_column("Phase")
    table.add_column("Time (s)", justify="right")
    table.add_column("%", justify="right")
    for label, secs in sorted(_records.items(), key=lambda kv: -kv[1]):
        pct = 100.0 * secs / total if total > 0 else 0.0
        table.add_row(label, f"{secs:.3f}", f"{pct:.1f}")
    table.add_row("[bold]Total[/bold]", f"[bold]{total:.3f}[/bold]", "100.0")
    ui.console.print(Panel(table, title="[bold]Profile[/bold]", border_style="yellow"))
    hw_table = Table(show_header=False, box=None)
    hw_table.add_column("Key", style="dim")
    hw_table.add_column("Value")
    for k, v in hw.items():
        hw_table.add_row(k, v)
    ui.console.print(Panel(hw_table, title="Hardware", border_style="dim"))


def records() -> dict[str, float]:
    return dict(_records)
