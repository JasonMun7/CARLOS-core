"""PyTorch device selection."""

from __future__ import annotations

import torch

_forced_cpu = False


def force_cpu(forced: bool = True) -> None:
    """Force CPU for reproducible benchmark scoring."""
    global _forced_cpu
    _forced_cpu = forced


def get_device() -> torch.device:
    if _forced_cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
