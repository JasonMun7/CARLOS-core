"""Terminal UI helpers (Rich)."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text

from carlos.config import CarlosConfig

console = Console()
_quiet = False
_run_t0: float | None = None


def set_quiet(quiet: bool) -> None:
    global _quiet
    _quiet = quiet


def is_quiet() -> bool:
    return _quiet


def start_run() -> None:
    global _run_t0
    _run_t0 = time.perf_counter()


def elapsed_s() -> float:
    if _run_t0 is None:
        return 0.0
    return time.perf_counter() - _run_t0


def welcome() -> None:
    title = Text("CARLOS", style="bold cyan")
    title.append(" — Continuous-time Adaptive RL for Optimal Stopping", style="dim")
    console.print(
        Panel(
            title,
            subtitle="[dim]arxiv:2606.17545[/dim]",
            border_style="cyan",
            padding=(0, 1),
        )
    )
    table = Table(box=box.SIMPLE, show_header=True, header_style="bold")
    table.add_column("Command", style="cyan")
    table.add_column("What it does")
    table.add_row("benchmark b1", "Official scored B1 run (pass/fail exit code)")
    table.add_row("train [--dev]", "Full pipeline — use [dim]--dev[/dim] for smoke tests")
    table.add_row("stage1", "LSMC initialization only → R^[0]")
    table.add_row("price [--dev]", "Stage 1 + validation price")
    table.add_row("agent", "Legacy v1 smoke test (fixed-k delay)")
    console.print(table)
    console.print(
        "\n[bold]Quick start[/bold]\n"
        "  [cyan]python -m carlos benchmark b1[/cyan]     scored benchmark\n"
        "  [cyan]python -m carlos train --dev --loops 3[/cyan]  fast smoke test\n"
        "  [cyan]python -m carlos --help[/cyan]           all options\n"
    )


def run_config_panel(cfg: CarlosConfig, *, mode: str, seed: int | None = None) -> None:
    if _quiet:
        return
    table = Table(box=box.ROUNDED, show_header=False, padding=(0, 1))
    table.add_column("Key", style="dim")
    table.add_column("Value")
    table.add_row("Mode", mode)
    if seed is not None:
        table.add_row("Training seed", str(seed))
    table.add_row("Paths (stage1 / RL / val)", f"{cfg.stage1_paths:,} / {cfg.rl_training_inputs:,} / {cfg.val_paths:,}")
    table.add_row("Grid", f"{cfg.num_steps} steps → finest {cfg.finest_grid_steps()}")
    if not cfg.dev_mode and cfg.target_price:
        lo = cfg.target_price - cfg.target_tolerance
        hi = cfg.target_price + cfg.target_tolerance
        table.add_row("B1 target", f"{cfg.target_price:.3f}  [dim](accept {lo:.3f}–{hi:.3f})[/dim]")
    console.print(Panel(table, title="[bold]Run configuration[/bold]", border_style="blue"))


def section(title: str, subtitle: str = "") -> None:
    if _quiet:
        console.print(f"\n=== {title} ===" + (f" — {subtitle}" if subtitle else ""))
        return
    console.rule(f"[bold]{title}[/bold]", style="cyan")
    if subtitle:
        console.print(f"[dim]{subtitle}[/dim]")


def stage1_epoch(epoch: int, total: int, loss: float) -> None:
    msg = f"stage1 epoch {epoch}/{total}  loss={loss:.6f}"
    if _quiet:
        console.print(msg)
    else:
        console.print(f"  [dim]epoch[/dim] {epoch}/{total}  loss={loss:.6f}")


def stage1_summary(bermudan: float, n_samples: int) -> None:
    if _quiet:
        console.print(f"LSMC Bermudan price: {bermudan:.4f}")
        console.print(f"Stage 1 training samples: {n_samples}")
        return
    table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    table.add_row("LSMC Bermudan", f"[bold]{bermudan:.4f}[/bold]")
    table.add_row("Training samples", f"{n_samples:,}")
    console.print(Panel(table, title="[bold]Stage 1[/bold]", border_style="green"))


def grid_overview(max_level: int) -> None:
    if _quiet:
        console.print(f"Grid levels: 0..{max_level}")
    else:
        console.print(f"[dim]Grid levels[/dim]  [bold]0[/bold] → [bold]{max_level}[/bold]")


def grid_level_header(level: int, dt: float, steps: int) -> None:
    section(f"Grid level {level}", f"Δt = {dt:.5f}  ·  {steps} steps")


def loop_metrics(loop: int, loss: float, val_reward: float) -> None:
    if _quiet:
        console.print(f"loop {loop}  loss={loss:.6f}  val_reward={val_reward:.4f}")
        return
    console.print(
        f"  [dim]loop[/dim] [bold]{loop:>3}[/bold]  "
        f"loss [yellow]{loss:.6f}[/yellow]  "
        f"val_reward [green]{val_reward:.4f}[/green]"
    )


def grid_saturated() -> None:
    if _quiet:
        console.print("Grid saturation detected (Eq. 32)")
    else:
        console.print("  [bold magenta]▸[/bold magenta] Grid saturation detected [dim](Eq. 32)[/dim]")


def level_price(level: int, price: float) -> None:
    if _quiet:
        console.print(f"Level {level} validate_price: {price:.4f}")
    else:
        console.print(f"  [dim]level {level}[/dim] validate_price [bold]{price:.4f}[/bold]")


def validation_price(price: float, target: float | None = None) -> None:
    if _quiet:
        if target is not None:
            console.print(f"validate_price: {price:.4f}  (Table 3 CARLOS target: {target})")
        else:
            console.print(f"validate_price: {price:.4f}")
        return
    if target is not None:
        console.print(
            f"  validate_price [bold]{price:.4f}[/bold]  "
            f"[dim]target {target:.3f}[/dim]"
        )
    else:
        console.print(f"  validate_price [bold]{price:.4f}[/bold]")


def benchmark_verdict(
    *,
    passed: bool,
    price: float,
    lo: float,
    hi: float,
    seed: int,
    elapsed: float,
) -> None:
    status = "PASS" if passed else "FAIL"
    if _quiet:
        console.print(
            f"B1 benchmark: {status}  price={price:.4f}  "
            f"acceptance=[{lo:.3f}, {hi:.3f}]  seed={seed}"
        )
        return
    color = "green" if passed else "red"
    table = Table(box=box.DOUBLE_EDGE, show_header=False, padding=(0, 1))
    table.add_column("", style="dim", width=14)
    table.add_column("")
    table.add_row("Result", f"[bold {color}]{status}[/bold {color}]")
    table.add_row("Price", f"[bold]{price:.4f}[/bold]")
    table.add_row("Acceptance", f"{lo:.3f} – {hi:.3f}")
    table.add_row("Seed", str(seed))
    table.add_row("Elapsed", f"{elapsed:.1f}s")
    console.print(Panel(table, title="[bold]B1 Benchmark[/bold]", border_style=color))


@contextmanager
def target_progress(total: int, description: str = "Computing targets") -> Iterator[object | None]:
    if _quiet or total < 64:
        yield None
        return
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=32),
        MofNCompleteColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task(description, total=total)

        def advance() -> None:
            progress.advance(task)

        yield advance


def run_footer() -> None:
    if _quiet:
        return
    console.print(f"\n[dim]Finished in {elapsed_s():.1f}s[/dim]\n")
