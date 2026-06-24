"""CLI argument parsing and command dispatch."""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from carlos import ui

EPILOG = """
examples:
  python -m carlos                         show this guide
  python -m carlos benchmark b1            official B1 benchmark (exit 0/1)
  python -m carlos train --dev --loops 3   fast smoke test
  python -m carlos stage1                  LSMC → ADNN init only
  python -m carlos price --dev             stage 1 + validate price

smoke vs benchmark:
  train --dev     reduced paths, not scored against 4.592
  benchmark b1    Table 6 path counts, pass/fail on finest grid
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="carlos",
        description="CARLOS — optimal stopping via deep RL (arxiv:2606.17545)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EPILOG,
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="plain output (no colors, panels, or progress bars)",
    )

    sub = parser.add_subparsers(dest="command", metavar="command")

    sub.add_parser("stage1", help="Stage 1: LSMC + ADNN initialization")

    p_train = sub.add_parser("train", help="Full Stage 2 RL pipeline")
    p_train.add_argument(
        "--loops",
        type=int,
        default=5,
        metavar="N",
        help="max RL loops per grid level (default: 5)",
    )
    p_train.add_argument(
        "--dev",
        action="store_true",
        help="smoke test with reduced path counts",
    )
    p_train.add_argument("--seed", type=int, default=0, help="training seed (default: 0)")

    p_bench = sub.add_parser("benchmark", help="Scored benchmark (pass/fail exit code)")
    bench_sub = p_bench.add_subparsers(dest="benchmark", metavar="preset")
    p_b1 = bench_sub.add_parser("b1", help="B1 basket put (Table 2 + Table 6)")
    p_b1.add_argument("--loops", type=int, default=5, metavar="N", help="max loops per level")
    p_b1.add_argument("--seed", type=int, default=0, help="training seed (default: 0)")

    sub.add_parser("agent", help="Legacy v1 smoke test (fixed-k delay)")

    p_price = sub.add_parser("price", help="Stage 1 init + validation price")
    p_price.add_argument("--dev", action="store_true", help="smoke test path counts")
    p_price.add_argument("--seed", type=int, default=42, help="stage 1 path seed")

    return parser


def dispatch(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    ui.set_quiet(args.quiet)

    if args.command is None:
        ui.welcome()
        return 0

    ui.start_run()

    if args.command == "stage1":
        from carlos.stage1 import main as stage1_main

        stage1_main()
        ui.run_footer()
        return 0

    if args.command == "train":
        from carlos.config import CarlosConfig
        from carlos.rl_loop import run_rl_loop

        cfg = CarlosConfig(dev_mode=args.dev)
        mode = "smoke test" if args.dev else "training"
        ui.run_config_panel(cfg, mode=mode, seed=args.seed)
        run_rl_loop(cfg, seed=args.seed, max_loops_per_level=args.loops)
        ui.run_footer()
        return 0

    if args.command == "benchmark":
        if args.benchmark == "b1":
            from carlos.benchmark import run_b1_benchmark
            from carlos.config import b1_benchmark

            cfg = b1_benchmark()
            ui.run_config_panel(cfg, mode="B1 benchmark", seed=args.seed)
            code = run_b1_benchmark(seed=args.seed, max_loops_per_level=args.loops)
            ui.run_footer()
            return code
        parser.error("benchmark requires a preset: b1")

    if args.command == "agent":
        from carlos.agent import main as agent_main

        agent_main()
        ui.run_footer()
        return 0

    if args.command == "price":
        from carlos.config import CarlosConfig
        from carlos.pricing import validate_price
        from carlos.stage1 import run_stage1

        cfg = CarlosConfig(dev_mode=args.dev)
        mode = "smoke test" if args.dev else "stage 1 + price"
        ui.run_config_panel(cfg, mode=mode, seed=args.seed)
        model = run_stage1(cfg, seed=args.seed)
        validate_price(model, cfg)
        ui.run_footer()
        return 0

    parser.error(f"unknown command: {args.command}")
    return 1


def main(argv: Sequence[str] | None = None) -> None:
    sys.exit(dispatch(argv))
