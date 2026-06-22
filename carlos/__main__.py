"""CARLOS CLI entry points."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="carlos", description="CARLOS optimal stopping")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("stage1", help="Run Stage 1 LSMC + ADNN init")
    p_train = sub.add_parser("train", help="Run full Stage 2 RL loop")
    p_train.add_argument("--loops", type=int, default=5)
    p_train.add_argument("--dev", action="store_true")
    sub.add_parser("agent", help="Run v1 simplified agent smoke test")
    p_price = sub.add_parser("price", help="Validate price after stage1 only")
    p_price.add_argument("--dev", action="store_true")

    args = parser.parse_args(argv)

    if args.command == "stage1":
        from carlos.stage1 import main as stage1_main

        stage1_main()
    elif args.command == "train":
        from carlos.rl_loop import run_rl_loop
        from carlos.config import B1Config
        from carlos.pricing import validate_price

        cfg = B1Config(dev_mode=args.dev)
        model = run_rl_loop(cfg, max_loops_per_level=args.loops)
        validate_price(model, cfg)
    elif args.command == "agent":
        from carlos.agent import main as agent_main

        agent_main()
    elif args.command == "price":
        from carlos.config import B1Config
        from carlos.stage1 import run_stage1
        from carlos.pricing import validate_price

        cfg = B1Config(dev_mode=args.dev)
        model = run_stage1(cfg)
        validate_price(model, cfg)


if __name__ == "__main__":
    main()
