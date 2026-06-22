"""CARLOS CLI entry points."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="carlos", description="CARLOS optimal stopping")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("stage1", help="Run Stage 1 LSMC + ADNN init")
    p_train = sub.add_parser("train", help="Run full Stage 2 RL loop (experimental)")
    p_train.add_argument("--loops", type=int, default=5)
    p_train.add_argument("--dev", action="store_true", help="Smoke test (reduced paths)")
    p_train.add_argument("--seed", type=int, default=0)

    p_bench = sub.add_parser("benchmark", help="Scored benchmark run (pass/fail exit code)")
    bench_sub = p_bench.add_subparsers(dest="benchmark", required=True)
    p_b1 = bench_sub.add_parser("b1", help="B1 basket put benchmark (Table 6 path counts)")
    p_b1.add_argument("--loops", type=int, default=5)
    p_b1.add_argument("--seed", type=int, default=0)

    sub.add_parser("agent", help="Run v1 simplified agent smoke test")
    p_price = sub.add_parser("price", help="Validate price after stage1 only")
    p_price.add_argument("--dev", action="store_true")

    args = parser.parse_args(argv)

    if args.command == "stage1":
        from carlos.stage1 import main as stage1_main

        stage1_main()
    elif args.command == "train":
        from carlos.config import CarlosConfig
        from carlos.rl_loop import run_rl_loop

        cfg = CarlosConfig(dev_mode=args.dev)
        run_rl_loop(cfg, seed=args.seed, max_loops_per_level=args.loops)
    elif args.command == "benchmark":
        if args.benchmark == "b1":
            from carlos.benchmark import run_b1_benchmark

            sys.exit(run_b1_benchmark(seed=args.seed, max_loops_per_level=args.loops))
    elif args.command == "agent":
        from carlos.agent import main as agent_main

        agent_main()
    elif args.command == "price":
        from carlos.config import CarlosConfig
        from carlos.pricing import validate_price
        from carlos.stage1 import run_stage1

        cfg = CarlosConfig(dev_mode=args.dev)
        model = run_stage1(cfg)
        validate_price(model, cfg)


if __name__ == "__main__":
    main()
