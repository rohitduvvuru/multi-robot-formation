#!/usr/bin/env python3
"""Run lightweight hyperparameter sweeps for hallway PPO.

Supports:
- comm-range sweep
- num-sgd-iter sweep
- optional disturbance curriculum toggle
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> int:
    print("[run]", " ".join(cmd))
    return subprocess.call(cmd)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--iterations", type=int, default=200)
    ap.add_argument("--num-envs", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--base-tag", default="sweep")
    ap.add_argument("--comm-ranges", default="2.0,3.0,4.0")
    ap.add_argument("--sgd-iters", default="4,6,8")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--curriculum-teleop", action="store_true")
    ap.add_argument("--curriculum-start-scale", type=float, default=0.25)
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    train_py = repo_root / "code" / "train_hallway.py"

    comms = [x.strip() for x in args.comm_ranges.split(",") if x.strip()]
    sgd_iters = [x.strip() for x in args.sgd_iters.split(",") if x.strip()]

    rc = 0
    # Sweep 1: comm range with fixed SGD iters at middle value (default 6)
    base_sgd = sgd_iters[min(len(sgd_iters) - 1, 1)]
    for cr in comms:
        tag = f"{args.base_tag}_comm{cr.replace('.', 'p')}_sgd{base_sgd}"
        cmd = [
            args.python,
            str(train_py),
            "--iterations", str(args.iterations),
            "--num-envs", str(args.num_envs),
            "--max-steps", str(args.max_steps),
            "--num-sgd-iter", str(base_sgd),
            "--comm-range", str(cr),
            "--seed", str(args.seed),
            "--tag", tag,
            "--anneal-lr",
        ]
        if args.curriculum_teleop:
            cmd += ["--curriculum-teleop", "--curriculum-start-scale", str(args.curriculum_start_scale)]
        rc |= run(cmd)

    # Sweep 2: SGD iters with fixed comm range at middle value (default 3.0)
    base_cr = comms[min(len(comms) - 1, 1)]
    for si in sgd_iters:
        tag = f"{args.base_tag}_comm{base_cr.replace('.', 'p')}_sgd{si}"
        cmd = [
            args.python,
            str(train_py),
            "--iterations", str(args.iterations),
            "--num-envs", str(args.num_envs),
            "--max-steps", str(args.max_steps),
            "--num-sgd-iter", str(si),
            "--comm-range", str(base_cr),
            "--seed", str(args.seed),
            "--tag", tag,
            "--anneal-lr",
        ]
        if args.curriculum_teleop:
            cmd += ["--curriculum-teleop", "--curriculum-start-scale", str(args.curriculum_start_scale)]
        rc |= run(cmd)

    return rc


if __name__ == "__main__":
    raise SystemExit(main())