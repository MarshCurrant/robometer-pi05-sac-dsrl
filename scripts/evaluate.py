#!/usr/bin/env python3
"""Evaluate a frozen Pi0.5 plus SAC residual checkpoint in LIBERO."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--evaluation-step", type=int,
        help="Low-level env step for a legacy final/renamed checkpoint lacking a manifest",
    )
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    if args.evaluation_step is not None and args.evaluation_step < 0:
        parser.error("--evaluation-step must be nonnegative")
    root = Path(__file__).resolve().parents[1]
    python_bin = Path(os.environ.get("PYTHON_BIN", root / ".venv-policy/bin/python"))
    if not python_bin.is_file():
        raise SystemExit(f"Policy Python not found: {python_bin}; source scripts/load_env.sh")
    command = [
        str(python_bin),
        str(root / "scripts/train.py"),
        "--config",
        str(args.config),
        "--set",
        f"training.load_dir={args.checkpoint.resolve()}",
        "--set",
        "training.load_mode=evaluate",
        "--set",
        "training.num_rollouts=0",
        "--set",
        "eval.eval_on_first_step=true",
        "--set",
        f"eval.eval_num_episodes={args.episodes}",
        "--set",
        f"env.eval_seed={args.seed}",
        "--set",
        "logging.wandb_offline=true",
        "--set",
        "runtime.run_name=eval",
    ]
    if args.evaluation_step is not None:
        command.extend(["--set", f"eval.evaluation_step={args.evaluation_step}"])
    raise SystemExit(subprocess.run(command, cwd=root, check=False).returncode)


if __name__ == "__main__":
    main()
