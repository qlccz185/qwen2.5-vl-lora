#!/usr/bin/env python3
"""Run Task3 (hybrid head+LM LoRA) pipeline."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Run Task3 hybrid training and evaluation")
    parser.add_argument(
        "--train-config",
        type=Path,
        default=base_dir / "config_hybrid_train.json",
        help="Configuration file for hybrid training",
    )
    parser.add_argument(
        "--eval-config",
        type=Path,
        default=base_dir / "config_hybrid_eval.json",
        help="Configuration file for hybrid evaluation",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python interpreter for subprocess calls",
    )
    return parser.parse_args()


def run_step(description: str, command: list[str]) -> None:
    print(f"\n[RUN] {description}")
    print("Command:", " ".join(command))
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    base_dir = Path(__file__).resolve().parent

    run_step(
        "Training Task3 hybrid LoRA adapters",
        [args.python, str(base_dir / "hybrid_train.py"), "--config", str(args.train_config.resolve())],
    )

    run_step(
        "Evaluating Task3 hybrid model",
        [args.python, str(base_dir / "hybrid_eval.py"), "--config", str(args.eval_config.resolve())],
    )

    print("\n[SUCCESS] Task3 experiment completed.")


if __name__ == "__main__":
    main()
