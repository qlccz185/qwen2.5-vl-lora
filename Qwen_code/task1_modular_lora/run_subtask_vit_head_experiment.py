#!/usr/bin/env python3
"""Run Task1 Subtask1 (ViT+head evaluation) end-to-end."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Run Task1 Subtask1 training and evaluation")
    parser.add_argument(
        "--train-config",
        type=Path,
        default=base_dir / "config_lora.json",
        help="Path to the training configuration JSON",
    )
    parser.add_argument(
        "--eval-config",
        type=Path,
        default=base_dir / "subtask_vit_head_eval" / "config_eval.json",
        help="Path to the evaluation configuration JSON",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python interpreter to use for the subprocess calls",
    )
    return parser.parse_args()


def run_step(description: str, command: list[str]) -> None:
    print(f"\n[RUN] {description}")
    print("Command:", " ".join(command))
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    base_dir = Path(__file__).resolve().parent

    train_script = base_dir / "lora_train.py"
    eval_script = base_dir / "subtask_vit_head_eval" / "lora_eval.py"

    run_step(
        "Training Task1 ViT + head LoRA adapters",
        [args.python, str(train_script), "--config", str(args.train_config.resolve())],
    )

    run_step(
        "Evaluating Task1 ViT + head adapters",
        [args.python, str(eval_script), "--config", str(args.eval_config.resolve())],
    )

    print("\n[SUCCESS] Task1 Subtask1 experiment completed.")


if __name__ == "__main__":
    main()
