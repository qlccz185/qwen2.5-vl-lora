#!/usr/bin/env python3
"""Run Task2 (LM-guided LoRA) full pipeline."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Run Task2 dataset mapping, training, and inference")
    parser.add_argument(
        "--map-config",
        type=Path,
        default=base_dir / "config_map.json",
        help="Configuration for dataset preprocessing",
    )
    parser.add_argument(
        "--train-config",
        type=Path,
        default=base_dir / "config_lora_trainer.json",
        help="Configuration for LoRA training",
    )
    parser.add_argument(
        "--infer-config",
        type=Path,
        default=base_dir / "config_lora_infer.json",
        help="Configuration for evaluation/inference",
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
        "Mapping datasets for Task2",
        [args.python, str(base_dir / "map.py"), "--config", str(args.map_config.resolve())],
    )

    run_step(
        "Training Task2 LoRA adapters",
        [args.python, str(base_dir / "LORA.py"), "--config", str(args.train_config.resolve())],
    )

    run_step(
        "Evaluating Task2 adapters via language model outputs",
        [args.python, str(base_dir / "lora_infer.py"), "--config", str(args.infer_config.resolve())],
    )

    print("\n[SUCCESS] Task2 experiment completed.")


if __name__ == "__main__":
    main()
