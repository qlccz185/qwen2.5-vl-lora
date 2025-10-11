"""One-click pipeline for Task2 LM-supervised LoRA training + evaluation."""
from __future__ import annotations

import argparse
from pathlib import Path

from train import load_config as load_train_config, train
from evaluate import load_config as load_eval_config, evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Task2 training and evaluation end-to-end")
    parser.add_argument(
        "--train-config",
        type=Path,
        default=Path(__file__).with_name("config_train.json"),
        help="Path to the training configuration JSON",
    )
    parser.add_argument(
        "--eval-config",
        type=Path,
        default=Path(__file__).with_name("config_eval.json"),
        help="Path to the evaluation configuration JSON",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Only run training without launching evaluation",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_cfg = load_train_config(args.train_config)
    train(train_cfg)

    if not args.skip_eval:
        eval_cfg = load_eval_config(args.eval_config)
        evaluate(eval_cfg)


if __name__ == "__main__":
    main()
