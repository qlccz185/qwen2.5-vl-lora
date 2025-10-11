"""Evaluate Task2 LoRA checkpoints using the Task1-style configuration."""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from peft import PeftModel
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

try:  # Allow running both as a module and as a script
    from .datasets import ForgeryBinaryDataset, Task2Collator
    from .train import get_dtype, prepare_token_id
except ImportError:  # pragma: no cover - fallback when executed directly
    from datasets import ForgeryBinaryDataset, Task2Collator
    from train import get_dtype, prepare_token_id
def _binary_f1(preds: Sequence[float], labels: Sequence[float]) -> float:
    tp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 1)
    fp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 0)
    fn = sum(1 for p, l in zip(preds, labels) if p == 0 and l == 1)
    denom = 2 * tp + fp + fn
    return 0.0 if denom == 0 else (2 * tp) / denom


def _binary_auroc(scores: Sequence[float], labels: Sequence[float]) -> float:
    pos = sum(1 for l in labels if l == 1)
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return float("nan")

    # Rank data while averaging ties, adapted from the Mann–Whitney U statistic
    sorted_pairs = sorted(zip(scores, labels))
    rank_sum = 0.0
    n = len(sorted_pairs)
    i = 0
    while i < n:
        j = i
        while j < n and sorted_pairs[j][0] == sorted_pairs[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            if sorted_pairs[k][1] == 1:
                rank_sum += avg_rank
        i = j

    u = rank_sum - (pos * (pos + 1) / 2.0)
    return u / (pos * neg)


@dataclass
class EvalConfig:
    model_path: str
    data_root: str
    ann_file: str
    out_dir: str
    lora_weights: Optional[str] = None
    working_dir: Optional[str] = None

    batch_size: int = 2
    num_workers: int = 4
    amp_dtype: str = "bf16"
    image_size: int = 448

    prompt_text: str = (
        "Please determine whether this image is Real or Fake. Answer only with Fake or Real."
    )
    positive_response: str = " Fake"
    negative_response: str = " Real"
    dump_predictions: Optional[str] = None
    metrics_filename: str = "metrics.csv"
    inference_filename: str = "inference_results.json"


def load_config(path: Path) -> EvalConfig:
    data = json.loads(path.read_text(encoding="utf-8"))
    return EvalConfig(**data)


def prepare_environment(cfg: EvalConfig) -> None:
    if cfg.working_dir:
        os_cwd = Path(cfg.working_dir).expanduser().resolve()
        os_cwd.mkdir(parents=True, exist_ok=True)
        import os

        os.chdir(os_cwd)
        print(f"Current working directory: {os.getcwd()}")


def evaluate(cfg: EvalConfig) -> None:
    prepare_environment(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = get_dtype(cfg.amp_dtype)
    amp_enabled = device.type == "cuda" and dtype in (torch.float16, torch.bfloat16)
    autocast_dtype = dtype if dtype in (torch.float16, torch.bfloat16) else torch.float32

    processor = AutoProcessor.from_pretrained(cfg.model_path, local_files_only=True)
    positive_token = prepare_token_id(processor, cfg.positive_response)
    negative_token = prepare_token_id(processor, cfg.negative_response)

    dataset = ForgeryBinaryDataset(cfg.ann_file, cfg.data_root)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=Task2Collator(processor, cfg.prompt_text, image_size=cfg.image_size),
    )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        cfg.model_path,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        local_files_only=True,
    )
    if cfg.lora_weights:
        model = PeftModel.from_pretrained(model, cfg.lora_weights)
    model.eval()

    output_dir = Path(cfg.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prediction_rows = []
    inference_records = []
    all_probs = []
    all_preds = []
    all_labels = []
    total_loss = 0.0
    total_samples = 0
    total_correct = 0

    for batch in tqdm(dataloader, desc="Evaluating", dynamic_ncols=True):
        paths = batch.pop("paths")
        labels = batch.pop("labels").to(device)
        inputs = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=amp_enabled):
            outputs = model(**inputs)
            logits = outputs.logits[:, -1, :]
            pos_logits = logits[:, positive_token]
            neg_logits = logits[:, negative_token]
            margin = pos_logits - neg_logits
            loss = F.binary_cross_entropy_with_logits(margin, labels)
        probs = torch.sigmoid(margin)
        preds = (probs >= 0.5).float()

        total_loss += loss.item() * labels.size(0)
        total_samples += labels.size(0)
        total_correct += (preds == labels).sum().item()

        batch_probs = probs.tolist()
        batch_preds = preds.tolist()
        batch_labels = labels.tolist()
        all_probs.extend(batch_probs)
        all_preds.extend(batch_preds)
        all_labels.extend(batch_labels)

        for path, prob, pred, label in zip(paths, batch_probs, batch_preds, batch_labels):
            record = {
                "image_path": path,
                "prob_fake": float(prob),
                "pred_label": int(pred),
                "gt_label": int(label),
            }
            prediction_rows.append(record)
            inference_records.append(record)

    metrics = {
        "samples": int(total_samples),
        "loss": float(total_loss / max(1, total_samples)),
        "accuracy": float(total_correct / max(1, total_samples)),
        "auroc": float(_binary_auroc(all_probs, all_labels)) if total_samples else float("nan"),
        "f1": float(_binary_f1(all_preds, all_labels)) if total_samples else 0.0,
    }

    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    metrics_path = output_dir / cfg.metrics_filename
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not metrics_path.exists()
    dataset_name = Path(cfg.ann_file).resolve().parent.name or Path(cfg.ann_file).stem
    with metrics_path.open("a", newline="", encoding="utf-8") as f:
        fieldnames = [
            "timestamp",
            "dataset",
            "prompt",
            "num_samples",
            "loss",
            "accuracy",
            "auroc",
            "f1",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "dataset": dataset_name,
                "prompt": cfg.prompt_text,
                "num_samples": metrics["samples"],
                "loss": metrics["loss"],
                "accuracy": metrics["accuracy"],
                "auroc": metrics["auroc"],
                "f1": metrics["f1"],
            }
        )

    inference_path = output_dir / cfg.inference_filename
    inference_path.parent.mkdir(parents=True, exist_ok=True)
    inference_path.write_text(json.dumps(inference_records, indent=2), encoding="utf-8")

    if cfg.dump_predictions:
        pred_path = Path(cfg.dump_predictions)
        if not pred_path.is_absolute():
            pred_path = output_dir / pred_path
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        with pred_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["image_path", "prob_fake", "pred_label", "gt_label"])
            writer.writeheader()
            writer.writerows(prediction_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Task2 LoRA checkpoints")
    default_cfg = Path(__file__).with_name("config_eval.json")
    parser.add_argument("--config", type=Path, default=default_cfg)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    evaluate(cfg)


if __name__ == "__main__":
    main()
