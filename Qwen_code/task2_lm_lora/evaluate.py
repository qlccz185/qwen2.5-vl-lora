"""Evaluate a Task2 LM-supervised LoRA checkpoint on raw annotations."""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

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


@dataclass
class EvalConfig:
    base_model_path: str
    annotation_file: str
    data_root: str
    prompt: str
    output_dir: str
    lora_weights: Optional[str] = None
    image_size: int = 448
    batch_size: int = 2
    torch_dtype: str | None = "bfloat16"
    positive_response: str = " yes"
    negative_response: str = " no"
    dump_predictions: Optional[str] = None


def load_config(path: Path) -> EvalConfig:
    data = json.loads(path.read_text(encoding="utf-8"))
    return EvalConfig(**data)


def evaluate(cfg: EvalConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = get_dtype(cfg.torch_dtype)
    amp_enabled = device.type == "cuda" and dtype in (torch.float16, torch.bfloat16)
    autocast_dtype = dtype if dtype in (torch.float16, torch.bfloat16) else torch.float32

    processor = AutoProcessor.from_pretrained(cfg.base_model_path, local_files_only=True)
    yes_token_id = prepare_token_id(processor, cfg.positive_response)
    no_token_id = prepare_token_id(processor, cfg.negative_response)

    dataset = ForgeryBinaryDataset(cfg.annotation_file, cfg.data_root)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=Task2Collator(processor, cfg.prompt, image_size=cfg.image_size),
    )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        cfg.base_model_path,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        local_files_only=True,
    )
    if cfg.lora_weights:
        model = PeftModel.from_pretrained(model, cfg.lora_weights)
    model.eval()

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prediction_rows = []
    total_loss = 0.0
    total_samples = 0
    total_correct = 0

    for batch in tqdm(dataloader, desc="Evaluating"):
        paths = batch.pop("paths")
        labels = batch.pop("labels").to(device)
        inputs = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=amp_enabled):
            outputs = model(**inputs)
            logits = outputs.logits[:, -1, :]
            yes_logits = logits[:, yes_token_id]
            no_logits = logits[:, no_token_id]
            margin = yes_logits - no_logits
            loss = F.binary_cross_entropy_with_logits(margin, labels)
        probs = torch.sigmoid(margin)
        preds = (probs >= 0.5).float()

        total_loss += loss.item() * labels.size(0)
        total_samples += labels.size(0)
        total_correct += (preds == labels).sum().item()

        for path, prob, pred, label in zip(paths, probs.tolist(), preds.tolist(), labels.tolist()):
            prediction_rows.append(
                {
                    "image_path": path,
                    "prob_fake": float(prob),
                    "pred_label": int(pred),
                    "gt_label": int(label),
                }
            )

    metrics = {
        "samples": int(total_samples),
        "loss": float(total_loss / max(1, total_samples)),
        "accuracy": float(total_correct / max(1, total_samples)),
    }

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

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
