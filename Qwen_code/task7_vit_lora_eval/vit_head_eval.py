# -*- coding: utf-8 -*-
"""Task7 evaluation: apply ViT LoRA adapters and a frozen forensic head."""

from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from peft import PeftModel
from peft.tuners.lora import LoraLayer
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

import sys
TASK1_DIR = Path(__file__).resolve().parents[1] / "task1_modular_lora" / "subtask_vit_head_eval"
if str(TASK1_DIR) not in sys.path:
    sys.path.append(str(TASK1_DIR))

from test import (  # noqa: E402
    ForgeryJointValDataset,
    QwenVisualTap,
    ForensicJoint,
    collate_joint_test,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Task7 vision LoRA + frozen head")
    default_cfg = Path(__file__).with_name("config_eval.json")
    parser.add_argument("--config", type=Path, default=default_cfg, help="Path to config JSON")
    return parser.parse_args()


def load_config(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def prepare_environment(cfg: Dict) -> None:
    work = cfg.get("working_dir")
    if work:
        work_dir = Path(work).expanduser()
        work_dir.mkdir(parents=True, exist_ok=True)
        os.chdir(work_dir)
    print("Current working directory:", os.getcwd())
    seed = cfg.get("seed")
    if seed is not None:
        set_seed(seed)


def _log_lora_injection(peft_model: PeftModel, scope_keyword: Optional[str] = "visual") -> None:
    try:
        base_model = peft_model.get_base_model()
    except AttributeError:
        base_model = peft_model

    injected: List[str] = []
    for name, module in base_model.named_modules():
        if isinstance(module, LoraLayer):
            if scope_keyword and scope_keyword not in name:
                continue
            injected.append(name)

    unique_modules = sorted(set(injected))
    if not unique_modules:
        scope_msg = f" containing '{scope_keyword}'" if scope_keyword else ""
        print(f"[WARN] No LoRA modules detected on the base model{scope_msg}.")
        return

    print(f"Visual LoRA injected into {len(unique_modules)} modules:")
    for module_name in unique_modules:
        print(" -", module_name)


def load_visual_lora_weights(model: nn.Module, cfg: Dict) -> nn.Module:
    visual_cfg = cfg.get("visual_lora", {})
    adapter_path = visual_cfg.get("path") or cfg.get("visual_lora_path")
    if not adapter_path:
        raise ValueError("Configuration must include visual_lora.path for pretrained vision adapters.")

    adapter_dir = Path(adapter_path).expanduser()
    if adapter_dir.is_file():
        adapter_dir = adapter_dir.parent
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Visual LoRA adapter directory not found: {adapter_dir}")

    print("Loading pretrained visual LoRA from", adapter_dir)
    peft_model = PeftModel.from_pretrained(model, adapter_dir, is_trainable=False)
    _log_lora_injection(peft_model, scope_keyword="visual")
    print("Merging visual LoRA weights into the base model for evaluation.")
    merged_model = peft_model.merge_and_unload()
    return merged_model


def build_model_and_head(cfg: Dict) -> Tuple[nn.Module, AutoProcessor, QwenVisualTap, ForensicJoint, torch.device]:
    dtype_name = cfg.get("torch_dtype")
    torch_dtype = getattr(torch, dtype_name) if dtype_name else torch.bfloat16
    base_model_path = cfg["base_model_path"]

    print("Loading base model from", base_model_path)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model_path,
        torch_dtype=torch_dtype,
    )

    model = load_visual_lora_weights(model, cfg)

    for param in model.parameters():
        param.requires_grad = False

    processor = AutoProcessor.from_pretrained(base_model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    base_model = model.get_base_model() if hasattr(model, "get_base_model") else model
    visual_cfg = cfg.get("visual_lora", {})
    visual_layers: Iterable[int] = visual_cfg.get("layers") or cfg.get("visual_layers") or [7, 15, 23, 31]
    visual_layers = tuple(int(v) for v in visual_layers)
    visual_tap = QwenVisualTap(base_model.visual, layers=visual_layers).to(device)
    heads = ForensicJoint(layers=visual_layers).to(device)

    if cfg.get("head_checkpoint"):
        head_path = Path(cfg["head_checkpoint"]).expanduser()
        print("Loading frozen head checkpoint from", head_path)
        head_state = torch.load(head_path, map_location="cpu")
        if isinstance(head_state, dict) and "state_dict" in head_state:
            head_state = head_state["state_dict"]
        missing, unexpected = heads.load_state_dict(head_state, strict=False)
        if missing:
            print(f"[WARN] Missing keys while loading head: {missing}")
        if unexpected:
            print(f"[WARN] Unexpected keys while loading head: {unexpected}")
    else:
        raise ValueError("head_checkpoint must be provided for evaluation.")

    heads.eval()
    for param in heads.parameters():
        param.requires_grad = False

    return model, processor, visual_tap, heads, device


def build_dataloader(processor: AutoProcessor, cfg: Dict) -> DataLoader:
    ann_path = cfg.get("ann_eval") or cfg.get("annotation_file")
    if not ann_path:
        raise ValueError("Evaluation annotation path is required (ann_eval in config).")
    dataset = ForgeryJointValDataset(ann_path, data_root=cfg.get("data_root"))
    batch_size = cfg.get("batch_size", 2)
    num_workers = cfg.get("num_workers", 4)
    image_size = cfg.get("image_size", 448)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=lambda batch: collate_joint_test(
            batch,
            processor,
            image_size,
            prompt_text=cfg.get("prompt", "."),
        ),
    )


def evaluate(cfg: Dict) -> Tuple[Dict, List[Dict]]:
    prepare_environment(cfg)
    model, processor, visual_tap, heads, device = build_model_and_head(cfg)
    dataloader = build_dataloader(processor, cfg)

    thr_cls = float(cfg.get("classification_threshold", 0.5))
    thr_map = float(cfg.get("heatmap_threshold", 0.5))
    only_fake = bool(cfg.get("heatmap_only_fake", True))
    skip_empty = bool(cfg.get("heatmap_skip_empty", True))

    all_probs: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []
    records: List[Dict] = []

    inter_sum = union_sum = 0.0
    dice_num = dice_den = 0.0
    heatmap_samples = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Eval", dynamic_ncols=True):
            if len(batch) == 5:
                inputs, labels, masks, paths, _ = batch
            else:
                inputs, labels, masks, paths = batch
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)
            masks = masks.to(device)

            grids = visual_tap(inputs["pixel_values"], inputs["image_grid_thw"])
            grids = {k: v.float() for k, v in grids.items()}
            cls_logits, hm_logits = heads(grids)

            prob = torch.sigmoid(cls_logits).detach().to(device="cpu", dtype=torch.float32)
            labels_cpu = labels.detach().to(device="cpu", dtype=torch.float32)

            all_probs.append(prob)
            all_labels.append(labels_cpu)

            prob_np = prob.numpy()
            label_np = labels_cpu.numpy().astype(int)
            pred_np = (prob_np >= thr_cls).astype(int)
            for path, label_val, prob_val, pred_val in zip(paths, label_np, prob_np, pred_np):
                records.append(
                    {
                        "image_path": str(path),
                        "image_label": int(label_val),
                        "fake_probability": float(prob_val),
                        "predicted_label": int(pred_val),
                    }
                )

            if hm_logits.dim() == 3:
                hm_logits = hm_logits.unsqueeze(1)
            hm_prob = torch.sigmoid(hm_logits)
            hm_prob = F.interpolate(hm_prob, size=(448, 448), mode="bilinear", align_corners=False)
            gt = masks.unsqueeze(1).float()

            if only_fake:
                keep = (labels > 0.5).view(-1)
                hm_prob = hm_prob[keep]
                gt = gt[keep]
            if skip_empty and hm_prob.numel() > 0:
                has_gt = (gt.view(gt.size(0), -1).sum(dim=1) > 0)
                hm_prob = hm_prob[has_gt]
                gt = gt[has_gt]

            if hm_prob.numel() == 0:
                continue

            pred = (hm_prob >= thr_map).float()
            gt_bin = (gt >= thr_map).float()

            inter = (pred * gt_bin).sum().item()
            union = (pred + gt_bin - pred * gt_bin).sum().item() + 1e-6
            dice_n = 2 * inter
            dice_d = (pred.sum() + gt_bin.sum()).item() + 1e-6

            inter_sum += inter
            union_sum += union
            dice_num += dice_n
            dice_den += dice_d
            heatmap_samples += int(pred.size(0))

    if not all_probs:
        raise RuntimeError("No samples were evaluated. Check the evaluation annotation path.")

    y_prob = torch.cat(all_probs).numpy()
    y_true = torch.cat(all_labels).numpy().astype(int)

    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    try:
        auroc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        auroc = float("nan")
    y_pred = (y_prob >= thr_cls).astype(int)
    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    metrics: Dict[str, float] = {
        "timestamp": datetime.utcnow().isoformat(),
        "auroc": auroc,
        "acc": acc,
        "f1": f1,
    }

    if heatmap_samples > 0:
        metrics["mean_iou"] = float(inter_sum / union_sum)
        metrics["mean_dice"] = float(dice_num / dice_den)
        metrics["heatmap_samples"] = int(heatmap_samples)
    else:
        metrics["mean_iou"] = 0.0
        metrics["mean_dice"] = 0.0
        metrics["heatmap_samples"] = 0

    print("Evaluation metrics:", metrics)
    return metrics, records


def append_metrics_to_csv(cfg: Dict, metrics: Dict) -> Path:
    csv_path = Path(cfg.get("metrics_csv", "task7_eval_metrics.csv"))
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["timestamp", "auroc", "acc", "f1", "mean_iou", "mean_dice", "heatmap_samples"],
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(metrics)
    return csv_path


def save_inference(cfg: Dict, records: List[Dict]) -> Path:
    output_path = Path(cfg.get("inference_output", "task7_inference_results.json"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    return output_path


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    metrics, records = evaluate(cfg)
    csv_path = append_metrics_to_csv(cfg, metrics)
    inference_path = save_inference(cfg, records)
    print(f"Metrics appended to {csv_path}")
    print(f"Inference results saved to {inference_path}")


if __name__ == "__main__":
    main()
