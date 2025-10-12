# -*- coding: utf-8 -*-
"""Evaluate Task3 hybrid LoRA by fusing head outputs into LM decisions."""

import argparse
import csv
import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from peft import LoraConfig, get_peft_model
from transformers import AutoProcessor, AutoTokenizer, Qwen2_5_VLForConditionalGeneration

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


class FusionProjector(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, token_count: int, dropout: float = 0.1):
        super().__init__()
        self.token_count = token_count
        self.hidden_size = hidden_size
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size * token_count),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * token_count, hidden_size * token_count),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fused = self.net(x)
        batch = x.size(0)
        return fused.view(batch, self.token_count, self.hidden_size)


@dataclass
class HybridEvalConfig:
    base_model_path: str
    data_root: str
    annotation_file: str
    csv_path: str = "task3_eval_metrics.csv"
    output_json: str = "inference_results.json"
    lora_checkpoint: Optional[str] = None
    head_checkpoint: Optional[str] = None
    fusion_checkpoint: Optional[str] = None
    working_dir: Optional[str] = None
    seed: Optional[int] = None

    batch_size: int = 2
    num_workers: int = 2
    image_size: int = 448
    torch_dtype: str = "bfloat16"
    fusion_tokens: int = 4
    fusion_hidden_dropout: float = 0.1
    visual_layers: Sequence[int] = (7, 15, 23, 31)
    lora: Dict[str, object] = field(default_factory=dict)
    prompt: str = (
        "Please determine whether this image is Real or Fake. Answer only with Fake or Real."
    )
    target_text: Dict[str, str] = field(
        default_factory=lambda: {"fake": "Fake", "real": "Real"}
    )
    fake_token: Optional[str] = None
    real_token: Optional[str] = None
    mapped_dataset_path: Optional[str] = None  # deprecated, kept for compatibility


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task3 hybrid evaluation")
    default_cfg = Path(__file__).with_name("config_hybrid_eval.json")
    parser.add_argument("--config", type=Path, default=default_cfg, help="Path to config JSON")
    return parser.parse_args()


def load_config(path: Path) -> HybridEvalConfig:
    data = json.loads(path.read_text(encoding="utf-8"))
    return HybridEvalConfig(**data)


def prepare_environment(cfg: HybridEvalConfig) -> None:
    if cfg.working_dir:
        work_dir = Path(cfg.working_dir).expanduser().resolve()
        work_dir.mkdir(parents=True, exist_ok=True)
        os.chdir(work_dir)
    print("Current working directory:", os.getcwd())
    if cfg.seed is not None:
        set_seed(cfg.seed)


def find_modules_by_patterns(
    model: nn.Module, patterns: Optional[Sequence[str]]
) -> list[str]:
    matches: set[str] = set()
    for name, module in model.named_modules():
        if not hasattr(module, "weight"):
            continue
        for pattern in patterns or []:
            if pattern and pattern in name:
                matches.add(name)
                break
    return sorted(matches)


def build_lora_config(model: nn.Module, cfg: HybridEvalConfig) -> Optional[LoraConfig]:
    lora_cfg: Dict[str, object] = cfg.lora or {}
    if not lora_cfg:
        return None

    target_modules: set[str] = set(lora_cfg.get("target_modules", []))

    visual_cfg = lora_cfg.get("visual", {})
    if visual_cfg.get("enabled", False):
        for idx in visual_cfg.get("blocks", []):
            target_modules.update(
                [
                    f"visual.blocks.{idx}.attn.qkv",
                    f"visual.blocks.{idx}.attn.proj",
                    f"visual.blocks.{idx}.mlp.gate_proj",
                    f"visual.blocks.{idx}.mlp.up_proj",
                    f"visual.blocks.{idx}.mlp.down_proj",
                ]
            )
        target_modules.update(find_modules_by_patterns(model, visual_cfg.get("extra_patterns", [])))
        target_modules.update(visual_cfg.get("extra_modules", []))

    merge_cfg = lora_cfg.get("merge", {})
    if merge_cfg.get("enabled", False):
        target_modules.update(find_modules_by_patterns(model, merge_cfg.get("module_patterns", [])))
        target_modules.update(merge_cfg.get("extra_modules", []))

    lm_cfg = lora_cfg.get("lm", {})
    if lm_cfg.get("enabled", False):
        target_modules.update(find_modules_by_patterns(model, lm_cfg.get("module_patterns", [])))
        target_modules.update(lm_cfg.get("extra_modules", []))

    if not target_modules:
        raise ValueError("No LoRA target modules were resolved. Please check config.lora settings.")

    print(f"LoRA target modules ({len(target_modules)}):")
    for name in sorted(target_modules):
        print(" -", name)

    return LoraConfig(
        r=lora_cfg.get("r", 8),
        lora_alpha=lora_cfg.get("alpha", 16),
        target_modules=sorted(target_modules),
        lora_dropout=lora_cfg.get("dropout", 0.05),
        bias=lora_cfg.get("bias", "none"),
        task_type="CAUSAL_LM",
    )


def load_model(cfg: HybridEvalConfig):
    dtype = getattr(torch, cfg.torch_dtype) if cfg.torch_dtype else torch.bfloat16
    print("Loading base model from", cfg.base_model_path)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        cfg.base_model_path,
        torch_dtype=dtype,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lora_config = build_lora_config(model, cfg)
    if lora_config is not None:
        model = get_peft_model(model, lora_config)
        if cfg.lora_checkpoint:
            ckpt_path = Path(cfg.lora_checkpoint)
            if ckpt_path.is_dir():
                adapter_path = ckpt_path / "adapter_model.bin"
                if not adapter_path.exists():
                    adapter_path = ckpt_path / "pytorch_model.bin"
            else:
                adapter_path = ckpt_path
            print("Loading LoRA adapters from", adapter_path)
            state = torch.load(adapter_path, map_location="cpu")
            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing:
                print("Missing keys when loading LoRA state:", missing)
            if unexpected:
                print("Unexpected keys when loading LoRA state:", unexpected)
    elif cfg.lora_checkpoint:
        raise ValueError("LoRA checkpoint provided but no lora configuration was specified.")

    model.to(device)
    model.eval()

    processor = AutoProcessor.from_pretrained(cfg.base_model_path)
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    base_model = model.get_base_model() if hasattr(model, "get_base_model") else model

    return model, base_model, processor, tokenizer, device


def build_modules(base_model, full_model, cfg: HybridEvalConfig, device):
    visual_layers = tuple(int(v) for v in cfg.visual_layers)
    visual_tap = QwenVisualTap(base_model.visual, layers=visual_layers).to(device)
    heads = ForensicJoint(layers=visual_layers).to(device)
    fusion_dim = heads.cls.fc1.in_features + 4
    fusion_projector = FusionProjector(
        input_dim=fusion_dim,
        hidden_size=full_model.config.hidden_size,
        token_count=int(cfg.fusion_tokens),
        dropout=cfg.fusion_hidden_dropout,
    ).to(device)

    if cfg.head_checkpoint:
        print("Loading head checkpoint from", cfg.head_checkpoint)
        heads.load_state_dict(torch.load(cfg.head_checkpoint, map_location=device))
    heads.eval()

    if cfg.fusion_checkpoint:
        print("Loading fusion projector from", cfg.fusion_checkpoint)
        fusion_projector.load_state_dict(torch.load(cfg.fusion_checkpoint, map_location=device))
    fusion_projector.eval()

    return visual_tap, heads, fusion_projector


def build_dataloader(processor, cfg: HybridEvalConfig) -> DataLoader:
    ann_path = cfg.annotation_file or cfg.mapped_dataset_path
    if not ann_path:
        raise ValueError("Annotation file path must be provided in the evaluation config.")
    dataset = ForgeryJointValDataset(ann_path, data_root=cfg.data_root)
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=lambda batch: collate_joint_test(batch, processor, cfg.image_size),
    )


def build_fusion_features(heads: ForensicJoint, fused_feature: torch.Tensor, cls_logits: torch.Tensor, hm_logits: torch.Tensor):
    cls_prob = torch.sigmoid(cls_logits).unsqueeze(1)
    if hm_logits.dim() == 3:
        hm_logits = hm_logits.unsqueeze(1)
    hm_prob = torch.sigmoid(hm_logits)
    hm_mean = hm_prob.mean(dim=(1, 2, 3), keepdim=True)
    hm_max = hm_prob.amax(dim=(1, 2, 3), keepdim=True)
    pooled = heads.cls.pool(fused_feature)
    return torch.cat([cls_prob, cls_logits.unsqueeze(1), hm_mean, hm_max, pooled], dim=1)


def first_token_id(tokenizer, text: str) -> int:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if not ids:
        raise ValueError(f"Tokenization produced empty ids for text: {text}")
    return ids[0]


def evaluate(cfg: HybridEvalConfig) -> tuple[dict, list[dict]]:
    prepare_environment(cfg)
    full_model, base_model, processor, tokenizer, device = load_model(cfg)
    visual_tap, heads, fusion_projector = build_modules(base_model, full_model, cfg, device)
    dataloader = build_dataloader(processor, cfg)

    fusion_tokens = int(cfg.fusion_tokens)
    target_cfg = cfg.target_text or {"fake": "Fake", "real": "Real"}
    fake_token_text = cfg.fake_token or target_cfg.get("fake", "Fake")
    real_token_text = cfg.real_token or target_cfg.get("real", "Real")
    fake_token = first_token_id(tokenizer, fake_token_text)
    real_token = first_token_id(tokenizer, real_token_text)

    all_probs = []
    all_labels = []
    records: list[dict] = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Eval", dynamic_ncols=True):
            inputs, labels, _, paths, _ = batch
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)

            grids = visual_tap(inputs["pixel_values"], inputs["image_grid_thw"])
            grids = {k: v.float() for k, v in grids.items()}
            layer_grids = [grids[i] for i in heads.layers]
            fused_feature = heads.fuser(layer_grids)
            cls_logits = heads.cls(fused_feature)
            hm_logits = heads.evi(fused_feature)
            fusion_features = build_fusion_features(heads, fused_feature, cls_logits, hm_logits)

            fusion_embeds = fusion_projector(fusion_features)
            prompt_embeds = full_model.get_input_embeddings()(inputs["input_ids"])
            inputs_embeds = torch.cat([fusion_embeds, prompt_embeds], dim=1)

            attention_mask = torch.cat(
                [
                    torch.ones((inputs_embeds.size(0), fusion_tokens), dtype=inputs["attention_mask"].dtype, device=device),
                    inputs["attention_mask"],
                ],
                dim=1,
            )

            outputs = full_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                pixel_values=inputs["pixel_values"],
                image_grid_thw=inputs["image_grid_thw"],
            )
            logits = outputs.logits
            prompt_len = inputs["input_ids"].size(1)
            next_token_logits = logits[:, fusion_tokens + prompt_len - 1, :]
            probs = torch.softmax(next_token_logits[:, [fake_token, real_token]], dim=-1)
            fake_prob = probs[:, 0].detach().cpu()
            labels_cpu = labels.detach().cpu()
            all_probs.append(fake_prob)
            all_labels.append(labels_cpu)

            prob_np = fake_prob.numpy()
            label_np = labels_cpu.numpy().astype(int)
            pred_np = (prob_np >= 0.5).astype(int)
            for path, label_val, prob_val, pred_val in zip(paths, label_np, prob_np, pred_np):
                records.append({
                    "image_path": str(path),
                    "image_label": int(label_val),
                    "fake_probability": float(prob_val),
                    "predicted_label": int(pred_val),
                })

    if not all_probs:
        raise RuntimeError("No samples were evaluated. Check the annotation file path.")

    y_prob = torch.cat(all_probs).numpy()
    y_true = torch.cat(all_labels).numpy().astype(int)

    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    auroc = float(roc_auc_score(y_true, y_prob))
    y_pred = (y_prob >= 0.5).astype(int)
    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred))

    metrics = {
        "timestamp": datetime.utcnow().isoformat(),
        "auroc": auroc,
        "acc": acc,
        "f1": f1,
    }
    print("Evaluation metrics:", metrics)
    return metrics, records


def append_metrics_to_csv(cfg: HybridEvalConfig, metrics: dict) -> Path:
    csv_path = Path(cfg.csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "auroc", "acc", "f1"])
        if not file_exists:
            writer.writeheader()
        writer.writerow(metrics)
    return csv_path


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    metrics, records = evaluate(cfg)
    csv_path = append_metrics_to_csv(cfg, metrics)
    inference_path = Path(cfg.output_json)
    inference_path.parent.mkdir(parents=True, exist_ok=True)
    with inference_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"Metrics appended to {csv_path}")
    print(f"Inference results saved to {inference_path}")


if __name__ == "__main__":
    main()
