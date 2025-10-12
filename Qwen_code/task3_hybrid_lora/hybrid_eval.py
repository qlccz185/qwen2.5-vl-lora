# -*- coding: utf-8 -*-
"""Hybrid Task3 evaluation pipeline aligned with the training configuration."""

import argparse
import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file as load_safetensors
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
    """Project fused ViT/head signals into LM token embeddings."""

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

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch = features.size(0)
        fused = self.net(features)
        return fused.view(batch, self.token_count, self.hidden_size)


def parse_args():
    parser = argparse.ArgumentParser(description="Task3 hybrid evaluation")
    default_cfg = Path(__file__).with_name("config_hybrid_eval.json")
    parser.add_argument("--config", type=Path, default=default_cfg, help="Path to config JSON")
    return parser.parse_args()


def load_config(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def prepare_environment(cfg: Dict):
    work = cfg.get("working_dir")
    if work:
        work_dir = Path(work).expanduser()
        work_dir.mkdir(parents=True, exist_ok=True)
        os.chdir(work_dir)
    print("Current working directory:", os.getcwd())
    seed = cfg.get("seed")
    if seed is not None:
        set_seed(seed)


def find_modules_by_patterns(model: nn.Module, patterns: Sequence[str]) -> list[str]:
    matches: set[str] = set()
    if not patterns:
        return []
    for name, module in model.named_modules():
        if not hasattr(module, "weight"):
            continue
        for pat in patterns:
            if pat and pat in name:
                matches.add(name)
                break
    return sorted(matches)


def build_lora_config(model: nn.Module, cfg: Dict) -> LoraConfig:
    lora_cfg = cfg.get("lora", {})
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

    print("LoRA target modules ({}):".format(len(target_modules)))
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


def load_lora_checkpoint(model, ckpt_path: Path):
    if not ckpt_path.exists():
        raise FileNotFoundError(f"LoRA checkpoint not found: {ckpt_path}")

    def load_from_path(path: Path):
        if path.suffix == ".safetensors":
            return load_safetensors(path)
        return torch.load(path, map_location="cpu")

    if ckpt_path.is_dir():
        candidates = [
            ckpt_path / "adapter_model.safetensors",
            ckpt_path / "adapter_model.bin",
            ckpt_path / "pytorch_model.bin",
        ]
        adapter_path = next((p for p in candidates if p.exists()), None)
        if adapter_path is None:
            raise FileNotFoundError(
                f"No adapter checkpoint found inside directory {ckpt_path}"
            )
        state = load_from_path(adapter_path)
    else:
        state = load_from_path(ckpt_path)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print("Missing keys when loading LoRA state:", missing)
    if unexpected:
        print("Unexpected keys when loading LoRA state:", unexpected)


def build_model_and_modules(cfg: Dict):
    dtype_name = cfg.get("torch_dtype")
    torch_dtype = getattr(torch, dtype_name) if dtype_name else torch.bfloat16
    base_model_path = cfg["base_model_path"]

    print("Loading base model from", base_model_path)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model_path,
        torch_dtype=torch_dtype,
    )

    lora_config = build_lora_config(model, cfg)
    model = get_peft_model(model, lora_config)

    if cfg.get("lora_checkpoint"):
        load_lora_checkpoint(model, Path(cfg["lora_checkpoint"]).expanduser())

    processor = AutoProcessor.from_pretrained(base_model_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    base_model = model.get_base_model() if hasattr(model, "get_base_model") else model
    visual_layers = tuple(int(v) for v in cfg.get("visual_layers", [7, 15, 23, 31]))
    visual_tap = QwenVisualTap(base_model.visual, layers=visual_layers).to(device)
    heads = ForensicJoint(layers=visual_layers).to(device)

    fusion_tokens = int(cfg.get("fusion_tokens", 4))
    fusion_dim = heads.cls.fc1.in_features + 4
    fusion_projector = FusionProjector(
        input_dim=fusion_dim,
        hidden_size=model.config.hidden_size,
        token_count=fusion_tokens,
        dropout=cfg.get("fusion_hidden_dropout", 0.1),
    ).to(device)

    if cfg.get("head_checkpoint"):
        head_path = Path(cfg["head_checkpoint"]).expanduser()
        print("Loading head checkpoint from", head_path)
        heads.load_state_dict(torch.load(head_path, map_location=device))
    heads.eval()

    if cfg.get("fusion_checkpoint"):
        fusion_path = Path(cfg["fusion_checkpoint"]).expanduser()
        print("Loading fusion projector from", fusion_path)
        fusion_projector.load_state_dict(torch.load(fusion_path, map_location=device))
    fusion_projector.eval()

    return model, processor, tokenizer, visual_tap, heads, fusion_projector, device


def build_dataloader(processor, cfg: Dict) -> DataLoader:
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
        collate_fn=lambda batch: collate_joint_test(batch, processor, image_size),
    )


def build_fusion_features(
    heads: ForensicJoint,
    fused_feature: torch.Tensor,
    cls_logits: torch.Tensor,
    hm_logits: torch.Tensor,
) -> torch.Tensor:
    cls_prob = torch.sigmoid(cls_logits).unsqueeze(1)
    if hm_logits.dim() == 3:
        hm_logits = hm_logits.unsqueeze(1)
    hm_prob = torch.sigmoid(hm_logits)
    hm_mean = hm_prob.mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1)
    hm_max = hm_prob.amax(dim=(1, 2, 3), keepdim=False).unsqueeze(1)
    pooled = heads.cls.pool(fused_feature)
    return torch.cat([cls_prob, cls_logits.unsqueeze(1), hm_mean, hm_max, pooled], dim=1)


def resolve_label_tokens(tokenizer, cfg: Dict) -> tuple[int, int]:
    target_cfg = cfg.get("target_text", {"fake": "Fake", "real": "Real"})
    fake_text = target_cfg.get("fake", "Fake")
    real_text = target_cfg.get("real", "Real")

    fake_ids = tokenizer(fake_text, add_special_tokens=False)["input_ids"]
    real_ids = tokenizer(real_text, add_special_tokens=False)["input_ids"]
    if not fake_ids:
        raise ValueError(f"Empty tokenization result for fake label text: {fake_text}")
    if not real_ids:
        raise ValueError(f"Empty tokenization result for real label text: {real_text}")
    return fake_ids[0], real_ids[0]


def evaluate(cfg: Dict):
    prepare_environment(cfg)
    model, processor, tokenizer, visual_tap, heads, fusion_projector, device = build_model_and_modules(cfg)
    dataloader = build_dataloader(processor, cfg)

    fusion_tokens = int(cfg.get("fusion_tokens", 4))
    fake_token, real_token = resolve_label_tokens(tokenizer, cfg)

    all_probs = []
    all_labels = []
    records: list[Dict] = []

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
            

            prompt_embeds = model.get_input_embeddings()(inputs["input_ids"])
            target_dtype = prompt_embeds.dtype
            fusion_embeds = fusion_projector(fusion_features).to(dtype=target_dtype)
            prompt_embeds = prompt_embeds.to(dtype=target_dtype)
            inputs_embeds = torch.cat([fusion_embeds, prompt_embeds], dim=1)

            attention_mask = torch.cat(
                [
                    torch.ones((inputs_embeds.size(0), fusion_tokens), dtype=inputs["attention_mask"].dtype, device=device),
                    inputs["attention_mask"],
                ],
                dim=1,
            )

            outputs = model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                pixel_values=inputs["pixel_values"],
                image_grid_thw=inputs["image_grid_thw"],
            )
            logits = outputs.logits
            prompt_len = inputs["input_ids"].size(1)
            next_token_logits = logits[:, fusion_tokens + prompt_len - 1, :]
            probs = torch.softmax(next_token_logits[:, [fake_token, real_token]], dim=-1)
            fake_prob = probs[:, 0].detach().to(dtype=torch.float32, device="cpu")
            labels_cpu = labels.detach().cpu()

            all_probs.append(fake_prob)
            all_labels.append(labels_cpu)

            prob_np = fake_prob.numpy()
            label_np = labels_cpu.numpy().astype(int)
            pred_np = (prob_np >= 0.5).astype(int)
            for path, label_val, prob_val, pred_val in zip(paths, label_np, prob_np, pred_np):
                records.append(
                    {
                        "image_path": str(path),
                        "image_label": int(label_val),
                        "fake_probability": float(prob_val),
                        "predicted_label": int(pred_val),
                    }
                )

    if not all_probs:
        raise RuntimeError("No samples were evaluated. Check the evaluation annotation path.")

    y_prob = torch.cat(all_probs).numpy()
    y_true = torch.cat(all_labels).numpy().astype(int)

    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    try:
        auroc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        auroc = float("nan")
    y_pred = (y_prob >= 0.5).astype(int)
    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    metrics = {
        "timestamp": datetime.utcnow().isoformat(),
        "auroc": auroc,
        "acc": acc,
        "f1": f1,
    }
    print("Evaluation metrics:", metrics)
    return metrics, records


def append_metrics_to_csv(cfg: Dict, metrics: Dict) -> Path:
    csv_path = Path(cfg.get("metrics_csv", "task3_eval_metrics.csv"))
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "auroc", "acc", "f1"])
        if not file_exists:
            writer.writeheader()
        writer.writerow(metrics)
    return csv_path


def save_inference(cfg: Dict, records: list[Dict]) -> Path:
    output_path = Path(cfg.get("inference_output", "task3_inference_results.json"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    return output_path


def main():
    args = parse_args()
    cfg = load_config(args.config)
    metrics, records = evaluate(cfg)
    csv_path = append_metrics_to_csv(cfg, metrics)
    inference_path = save_inference(cfg, records)
    print(f"Metrics appended to {csv_path}")
    print(f"Inference results saved to {inference_path}")


if __name__ == "__main__":
    main()
