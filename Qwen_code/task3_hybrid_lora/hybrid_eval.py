# -*- coding: utf-8 -*-
"""Evaluate Task3 hybrid LoRA by fusing head outputs into LM decisions."""

import argparse
import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from peft import PeftModel
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
        B = x.size(0)
        fused = self.net(x)
        return fused.view(B, self.token_count, self.hidden_size)


def parse_args():
    parser = argparse.ArgumentParser(description="Task3 hybrid evaluation")
    default_cfg = Path(__file__).with_name("config_hybrid_eval.json")
    parser.add_argument("--config", type=Path, default=default_cfg, help="Path to config JSON")
    return parser.parse_args()


def load_config(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def prepare_environment(cfg: Dict):
    work = cfg.get("working_dir")
    if work:
        os.chdir(work)
    print("Current working directory:", os.getcwd())
    seed = cfg.get("seed")
    if seed is not None:
        set_seed(seed)


def load_model(cfg: Dict):
    dtype_name = cfg.get("torch_dtype")
    torch_dtype = getattr(torch, dtype_name) if dtype_name else torch.bfloat16
    base_model_path = cfg["base_model_path"]

    print("Loading base model from", base_model_path)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model_path,
        torch_dtype=torch_dtype,
    )

    lora_ckpt = cfg.get("lora_checkpoint")
    if lora_ckpt:
        print("Loading LoRA adapters from", lora_ckpt)
        model = PeftModel.from_pretrained(model, lora_ckpt)
        model = model.merge_and_unload()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    processor = AutoProcessor.from_pretrained(base_model_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, processor, tokenizer, device


def build_modules(model, cfg: Dict, device):
    visual_layers = tuple(cfg.get("visual_layers", [7, 15, 23, 31]))
    visual_tap = QwenVisualTap(model.visual, layers=visual_layers).to(device)
    heads = ForensicJoint(layers=visual_layers).to(device)
    fusion_dim = heads.cls.fc1.in_features + 4
    fusion_tokens = int(cfg.get("fusion_tokens", 4))
    fusion_projector = FusionProjector(
        input_dim=fusion_dim,
        hidden_size=model.config.hidden_size,
        token_count=fusion_tokens,
        dropout=cfg.get("fusion_hidden_dropout", 0.1),
    ).to(device)

    head_ckpt = cfg.get("head_checkpoint")
    if head_ckpt:
        print("Loading head checkpoint from", head_ckpt)
        heads.load_state_dict(torch.load(head_ckpt, map_location=device))
    heads.eval()

    fusion_ckpt = cfg.get("fusion_checkpoint")
    if fusion_ckpt:
        print("Loading fusion projector from", fusion_ckpt)
        fusion_projector.load_state_dict(torch.load(fusion_ckpt, map_location=device))
    fusion_projector.eval()

    return visual_tap, heads, fusion_projector


def build_dataloader(processor, cfg: Dict) -> DataLoader:
    dataset = ForgeryJointValDataset(cfg["annotation_file"], data_root=cfg.get("data_root"))
    batch_size = cfg.get("batch_size", 2)
    num_workers = cfg.get("num_workers", 2)
    image_size = cfg.get("image_size", 448)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=lambda batch: collate_joint_test(batch, processor, image_size),
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


def evaluate(cfg: Dict):
    prepare_environment(cfg)
    model, processor, tokenizer, device = load_model(cfg)
    visual_tap, heads, fusion_projector = build_modules(model, cfg, device)
    dataloader = build_dataloader(processor, cfg)

    fusion_tokens = int(cfg.get("fusion_tokens", 4))
    fake_token = first_token_id(tokenizer, cfg.get("fake_token", " Fake"))
    real_token = first_token_id(tokenizer, cfg.get("real_token", " Real"))

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Eval", dynamic_ncols=True):
            inputs, labels, masks = batch[0], batch[1], batch[2]
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
            prompt_embeds = model.get_input_embeddings()(inputs["input_ids"])
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
            fake_prob = probs[:, 0].detach().cpu()
            all_probs.append(fake_prob)
            all_labels.append(labels.detach().cpu())

    y_prob = torch.cat(all_probs).numpy()
    y_true = torch.cat(all_labels).numpy().astype(int)

    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    auroc = float(roc_auc_score(y_true, y_prob))
    y_pred = (y_prob >= 0.5).astype(int)
    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred))

    metrics = {"timestamp": datetime.utcnow().isoformat(), "auroc": auroc, "acc": acc, "f1": f1}
    print("Evaluation metrics:", metrics)

    csv_path = Path(cfg.get("csv_path", "task3_eval_metrics.csv"))
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "auroc", "acc", "f1"])
        if not file_exists:
            writer.writeheader()
        writer.writerow(metrics)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    evaluate(cfg)


if __name__ == "__main__":
    main()
