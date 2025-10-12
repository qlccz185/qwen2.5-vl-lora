# -*- coding: utf-8 -*-
import argparse
import os
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    get_cosine_schedule_with_warmup,
)

from test import (
    ForgeryJointValDataset,
    collate_joint_test,
    QwenVisualTap,
    ForensicJoint,
    set_seed,
)
from peft import LoraConfig, get_peft_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune Qwen2.5-VL forensic heads with LoRA using configuration file",
    )
    default_cfg = Path(__file__).with_name("config_lora.json")
    parser.add_argument(
        "--config",
        type=Path,
        default=default_cfg,
        help="Path to the JSON configuration file.",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def prepare_environment(cfg: dict):
    working_dir = cfg.get("working_dir")
    if working_dir:
        os.chdir(working_dir)
    print("Current working directory:", os.getcwd())
# --------------------------
# LoRA utility
# --------------------------

def build_lora_on_qwen_visual(qwen_visual, r=8, alpha=16, dropout=0.05, target_layers=[7, 15, 23, 31]):
    """
    在 Qwen2.5-VL 的视觉 backbone (qwen.visual.blocks)
    的第 7、15、23、31 层注入 LoRA。
    """
    from peft import LoraConfig, get_peft_model

    target_layers = target_layers
    target_modules = []

    for i in target_layers:
        target_modules += [
            f"blocks.{i}.attn.qkv",
            f"blocks.{i}.attn.proj",
            f"blocks.{i}.mlp.gate_proj",
            f"blocks.{i}.mlp.up_proj",
            f"blocks.{i}.mlp.down_proj",
        ]

    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        target_modules=target_modules,
        task_type="FEATURE_EXTRACTION",
    )

    print("✅ Injecting LoRA into following modules:")
    for name in target_modules:
        print("   ", name)

    qwen_visual_lora = get_peft_model(qwen_visual, config)
    return qwen_visual_lora


def count_trainable_parameters(model):
    t = sum(p.numel() for p in model.parameters())
    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return tr, t


def load_frozen_head_weights(heads: nn.Module, checkpoint_path: str, device: str) -> None:
    ckpt_path = Path(checkpoint_path).expanduser()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Head checkpoint not found: {ckpt_path}")

    print(f"Loading forensic head weights from {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    missing, unexpected = heads.load_state_dict(state, strict=False)
    if missing:
        print(f"[WARN] Missing keys when loading head: {missing}")
    if unexpected:
        print(f"[WARN] Unexpected keys when loading head: {unexpected}")

    heads.to(device)
    heads.eval()
    for param in heads.parameters():
        param.requires_grad = False

# --------------------------
# Training loop
# --------------------------
def train_one_epoch(
    qwen_vis_tap,
    heads,
    dl_train,
    device,
    optimizer,
    scheduler,
    scaler,
    evi_alpha=0.5,
    lambda_evi=2.0,
    lambda_sparse=1e-3,
    lambda_contrast=1e-2,
    grad_accum=1,
    max_norm=1.0,
    amp_dtype=torch.bfloat16,
):
    qwen_vis_tap.train()
    heads.eval()
    optimizer.zero_grad(set_to_none=True)
    running = {"loss": 0.0, "cls": 0.0, "evi": 0.0, "sparse": 0.0, "contrast": 0.0}

    pbar = tqdm(dl_train, desc="Train", dynamic_ncols=True)
    for step, batch in enumerate(pbar, 1):
        if len(batch) == 5:
            (inputs, labels, masks, paths, _) = batch
        else:
            (inputs, labels, masks, paths) = batch

        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = labels.to(device)
        masks = masks.to(device)

        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=(device=="cuda")):
            # 特征提取
            grids = qwen_vis_tap(inputs["pixel_values"], inputs["image_grid_thw"])
            grids = {k: v.float() for k, v in grids.items()}
            logits, hm_logits = heads(grids)   # (B,), (B,Ht,Wt)

            # 1️⃣ 分类损失
            L_cls = F.binary_cross_entropy_with_logits(logits, labels.float())

            # 2️⃣ 证据分支
            if hm_logits.dim() == 3:
                hm_logits = hm_logits.unsqueeze(1)
            prob32 = torch.sigmoid(hm_logits)                         # [B,1,Ht,Wt]
            prob64 = F.interpolate(prob32, scale_factor=2.0, mode="bilinear", align_corners=False)
            mask64 = F.interpolate(masks.unsqueeze(1).float(), size=prob64.shape[-2:], mode="bilinear", align_corners=False).clamp(0,1)

            # BCE + Dice 混合
            L_bce = F.binary_cross_entropy_with_logits(
                torch.logit(prob64.clamp(1e-6,1-1e-6)), mask64)
            inter = (prob64 * mask64).sum(dim=(1,2,3))
            denom = prob64.sum(dim=(1,2,3)) + mask64.sum(dim=(1,2,3)) + 1e-6
            L_dice = 1 - (2 * inter / denom).mean()
            L_evi = evi_alpha * L_bce + (1 - evi_alpha) * L_dice

            # 3️⃣ 稀疏约束（让热图整体更暗）
            L_sparse = prob64.mean()

            # 4️⃣ 对比项（让假图热图更亮，真图更暗）
            real_idx = (labels < 0.5)
            fake_idx = (labels > 0.5)
            if real_idx.any() and fake_idx.any():
                L_contrast = prob64[real_idx].mean() - prob64[fake_idx].mean()
            else:
                L_contrast = torch.zeros((), device=device)

            # 5️⃣ 合成总损失
            loss = (
                L_cls
                + lambda_evi * L_evi
                + lambda_sparse * L_sparse
                + lambda_contrast * L_contrast
            ) / grad_accum

        # backward
        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if step % grad_accum == 0:
            if scaler:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(heads.parameters(), max_norm)
                torch.nn.utils.clip_grad_norm_(qwen_vis_tap.parameters(), max_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(heads.parameters(), max_norm)
                torch.nn.utils.clip_grad_norm_(qwen_vis_tap.parameters(), max_norm)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        # logging
        running["loss"] += loss.item() * grad_accum
        running["cls"] += L_cls.item()
        running["evi"] += L_evi.item()
        running["sparse"] += L_sparse.item()
        running["contrast"] += L_contrast.item()
        pbar.set_postfix({
            "loss": f"{running['loss']/step:.4f}",
            "cls": f"{running['cls']/step:.4f}",
            "evi": f"{running['evi']/step:.4f}",
            "spr": f"{running['sparse']/step:.4f}",
            "con": f"{running['contrast']/step:.4f}",
        })

    for k in running:
        running[k] /= len(dl_train)
    return running


# --------------------------
# Main
# --------------------------
def main():
    cli_args = parse_args()
    cfg = load_config(cli_args.config)
    prepare_environment(cfg)

    set_seed(cfg.get("seed", 42))
    out_dir = Path(cfg["out_dir"]) if "out_dir" in cfg else Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = AutoProcessor.from_pretrained(cfg["model_path"], local_files_only=True)
    ds_train = ForgeryJointValDataset(cfg["ann_train"], data_root=cfg["data_root"])
    dl_train = DataLoader(
        ds_train,
        batch_size=cfg.get("batch_size", 1),
        shuffle=True,
        num_workers=cfg.get("num_workers", 0),
        pin_memory=True,
        collate_fn=lambda b: collate_joint_test(b, processor, cfg.get("image_size", 448)),
        drop_last=True,
    )

    # model
    base_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    dtype_name = cfg.get("torch_dtype") or cfg.get("model_dtype")
    if dtype_name:
        base_dtype = getattr(torch, dtype_name)

    qwen = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        cfg["model_path"],
        torch_dtype=base_dtype,
        device_map="auto",
    )
    for p in qwen.parameters():
        p.requires_grad = False
    qwen.eval()

    qwen.visual = build_lora_on_qwen_visual(
        qwen.visual,
        r=cfg.get("lora_r", 8),
        alpha=cfg.get("lora_alpha", 16),
        dropout=cfg.get("lora_dropout", 0.05),
        target_layers=cfg.get("lora_target_layers", [7, 15, 23, 31]),
    )
    qwen.visual.train()
    visual_layers = tuple(cfg.get("visual_layers", [7, 15, 23, 31]))
    visual_tap = QwenVisualTap(qwen.visual, layers=visual_layers).to(device)
    heads = ForensicJoint(fuse_in_ch=1280, fuse_out_ch=512, layers=visual_layers).to(device)

    head_ckpt = cfg.get("head_checkpoint")
    if not head_ckpt:
        raise ValueError("Config must provide head_checkpoint for frozen head weights.")
    load_frozen_head_weights(heads, head_ckpt, device)

    # optimizer (LoRA only)
    lora_params = [p for p in visual_tap.parameters() if p.requires_grad]
    grad_accum = cfg.get("grad_accum", 1)
    epochs = cfg.get("epochs", 1)
    warmup_ratio = cfg.get("warmup_ratio", 0.0)
    steps_per_epoch = math.ceil(len(dl_train) / max(1, grad_accum))
    total_steps = max(1, epochs * steps_per_epoch)
    warmup_steps = int(total_steps * warmup_ratio)

    optimizer = torch.optim.AdamW(
        [
            {
                "params": lora_params,
                "lr": cfg.get("lr_lora", 1e-4),
                "weight_decay": cfg.get("weight_decay", 0.0),
            }
        ]
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # AMP setup
    amp_mode = str(cfg.get("amp_dtype", "bf16")).lower()
    if amp_mode == "bf16":
        amp_dtype = torch.bfloat16
        scaler = None
    elif amp_mode == "fp16":
        amp_dtype = torch.float16
        scaler = torch.cuda.amp.GradScaler()
    else:
        amp_dtype = torch.float32
        scaler = None

    # 🔹 创建日志 CSV 文件
    import csv
    from datetime import datetime

    csv_path = out_dir / cfg.get("train_log", "train_log_lora_joint.csv")
    file_exists = csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["timestamp", "epoch", "loss", "cls", "evi", "sparse", "contrast"],
        )
        if not file_exists:
            writer.writeheader()

    # train loop
    print("\n=== Start Training (No Eval) ===")
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")

        stats = train_one_epoch(
            visual_tap,
            heads,
            dl_train,
            device,
            optimizer,
            scheduler,
            scaler,
            evi_alpha=cfg.get("evi_alpha", 0.5),
            lambda_evi=cfg.get("lambda_evi", 2.0),
            lambda_sparse=cfg.get("lambda_sparse", 1e-3),
            lambda_contrast=cfg.get("lambda_contrast", 1e-2),
            grad_accum=grad_accum,
            max_norm=cfg.get("max_norm", 1.0),
            amp_dtype=amp_dtype,
        )

        print(
            f"Epoch {epoch} done | loss={stats['loss']:.4f} cls={stats['cls']:.4f} evi={stats['evi']:.4f}"
        )

        # 🔹 将结果追加写入 CSV
        row = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "epoch": epoch,
            "loss": stats["loss"],
            "cls": stats["cls"],
            "evi": stats["evi"],
            "sparse": stats["sparse"],
            "contrast": stats["contrast"],
        }
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writerow(row)

    # save model at end
    save_path = out_dir / cfg.get("final_trained_heads", "final_trained_heads.pt")
    torch.save(
        {
            "epoch": epochs,
            "state_dict": heads.state_dict(),
            "visual_lora": qwen.visual.state_dict(),
            "head_checkpoint": str(Path(head_ckpt).expanduser()),
            "args": cfg,
        },
        save_path,
    )
    print(f"[SAVE] Model saved to {save_path}")

if __name__ == "__main__":
    main()

