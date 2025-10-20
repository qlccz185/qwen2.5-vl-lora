# -*- coding: utf-8 -*-
"""Hybrid Task3 training pipeline that fuses Task1 heads with LM supervision."""

import argparse
import csv
import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    Qwen2_5_VLForConditionalGeneration,
    get_cosine_schedule_with_warmup,
)

# --- import Task1 utilities -------------------------------------------------
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
    """Project fused ViT/head signals into trainable LM token embeddings."""

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
        B = features.size(0)
        fused = self.net(features)
        return fused.view(B, self.token_count, self.hidden_size)


def parse_args():
    parser = argparse.ArgumentParser(description="Task3 hybrid LoRA trainer")
    default_cfg = Path(__file__).with_name("config_hybrid_train.json")
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


def find_modules_by_patterns(model: nn.Module, patterns: Sequence[str]) -> List[str]:
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


def build_lm_lora_config(model: nn.Module, cfg: Dict) -> Optional[LoraConfig]:
    lora_cfg = cfg.get("lora", {})
    lm_cfg = lora_cfg.get("lm", {})
    if not lm_cfg.get("enabled", False):
        print("[INFO] LM LoRA is disabled via configuration.")
        return None

    target_modules: set[str] = set(lm_cfg.get("target_modules", []))
    target_modules.update(find_modules_by_patterns(model, lm_cfg.get("module_patterns", [])))
    target_modules.update(lm_cfg.get("extra_modules", []))

    if not target_modules:
        raise ValueError("LM LoRA enabled but no target modules were resolved from configuration.")

    print("LM LoRA target modules ({}):".format(len(target_modules)))
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
    if visual_cfg.get("layers"):
        print(" - Visual LoRA layers:", visual_cfg.get("layers"))
    peft_model = PeftModel.from_pretrained(model, adapter_dir, is_trainable=False)
    print("Merging visual LoRA weights into the base model and freezing them.")
    merged_model = peft_model.merge_and_unload()
    return merged_model


def load_frozen_head_weights(heads: ForensicJoint, checkpoint_path: str, device: torch.device) -> None:
    ckpt_path = Path(checkpoint_path).expanduser()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Frozen head checkpoint not found: {ckpt_path}")

    print("Loading frozen forensic head from", ckpt_path)
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    missing, unexpected = heads.load_state_dict(state, strict=False)
    if missing:
        print(f"[WARN] Missing keys while loading head: {missing}")
    if unexpected:
        print(f"[WARN] Unexpected keys while loading head: {unexpected}")

    heads.to(device)
    heads.eval()
    for param in heads.parameters():
        param.requires_grad = False


def build_model_and_tools(cfg: Dict):
    dtype_name = cfg.get("torch_dtype")
    torch_dtype = getattr(torch, dtype_name) if dtype_name else torch.bfloat16
    base_model_path = cfg["base_model_path"]

    print("Loading base model from", base_model_path)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model_path,
        torch_dtype=torch_dtype,
    )

    model = load_visual_lora_weights(model, cfg)

    lm_lora_config = build_lm_lora_config(model, cfg)
    lm_lora_enabled = lm_lora_config is not None
    if lm_lora_enabled:
        model.enable_input_require_grads()
        model = get_peft_model(model, lm_lora_config)
    else:
        for param in model.parameters():
            param.requires_grad = False

    # ensure forward pass keeps gradients for custom embeddings even when LM LoRA disabled
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    processor = AutoProcessor.from_pretrained(base_model_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    base_model = model.get_base_model() if hasattr(model, "get_base_model") else model
    visual_cfg = cfg.get("visual_lora", {})
    visual_layers = tuple(visual_cfg.get("layers") or cfg.get("visual_layers", [7, 15, 23, 31]))
    visual_tap = QwenVisualTap(base_model.visual, layers=visual_layers).to(device)
    heads = ForensicJoint(layers=visual_layers).to(device)

    frozen_head_path = cfg.get("frozen_head_path") or cfg.get("head_checkpoint")
    if not frozen_head_path:
        raise ValueError("Configuration must include frozen_head_path for task5 training.")
    load_frozen_head_weights(heads, frozen_head_path, device)

    fusion_tokens = int(cfg.get("fusion_tokens", 4))
    fusion_dim = heads.cls.fc1.in_features + 4  # fused channels + cls/evidence stats
    fusion_projector = FusionProjector(
        input_dim=fusion_dim,
        hidden_size=model.config.hidden_size,
        token_count=fusion_tokens,
        dropout=cfg.get("fusion_hidden_dropout", 0.1),
    ).to(device)

    return model, processor, tokenizer, visual_tap, heads, fusion_projector, device, lm_lora_enabled


def build_dataloader(processor, cfg: Dict) -> DataLoader:
    dataset = ForgeryJointValDataset(cfg["ann_train"], data_root=cfg.get("data_root"))
    batch_size = cfg.get("batch_size", 2)
    num_workers = cfg.get("num_workers", 4)
    image_size = cfg.get("image_size", 448)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=lambda batch: collate_joint_test(batch, processor, image_size),
    )


def compute_head_losses(
    cls_logits: torch.Tensor,
    hm_logits: torch.Tensor,
    labels: torch.Tensor,
    masks: torch.Tensor,
    cfg: Dict,
):
    weights = cfg.get("loss_weights", {})
    lambda_evi = weights.get("head_evi", 2.0)
    lambda_sparse = weights.get("head_sparse", 1e-3)
    lambda_contrast = weights.get("head_contrast", 1e-2)
    evi_alpha = cfg.get("evi_alpha", 0.5)

    loss_cls = F.binary_cross_entropy_with_logits(cls_logits, labels.float())

    if hm_logits.dim() == 3:
        hm_logits = hm_logits.unsqueeze(1)

    prob_map = torch.sigmoid(hm_logits)
    prob_up = F.interpolate(prob_map, size=masks.shape[-2:], mode="bilinear", align_corners=False)
    mask_up = masks.unsqueeze(1).float()

    logit_map = torch.logit(prob_up.clamp(1e-6, 1 - 1e-6))
    loss_bce = F.binary_cross_entropy_with_logits(logit_map, mask_up)
    inter = (prob_up * mask_up).sum(dim=(1, 2, 3))
    denom = prob_up.sum(dim=(1, 2, 3)) + mask_up.sum(dim=(1, 2, 3)) + 1e-6
    loss_dice = 1 - (2 * inter / denom).mean()
    loss_evi = evi_alpha * loss_bce + (1 - evi_alpha) * loss_dice

    loss_sparse = prob_up.mean()

    real_idx = (labels < 0.5)
    fake_idx = (labels > 0.5)
    if real_idx.any() and fake_idx.any():
        loss_contrast = prob_up[real_idx].mean() - prob_up[fake_idx].mean()
    else:
        loss_contrast = torch.zeros((), device=prob_up.device)

    total = (
        weights.get("head_cls", 1.0) * loss_cls
        + lambda_evi * loss_evi
        + lambda_sparse * loss_sparse
        + lambda_contrast * loss_contrast
    )

    return {
        "total": total,
        "cls": loss_cls,
        "evi": loss_evi,
        "sparse": loss_sparse,
        "contrast": loss_contrast,
    }



def build_fusion_features(
    heads: ForensicJoint,
    fused_feature: torch.Tensor,
    cls_logits: torch.Tensor,
    hm_logits: torch.Tensor,
) -> torch.Tensor:
    cls_prob = torch.sigmoid(cls_logits).unsqueeze(1)  # [B, 1]
    if hm_logits.dim() == 3:
        hm_logits = hm_logits.unsqueeze(1)
    hm_prob = torch.sigmoid(hm_logits)
    hm_mean = hm_prob.mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1)  # [B, 1]
    hm_max = hm_prob.amax(dim=(1, 2, 3), keepdim=False).unsqueeze(1)   # [B, 1]
    pooled = heads.cls.pool(fused_feature)  # [B, C]
    return torch.cat([cls_prob, cls_logits.unsqueeze(1), hm_mean, hm_max, pooled], dim=1)


def prepare_lm_inputs(
    model,
    tokenizer,
    fusion_projector,
    fusion_features,
    prompt_inputs: Dict[str, torch.Tensor],
    target_texts: Sequence[str],
    fusion_token_count: int,
):
    device = fusion_features.device
    target_tokens = tokenizer(
        list(target_texts),
        padding=True,
        return_tensors="pt",
        add_special_tokens=False,
    )
    target_input_ids = target_tokens["input_ids"].to(device)
    target_attention = target_tokens["attention_mask"].to(device)

    prompt_ids = prompt_inputs["input_ids"].to(device)
    prompt_attention = prompt_inputs["attention_mask"].to(device)

    full_input_ids = torch.cat([prompt_ids, target_input_ids], dim=1)
    full_attention = torch.cat([prompt_attention, target_attention], dim=1)

    prompt_embeds = model.get_input_embeddings()(full_input_ids)
    fusion_embeds = fusion_projector(fusion_features)

    fusion_attention = torch.ones(
        (fusion_embeds.size(0), fusion_token_count),
        dtype=full_attention.dtype,
        device=device,
    )

    attention_mask = torch.cat([fusion_attention, full_attention], dim=1)

    ignore_prompt = torch.full_like(prompt_ids, -100)
    labels_target = target_input_ids.masked_fill(target_attention == 0, -100)
    labels = torch.cat(
        [
            torch.full((fusion_embeds.size(0), fusion_token_count), -100, dtype=torch.long, device=device),
            ignore_prompt,
            labels_target,
        ],
        dim=1,
    )

    inputs_embeds = torch.cat([fusion_embeds, prompt_embeds], dim=1)

    return inputs_embeds, attention_mask, labels


def train(cfg: Dict):
    prepare_environment(cfg)
    (
        model,
        processor,
        tokenizer,
        visual_tap,
        heads,
        fusion_projector,
        device,
        lm_lora_enabled,
    ) = build_model_and_tools(cfg)
    dataloader = build_dataloader(processor, cfg)

    grad_accum = cfg.get("grad_accum", 1)
    epochs = cfg.get("epochs", 1)
    warmup_ratio = cfg.get("warmup_ratio", 0.0)
    steps_per_epoch = math.ceil(len(dataloader) / max(1, grad_accum))
    total_steps = max(1, epochs * steps_per_epoch)
    warmup_steps = int(total_steps * warmup_ratio)

    params_fusion = [p for p in fusion_projector.parameters() if p.requires_grad]
    params_lora = [p for p in model.parameters() if p.requires_grad] if lm_lora_enabled else []

    optim_groups = [
        {
            "params": params_fusion,
            "lr": cfg.get("lr_fusion", 5e-4),
            "weight_decay": cfg.get("weight_decay", 0.0),
        }
    ]
    if params_lora:
        optim_groups.append(
            {
                "params": params_lora,
                "lr": cfg.get("lr_lora", 1e-4),
                "weight_decay": 0.0,
            }
        )

    optimizer = torch.optim.AdamW(optim_groups)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

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

    fusion_token_count = int(cfg.get("fusion_tokens", 4))
    loss_weights = cfg.get("loss_weights", {})

    out_dir = Path(cfg.get("output_dir", "outputs_task3"))
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / cfg.get("train_log", "task3_train_log.csv")
    csv_exists = csv_path.exists()

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["timestamp", "epoch", "step", "loss", "lm", "cls", "evi", "sparse", "contrast"],
        )
        if not csv_exists:
            writer.writeheader()

        print("\n=== Start Hybrid Training ===")
        global_step = 0
        for epoch in range(1, epochs + 1):
            print(f"\nEpoch {epoch}/{epochs}")
            model.train()
            heads.eval()
            fusion_projector.train()

            running = {"loss": 0.0, "lm": 0.0, "cls": 0.0, "evi": 0.0, "sparse": 0.0, "contrast": 0.0}
            optimizer.zero_grad(set_to_none=True)
            pbar = tqdm(enumerate(dataloader, 1), total=len(dataloader), desc="Train", dynamic_ncols=True)

            for step_idx, batch in pbar:
                inputs, batch_labels, batch_masks = batch[0], batch[1], batch[2]
                inputs = {k: v.to(device) for k, v in inputs.items()}
                labels = batch_labels.to(device)
                masks = batch_masks.to(device)

                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda" and amp_dtype != torch.float32)):
                    grids = visual_tap(inputs["pixel_values"], inputs["image_grid_thw"])
                    grids = {k: v.float() for k, v in grids.items()}

                    layer_grids = [grids[i] for i in heads.layers]
                    fused_feature = heads.fuser(layer_grids)
                    cls_logits = heads.cls(fused_feature)
                    hm_logits = heads.evi(fused_feature)

                    head_losses = compute_head_losses(cls_logits, hm_logits, labels, masks, cfg)

                    fusion_features = build_fusion_features(heads, fused_feature, cls_logits, hm_logits)

                    target_cfg = cfg.get("target_text", {"fake": "Fake", "real": "Real"})
                    fake_text = target_cfg.get("fake", "Fake")
                    real_text = target_cfg.get("real", "Real")
                    label_list = batch_labels.tolist()
                    target_texts = [fake_text if y >= 0.5 else real_text for y in label_list]
                    inputs_embeds, attention_mask, lm_labels = prepare_lm_inputs(
                        model,
                        tokenizer,
                        fusion_projector,
                        fusion_features,
                        {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]},
                        target_texts,
                        fusion_token_count,
                    )

                    outputs = model(
                        inputs_embeds=inputs_embeds,
                        attention_mask=attention_mask,
                        pixel_values=inputs["pixel_values"],
                        image_grid_thw=inputs["image_grid_thw"],
                        labels=lm_labels,
                    )
                    lm_loss = outputs.loss

                    total_loss = head_losses["total"] + loss_weights.get("lm", 1.0) * lm_loss
                    loss = total_loss / grad_accum

                if scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                if step_idx % grad_accum == 0:
                    if cfg.get("max_norm"):
                        if scaler:
                            scaler.unscale_(optimizer)
                        trainable_params = list(
                            filter(
                                lambda p: p.requires_grad,
                                list(model.parameters()) + list(fusion_projector.parameters()),
                            )
                        )
                        if trainable_params:
                            nn.utils.clip_grad_norm_(trainable_params, cfg["max_norm"])
                    if scaler:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                running["loss"] += total_loss.detach().item()
                running["lm"] += lm_loss.detach().item()
                running["cls"] += head_losses["cls"].detach().item()
                running["evi"] += head_losses["evi"].detach().item()
                running["sparse"] += head_losses["sparse"].detach().item()
                running["contrast"] += head_losses["contrast"].detach().item()

                avg_loss = running["loss"] / step_idx
                pbar.set_postfix({"loss": f"{avg_loss:.4f}", "lm": f"{running['lm']/step_idx:.4f}"})

                writer.writerow(
                    {
                        "timestamp": datetime.utcnow().isoformat(),
                        "epoch": epoch,
                        "step": step_idx,
                        "loss": total_loss.detach().item(),
                        "lm": lm_loss.detach().item(),
                        "cls": head_losses["cls"].detach().item(),
                        "evi": head_losses["evi"].detach().item(),
                        "sparse": head_losses["sparse"].detach().item(),
                        "contrast": head_losses["contrast"].detach().item(),
                    }
                )
                f.flush()

    save_artifacts(cfg, model, heads, fusion_projector, lm_lora_enabled)


def save_artifacts(cfg: Dict, model, heads, fusion_projector, lm_lora_enabled: bool):
    out_dir = Path(cfg.get("output_dir", "outputs_task3"))
    out_dir.mkdir(parents=True, exist_ok=True)

    lora_dir = out_dir / cfg.get("lora_checkpoint", "task3_lora")
    if lm_lora_enabled:
        print("Saving LM LoRA adapters to", lora_dir)
        model.save_pretrained(lora_dir)
    else:
        print("[INFO] LM LoRA disabled; skipping adapter serialization.")

    head_snapshot_name = cfg.get("head_snapshot")
    if head_snapshot_name:
        head_path = out_dir / head_snapshot_name
        print("Saving frozen forensic head copy to", head_path)
        torch.save(heads.state_dict(), head_path)
    else:
        print("[INFO] head_snapshot not set; skipping frozen head serialization")

    fusion_path = out_dir / cfg.get("fusion_checkpoint", "task3_fusion.pt")
    print("Saving fusion projector to", fusion_path)
    torch.save(fusion_projector.state_dict(), fusion_path)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    train(cfg)


if __name__ == "__main__":
    main()
