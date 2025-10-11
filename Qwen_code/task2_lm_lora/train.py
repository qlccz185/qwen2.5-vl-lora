"""Train LoRA adapters on the Qwen2.5-VL visual stack using LM supervision."""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from peft import LoraConfig, get_peft_model
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

try:  # Local package execution vs. module execution
    from .datasets import ForgeryBinaryDataset, Task2Collator
except ImportError:  # pragma: no cover - fallback when run as a script
    from datasets import ForgeryBinaryDataset, Task2Collator


@dataclass
class TrainConfig:
    base_model_path: str
    train_annotation: str
    val_annotation: Optional[str]
    data_root: str
    output_dir: str
    prompt: str
    train_batch_size: int = 1
    eval_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    num_epochs: int = 1
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    seed: int = 42
    image_size: int = 448
    torch_dtype: str | None = "bfloat16"
    log_interval: int = 10
    eval_interval: int = 200
    save_interval: int = 1000
    positive_response: str = " yes"
    negative_response: str = " no"
    vision_target_blocks: Optional[List[int]] = None
    include_all_vision_blocks: bool = True
    extra_lora_modules: Optional[List[str]] = None
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05


def load_config(path: Path) -> TrainConfig:
    data = json.loads(path.read_text(encoding="utf-8"))
    return TrainConfig(**data)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_dtype(name: str | None) -> torch.dtype:
    if name is None:
        return torch.bfloat16
    name = name.lower()
    if name == "fp16" or name == "float16":
        return torch.float16
    if name == "bf16" or name == "bfloat16":
        return torch.bfloat16
    if name == "fp32" or name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def collect_target_modules(model: Qwen2_5_VLForConditionalGeneration, cfg: TrainConfig) -> List[str]:
    visual = getattr(model, "visual", None)
    if visual is None:
        raise RuntimeError("Model does not expose a visual transformer stack")
    num_blocks = len(visual.blocks)
    if cfg.include_all_vision_blocks or not cfg.vision_target_blocks:
        blocks = list(range(num_blocks))
    else:
        blocks = [b for b in cfg.vision_target_blocks if 0 <= b < num_blocks]
        if not blocks:
            raise ValueError("No valid vision blocks specified for LoRA")
    modules: List[str] = []
    for idx in blocks:
        modules.extend(
            [
                f"visual.blocks.{idx}.attn.qkv",
                f"visual.blocks.{idx}.attn.proj",
                f"visual.blocks.{idx}.mlp.gate_proj",
                f"visual.blocks.{idx}.mlp.up_proj",
                f"visual.blocks.{idx}.mlp.down_proj",
            ]
        )
    modules.append("multi_modal_projector")
    if cfg.extra_lora_modules:
        modules.extend(cfg.extra_lora_modules)
    # Deduplicate while preserving order
    seen = set()
    ordered: List[str] = []
    for name in modules:
        if name not in seen:
            ordered.append(name)
            seen.add(name)
    return ordered


def build_lora_config(model: Qwen2_5_VLForConditionalGeneration, cfg: TrainConfig) -> LoraConfig:
    targets = collect_target_modules(model, cfg)
    return LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        target_modules=targets,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )


def prepare_token_id(processor, text: str) -> int:
    ids = processor.tokenizer(text, add_special_tokens=False).input_ids
    if not ids:
        raise ValueError(f"Token '{text}' produced no ids; adjust positive/negative responses")
    if len(ids) > 1:
        print(
            f"[Task2] Warning: token '{text}' splits into {len(ids)} pieces; using the first id only."
        )
    return ids[0]


def run_eval(
    model: Qwen2_5_VLForConditionalGeneration,
    dataloader: DataLoader,
    device: torch.device,
    yes_token_id: int,
    no_token_id: int,
) -> Dict[str, float]:
    model.eval()
    total = 0
    correct = 0
    losses: List[float] = []
    with torch.no_grad():
        for batch in dataloader:
            paths = batch.pop("paths")
            labels = batch.pop("labels").to(device)
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**inputs)
            logits = outputs.logits[:, -1, :]
            yes_logits = logits[:, yes_token_id]
            no_logits = logits[:, no_token_id]
            margin = yes_logits - no_logits
            loss = F.binary_cross_entropy_with_logits(margin, labels)
            losses.append(loss.item() * labels.size(0))
            probs = torch.sigmoid(margin)
            preds = (probs >= 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    model.train()
    avg_loss = sum(losses) / max(total, 1)
    acc = correct / max(total, 1)
    return {"loss": avg_loss, "accuracy": acc, "samples": total}


def cosine_scheduler(optimizer, num_warmup: int, num_training_steps: int):
    def lr_lambda(step: int) -> float:
        if step < num_warmup:
            return float(step) / float(max(1, num_warmup))
        progress = (step - num_warmup) / float(max(1, num_training_steps - num_warmup))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train(cfg: TrainConfig) -> None:
    set_seed(cfg.seed)

    dtype = get_dtype(cfg.torch_dtype)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = AutoProcessor.from_pretrained(cfg.base_model_path, local_files_only=True)
    yes_token_id = prepare_token_id(processor, cfg.positive_response)
    no_token_id = prepare_token_id(processor, cfg.negative_response)

    train_dataset = ForgeryBinaryDataset(cfg.train_annotation, cfg.data_root)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train_batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=Task2Collator(processor, cfg.prompt, image_size=cfg.image_size),
    )

    if cfg.val_annotation:
        val_dataset = ForgeryBinaryDataset(cfg.val_annotation, cfg.data_root)
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.eval_batch_size,
            shuffle=False,
            num_workers=2,
            collate_fn=Task2Collator(processor, cfg.prompt, image_size=cfg.image_size),
        )
    else:
        val_loader = None

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        cfg.base_model_path,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        local_files_only=True,
    )
    model.enable_input_require_grads()

    lora_config = build_lora_config(model, cfg)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    optim = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.999),
    )

    num_training_steps = math.ceil(len(train_loader) / cfg.gradient_accumulation_steps) * cfg.num_epochs
    warmup_steps = int(num_training_steps * cfg.warmup_ratio)
    scheduler = cosine_scheduler(optim, warmup_steps, num_training_steps)

    amp_enabled = device.type == "cuda" and dtype in (torch.float16, torch.bfloat16)
    autocast_dtype = dtype if dtype in (torch.float16, torch.bfloat16) else torch.float32
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda" and dtype == torch.float16)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    best_val_acc = -1.0

    for epoch in range(cfg.num_epochs):
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.num_epochs}")
        optim.zero_grad()
        for step, batch in enumerate(progress, start=1):
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
            loss = loss / cfg.gradient_accumulation_steps
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
            if step % cfg.gradient_accumulation_steps == 0:
                if cfg.max_grad_norm and cfg.max_grad_norm > 0:
                    if scaler.is_enabled():
                        scaler.unscale_(optim)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                if scaler.is_enabled():
                    scaler.step(optim)
                    scaler.update()
                else:
                    optim.step()
                scheduler.step()
                optim.zero_grad()
                global_step += 1

                if global_step % cfg.log_interval == 0:
                    progress.set_postfix({"loss": loss.item() * cfg.gradient_accumulation_steps})

                if cfg.save_interval and global_step % cfg.save_interval == 0:
                    checkpoint_dir = output_dir / f"step_{global_step}"
                    checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    model.save_pretrained(checkpoint_dir)

                if val_loader and cfg.eval_interval and global_step % cfg.eval_interval == 0:
                    metrics = run_eval(model, val_loader, device, yes_token_id, no_token_id)
                    acc = metrics["accuracy"]
                    if acc > best_val_acc:
                        best_val_acc = acc
                        model.save_pretrained(output_dir / "best")
                    metrics_path = output_dir / "val_metrics.json"
                    with metrics_path.open("w", encoding="utf-8") as f:
                        json.dump({"step": global_step, **metrics}, f, indent=2)

        # End of epoch evaluation
        if val_loader:
            metrics = run_eval(model, val_loader, device, yes_token_id, no_token_id)
            acc = metrics["accuracy"]
            if acc > best_val_acc:
                best_val_acc = acc
                model.save_pretrained(output_dir / "best")
            metrics_path = output_dir / f"val_epoch_{epoch+1}.json"
            with metrics_path.open("w", encoding="utf-8") as f:
                json.dump({"epoch": epoch + 1, **metrics}, f, indent=2)

    model.save_pretrained(output_dir / "last")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task2 LM-supervised LoRA training")
    default_cfg = Path(__file__).with_name("config_train.json")
    parser.add_argument("--config", type=Path, default=default_cfg)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    train(cfg)


if __name__ == "__main__":
    main()
