"""Train Task2 LoRA adapters following the Task1 configuration style."""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

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
    """Configuration entries aligned with Task1's LoRA trainer."""

    model_path: str
    data_root: str
    ann_train: str
    out_dir: str
    ann_val: Optional[str] = None
    working_dir: Optional[str] = None

    seed: int = 42
    epochs: int = 1
    batch_size: int = 1
    num_workers: int = 4
    grad_accum: int = 1

    lr_lora: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0

    lora_target_layers: Optional[List[Union[int, str]]] = None
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05

    amp_dtype: str = "bf16"
    image_size: int = 448

    prompt_text: str = (
        "Please determine whether this image is Real or Fake. Answer only with Fake or Real."
    )
    positive_response: str = " Fake"
    negative_response: str = " Real"

    log_interval: int = 10
    eval_interval: Optional[int] = None
    save_interval: Optional[int] = None


def load_config(path: Path) -> TrainConfig:
    data = json.loads(path.read_text(encoding="utf-8"))
    return TrainConfig(**data)


def prepare_environment(cfg: TrainConfig) -> None:
    if cfg.working_dir:
        os_cwd = Path(cfg.working_dir).expanduser().resolve()
        os_cwd.mkdir(parents=True, exist_ok=True)
        import os

        os.chdir(os_cwd)
        print(f"Current working directory: {os.getcwd()}")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_dtype(name: str | None) -> torch.dtype:
    if name is None:
        return torch.bfloat16
    name = name.lower()
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp32", "float32", "single"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def build_lora_targets(layer_specs: Iterable[Union[int, str]], num_blocks: int) -> List[str]:
    modules: List[str] = []
    seen_blocks: set[int] = set()
    for spec in layer_specs:
        if isinstance(spec, str) and spec.strip().isdigit():
            spec = int(spec.strip())
        if isinstance(spec, int):
            if spec < 0 or spec >= num_blocks:
                raise ValueError(
                    f"LoRA target layer index {spec} is out of range for visual blocks ({num_blocks})."
                )
            if spec in seen_blocks:
                continue
            seen_blocks.add(spec)
            modules.extend(
                [
                    f"visual.blocks.{spec}.attn.qkv",
                    f"visual.blocks.{spec}.attn.proj",
                    f"visual.blocks.{spec}.mlp.gate_proj",
                    f"visual.blocks.{spec}.mlp.up_proj",
                    f"visual.blocks.{spec}.mlp.down_proj",
                ]
            )
        else:
            name = str(spec).strip()
            if not name:
                continue
            modules.append(name)
    seen = set()
    ordered: List[str] = []
    for name in modules:
        if name not in seen:
            ordered.append(name)
            seen.add(name)
    return ordered


def build_lora_config(model: Qwen2_5_VLForConditionalGeneration, cfg: TrainConfig) -> LoraConfig:
    visual = getattr(model, "visual", None)
    if visual is None:
        raise RuntimeError("Model does not expose a visual transformer stack")

    if cfg.lora_target_layers is None:
        layer_specs: List[Union[int, str]] = list(range(len(visual.blocks)))
    else:
        layer_specs = cfg.lora_target_layers

    target_modules = build_lora_targets(layer_specs, len(visual.blocks))
    if not target_modules:
        raise ValueError("No valid LoRA target modules resolved from configuration")
    print("✅ Injecting LoRA into following modules:")
    for name in target_modules:
        print("   ", name)

    return LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )


def prepare_token_id(processor, text: str) -> int:
    ids = processor.tokenizer(text, add_special_tokens=False).input_ids
    if not ids:
        raise ValueError(f"Token '{text}' produced no ids; adjust responses in the config")
    if len(ids) > 1:
        print(f"[Task2] Warning: '{text}' splits into {len(ids)} tokens; using the first id only.")
    return ids[0]


def run_eval(
    model: Qwen2_5_VLForConditionalGeneration,
    dataloader: DataLoader,
    device: torch.device,
    positive_token: int,
    negative_token: int,
) -> Dict[str, float]:
    model.eval()
    total = 0
    correct = 0
    losses: List[float] = []
    with torch.no_grad():
        for batch in dataloader:
            batch = dict(batch)
            batch.pop("paths", None)
            labels = batch.pop("labels").to(device)
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**inputs)
            logits = outputs.logits[:, -1, :]
            pos_logits = logits[:, positive_token]
            neg_logits = logits[:, negative_token]
            margin = pos_logits - neg_logits
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
    prepare_environment(cfg)
    set_seed(cfg.seed)

    dtype = get_dtype(cfg.amp_dtype)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = AutoProcessor.from_pretrained(cfg.model_path, local_files_only=True)
    positive_token = prepare_token_id(processor, cfg.positive_response)
    negative_token = prepare_token_id(processor, cfg.negative_response)

    train_dataset = ForgeryBinaryDataset(cfg.ann_train, cfg.data_root)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=Task2Collator(processor, cfg.prompt_text, image_size=cfg.image_size),
    )

    val_loader = None
    if cfg.ann_val:
        val_dataset = ForgeryBinaryDataset(cfg.ann_val, cfg.data_root)
        val_loader = DataLoader(
            val_dataset,
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
    model.enable_input_require_grads()

    lora_config = build_lora_config(model, cfg)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr_lora,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.999),
    )

    steps_per_epoch = math.ceil(len(train_loader) / max(1, cfg.grad_accum))
    total_steps = steps_per_epoch * cfg.epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    scheduler = cosine_scheduler(optimizer, warmup_steps, total_steps)

    amp_enabled = device.type == "cuda" and dtype in (torch.float16, torch.bfloat16)
    autocast_dtype = dtype if dtype in (torch.float16, torch.bfloat16) else torch.float32
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda" and dtype == torch.float16)

    output_dir = Path(cfg.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    best_val_acc = -1.0

    for epoch in range(cfg.epochs):
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg.epochs}", dynamic_ncols=True)
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(progress, start=1):
            batch = dict(batch)
            batch.pop("paths", None)
            labels = batch.pop("labels").to(device)
            inputs = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=amp_enabled):
                outputs = model(**inputs)
                logits = outputs.logits[:, -1, :]
                pos_logits = logits[:, positive_token]
                neg_logits = logits[:, negative_token]
                margin = pos_logits - neg_logits
                loss = F.binary_cross_entropy_with_logits(margin, labels)

            loss = loss / cfg.grad_accum
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if step % cfg.grad_accum == 0:
                if cfg.max_grad_norm and cfg.max_grad_norm > 0:
                    if scaler.is_enabled():
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if cfg.log_interval and global_step % cfg.log_interval == 0:
                    progress.set_postfix({"loss": f"{loss.item() * cfg.grad_accum:.4f}"})

                if cfg.save_interval and global_step % cfg.save_interval == 0:
                    ckpt_dir = output_dir / f"step_{global_step}"
                    ckpt_dir.mkdir(parents=True, exist_ok=True)
                    model.save_pretrained(ckpt_dir)

                if val_loader and cfg.eval_interval and global_step % cfg.eval_interval == 0:
                    metrics = run_eval(model, val_loader, device, positive_token, negative_token)
                    acc = metrics["accuracy"]
                    if acc > best_val_acc:
                        best_val_acc = acc
                        model.save_pretrained(output_dir / "best")
                    (output_dir / "val_metrics.json").write_text(
                        json.dumps({"step": global_step, **metrics}, indent=2),
                        encoding="utf-8",
                    )

        if val_loader:
            metrics = run_eval(model, val_loader, device, positive_token, negative_token)
            acc = metrics["accuracy"]
            if acc > best_val_acc:
                best_val_acc = acc
                model.save_pretrained(output_dir / "best")
            (output_dir / f"val_epoch_{epoch + 1}.json").write_text(
                json.dumps({"epoch": epoch + 1, **metrics}, indent=2),
                encoding="utf-8",
            )

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
