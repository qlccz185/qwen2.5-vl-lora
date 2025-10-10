#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Evaluate the full Qwen2.5-VL model after injecting LoRA weights."""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm

from transformers import AutoProcessor, AutoTokenizer, Qwen2_5_VLForConditionalGeneration

from test import ForgeryJointValDataset, resize_square_pad_rgb, set_seed


@dataclass
class FullModelEvalConfig:
    working_dir: str | None
    model_path: str
    lora_checkpoint: str
    annotation_file: str
    data_root: str
    csv_path: str
    prompt_text: str
    fake_token: str
    real_token: str
    max_new_tokens: int
    seed: int
    image_size: int
    torch_dtype: str | None = None
    dataset_limit: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    lora_r: int | None = None
    lora_alpha: int | None = None
    lora_dropout: float | None = None
    lora_target_layers: Sequence[int] | None = None

    @staticmethod
    def from_dict(raw: dict) -> "FullModelEvalConfig":
        return FullModelEvalConfig(
            working_dir=raw.get("working_dir"),
            model_path=raw["model_path"],
            lora_checkpoint=raw["lora_checkpoint"],
            annotation_file=raw["annotation_file"],
            data_root=raw.get("data_root", "."),
            csv_path=raw.get("csv_path", "./full_model_eval_metrics.csv"),
            prompt_text=raw.get("prompt_text", "Is this image authentic? Answer:"),
            fake_token=raw.get("fake_token", " Fake"),
            real_token=raw.get("real_token", " Real"),
            max_new_tokens=int(raw.get("max_new_tokens", 4)),
            seed=int(raw.get("seed", 42)),
            image_size=int(raw.get("image_size", 448)),
            torch_dtype=raw.get("torch_dtype"),
            dataset_limit=raw.get("dataset_limit"),
            temperature=raw.get("temperature"),
            top_p=raw.get("top_p"),
            lora_r=raw.get("lora_r"),
            lora_alpha=raw.get("lora_alpha"),
            lora_dropout=raw.get("lora_dropout"),
            lora_target_layers=raw.get("lora_target_layers"),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inject LoRA weights into Qwen2.5-VL and evaluate using full-model generations.",
    )
    default_cfg = Path(__file__).with_name("config_full_model_eval.json")
    parser.add_argument(
        "--config",
        type=Path,
        default=default_cfg,
        help="Path to the JSON configuration file.",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> FullModelEvalConfig:
    with config_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return FullModelEvalConfig.from_dict(raw)


def prepare_environment(cfg: FullModelEvalConfig) -> None:
    if cfg.working_dir:
        os.chdir(cfg.working_dir)
    print("Current working directory:", os.getcwd())


def load_dataset(cfg: FullModelEvalConfig) -> List[dict]:
    dataset = ForgeryJointValDataset(
        cfg.annotation_file,
        data_root=cfg.data_root,
    )
    if cfg.dataset_limit is not None:
        limit = max(int(cfg.dataset_limit), 0)
        dataset.items = dataset.items[:limit]
    return dataset.items


def build_model(cfg: FullModelEvalConfig):
    dtype = torch.bfloat16
    if cfg.torch_dtype:
        dtype = getattr(torch, cfg.torch_dtype)

    print(f"Loading base model from {cfg.model_path}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        cfg.model_path,
        device_map="auto",
        torch_dtype=dtype,
    )
    model.eval()

    ckpt_path = Path(cfg.lora_checkpoint).expanduser().resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"LoRA checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    ckpt_args = ckpt.get("args", {})

    from lora_train import build_lora_on_qwen_visual

    model.visual = build_lora_on_qwen_visual(
        model.visual,
        r=ckpt_args.get("lora_r", cfg.lora_r or 8),
        alpha=ckpt_args.get("lora_alpha", cfg.lora_alpha or 16),
        dropout=ckpt_args.get("lora_dropout", cfg.lora_dropout or 0.05),
        target_layers=ckpt_args.get(
            "lora_target_layers",
            cfg.lora_target_layers or [7, 15, 23, 31],
        ),
    )
    model.visual.load_state_dict(ckpt["visual_lora"], strict=False)
    print("✅ LoRA adapters injected into the visual backbone")
    return model


def ensure_single_token(tokenizer, token_str: str) -> int:
    token_ids = tokenizer.encode(token_str, add_special_tokens=False)
    if len(token_ids) != 1:
        raise ValueError(
            f"Token '{token_str}' is split into {len(token_ids)} pieces: {token_ids}. "
            "Please adjust the token string in the config so that it maps to a single token.",
        )
    return token_ids[0]


def prepare_inputs(processor, image: Image.Image, prompt: str, device: str):
    message = {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ],
    }
    text = processor.apply_chat_template(
        [message],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt",
    )
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}


def compute_probs_from_scores(scores: torch.Tensor, fake_id: int, real_id: int) -> Tuple[float, float]:
    probs = torch.softmax(scores, dim=-1)
    fake_prob = probs[fake_id].item()
    real_prob = probs[real_id].item()
    denom = fake_prob + real_prob
    if denom <= 0:
        return 0.5, 0.5
    return fake_prob / denom, real_prob / denom


def evaluate_samples(
    model,
    processor,
    tokenizer,
    samples: Iterable[dict],
    cfg: FullModelEvalConfig,
    device: str,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    fake_token_id = ensure_single_token(tokenizer, cfg.fake_token)
    real_token_id = ensure_single_token(tokenizer, cfg.real_token)

    y_true: List[int] = []
    y_prob_fake: List[float] = []
    generations: List[str] = []

    generation_kwargs = {
        "max_new_tokens": cfg.max_new_tokens,
        "do_sample": False,
        "return_dict_in_generate": True,
        "output_scores": True,
    }
    if cfg.temperature is not None:
        generation_kwargs["temperature"] = cfg.temperature
    if cfg.top_p is not None:
        generation_kwargs["top_p"] = cfg.top_p

    for rec in tqdm(samples, desc="Full-model Eval"):
        img_path = Path(rec["path"]) if "path" in rec else Path(rec["image_path"])
        with Image.open(img_path).convert("RGB") as img:
            image = resize_square_pad_rgb(img, size=cfg.image_size)

        inputs = prepare_inputs(processor, image, cfg.prompt_text, device)
        with torch.no_grad():
            outputs = model.generate(**inputs, **generation_kwargs)

        logits = outputs.scores[0][0].to(torch.float32)
        fake_prob, real_prob = compute_probs_from_scores(logits, fake_token_id, real_token_id)
        new_tokens = outputs.sequences[:, inputs["input_ids"].shape[-1]:]
        generated_text = tokenizer.decode(new_tokens[0], skip_special_tokens=True).strip()

        y_true.append(int(rec.get("label", 0)))
        y_prob_fake.append(fake_prob)
        generations.append(generated_text)

    return np.array(y_true, dtype=np.int64), np.array(y_prob_fake, dtype=np.float32), generations


def compute_metrics(y_true: np.ndarray, y_prob_fake: np.ndarray) -> Tuple[float, float, float]:
    y_pred = (y_prob_fake >= 0.5).astype(np.int64)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    try:
        auroc = roc_auc_score(y_true, y_prob_fake)
    except ValueError:
        auroc = float("nan")
    return auroc, acc, f1


def append_metrics_to_csv(cfg: FullModelEvalConfig, metrics: Tuple[float, float, float], sample_count: int) -> None:
    csv_path = Path(cfg.csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    auroc, acc, f1 = metrics
    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": Path(cfg.annotation_file).resolve().parent.name,
        "prompt": cfg.prompt_text,
        "num_samples": sample_count,
        "auroc": auroc,
        "acc": acc,
        "f1": f1,
    }
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print(f"[LOG] Metrics appended to {csv_path}")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    prepare_environment(cfg)
    set_seed(cfg.seed)

    samples = load_dataset(cfg)
    if len(samples) == 0:
        raise RuntimeError("No samples available for evaluation. Check the annotation file paths.")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    processor = AutoProcessor.from_pretrained(
        cfg.model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    model = build_model(cfg)

    device = next(model.parameters()).device

    # 🔹 模型推理与指标计算
    y_true, y_prob_fake, generations = evaluate_samples(model, processor, tokenizer, samples, cfg, device)
    auroc, acc, f1 = compute_metrics(y_true, y_prob_fake)

    print("[METRICS] AUROC={:.4f} ACC={:.4f} F1={:.4f}".format(auroc, acc, f1))
    append_metrics_to_csv(cfg, (auroc, acc, f1), len(samples))

    # ============================================================
    # 🔹 读取输出文件名：从 config 中获取（默认 inference_results.json）
    # ============================================================
    output_name = getattr(cfg, "output_json", None) or "inference_results.json"
    infer_result_path = Path(cfg.csv_path).with_name(output_name)

    # 🔹 收集所有样本结果
    all_records = []
    for rec, prob_fake, gen in zip(samples, y_prob_fake.tolist(), generations):
        record = {
            "image_path": rec.get("path") or rec.get("image_path"),
            "image_label": int(rec.get("label", 0)),
            "fake_probability": float(prob_fake),
            "generated_text": gen,
        }
        all_records.append(record)

    # 🔹 保存为 JSON 文件
    with infer_result_path.open("w", encoding="utf-8") as f:
        json.dump(all_records, f, ensure_ascii=False, indent=2)

    print(f"[LOG] Inference results saved to {infer_result_path}")

if __name__ == "__main__":
    main()
