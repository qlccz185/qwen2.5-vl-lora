#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run inference on Qwen2.5-VL with LoRA adapters and compute classification metrics."""
from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm

from transformers import AutoProcessor, AutoTokenizer, Qwen2_5_VLForConditionalGeneration

from peft import PeftModel

from qwen_vl_utils import process_vision_info


@dataclass
class InferenceConfig:
    working_dir: str | None
    base_model_path: str
    lora_path: str
    annotation_file: str
    data_root: str
    csv_path: str
    prompt_text: str
    fake_token: str
    real_token: str
    max_new_tokens: int
    torch_dtype: str | None = None
    dataset_limit: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    output_json: str | None = None
    mapped_dataset_path: str | None = None

    @staticmethod
    def from_dict(raw: dict) -> "InferenceConfig":
        return InferenceConfig(
            working_dir=raw.get("working_dir"),
            base_model_path=raw["base_model_path"],
            lora_path=raw["lora_path"],
            annotation_file=raw["annotation_file"],
            data_root=raw.get("data_root", "."),
            csv_path=raw.get("csv_path", "./lora_infer_metrics.csv"),
            prompt_text=raw.get("prompt_text", "Is this image authentic? Answer:"),
            fake_token=raw.get("fake_token", " Fake"),
            real_token=raw.get("real_token", " Real"),
            max_new_tokens=int(raw.get("max_new_tokens", 4)),
            torch_dtype=raw.get("torch_dtype"),
            dataset_limit=raw.get("dataset_limit"),
            temperature=raw.get("temperature"),
            top_p=raw.get("top_p"),
            output_json=raw.get("output_json"),
            mapped_dataset_path=raw.get("mapped_dataset_path"),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inject LoRA weights into Qwen2.5-VL and evaluate on a dataset",
    )
    default_cfg = Path(__file__).with_name("config_lora_infer.json")
    parser.add_argument(
        "--config",
        type=Path,
        default=default_cfg,
        help="Path to the JSON configuration file.",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> InferenceConfig:
    with open(config_path, "r", encoding="utf-8") as f:
        raw_cfg = json.load(f)
    return InferenceConfig.from_dict(raw_cfg)


def prepare_environment(cfg: InferenceConfig) -> None:
    if cfg.working_dir:
        os.chdir(cfg.working_dir)
    print("Current working directory:", os.getcwd())


def resolve_path(path_str: str, root: Path) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def load_dataset(cfg: InferenceConfig) -> list[dict]:
    if cfg.mapped_dataset_path:
        mapped_path = Path(cfg.mapped_dataset_path).expanduser().resolve()
        if not mapped_path.exists():
            raise FileNotFoundError(f"Mapped dataset not found: {mapped_path}")

        dataset = load_from_disk(str(mapped_path))
        if cfg.dataset_limit is not None:
            limit = max(int(cfg.dataset_limit), 0)
            dataset = dataset.select(range(min(limit, len(dataset))))

        samples: list[dict] = []
        for rec in dataset:
            image_path = rec.get("image_path")
            if not image_path:
                continue
            sample: dict[str, object] = {
                "image_path": Path(image_path),
                "label": int(rec.get("label", 0)),
            }
            prompt = rec.get("prompt")
            if prompt:
                sample["prompt"] = prompt

            cached_keys = ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")
            if all(key in rec for key in cached_keys):
                sample["preprocessed"] = {key: rec[key] for key in cached_keys}

            samples.append(sample)

        print(f"Loaded {len(samples)} samples from mapped dataset {mapped_path}")
        return samples

    ann_path = Path(cfg.annotation_file).expanduser().resolve()
    if not ann_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {ann_path}")

    with ann_path.open("r", encoding="utf-8") as f:
        records = json.load(f)

    data_root = Path(cfg.data_root).expanduser().resolve()
    samples: list[dict] = []

    for rec in records:
        img_rel = rec.get("image_path")
        if not img_rel:
            continue
        img_path = resolve_path(img_rel, data_root)
        if not img_path.exists():
            print(f"[WARN] Image not found, skipped: {img_path}")
            continue
        label = int(rec.get("label", 0))
        samples.append({"image_path": img_path, "label": label, "prompt": cfg.prompt_text})

    if cfg.dataset_limit is not None:
        limit = max(int(cfg.dataset_limit), 0)
        samples = samples[:limit]

    print(f"Loaded {len(samples)} samples from {ann_path}")
    return samples


def build_model_and_tools(cfg: InferenceConfig):
    dtype = getattr(torch, cfg.torch_dtype) if cfg.torch_dtype else torch.bfloat16
    print(f"Loading base model from {cfg.base_model_path}")
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        cfg.base_model_path,
        torch_dtype=dtype,
    )
    model = PeftModel.from_pretrained(base_model, cfg.lora_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_path)
    processor = AutoProcessor.from_pretrained(cfg.base_model_path)
    return model, tokenizer, processor, device


def prepare_inputs(image_path: Path, prompt: str, processor, device: str):
    with Image.open(image_path).convert("RGB") as img:
        width, height = img.size
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": str(image_path),
                    "resized_height": height,
                    "resized_width": width,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    )
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}


def prepare_cached_inputs(cached_inputs: dict, device: str) -> dict:
    prepared = {}
    for key in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw"):
        if key not in cached_inputs:
            continue
        value = cached_inputs[key]
        if not isinstance(value, torch.Tensor):
            value = torch.as_tensor(value)
        if value.dim() == 0:
            value = value.unsqueeze(0)
        value = value.unsqueeze(0)
        prepared[key] = value.to(device)
    return prepared


def compute_probabilities(scores: torch.Tensor, fake_token_id: int, real_token_id: int) -> tuple[float, float]:
    probs = torch.softmax(scores, dim=-1)
    fake_prob = probs[fake_token_id].item()
    real_prob = probs[real_token_id].item()
    denom = fake_prob + real_prob
    if denom <= 0:
        return 0.5, 0.5
    fake_norm = fake_prob / denom
    real_norm = real_prob / denom
    return fake_norm, real_norm


def infer_sample(
    model,
    processor,
    tokenizer,
    device: str,
    image_path: Path,
    prompt: str,
    fake_token_id: int,
    real_token_id: int,
    generation_kwargs: dict,
    cached_inputs: dict | None = None,
) -> tuple[float, str]:
    if cached_inputs:
        inputs = prepare_cached_inputs(cached_inputs, device)
    else:
        inputs = prepare_inputs(image_path, prompt, processor, device)
    gen_kwargs = {
        "max_new_tokens": generation_kwargs.get("max_new_tokens", 4),
        "do_sample": False,
        "return_dict_in_generate": True,
        "output_scores": True,
    }
    if generation_kwargs.get("temperature") is not None:
        gen_kwargs["temperature"] = generation_kwargs["temperature"]
    if generation_kwargs.get("top_p") is not None:
        gen_kwargs["top_p"] = generation_kwargs["top_p"]

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            **gen_kwargs,
        )
    scores = generated.scores[0][0]
    fake_prob, real_prob = compute_probabilities(scores, fake_token_id, real_token_id)

    generated_ids = generated.sequences[0, inputs["input_ids"].shape[1] :]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return fake_prob, generated_text


def keywords_to_label(text: str) -> int | None:
    lowered = text.lower()
    if any(token in lowered for token in ("fake", "fraud", "spoof")):
        return 1
    if any(token in lowered for token in ("real", "authentic", "genuine", "true")):
        return 0
    return None


def evaluate(cfg: InferenceConfig) -> tuple[dict, list[dict]]:
    prepare_environment(cfg)
    dataset = load_dataset(cfg)
    if not dataset:
        raise RuntimeError("No valid samples to evaluate.")

    model, tokenizer, processor, device = build_model_and_tools(cfg)
    fake_token_id = tokenizer(cfg.fake_token, add_special_tokens=False)["input_ids"][0]
    real_token_id = tokenizer(cfg.real_token, add_special_tokens=False)["input_ids"][0]

    generation_kwargs = {
        "max_new_tokens": cfg.max_new_tokens,
        "temperature": cfg.temperature,
        "top_p": cfg.top_p,
    }

    y_true: list[int] = []
    y_prob: list[float] = []
    y_pred: list[int] = []
    records: list[dict] = []

    for sample in tqdm(dataset, desc="Evaluating"):
        prompt = sample.get("prompt", cfg.prompt_text)
        cached_inputs = sample.get("preprocessed") if isinstance(sample, dict) else None
        prob_fake, text = infer_sample(
            model,
            processor,
            tokenizer,
            device,
            sample["image_path"],
            prompt,
            fake_token_id,
            real_token_id,
            generation_kwargs,
            cached_inputs=cached_inputs,
        )
        label_true = int(sample["label"])
        label_pred = 1 if prob_fake >= 0.5 else 0
        label_from_text = keywords_to_label(text)
        if label_from_text is not None:
            label_pred = label_from_text
        y_true.append(label_true)
        y_prob.append(prob_fake)
        y_pred.append(label_pred)

        records.append(
            {
                "image_path": str(sample["image_path"]),
                "image_label": int(label_true),
                "fake_probability": float(prob_fake),
                "predicted_label": int(label_pred),
                "generated_text": text,
            }
        )

    metrics = {}
    try:
        metrics["auroc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        metrics["auroc"] = float("nan")
    metrics["acc"] = float(accuracy_score(y_true, y_pred))
    metrics["f1"] = float(f1_score(y_true, y_pred))
    metrics["samples"] = len(y_true)
    metrics["positive_ratio"] = float(np.mean(y_true))
    return metrics, records


def append_metrics_to_csv(cfg: InferenceConfig, metrics: dict) -> Path:
    csv_path = Path(cfg.csv_path).expanduser().resolve()
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": Path(cfg.annotation_file).resolve().parent.name,
        "samples": metrics.get("samples"),
        "positive_ratio": metrics.get("positive_ratio"),
        "auroc": metrics.get("auroc"),
        "acc": metrics.get("acc"),
        "f1": metrics.get("f1"),
        "lora_path": cfg.lora_path,
        "base_model_path": cfg.base_model_path,
    }

    file_exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    return csv_path


def main():
    args = parse_args()
    cfg = load_config(args.config)
    metrics, records = evaluate(cfg)
    csv_path = append_metrics_to_csv(cfg, metrics)
    print("Evaluation metrics:", metrics)
    print(f"Metrics appended to {csv_path}")

    output_name = cfg.output_json or "inference_results.json"
    inference_path = csv_path.with_name(output_name)
    with inference_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"Inference results saved to {inference_path}")


if __name__ == "__main__":
    main()
