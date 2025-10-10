"""Centralized dataset mapping utilities for Qwen multi-task pipelines."""

from __future__ import annotations

import argparse
import gc
import json
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, concatenate_datasets, load_from_disk
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoTokenizer

from qwen_vl_utils import process_vision_info


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-process datasets for Qwen2.5-VL tasks")
    default_cfg = Path(__file__).with_name("config_dataset_map.json")
    parser.add_argument(
        "--config",
        type=Path,
        default=default_cfg,
        help="Path to the JSON configuration file.",
    )
    return parser.parse_args(argv)


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def prepare_environment(cfg: dict[str, Any]) -> None:
    working_dir = cfg.get("working_dir")
    if working_dir:
        os.chdir(working_dir)
    print("Current working directory:", os.getcwd())


def build_tokenizer_and_processor(base_model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    processor = AutoProcessor.from_pretrained(base_model_path)
    return tokenizer, processor


def _parse_size_str(value: str | None) -> tuple[int, int] | None:
    if not value or not isinstance(value, str):
        return None
    lower = value.lower()
    if "x" not in lower:
        return None
    try:
        height_str, width_str = lower.split("x")
        return int(height_str), int(width_str)
    except (ValueError, TypeError):
        return None


def format_prompt(template: str, annotation: dict[str, Any]) -> tuple[str, str]:
    label_value = annotation.get("label")
    if label_value is None:
        label_text = annotation.get("label_text", "")
    else:
        label_text = "Fake" if int(label_value) == 1 else "Real"

    safe_context = {**annotation, "label_text": label_text}
    try:
        prompt = template.format(**safe_context)
    except KeyError as exc:
        missing_key = exc.args[0]
        raise KeyError(
            f"Prompt template references unknown field '{missing_key}'. "
            "Please add it to the annotation or adjust the template."
        ) from exc
    return prompt, label_text


def resolve_image_path(image_ref: str, data_root: Path) -> Path:
    img_path = Path(image_ref)
    if not img_path.is_absolute():
        img_path = data_root / img_path
    return img_path.resolve()


def resolve_mask_path(mask_ref: str | None, data_root: Path) -> tuple[str | None, tuple[int, int] | None]:
    if not mask_ref:
        return None, None
    parsed_size = _parse_size_str(mask_ref)
    if parsed_size:
        return mask_ref, parsed_size
    mask_path = Path(mask_ref)
    if not mask_path.is_absolute():
        mask_path = data_root / mask_path
    return str(mask_path.resolve()), None


def build_record(
    ann: dict[str, Any],
    data_root: Path,
    prompt_template: str,
    dataset_name: str,
) -> dict[str, Any]:
    image_rel = ann.get("image_path")
    if not image_rel:
        raise ValueError("Annotation record is missing 'image_path'.")

    prompt_text, label_text = format_prompt(prompt_template, ann)
    image_path = resolve_image_path(image_rel, data_root)
    label_val = ann.get("label")

    record: dict[str, Any] = {
        "dataset_name": dataset_name,
        "image_path": str(image_path),
        "image_rel_path": str(image_rel),
        "prompt": prompt_text,
        "label_text": label_text,
        "data_root": str(data_root),
    }
    if label_val is not None:
        record["label"] = int(label_val)

    mask_ref = ann.get("mask_path")
    resolved_mask, mask_size = resolve_mask_path(mask_ref, data_root)
    if mask_ref is not None:
        record["mask_ref"] = mask_ref
    if resolved_mask is not None:
        record["mask_path_resolved"] = resolved_mask
    if mask_size is not None:
        record["mask_size"] = list(mask_size)

    for key, value in ann.items():
        if key in {"image_path", "label", "mask_path"}:
            continue
        if key not in record:
            record[key] = value

    record["annotation_json"] = json.dumps(ann, ensure_ascii=False)
    return record


def load_dataset_from_json(
    annotation_file: str,
    data_root: str,
    dataset_limit: int | None,
    prompt_template: str,
    dataset_name: str,
) -> Dataset:
    with open(annotation_file, "r", encoding="utf-8") as f:
        annotations = json.load(f)

    root_path = Path(data_root).resolve()
    data_list: list[dict[str, Any]] = []
    for ann in annotations:
        try:
            record = build_record(ann, root_path, prompt_template, dataset_name)
        except Exception as exc:  # pragma: no cover - safety log
            print(f"[WARN] Skipped annotation due to error: {exc}")
            continue
        data_list.append(record)

    dataset = Dataset.from_list(data_list)
    if dataset_limit is not None:
        dataset = dataset.select(range(min(dataset_limit, len(dataset))))
    return dataset


def process_func_safe(examples):
    results = {
        "input_ids": [],
        "attention_mask": [],
        "labels": [],
        "pixel_values": [],
        "image_grid_thw": [],
        "image_path": [],
        "prompt": [],
    }
    if "label" in examples:
        results["label"] = []

    example_count = len(examples["image_path"])
    for i in range(example_count):
        try:
            img_path = examples["image_path"][i]
            prompt = examples["prompt"][i]
            label_value = examples.get("label", [None] * example_count)[i]

            with Image.open(img_path).convert("RGB") as img:
                width, height = img.size

            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": img_path,
                            "resized_height": height,
                            "resized_width": width,
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, _ = process_vision_info(messages)
            inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors="pt")
            inputs = {k: v.squeeze(0) for k, v in inputs.items()}

            response = tokenizer(prompt, add_special_tokens=False)

            input_ids = torch.cat([inputs["input_ids"], torch.tensor(response["input_ids"])])
            attention_mask = torch.cat([inputs["attention_mask"], torch.tensor(response["attention_mask"])])
            labels = torch.cat([torch.full_like(inputs["input_ids"], -100), torch.tensor(response["input_ids"])])

            results["input_ids"].append(input_ids)
            results["attention_mask"].append(attention_mask)
            results["labels"].append(labels)
            results["pixel_values"].append(inputs["pixel_values"])
            results["image_grid_thw"].append(inputs["image_grid_thw"])
            results["image_path"].append(img_path)
            results["prompt"].append(prompt)
            if "label" in results:
                results["label"].append(None if label_value is None else int(label_value))

        except Exception as exc:  # pragma: no cover - logging safety
            print(f"[WARN] Skipped {examples['image_path'][i]} due to error: {exc}")
            continue
    return results


def ensure_iterable(obj: Any) -> Iterable:
    if isinstance(obj, Iterable) and not isinstance(obj, (str, bytes, dict)):
        return obj
    return [obj]


def map_and_merge_dataset(dataset: Dataset, dataset_cfg: dict[str, Any], global_cfg: dict[str, Any]):
    name = dataset_cfg.get("name", "dataset")
    map_batch_size = dataset_cfg.get("map_batch_size", global_cfg.get("map_batch_size", 50))
    batch_output_dir = Path(
        dataset_cfg.get(
            "batch_output_dir",
            Path(global_cfg.get("batch_output_dir", "data/map_batches")) / name,
        )
    )
    batch_output_dir.mkdir(parents=True, exist_ok=True)

    map_kwargs = dict(global_cfg.get("map_kwargs", {}))
    map_kwargs.update(dataset_cfg.get("map_kwargs", {}))
    map_batch_size_inner = map_kwargs.get("batch_size", 1)
    map_num_proc = map_kwargs.get("num_proc", 1)

    num_batches = (len(dataset) + map_batch_size - 1) // map_batch_size
    for batch_idx, start_idx in enumerate(
        tqdm(range(0, len(dataset), map_batch_size), total=num_batches, desc=f"Mapping {name} dataset")
    ):
        end_idx = min(start_idx + map_batch_size, len(dataset))
        batch_dataset = dataset.select(range(start_idx, end_idx))

        processed_batch = batch_dataset.map(
            process_func_safe,
            batched=True,
            batch_size=map_batch_size_inner,
            num_proc=map_num_proc,
            remove_columns=[],
        )

        processed_batch.save_to_disk(batch_output_dir / f"{name}_batch_{batch_idx}")

        del batch_dataset, processed_batch
        gc.collect()

    all_batches = []
    for batch_idx in range(num_batches):
        batch_ds = load_from_disk(batch_output_dir / f"{name}_batch_{batch_idx}")
        all_batches.append(batch_ds)

    if not all_batches:
        raise RuntimeError(f"No batches were created for dataset '{name}'.")

    final_dataset = concatenate_datasets(all_batches)
    final_output_path = Path(
        dataset_cfg.get(
            "final_dataset_path",
            Path(global_cfg.get("final_dataset_path", "data/mapped_datasets")) / f"{name}_dataset_mapped",
        )
    )
    final_output_path.parent.mkdir(parents=True, exist_ok=True)
    final_dataset.save_to_disk(str(final_output_path))
    print(f"All batches for {name} merged and saved to {final_output_path}")


def main(argv: list[str] | None = None):
    args = parse_args(argv)
    cfg = load_config(args.config)
    prepare_environment(cfg)

    global tokenizer, processor
    tokenizer, processor = build_tokenizer_and_processor(cfg["base_model_path"])

    prompt_template = cfg.get(
        "prompt_template",
        "Please determine whether this image is Real or Fake. Answer only with Fake or Real. Answer: {label_text}",
    )

    datasets_cfg = cfg.get("datasets")
    if not datasets_cfg:
        datasets_cfg = [
            {
                "name": cfg.get("dataset_name", "train"),
                "annotation_file": cfg["annotation_file"],
                "data_root": cfg.get("data_root", "."),
                "dataset_limit": cfg.get("dataset_limit"),
                "map_batch_size": cfg.get("map_batch_size"),
                "batch_output_dir": cfg.get("batch_output_dir"),
                "final_dataset_path": cfg.get("final_dataset_path"),
            }
        ]

    for dataset_cfg in ensure_iterable(datasets_cfg):
        dataset_prompt = dataset_cfg.get("prompt_template", prompt_template)
        dataset_limit = dataset_cfg.get("dataset_limit", cfg.get("dataset_limit"))
        dataset_name = dataset_cfg.get("name", "dataset")
        dataset = load_dataset_from_json(
            dataset_cfg["annotation_file"],
            dataset_cfg.get("data_root", cfg.get("data_root", ".")),
            dataset_limit,
            dataset_prompt,
            dataset_name,
        )

        if len(dataset) == 0:
            print(f"[WARN] Dataset '{dataset_name}' is empty. Skipping.")
            continue

        map_and_merge_dataset(dataset, dataset_cfg, cfg)


if __name__ == "__main__":
    main()
