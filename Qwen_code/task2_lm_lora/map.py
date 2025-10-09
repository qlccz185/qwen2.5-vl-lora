import argparse
import gc
import json
import os
from pathlib import Path

import torch
from datasets import Dataset, concatenate_datasets
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoTokenizer

from qwen_vl_utils import process_vision_info


def parse_args():
    parser = argparse.ArgumentParser(description="Pre-process datasets for Qwen2.5-VL training")
    default_cfg = Path(__file__).with_name("config_map.json")
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


def build_tokenizer_and_processor(base_model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    processor = AutoProcessor.from_pretrained(base_model_path)
    return tokenizer, processor


def load_dataset_from_json(annotation_file: str, data_root: str, dataset_limit: int | None):
    with open(annotation_file, "r", encoding="utf-8") as f:
        annotations = json.load(f)

    data_list = []
    for ann in annotations:
        img_path = str(Path(data_root) / ann["image_path"])
        label_text = "Fake" if ann["label"] == 1 else "Real"
        prompt_text = f"Please determine whether this image is Real or Fake. Answer only with Fake or Real. Answer: {label_text}"
        data_list.append({"image_path": img_path, "prompt": prompt_text, "label_text": label_text})

    dataset = Dataset.from_list(data_list)
    if dataset_limit is not None:
        dataset = dataset.select(range(min(dataset_limit, len(dataset))))
    return dataset


def process_func_safe(examples):
    results = {"input_ids": [], "attention_mask": [], "labels": [], "pixel_values": [], "image_grid_thw": []}
    for i in range(len(examples["image_path"])):
        try:
            img_path = examples["image_path"][i]
            prompt = examples["prompt"][i]

            with Image.open(img_path).convert("RGB") as img:
                width, height = img.size

            messages = [
                {"role": "user", "content":[
                    {"type":"image","image":img_path,"resized_height":height,"resized_width":width},
                    {"type":"text","text":prompt}
                ]}
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

        except Exception as e:
            print(f"Skipped {examples['image_path'][i]} due to error: {e}")
            continue
    return results

def main():
    args = parse_args()
    cfg = load_config(args.config)
    prepare_environment(cfg)

    global tokenizer, processor
    tokenizer, processor = build_tokenizer_and_processor(cfg["base_model_path"])

    full_train_dataset = load_dataset_from_json(
        cfg["annotation_file"],
        cfg["data_root"],
        cfg.get("dataset_limit"),
    )

    batch_size = cfg.get("map_batch_size", 50)
    batch_output_dir = Path(cfg.get("batch_output_dir", "data/train_map"))
    batch_output_dir.mkdir(parents=True, exist_ok=True)

    num_batches = (len(full_train_dataset) + batch_size - 1) // batch_size
    map_kwargs = cfg.get("map_kwargs", {})
    map_batch_size = map_kwargs.get("batch_size", 1)
    map_num_proc = map_kwargs.get("num_proc", 1)

    for batch_idx, start_idx in enumerate(
        tqdm(range(0, len(full_train_dataset), batch_size), total=num_batches, desc="Mapping train dataset")
    ):
        end_idx = min(start_idx + batch_size, len(full_train_dataset))
        batch_dataset = full_train_dataset.select(range(start_idx, end_idx))

        processed_batch = batch_dataset.map(
            process_func_safe,
            batched=True,
            batch_size=map_batch_size,
            num_proc=map_num_proc,
            remove_columns=batch_dataset.column_names,
        )

        processed_batch.save_to_disk(batch_output_dir / f"train_batch_{batch_idx}")

        del batch_dataset, processed_batch
        gc.collect()

    from datasets import load_from_disk

    all_batches = []
    for batch_idx in range(num_batches):
        batch_ds = load_from_disk(batch_output_dir / f"train_batch_{batch_idx}")
        all_batches.append(batch_ds)

    final_train_dataset = concatenate_datasets(all_batches)
    final_output_path = Path(cfg.get("final_dataset_path", "data/train_dataset_mapped"))
    final_output_path.parent.mkdir(parents=True, exist_ok=True)
    final_train_dataset.save_to_disk(str(final_output_path))
    print(f"All batches merged and saved to {final_output_path}")


if __name__ == "__main__":
    main()

