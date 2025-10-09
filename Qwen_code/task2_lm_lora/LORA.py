import argparse
import json
import os
from pathlib import Path

import torch
from datasets import Dataset, load_from_disk
from peft import LoraConfig, TaskType, get_peft_model
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

from qwen_vl_utils import process_vision_info


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2.5-VL with LoRA using configuration file")
    default_cfg = Path(__file__).with_name("config_lora_trainer.json")
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


def build_model_and_tools(base_model_path: str, torch_dtype: str | None):
    dtype = getattr(torch, torch_dtype) if torch_dtype else torch.bfloat16
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model_path,
        torch_dtype=dtype,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    processor = AutoProcessor.from_pretrained(base_model_path)
    model.enable_input_require_grads()
    return model, tokenizer, processor


def load_dataset_from_json(annotation_file: str, data_root: str, dataset_limit: int | None):
    with open(annotation_file, "r", encoding="utf-8") as f:
        annotations = json.load(f)

    data_list = []
    for ann in annotations:
        img_path = str(Path(data_root) / ann["image_path"])
        label_text = "Fake" if ann["label"] == 1 else "Real"
        prompt_text = f"Is this image authentic? Answer: {label_text}"
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
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img_path, "resized_height": height, "resized_width": width},
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

        except Exception as e:
            print(f"Skipped {examples['image_path'][i]} due to error: {e}")
            continue
    return results


def build_lora_config(cfg: dict) -> LoraConfig:
    target_modules = []
    for block_idx in cfg.get("lora_target_blocks", []):
        target_modules.extend(
            [
                f"visual.blocks.{block_idx}.attn.qkv",
                f"visual.blocks.{block_idx}.attn.proj",
                f"visual.blocks.{block_idx}.mlp.gate_proj",
                f"visual.blocks.{block_idx}.mlp.up_proj",
                f"visual.blocks.{block_idx}.mlp.down_proj",
            ]
        )

    target_modules.extend(cfg.get("lora_extra_modules", []))

    task_type_value = cfg.get("lora_task_type", "CAUSAL_LM")
    task_type = getattr(TaskType, task_type_value)

    return LoraConfig(
        r=cfg.get("lora_r", 64),
        lora_alpha=cfg.get("lora_alpha", 16),
        target_modules=target_modules,
        lora_dropout=cfg.get("lora_dropout", 0.05),
        bias=cfg.get("lora_bias", "none"),
        task_type=task_type,
    )


def main():
    args = parse_args()
    cfg = load_config(args.config)
    prepare_environment(cfg)

    global tokenizer, processor
    model, tokenizer, processor = build_model_and_tools(
        cfg["base_model_path"], cfg.get("torch_dtype")
    )

    train_dataset = load_from_disk(cfg["train_dataset_path"])

    test_dataset_cfg = cfg.get("test_dataset", {})
    test_dataset = load_dataset_from_json(
        test_dataset_cfg["annotation_file"],
        test_dataset_cfg["data_root"],
        test_dataset_cfg.get("dataset_limit"),
    )

    map_kwargs = test_dataset_cfg.get("map_kwargs", {})
    test_dataset = test_dataset.map(
        process_func_safe,
        batched=True,
        batch_size=map_kwargs.get("batch_size", 1),
        num_proc=map_kwargs.get("num_proc", 1),
        remove_columns=test_dataset.column_names,
    )

    lora_config = build_lora_config(cfg)
    model = get_peft_model(model, lora_config)

    training_args = TrainingArguments(**cfg["training_args"])

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )

    trainer.train()

    save_path = cfg.get("save_lora_path", cfg["training_args"].get("output_dir", "./qwen-vl-lora"))
    model.save_pretrained(save_path)


if __name__ == "__main__":
    main()

