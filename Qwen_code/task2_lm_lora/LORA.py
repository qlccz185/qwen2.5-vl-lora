import argparse
import json
import os
from pathlib import Path

import torch
from datasets import load_from_disk
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
)


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
    model.enable_input_require_grads()
    return model, tokenizer


def prepare_train_dataset(dataset):
    required_columns = ["input_ids", "attention_mask", "labels", "pixel_values", "image_grid_thw"]
    missing = [column for column in required_columns if column not in dataset.column_names]
    if missing:
        raise ValueError(
            "The mapped training dataset is missing required columns: " + ", ".join(missing)
        )

    removable = [col for col in dataset.column_names if col not in required_columns]
    if removable:
        dataset = dataset.remove_columns(removable)

    dataset.set_format(type="torch")
    return dataset


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

    model, tokenizer = build_model_and_tools(
        cfg["base_model_path"], cfg.get("torch_dtype")
    )

    train_dataset = load_from_disk(cfg["train_dataset_path"])
    train_dataset = prepare_train_dataset(train_dataset)

    lora_config = build_lora_config(cfg)
    model = get_peft_model(model, lora_config)

    training_args = TrainingArguments(**cfg["training_args"])

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )

    trainer.train()

    save_path = cfg.get("save_lora_path", cfg["training_args"].get("output_dir", "./qwen-vl-lora"))
    model.save_pretrained(save_path)


if __name__ == "__main__":
    main()

