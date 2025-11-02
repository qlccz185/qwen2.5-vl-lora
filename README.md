# Qwen2.5-VL LoRA Pipelines

This repository collects a family of LoRA-based training and evaluation pipelines built around [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct).  The codebase supports modular forensic heads on top of the vision backbone, language-model-guided LoRA tuning, hybrid fusion strategies, and frozen inference flows for explanation-oriented outputs.  It is designed for offline experiments that operate on a local copy of the base model and dataset.

> 📘 Looking for Chinese documentation? See [README.zh-CN.md](README.zh-CN.md).

## Table of Contents

1. [Key Features](#key-features)
2. [Repository Layout](#repository-layout)
3. [Prerequisites](#prerequisites)
4. [Data Preparation](#data-preparation)
5. [Running the Pipelines](#running-the-pipelines)
6. [Task Overview](#task-overview)
7. [Configuration Highlights](#configuration-highlights)
8. [Outputs and Logging](#outputs-and-logging)
9. [Troubleshooting and Tips](#troubleshooting-and-tips)

## Key Features

- **Modular visual LoRA training** that augments the Qwen2.5-VL vision transformer with detection heads for binary forgery classification and heatmap evidence (Task 1/4).
- **Language-model-supervised LoRA** workflows that rely purely on the autoregressive head to deliver binary decisions (Task 2).
- **Hybrid fusion training** that combines frozen vision heads with LM supervision and optionally injects LoRA adapters into the language stack (Task 3/5/6).
- **Evaluation and inference utilities** for benchmarking pretrained adapters and for generating multimodal textual explanations with heatmap overlays (Task 7/8).
- **Config-driven execution** with JSON files that capture all paths, hyperparameters, and LoRA scopes, making it easy to reproduce experiments on new machines.

## Repository Layout

```
qwen2.5-vl-lora/
├── Qwen_code/                  # Source code for all experiments
│   ├── task1_modular_lora/      # Vision LoRA + forensic heads (training + eval subtasks)
│   ├── task2_lm_lora/           # LM-supervised LoRA (train/evaluate scripts)
│   ├── task3_hybrid_lora/       # Hybrid vision+LM LoRA fusion pipelines
│   ├── task4_modular_lora/      # Variant of Task1 starting from pretrained forensic heads
│   ├── task5_hybrid_lora/       # Hybrid training initialized with pretrained adapters
│   ├── task6_no_lora/           # Hybrid fusion without adding new LoRA modules
│   ├── task7_vit_lora_eval/     # Evaluation of vision LoRA + frozen head
│   └── task8_inference/         # Frozen multimodal inference with textual explanations
├── Qwen_pretrain/               # Pretrained assets (vision heads, LoRA adapters)
│   ├── head/
│   ├── lora_adapter/
│   └── script/
├── data/                        # Example data folder (expects trainingset2/, testset2/, etc.)
└── README.md                    # This document
```

Each task directory ships with example configuration files and helper scripts so that experiments can be launched with a single command.

## Prerequisites

- **Hardware**: NVIDIA GPU with ≥24 GB VRAM recommended for LoRA training on 448×448 crops.  Most scripts default to automatic mixed precision with `bfloat16`.
- **Software**:
  - Python 3.10+
  - PyTorch 2.1+ with CUDA support
  - `transformers>=4.40`, `peft>=0.10`, `accelerate`, `datasets`, `tqdm`, `numpy`, `Pillow`
- **Model assets**: Download `Qwen/Qwen2.5-VL-7B-Instruct` to a local directory such as `/root/autodl-tmp/Qwen2.5-VL-7B-Instruct`.
- **Pretrained heads/adapters (optional)**: Some tasks rely on the checkpoints under `Qwen_pretrain/`.  Copy the contents to your experiment workspace if you plan to resume from the provided weights.

Create a virtual environment and install dependencies, for example:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install transformers peft accelerate datasets tqdm pillow numpy
```

## Data Preparation

The scripts expect a data root that contains `trainingset2/` and `testset2/` subfolders.  A typical layout looks like:

```
/root/autodl-tmp/data/
├── trainingset2/
│   ├── images/...               # Original images
│   ├── masks/...                # Optional segmentation masks (if available)
│   ├── annotation.json          # Full annotation file (Task2/3/5/6/8)
│   ├── train_idx.json           # Training split indices (Task1/4)
│   └── val_idx.json             # Validation split indices (Task1/4/7)
└── testset2/
    ├── images/...
    ├── masks/...
    └── annotation.json
```

Adjust the absolute paths inside every configuration file if your dataset lives elsewhere.  All scripts support a `--config` argument to point to a customized JSON file.

## Running the Pipelines

All experiments are launched from the project root.  Examples:

```bash
# Task1: train modular vision LoRA + forensic heads
python Qwen_code/task1_modular_lora/lora_train.py --config Qwen_code/task1_modular_lora/config_lora.json

# Task1 Subtask (vision head evaluation)
python Qwen_code/task1_modular_lora/subtask_vit_head_eval/lora_eval.py \
  --config Qwen_code/task1_modular_lora/subtask_vit_head_eval/config_eval.json

# Task2: LM-supervised LoRA training and evaluation
python -m Qwen_code.task2_lm_lora.train --config Qwen_code/task2_lm_lora/config_train.json
python -m Qwen_code.task2_lm_lora.evaluate --config Qwen_code/task2_lm_lora/config_eval.json

# Task3: hybrid vision+LM LoRA fusion
python Qwen_code/task3_hybrid_lora/hybrid_train.py --config Qwen_code/task3_hybrid_lora/config_hybrid_train.json
python Qwen_code/task3_hybrid_lora/hybrid_eval.py --config Qwen_code/task3_hybrid_lora/config_hybrid_eval.json

# Task7: evaluate pretrained vision LoRA with a frozen head
python Qwen_code/task7_vit_lora_eval/vit_head_eval.py --config Qwen_code/task7_vit_lora_eval/config_eval.json

# Task8: run frozen multimodal inference with explanations
python Qwen_code/task8_inference/multimodal_inference.py --config Qwen_code/task8_inference/config_inference.json
```

> 💡 When running from a notebook or another working directory, make sure the project root is on `PYTHONPATH` so that intra-task imports (for example, the shared `test.py` utilities) can be resolved.

## Task Overview

| Task | Scope | Key Scripts | Outputs |
|------|-------|-------------|---------|
| **Task 1 — Modular ViT LoRA with heads** | Inject LoRA adapters into selected vision blocks and jointly train binary/evidence heads. | `task1_modular_lora/lora_train.py`, `subtask_vit_head_eval/lora_eval.py`, `subtask_full_model_eval/full_model_eval.py` | Trained LoRA adapters and head checkpoints in `/root/autodl-tmp/task1_modular_lora/` (configurable). |
| **Task 2 — LM-supervised LoRA** | Optimize vision (and optional projector) LoRA parameters using LM token logit differences as a binary signal. | `task2_lm_lora/train.py`, `task2_lm_lora/evaluate.py` | LoRA weights under `output_dir/best/`, evaluation metrics/CSV dumps. |
| **Task 3 — Hybrid head + LM fusion** | Fuse pretrained forensic heads with language supervision while LoRA-tuning vision/merger/LM modules. | `task3_hybrid_lora/hybrid_train.py`, `task3_hybrid_lora/hybrid_eval.py` | Separate checkpoints for LoRA adapters, fusion projector, and validation logs. |
| **Task 4 — Modular ViT LoRA with frozen head** | Finetune vision LoRA while keeping a pretrained forensic head fixed (loaded from `Qwen_pretrain/head`). | `task4_modular_lora/lora_train.py` + evaluation subtasks | Updated vision adapters plus refreshed head snapshots. |
| **Task 5 — Hybrid LoRA with pretrained adapters** | Start from provided vision LoRA + head weights and continue joint fusion/LM tuning (optionally touching LM layers). | `task5_hybrid_lora/hybrid_train.py`, `task5_hybrid_lora/hybrid_eval.py` | LoRA (`lora_checkpoint`), fusion projector, and metrics under `/root/autodl-tmp/task5_hybrid_lora/`. |
| **Task 6 — Hybrid fusion without extra LoRA** | Keep pretrained LoRA frozen and only update the fusion projector / forensic head. | `task6_no_lora/hybrid_train.py`, `task6_no_lora/hybrid_eval.py` | Fusion/head checkpoints captured in `/root/autodl-tmp/task6_no_lora/`. |
| **Task 7 — Vision LoRA evaluation** | Merge pretrained vision LoRA adapters into the base model and score a frozen head on validation data. | `task7_vit_lora_eval/vit_head_eval.py` | CSV metrics, optional inference dumps, and heatmaps for qualitative review. |
| **Task 8 — Multimodal inference** | Produce textual explanations by combining model predictions with heatmap overlays and configurable prompts. | `task8_inference/multimodal_inference.py` | Evidence CSV, rendered heatmaps, and prompt/response transcripts. |

## Configuration Highlights

All JSON configs share a common structure:

- `working_dir`: Optional path to switch into before execution, which helps when absolute imports are used.
- `base_model_path` / `model_path`: Location of the downloaded Qwen2.5-VL model.
- `data_root`, `ann_train`, `ann_eval`: Dataset root and annotation files.  Splits are usually stored as JSON index lists.
- `output_dir` / `out_dir`: Destination for checkpoints, logs, and intermediate artifacts.
- `lora` or `lora_target_layers`: LoRA rank, alpha, dropout, and targeted modules.  Hybrid tasks allow separate control for visual, merge, and LM scopes.
- `image_size`: Square crop size applied by the processor.
- `loss_weights`: Relative weights for classification, evidence, sparsity, contrastive, and LM terms.
- `prompt`, `target_text`, `positive_response`, `negative_response`: Strings used to shape the LM supervision signal.
- `torch_dtype`, `amp_dtype`: Control precision for model loading and autocast.

Feel free to duplicate the provided configs and adjust learning rates, gradients accumulation, or LoRA blocks to experiment with different setups.

## Outputs and Logging

- **Model checkpoints**: Saved at `output_dir`/`out_dir` with subfolders such as `best/`, `last/`, `step_*`, or named snapshots (`task5_lora`, `task5_fusion.pt`).
- **Metrics**: Training scripts write CSV logs (e.g., `training_metrics.csv`) and JSON summaries (`metrics.json`, `val_metrics.json`).
- **Predictions**: Evaluation scripts can emit CSV dumps listing image IDs, predicted labels, probabilities, and file paths for generated heatmaps.
- **Evidence visuals**: Task 8 stores composite images in the configured `evidence_dir`, overlaying heatmaps over the source images.

Ensure that the output directories exist or are creatable by the running user—scripts will attempt to create missing folders, but shared filesystems may require manual preparation.

## Troubleshooting and Tips

- **Path errors**: Every config ships with absolute paths pointing to `/root/autodl-tmp/...`.  Update these to match your environment.
- **Missing pretrained assets**: Tasks 4–8 expect vision-head checkpoints (`Qwen_pretrain/head/*.pt`) and LoRA adapter weights (`Qwen_pretrain/lora_adapter/`).  Download or train them beforehand.
- **Tokenizer tokens for LM supervision**: Task 2/3/5/6 scripts emit warnings if the positive/negative labels split into multiple tokens.  Adjust labels to single-token strings if needed.
- **Precision mismatches**: When running on GPUs without `bfloat16`, set `torch_dtype`/`amp_dtype` to `float16` in the configs.
- **Resuming experiments**: LoRA adapters and fusion modules are saved separately—when resuming, make sure to load both the adapter directory and any auxiliary head checkpoints.

With these components you can reproduce the provided LoRA experiments or adapt them to new datasets and forensic tasks.
