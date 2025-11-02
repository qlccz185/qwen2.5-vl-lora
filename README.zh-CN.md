# Qwen2.5-VL LoRA 流水线

本仓库围绕 [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) 构建了一系列基于 LoRA 的训练与评估流程，涵盖视觉骨干插入法证头、语言模型监督的 LoRA 微调、视觉与语言融合的混合策略，以及冻结推理与可解释输出。适用于在本地环境中使用离线模型与数据集开展实验。

> 📘 Looking for the English version? See [README.md](README.md).

## 目录

1. [核心特性](#核心特性)
2. [目录结构](#目录结构)
3. [运行前提](#运行前提)
4. [数据准备](#数据准备)
5. [运行示例](#运行示例)
6. [任务概览](#任务概览)
7. [配置要点](#配置要点)
8. [输出与日志](#输出与日志)
9. [常见问题与提示](#常见问题与提示)

## 核心特性

- **模块化视觉 LoRA 训练**：在 Qwen2.5-VL 视觉 Transformer 中选定层引入 LoRA，并联合训练真假判别头与热力图证据头（Task 1/4）。
- **语言模型监督的 LoRA**：直接利用语言模型输出的 logits 差异作为二分类信号，驱动视觉模态 LoRA 优化（Task 2）。
- **混合融合训练**：在冻结或可训练的视觉头基础上叠加语言监督，可按需对视觉、融合层及语言堆栈注入 LoRA（Task 3/5/6）。
- **评估与推理工具**：提供加载预训练适配器的验证脚本，以及生成带热力图证据的多模态解释型输出（Task 7/8）。
- **配置驱动**：所有路径、超参数与 LoRA 范围均写入 JSON 配置，便于在不同机器上复现实验。

## 目录结构

```
qwen2.5-vl-lora/
├── Qwen_code/                  # 各项实验的源码
│   ├── task1_modular_lora/      # 视觉 LoRA + 法证头（训练与评估子任务）
│   ├── task2_lm_lora/           # 语言模型监督的 LoRA（训练 / 评估脚本）
│   ├── task3_hybrid_lora/       # 视觉 + 语言混合 LoRA 流水线
│   ├── task4_modular_lora/      # 基于预训练法证头的 Task1 变体
│   ├── task5_hybrid_lora/       # 预置适配器的混合训练流程
│   ├── task6_no_lora/           # 不再新增 LoRA，仅训练融合模块
│   ├── task7_vit_lora_eval/     # 视觉 LoRA + 冻结法证头评估
│   └── task8_inference/         # 冻结多模态推理与解释生成
├── Qwen_pretrain/               # 预训练资源（法证头、LoRA 适配器等）
├── data/                        # 示例数据目录（需包含 trainingset2/、testset2/ 等）
└── README.md, README.zh-CN.md   # 英文与中文版说明
```

每个任务目录均提供示例配置文件，可直接通过单条命令启动实验。

## 运行前提

- **硬件**：建议使用显存 ≥24 GB 的 NVIDIA GPU，以应对 448×448 图像裁剪和混合精度训练。
- **软件环境**：
  - Python 3.10 及以上版本
  - 支持 CUDA 的 PyTorch 2.1+
  - `transformers>=4.40`、`peft>=0.10`、`accelerate`、`datasets`、`tqdm`、`numpy`、`Pillow`
- **模型资源**：请先将 `Qwen/Qwen2.5-VL-7B-Instruct` 下载到本地，如 `/root/autodl-tmp/Qwen2.5-VL-7B-Instruct`。
- **可选的预训练权重**：Task 4–8 默认使用 `Qwen_pretrain/` 下提供的法证头与 LoRA 适配器，如需复现实验请提前准备。

在新环境中可按以下方式安装依赖：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install transformers peft accelerate datasets tqdm pillow numpy
```

## 数据准备

脚本默认的数据根目录包含 `trainingset2/` 与 `testset2/`，建议结构如下：

```
/root/autodl-tmp/data/
├── trainingset2/
│   ├── images/...               # 原始图像
│   ├── masks/...                # 可选：伪造区域掩码
│   ├── annotation.json          # 任务 2/3/5/6/8 使用的完整标注
│   ├── train_idx.json           # 任务 1/4 训练划分索引
│   └── val_idx.json             # 任务 1/4/7 验证划分索引
└── testset2/
    ├── images/...
    ├── masks/...
    └── annotation.json
```

如果数据存放位置不同，请修改对应配置文件中的绝对路径。所有脚本均支持 `--config` 参数以加载自定义 JSON。

## 运行示例

在项目根目录执行以下命令即可运行主要流程：

```bash
# Task1：训练视觉 LoRA 与法证头
python Qwen_code/task1_modular_lora/lora_train.py --config Qwen_code/task1_modular_lora/config_lora.json

# Task1 子任务：视觉头评估
python Qwen_code/task1_modular_lora/subtask_vit_head_eval/lora_eval.py \
  --config Qwen_code/task1_modular_lora/subtask_vit_head_eval/config_eval.json

# Task2：语言模型监督的 LoRA 训练与评估
python -m Qwen_code.task2_lm_lora.train --config Qwen_code/task2_lm_lora/config_train.json
python -m Qwen_code.task2_lm_lora.evaluate --config Qwen_code/task2_lm_lora/config_eval.json

# Task3：视觉 + 语言混合 LoRA
python Qwen_code/task3_hybrid_lora/hybrid_train.py --config Qwen_code/task3_hybrid_lora/config_hybrid_train.json
python Qwen_code/task3_hybrid_lora/hybrid_eval.py --config Qwen_code/task3_hybrid_lora/config_hybrid_eval.json

# Task7：加载预训练视觉 LoRA 评估冻结头
python Qwen_code/task7_vit_lora_eval/vit_head_eval.py --config Qwen_code/task7_vit_lora_eval/config_eval.json

# Task8：冻结多模态推理与解释生成
python Qwen_code/task8_inference/multimodal_inference.py --config Qwen_code/task8_inference/config_inference.json
```

> 💡 若在 Notebook 等其他工作目录运行，请确保项目根目录加入 `PYTHONPATH`，以便正确导入共享的 `test.py` 等工具模块。

## 任务概览

| 任务 | 内容 | 核心脚本 | 输出位置 |
|------|------|----------|----------|
| **Task 1 — 模块化 ViT LoRA + 法证头** | 在指定视觉层注入 LoRA，并联合训练真假判别与证据热图头。 | `task1_modular_lora/lora_train.py`、`subtask_vit_head_eval/lora_eval.py`、`subtask_full_model_eval/full_model_eval.py` | 默认写入 `/root/autodl-tmp/task1_modular_lora/`（可配置）。 |
| **Task 2 — 语言模型监督的 LoRA** | 依据语言模型对 " yes"/" no" 的 logits 差异进行二分类训练。 | `task2_lm_lora/train.py`、`task2_lm_lora/evaluate.py` | `output_dir` 下的 `best/`、`last/` 权重及评估 CSV。 |
| **Task 3 — 混合法证头 + LM 融合 LoRA** | 将预训练视觉头与语言监督结合，同时对视觉/融合/语言模块注入 LoRA。 | `task3_hybrid_lora/hybrid_train.py`、`task3_hybrid_lora/hybrid_eval.py` | 独立的 LoRA、融合投影、验证日志等。 |
| **Task 4 — 冻结法证头的视觉 LoRA** | 载入 `Qwen_pretrain/head` 中的法证头，仅微调视觉 LoRA。 | `task4_modular_lora/` 下脚本 | 输出更新的视觉适配器与头部快照。 |
| **Task 5 — 预训练适配器的混合 LoRA** | 基于现成的视觉 LoRA 与法证头继续进行融合与语言监督微调，可同时调节语言堆栈。 | `task5_hybrid_lora/hybrid_train.py`、`task5_hybrid_lora/hybrid_eval.py` | `/root/autodl-tmp/task5_hybrid_lora/` 下的 LoRA、融合、指标文件。 |
| **Task 6 — 不新增 LoRA 的混合训练** | 冻结已有 LoRA，仅优化融合模块 / 法证头。 | `task6_no_lora/hybrid_train.py`、`task6_no_lora/hybrid_eval.py` | `/root/autodl-tmp/task6_no_lora/` 中的融合与头部权重。 |
| **Task 7 — 视觉 LoRA 评估** | 将视觉 LoRA 合并入基座模型，评估冻结法证头在验证集上的表现。 | `task7_vit_lora_eval/vit_head_eval.py` | 输出指标 CSV、推理结果与可选热力图。 |
| **Task 8 — 多模态解释型推理** | 结合模型结论与热力图，生成带证据的文本说明，可自定义提示词策略。 | `task8_inference/multimodal_inference.py` | 证据 CSV、热力图合成图与完整对话记录。 |

## 配置要点

通用 JSON 字段说明：

- `working_dir`：执行脚本前切换到的目录，便于使用绝对导入。
- `base_model_path` / `model_path`：本地 Qwen2.5-VL 模型目录。
- `data_root`、`ann_train`、`ann_eval`：数据根路径与训练/验证标注文件，通常为 JSON 索引或标注列表。
- `output_dir` / `out_dir`：保存权重、日志及中间结果的位置。
- `lora` / `lora_target_layers`：LoRA 的秩、alpha、dropout 以及目标模块；混合任务可分别配置视觉、融合、语言部分。
- `image_size`：处理器裁剪尺寸。
- `loss_weights`：真假分类、证据、稀疏、对比、语言等损失项的权重。
- `prompt`、`target_text`、`positive_response`、`negative_response`：控制语言监督或提示词的文本。
- `torch_dtype`、`amp_dtype`：模型加载与自动混合精度的数值类型。

建议复制配置文件后再按需修改学习率、梯度累积、LoRA 层编号等参数，以便快速对比实验。

## 输出与日志

- **模型权重**：保存在 `output_dir`/`out_dir`，常见结构包括 `best/`、`last/`、`step_*` 或自定义文件（如 `task5_lora`、`task5_fusion.pt`）。
- **训练指标**：脚本会记录 CSV（如 `training_metrics.csv`）与 JSON（如 `metrics.json`、`val_metrics.json`）。
- **预测结果**：评估脚本可输出包含图像 ID、预测标签、概率及热力图路径的 CSV。
- **证据可视化**：Task 8 会在配置的 `evidence_dir` 下保存叠加热力图的合成图像。

请确保目标输出目录存在或具备创建权限，部分共享文件系统可能需要手动初始化目录结构。

## 常见问题与提示

- **路径错误**：示例配置默认使用 `/root/autodl-tmp/...` 路径，请根据实际环境逐项替换。
- **缺少预训练资源**：Task 4–8 依赖 `Qwen_pretrain/head/*.pt` 与 `Qwen_pretrain/lora_adapter/`，若不存在需先下载或自训练。
- **语言标签分词问题**：Task 2/3/5/6 若发现正负标签被 tokenizer 切成多个 token，会给出提示；可将标签改为单 token 形式。
- **精度设置**：若 GPU 不支持 `bfloat16`，可在配置中将 `torch_dtype`/`amp_dtype` 改为 `float16`。
- **继续训练**：LoRA 适配器与融合模块分别保存，恢复时需同时加载适配器目录与相关头部快照。

以上内容可帮助你复现仓库中的 LoRA 实验，或在此基础上拓展更多图像取证任务。
