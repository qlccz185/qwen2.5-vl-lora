# Qwen2.5-VL LoRA Pipelines

本仓库在 Qwen2.5-VL-7B-Instruct 基础上整理了 5 条完整的 LoRA 微调 / 评估方案，覆盖视觉骨干、检测头、融合层以及语言模型不同模块的组合训练需求。代码在实际项目中以 `/root/autodl-tmp` 作为工作目录，默认约定如下：

- `/root/autodl-tmp/qwen2.5-vl-lora/`：本仓库克隆路径；
- `/root/autodl-tmp/Qwen2.5-VL-7B-Instruct/`：基础模型权重；
- `/root/autodl-tmp/data/`：数据集根目录；
- `/root/autodl-tmp/task*_.../`：各任务输出的 LoRA 权重、头部权重与评估日志。

如果你需要在其他目录运行，只需在对应的配置 JSON 中修改上述绝对路径即可。

---

## 目录概览

```
qwen2.5-vl-lora/
├── Qwen_code/
│   ├── task1_modular_lora/      # 任务 1：视觉骨干 + 检测头联合训练
│   ├── task2_lm_lora/           # 任务 2：语言模型监督的视觉 LoRA
│   ├── task3_hybrid_lora/       # 任务 3：视觉 + 语言混合联合训练
│   ├── task4_modular_lora/      # 任务 4：冻结检测头，仅训练视觉 LoRA
│   └── task5_hybrid_lora/       # 任务 5：冻结检测头的混合训练
├── Qwen_head/                   # 检测头预训练脚本与权重
├── data/                        # 示例数据目录（需自行准备完整数据）
└── README.md
```

`image.png` 对数据目录层级做了可视化示意，可结合下文的数据准备章节查看。【F:image.png†L1-L4】

---

## 依赖环境

- Python ≥ 3.10，建议在具备 24GB 显存以上的 NVIDIA GPU 上运行。
- 核心依赖：
  - `torch`、`torchvision`
  - `transformers`、`accelerate`、`safetensors`
  - `peft`
  - `tqdm`
  - `numpy`、`scikit-learn`、`pandas`
  - `Pillow`

可以基于 `pip` 安装（示例）：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate safetensors peft tqdm numpy scikit-learn pandas pillow
```

如需分布式或多卡训练，可额外安装 `deepspeed` / `flash-attn` 等依赖，代码逻辑与单卡版本兼容。

---

## 数据准备

数据集默认组织为：

```
data/
├── trainingset2/
│   ├── image/                  # 训练图像
│   ├── spatial_localize/       # 假图对应的掩码（可为空）
│   ├── train_idx.json          # 训练标注
│   └── val_idx.json            # 验证标注
└── testset2/
    ├── image/
    ├── spatial_localize/
    └── annotation.json         # 测试标注
```

标注 JSON 为列表结构，每条记录需包含：

```json
{
  "image_path": "trainingset2/image/xxx.jpg",
  "label": 1,                      // 假图:1，真图:0
  "mask_path": "trainingset2/spatial_localize/xxx.png" 或 "512x512"
}
```

- 对于真图（`label=0`），`mask_path` 可以直接填入原始高 × 宽字符串，脚本会自动生成空掩码。
- 对于假图（`label=1`），`mask_path` 建议提供有效的掩码路径；缺失时评估脚本会记录告警并跳过热图指标。

> **提示**：所有脚本都会在读取配置后调用 `working_dir` 字段，将当前目录切换到 `Qwen_code/`，以便正确导入本地模块。

---

## 任务概览

| 任务 | 训练脚本 | 评估脚本 | 核心目标 | 输出目录 |
| ---- | -------- | -------- | -------- | -------- |
| Task1 | `python Qwen_code/task1_modular_lora/lora_train.py` | `python Qwen_code/task1_modular_lora/subtask_vit_head_eval/lora_eval.py`<br>`python Qwen_code/task1_modular_lora/subtask_full_model_eval/full_model_eval.py` | 视觉骨干 LoRA + 检测头联合训练，并分别评估「只读视觉骨干」与「加载完整模型」两种方式。【F:Qwen_code/task1_modular_lora/lora_train.py†L1-L147】【F:Qwen_code/task1_modular_lora/subtask_vit_head_eval/lora_eval.py†L1-L128】 | `/root/autodl-tmp/task1_modular_lora/`【F:Qwen_code/task1_modular_lora/config_lora.json†L1-L36】 |
| Task2 | `python -m Qwen_code.task2_lm_lora.train` | `python -m Qwen_code.task2_lm_lora.evaluate` | 使用语言模型对「真假」回答的对数几率监督视觉 + 合并模块的 LoRA。【F:Qwen_code/task2_lm_lora/train.py†L1-L120】 | `/root/autodl-tmp/task2_lm_lora/`【F:Qwen_code/task2_lm_lora/config_train.json†L1-L33】 |
| Task3 | `python Qwen_code/task3_hybrid_lora/hybrid_train.py` | `python Qwen_code/task3_hybrid_lora/hybrid_eval.py` | 同时训练视觉骨干、检测头、融合 token 与语言模型的混合 LoRA，融合分类概率与 LM 生成信号。【F:Qwen_code/task3_hybrid_lora/hybrid_train.py†L1-L128】 | `/root/autodl-tmp/task3_hybrid_lora/`【F:Qwen_code/task3_hybrid_lora/config_hybrid_train.json†L1-L43】 |
| Task4 | `python Qwen_code/task4_modular_lora/lora_train.py` | `python Qwen_code/task4_modular_lora/subtask_vit_head_eval/lora_eval.py`<br>`python Qwen_code/task4_modular_lora/subtask_full_model_eval/full_model_eval.py` | 加载预训练检测头，仅对视觉骨干注入 LoRA 并保持头部权重冻结。【F:Qwen_code/task4_modular_lora/config_lora.json†L1-L28】 | `/root/autodl-tmp/task4_modular_lora/`【F:Qwen_code/task4_modular_lora/config_lora.json†L1-L36】 |
| Task5 | `python Qwen_code/task5_hybrid_lora/hybrid_train.py` | `python Qwen_code/task5_hybrid_lora/hybrid_eval.py` | 基于冻结检测头进行视觉 + 语言混合 LoRA 训练，适合将 Task3 迁移到已有头部的场景。【F:Qwen_code/task5_hybrid_lora/hybrid_train.py†L147-L206】 | `/root/autodl-tmp/task5_hybrid_lora/`【F:Qwen_code/task5_hybrid_lora/config_hybrid_train.json†L1-L43】 |

所有命令均支持 `--config` 参数替换默认配置；如不传入则使用同目录下的示例 JSON。

---

## Task1：视觉骨干 + 检测头联合训练

1. **训练**
   ```bash
   python Qwen_code/task1_modular_lora/lora_train.py \
       --config Qwen_code/task1_modular_lora/config_lora.json
   ```
   - `lora_target_layers` 控制注入 LoRA 的视觉 block 序号，默认覆盖第 7/15/23/31 层。【F:Qwen_code/task1_modular_lora/config_lora.json†L18-L24】
   - 输出目录包含 `training_metrics.csv`、检测头权重 `modular_vit_lora.pt` 以及注入后的 LoRA adapter。【F:Qwen_code/task1_modular_lora/config_lora.json†L29-L34】

2. **子任务评估**
   - **Subtask 1：视觉骨干 + 检测头推理**
     ```bash
     python Qwen_code/task1_modular_lora/subtask_vit_head_eval/lora_eval.py \
         --config Qwen_code/task1_modular_lora/subtask_vit_head_eval/config_eval.json
     ```
   - **Subtask 2：完整模型推理（包含语言模型生成）**
     ```bash
     python Qwen_code/task1_modular_lora/subtask_full_model_eval/full_model_eval.py \
         --config Qwen_code/task1_modular_lora/subtask_full_model_eval/config_full_model_eval.json
     ```
   评估脚本会导出分类指标 CSV、可选的热图可视化 JSON，并支持 `dataset_limit` 等裁剪参数。【F:Qwen_code/task1_modular_lora/subtask_full_model_eval/full_model_eval.py†L42-L120】

---

## Task2：语言模型监督的视觉 LoRA

Task2 不再依赖预映射后的 HuggingFace 数据集，而是直接读取 `train_idx.json` / `val_idx.json`，通过聊天式 Prompt 让语言模型回答真假，并将回答概率作为监督信号。【F:Qwen_code/task2_lm_lora/README.md†L1-L37】

- **训练**
  ```bash
  python -m Qwen_code.task2_lm_lora.train --config Qwen_code/task2_lm_lora/config_train.json
  ```
  脚本会周期性评估验证集，并在 `out_dir` 下保存 `best/`、`last/` 与中间步的 LoRA 权重。【F:Qwen_code/task2_lm_lora/README.md†L19-L33】

- **评估**
  ```bash
  python -m Qwen_code.task2_lm_lora.evaluate --config Qwen_code/task2_lm_lora/config_eval.json
  ```
  将输出 `metrics.json` 与按需导出的预测 CSV，支持自定义正负样本触发词（`positive_response` / `negative_response`）。【F:Qwen_code/task2_lm_lora/README.md†L35-L56】

如需调整 LoRA 注入范围，可修改 `lora_target_layers` 或直接追加模块名称（例如 `visual.merger.mlp.0`）。【F:Qwen_code/task2_lm_lora/train.py†L73-L120】

---

## Task3：视觉 + 语言混合联合训练

Task3 在视觉骨干、检测头和语言模型之间引入融合 token，通过 `FusionProjector` 将视觉与头部特征映射到额外的可训练 token，再交由语言模型进一步生成并计算语言损失。【F:Qwen_code/task3_hybrid_lora/hybrid_train.py†L1-L76】

- **训练**
  ```bash
  python Qwen_code/task3_hybrid_lora/hybrid_train.py \
      --config Qwen_code/task3_hybrid_lora/config_hybrid_train.json
  ```
  主要超参数：
  - `visual_layers`：视觉骨干 LoRA 层；
  - `lora.visual.extra_patterns`：匹配额外需要 LoRA 的模块名称片段；
  - `fusion_tokens` 与 `fusion_hidden_dropout`：控制融合 token 数量与 dropout。【F:Qwen_code/task3_hybrid_lora/config_hybrid_train.json†L17-L41】

- **评估**
  ```bash
  python Qwen_code/task3_hybrid_lora/hybrid_eval.py \
      --config Qwen_code/task3_hybrid_lora/config_hybrid_eval.json
  ```
  评估脚本会同时输出检测头指标、语言模型困惑度以及可选的预测导出。【F:Qwen_code/task3_hybrid_lora/hybrid_eval.py†L229-L306】

---

## Task4：冻结检测头的视觉 LoRA

Task4 在 Task1 基础上加载 `Qwen_head/checkpoints/best_by_AUROC_joint.pt` 作为固定检测头，只训练视觉骨干 LoRA。【F:Qwen_code/task4_modular_lora/config_lora.json†L7-L28】

- **训练**
  ```bash
  python Qwen_code/task4_modular_lora/lora_train.py \
      --config Qwen_code/task4_modular_lora/config_lora.json
  ```
  若未在配置中提供 `head_checkpoint` 将直接报错提示。【F:Qwen_code/task4_modular_lora/lora_train.py†L289-L296】

- **评估**
  ```bash
  python Qwen_code/task4_modular_lora/subtask_vit_head_eval/lora_eval.py \
      --config Qwen_code/task4_modular_lora/subtask_vit_head_eval/config_eval.json
  python Qwen_code/task4_modular_lora/subtask_full_model_eval/full_model_eval.py \
      --config Qwen_code/task4_modular_lora/subtask_full_model_eval/config_full_model_eval.json
  ```
  评估输出结构与 Task1 保持一致，但会记录冻结头部的来源路径便于复现实验。【F:Qwen_code/task4_modular_lora/subtask_vit_head_eval/config_eval.json†L1-L19】

---

## Task5：冻结检测头的混合训练

Task5 结合 Task3 的融合思路与 Task4 的冻结策略，在训练前加载外部检测头权重，并在训练结束后可选择备份一份头部快照。【F:Qwen_code/task5_hybrid_lora/hybrid_train.py†L147-L205】【F:Qwen_code/task5_hybrid_lora/config_hybrid_train.json†L1-L41】

- **训练**
  ```bash
  python Qwen_code/task5_hybrid_lora/hybrid_train.py \
      --config Qwen_code/task5_hybrid_lora/config_hybrid_train.json
  ```
  脚本会自动调用 `load_frozen_head_weights`，若路径无效则终止运行以防止误用旧权重。【F:Qwen_code/task5_hybrid_lora/hybrid_train.py†L147-L206】

- **评估**
  ```bash
  python Qwen_code/task5_hybrid_lora/hybrid_eval.py \
      --config Qwen_code/task5_hybrid_lora/config_hybrid_eval.json
  ```
  评估指标与 Task3 相同，可直接对比冻结/非冻结策略的效果。

---

## 检测头预训练（可选）

`Qwen_head/script/` 提供了两个辅助脚本：

- `warmclassification.py`：对检测头进行热身预训练，可输出 `best_by_AUROC_joint.pt` 等权重供 Task4/Task5 使用。【F:Qwen_head/script/warmclassification.py†L1-L120】
- `classanddetect.py`：联合分类与热图检测训练，支持 ROC-AUC、F1、平均精度等指标，并保存最优检查点。【F:Qwen_head/script/classanddetect.py†L1-L120】

运行时同样依赖上述数据目录约定，可根据需要修改脚本内默认路径。

---

## 常见问题与小贴士

1. **CUDA OOM**：可调低 `batch_size` 并增大 `grad_accum`，或在配置中将 `amp_dtype` 调整为 `fp16` 以节省显存。【F:Qwen_code/task1_modular_lora/config_lora.json†L25-L28】
2. **路径找不到**：所有配置均支持相对路径，但建议使用绝对路径避免在 `working_dir` 切换后失效。
3. **自定义 Prompt**：Task2/Task3/Task5 允许通过 `prompt_text` / `target_text` 调整语言提示，以便适配不同风格的数据集。【F:Qwen_code/task2_lm_lora/train.py†L61-L87】【F:Qwen_code/task3_hybrid_lora/config_hybrid_train.json†L31-L41】
4. **扩展 LoRA 模块**：可以在配置中追加 `lora.extra_modules` 或 `lora.lm.extra_modules`，脚本会自动搜索并注入 LoRA。【F:Qwen_code/task3_hybrid_lora/hybrid_train.py†L60-L118】

---

如需进一步定制流程，可直接修改对应任务目录中的 Python 脚本。所有脚本均在训练/评估开始前打印当前工作目录与 LoRA 注入模块列表，便于排查配置是否生效。
