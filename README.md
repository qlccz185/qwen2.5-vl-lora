# qwen2.5-vl-lora

## 项目概览
本仓库包含针对 Qwen2.5-VL 的多条 LoRA 训练/评测流程：

- **Task 1 — Modular ViT LoRA with classification heads**：在视觉骨干上插入 LoRA、训练额外分类头，并在完整的 Qwen2.5-VL 模型中进行评估。
  - 相关脚本与配置位于 `Qwen_code/task1_modular_lora/`，其中按照评估方式再划分为两个子任务：
    - `subtask_vit_head_eval/`：将 LoRA 注入回 ViT，并加载分类头输出结果进行评估。
    - `subtask_full_model_eval/`：将 LoRA 注入完整的 Qwen2.5-VL 模型，通过语言模型输出进行评估。
- **Task 2 — Full-model LoRA driven by LM outputs**：直接使用语言模型输出对 ViT 进行 LoRA 训练与推理，无需分类头。
  - 相关脚本与配置位于 `Qwen_code/task2_lm_lora/`。
- **Task 3 — Hybrid head+LM fusion LoRA**：同时对 Qwen 的 ViT、融合层与 LM 进行 LoRA 微调，将任务 1 中的分类头/热图输出与任务 2 的语言模型监督融合。
  - 相关脚本与配置位于 `Qwen_code/task3_hybrid_lora/`。

两个任务都需要共用的脚本将保留在 `Qwen_code/` 根目录下。

## 目录结构
- `Qwen_code/task1_modular_lora/`
  - 根目录：`lora_train.py`, `config_lora.json`, `loraworktest.py`
  - `subtask_vit_head_eval/`：`lora_eval.py`, `config_eval.json`, `test.py`
  - `subtask_full_model_eval/`：`full_model_eval.py`, `config_full_model_eval.json`
- `Qwen_code/task2_lm_lora/`
  - `LORA.py`, `lora_infer.py`, `map.py`
  - `config_lora_trainer.json`, `config_lora_infer.json`, `config_map.json`
- `Qwen_code/task3_hybrid_lora/`
  - `hybrid_train.py`, `config_hybrid_train.json`
  - `hybrid_eval.py`, `config_hybrid_eval.json`
- `Qwen_code/test0.py`：通用模型结构检查脚本

请根据任务选择对应目录下的脚本和配置文件执行。
