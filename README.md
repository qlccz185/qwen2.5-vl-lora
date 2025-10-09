# qwen2.5-vl-lora

## 项目概览
本仓库包含针对 Qwen2.5-VL 的两条 LoRA 训练/评测流程：

- **Task 1 — Modular ViT LoRA with classification heads**：在视觉骨干上插入 LoRA、训练额外分类头，并在完整的 Qwen2.5-VL 模型中进行评估。
  - 相关脚本与配置位于 `Qwen_code/task1_modular_lora/`。
- **Task 2 — Full-model LoRA driven by LM outputs**：直接使用语言模型输出对 ViT 进行 LoRA 训练与推理，无需分类头。
  - 相关脚本与配置位于 `Qwen_code/task2_lm_lora/`。

两个任务都需要共用的脚本将保留在 `Qwen_code/` 根目录下。

## 目录结构
- `Qwen_code/task1_modular_lora/`
  - `lora_train.py`, `lora_eval.py`, `config_lora.json`, `config_eval.json`
  - `full_model_eval/`：完整模型评估脚本与配置
  - `test.py`, `loraworktest.py`
- `Qwen_code/task2_lm_lora/`
  - `LORA.py`, `lora_infer.py`, `map.py`
  - `config_lora_trainer.json`, `config_lora_infer.json`, `config_map.json`
- `Qwen_code/test0.py`：通用模型结构检查脚本

请根据任务选择对应目录下的脚本和配置文件执行。
