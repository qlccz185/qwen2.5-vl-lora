# Qwen2.5-VL LoRA Pipelines

将本项目使用git clone 到/root/autodl-tmp下，在/root/autodl-tmp下需要有data文件和Qwen2.5-VL-7B-Instruct文件。其中data文件中的文件格式是这样的：


![图片](image.png)

本仓库围绕 Qwen2.5-VL 搭建了三条 LoRA 训练/评估流程。所有训练得到的模型参数与评估 CSV 日志均按照任务划分保存在 `/root/autodl-tmp` 下的三个目录：

- `/root/autodl-tmp/task1_modular_lora/`
- `/root/autodl-tmp/task2_lm_lora/`
- `/root/autodl-tmp/task3_hybrid_lora/`

其中 Task1 又包含两个子任务，对应的模型权重与评估结果分别写入：

- `/root/autodl-tmp/task1_modular_lora/training/`
- `/root/autodl-tmp/task1_modular_lora/subtask_vit_head_eval/`
- `/root/autodl-tmp/task1_modular_lora/subtask_full_model_eval/`

## 任务划分与脚本

### Task 1 — Modular ViT LoRA with heads
- 目录：`Qwen_code/task1_modular_lora/`
- 作用：在 Qwen2.5-VL 视觉骨干上插入 LoRA，同时训练额外的分类头与热图头。
- 训练脚本：`lora_train.py`，配置文件 `config_lora.json`。
- 子任务脚本：
  - **Subtask 1（ViT + Head 评估）**：`subtask_vit_head_eval/lora_eval.py`，配置文件 `subtask_vit_head_eval/config_eval.json`。
  - **Subtask 2（完整模型评估）**：`subtask_full_model_eval/full_model_eval.py`，配置文件 `subtask_full_model_eval/config_full_model_eval.json`。
- 一键运行脚本：
  - Subtask1：`python Qwen_code/task1_modular_lora/run_subtask_vit_head_experiment.py`
  - Subtask2：`python Qwen_code/task1_modular_lora/run_subtask_full_model_experiment.py`

运行流程（Subtask1/Subtask2 两者的第一步相同）：
1. 训练：`python Qwen_code/task1_modular_lora/lora_train.py --config Qwen_code/task1_modular_lora/config_lora.json`
2. 评估：
   - Subtask1：`python Qwen_code/task1_modular_lora/subtask_vit_head_eval/lora_eval.py --config Qwen_code/task1_modular_lora/subtask_vit_head_eval/config_eval.json`
   - Subtask2：`python Qwen_code/task1_modular_lora/subtask_full_model_eval/full_model_eval.py --config Qwen_code/task1_modular_lora/subtask_full_model_eval/config_full_model_eval.json`

训练产生的 LoRA/头部权重保存在 `/root/autodl-tmp/task1_modular_lora/training/`；Subtask1 与 Subtask2 的评估指标 CSV 分别写入对应的子任务目录下。

### Task 2 — LM-supervised LoRA without heads
- 目录：`Qwen_code/task2_lm_lora/`
- 作用：直接用语言模型输出对 ViT 进行 LoRA 训练与推理。
- 数据预处理脚本：`map.py`，配置文件 `config_map.json`。
- 训练脚本：`LORA.py`，配置文件 `config_lora_trainer.json`。
- 评估脚本：`lora_infer.py`，配置文件 `config_lora_infer.json`。
- 一键运行脚本：`python Qwen_code/task2_lm_lora/run_task2_experiment.py`

运行流程：
1. 预处理：`python Qwen_code/task2_lm_lora/map.py --config Qwen_code/task2_lm_lora/config_map.json`
2. 训练：`python Qwen_code/task2_lm_lora/LORA.py --config Qwen_code/task2_lm_lora/config_lora_trainer.json`
3. 评估：`python Qwen_code/task2_lm_lora/lora_infer.py --config Qwen_code/task2_lm_lora/config_lora_infer.json`

所有 Task2 相关的模型权重与评估 CSV 均写入 `/root/autodl-tmp/task2_lm_lora/` 目录下（`training/`、`evaluation/`、`datasets/` 子目录）。

### Task 3 — Hybrid head + LM fusion LoRA
- 目录：`Qwen_code/task3_hybrid_lora/`
- 作用：同时对 Qwen2.5-VL 的 ViT、融合层与语言模型进行 LoRA，并融合 Task1 的头部输出与 Task2 的语言监督。
- 训练脚本：`hybrid_train.py`，配置文件 `config_hybrid_train.json`。
- 评估脚本：`hybrid_eval.py`，配置文件 `config_hybrid_eval.json`。
- 一键运行脚本：`python Qwen_code/task3_hybrid_lora/run_task3_experiment.py`

运行流程：
1. 训练：`python Qwen_code/task3_hybrid_lora/hybrid_train.py --config Qwen_code/task3_hybrid_lora/config_hybrid_train.json`
2. 评估：`python Qwen_code/task3_hybrid_lora/hybrid_eval.py --config Qwen_code/task3_hybrid_lora/config_hybrid_eval.json`

Task3 产生的 LoRA/头部/融合权重与评估日志均位于 `/root/autodl-tmp/task3_hybrid_lora/`，其中 `training/` 保存模型参数，`evaluation/` 保存指标 CSV。

---

如需更换配置，可通过传入 `--config` 参数指定新的 JSON 文件，或者修改配置文件中的绝对路径以适配不同的资源布局。新增的一键脚本默认按顺序执行完整流程，若任一步失败会抛出错误并停止后续步骤。
