# Task2 — LM-supervised Visual LoRA

本目录重新实现了 Task2 的实验流程，不再依赖预先 `map` 好的 HuggingFace 数据集，
而是直接读取原始 JSON 标注，通过语言模型输出的二分类结果来训练和评估视觉 + 合并模
块的 LoRA 参数。

## 目录结构

- `datasets.py`：读取 `trainingset2`/`testset2` 目录下的原始 JSON 标注，完成图片裁剪和
  Processor 打包。
- `train.py`：主训练脚本，在 Qwen2.5-VL 的视觉 Transformer 与（可选的）合并/投影模
  块上注入 LoRA，使用语言模型对 “图片是否为伪造” 的回答概率做二分类 BCE 损失。
- `evaluate.py`：加载基础模型与 LoRA 权重，计算验证集的损失、准确率并可导出预测 CSV。
- `config_train.json` / `config_eval.json`：示例配置文件，可按需修改路径或超参数。

## 运行方式

1. **训练**

   ```bash
   python -m Qwen_code.task2_lm_lora.train --config Qwen_code/task2_lm_lora/config_train.json
   ```

   训练脚本会：
   - 读取 `train_annotation` 与 `val_annotation` 指向的 JSON 标注；
   - 按需裁剪图片到固定分辨率，并构造聊天式 prompt：
     “You are a forensic assistant... Answer 'yes' if the image is fake ...”；
   - 根据 `lora_target_layers` 指定的模块注入 LoRA（默认覆盖第 7/15/23/31 个视觉 block、
     `visual.merger.mlp.{0,2}` 与 `multi_modal_projector`）；
   - 计算语言模型对 " yes" / " no" 的对数几率差，并使用 `binary_cross_entropy_with_logits`
     更新 LoRA 参数。

   训练过程中会按 `eval_interval` 触发验证评估，并在 `output_dir` 下保存 `best` 与
   `last` LoRA 权重。

2. **评估**

   ```bash
   python -m Qwen_code.task2_lm_lora.evaluate --config Qwen_code/task2_lm_lora/config_eval.json
   ```

   评估脚本会载入 `lora_weights`（如果为空则仅用基础模型），输出 `metrics.json` 与
   `dump_predictions` 指定的 CSV。

## 自定义提示词与标签

- `positive_response` / `negative_response` 控制 BCE 计算中代表 “伪造” 与 “真实” 的标记。
  默认分别为 `" yes"` 与 `" no"`，若 tokenizer 会将其切分成多个 token，脚本会给出
  提示并默认仅使用第一个 token。
- 可以通过 `prompt` 调整提示语，使得模型在生成阶段更倾向于给出期望的回答格式。

## 训练输出

`output_dir` 中会包含：

- `step_XXXX/`：按 `save_interval` 保存的中间 LoRA 权重；
- `best/`：验证准确率最高的 LoRA；
- `last/`：训练结束时的 LoRA；
- `val_metrics.json` 与 `val_epoch_*.json`：记录验证集指标。

这样即可在 Task2 里直接使用语言模型的输出概率对视觉模块进行 LoRA 训练，无需额外的
数据映射步骤。
