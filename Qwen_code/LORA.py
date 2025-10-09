import os
import json
from pathlib import Path
from PIL import Image
import torch
from datasets import Dataset, load_from_disk
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoTokenizer,
    AutoProcessor,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType
from qwen_vl_utils import process_vision_info

os.chdir("/root/autodl-tmp/qwen2.5-vl-lora")
print("Current working directory:", os.getcwd())

# -----------------------------
# 0️⃣ 模型路径
# -----------------------------
base_model_path = "Qwen2.5-VL-7B-Instruct"

# -----------------------------
# 1️⃣ 加载模型、tokenizer、processor
# -----------------------------
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
processor = AutoProcessor.from_pretrained(base_model_path)

# 允许梯度更新
model.enable_input_require_grads()

# -----------------------------
# 2️⃣ 加载已经 map 好的训练集
# -----------------------------
train_dataset = load_from_disk("data/train_dataset_mapped_2000")

# -----------------------------
# 3️⃣ 读取并处理测试集（保持原来的 map 方法）
# -----------------------------
def load_dataset_from_json(annotation_file, data_root):
    with open(annotation_file, "r", encoding="utf-8") as f:
        annotations = json.load(f)
    data_list = []
    for ann in annotations:
        img_path = str(Path(data_root) / ann["image_path"])
        label_text = "Fake" if ann["label"] == 1 else "Real"
        prompt_text = f"Is this image authentic? Answer: {label_text}"
        data_list.append({"image_path": img_path, "prompt": prompt_text, "label_text": label_text})
    return Dataset.from_list(data_list)

test_dataset = load_dataset_from_json("data/testset/annotation.json", "data").select(range(100))

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

test_dataset = test_dataset.map(
    process_func_safe,
    batched=True,
    batch_size=1,
    num_proc=1,
    remove_columns=test_dataset.column_names
)

# -----------------------------
# 4️⃣ LoRA 配置
# -----------------------------
target_modules = []
for i in [29, 30, 31]:
    target_modules += [
        f"visual.blocks.{i}.attn.qkv",
        f"visual.blocks.{i}.attn.proj",
        f"visual.blocks.{i}.mlp.gate_proj",
        f"visual.blocks.{i}.mlp.up_proj",
        f"visual.blocks.{i}.mlp.down_proj"
    ]
target_modules += ["visual.merger.mlp.0", "visual.merger.mlp.2"]

lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=target_modules,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)

# -----------------------------
# 5️⃣ 训练参数
# -----------------------------
training_args = TrainingArguments(
    output_dir="./qwen-vl-lora",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=2e-4,
    num_train_epochs=3,
    fp16=True,
    logging_steps=10,
    save_steps=500,
    save_total_limit=2,
    dataloader_drop_last=True,
    report_to="none",
    disable_tqdm=False,
    dataloader_num_workers=0
)

# -----------------------------
# 6️⃣ 创建 Trainer
# -----------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)

# -----------------------------
# 7️⃣ 训练
# -----------------------------
trainer.train()

# -----------------------------
# 8️⃣ 保存 LoRA 权重
# -----------------------------
model.save_pretrained("./qwen-vl-lora")

