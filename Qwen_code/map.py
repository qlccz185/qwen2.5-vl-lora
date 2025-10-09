import os
import json
from pathlib import Path
from PIL import Image
import torch
from datasets import Dataset, concatenate_datasets
from tqdm import tqdm
from transformers import AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import gc

os.chdir("/root/autodl-tmp/qwen2.5-vl-lora")

# -----------------------------
# 1️⃣ 加载 tokenizer 和 processor
# -----------------------------
base_model_path = "Qwen2.5-VL-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
processor = AutoProcessor.from_pretrained(base_model_path)

# -----------------------------
# 2️⃣ 读取 JSON 数据
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

full_train_dataset = load_dataset_from_json("data/trainingset2/annotation.json", "data").select(range(2000))

# -----------------------------
# 3️⃣ 数据预处理函数
# -----------------------------
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

# -----------------------------
# 4️⃣ 分批 map 并保存，每批处理完释放内存
# ----------------------------
batch_size = 50
num_batches = (len(full_train_dataset) + batch_size - 1) // batch_size

for batch_idx, start_idx in enumerate(tqdm(range(0, len(full_train_dataset), batch_size), total=num_batches, desc="Mapping train dataset")):
    end_idx = min(start_idx + batch_size, len(full_train_dataset))
    batch_dataset = full_train_dataset.select(range(start_idx, end_idx))

    processed_batch = batch_dataset.map(
        process_func_safe,
        batched=True,
        batch_size=1,
        num_proc=1,
        remove_columns=batch_dataset.column_names
    )

    # 保存当前批次
    processed_batch.save_to_disk(f"data/train_map/train_batch_{batch_idx}")

    # 删除变量释放内存
    del batch_dataset, processed_batch
    gc.collect()

# -----------------------------
# 5️⃣ 所有批次处理完成后再合并
# -----------------------------
from datasets import load_from_disk

all_batches = []
for batch_idx in range(num_batches):
    batch_ds = load_from_disk(f"data/train_map/train_batch_{batch_idx}")
    all_batches.append(batch_ds)

final_train_dataset = concatenate_datasets(all_batches)
final_train_dataset.save_to_disk("data/train_dataset_mapped_2000")
print("All batches merged and saved to data/train_dataset_mapped_2000")

