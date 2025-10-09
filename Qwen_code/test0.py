from transformers import Qwen2_5_VLForConditionalGeneration

model_path = "/root/autodl-tmp/Qwen2.5-VL-7B-Instruct"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path)

# ✅ 一行搞定：打印模型结构
print(model)
