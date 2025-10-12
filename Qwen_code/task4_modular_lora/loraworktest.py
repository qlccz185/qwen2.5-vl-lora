# -*- coding: utf-8 -*-
import torch
import numpy as np
from transformers import Qwen2_5_VLForConditionalGeneration

# ============================================================
# 配置区（请修改成你自己的路径）
# ============================================================
ckpt_path = "/root/autodl-tmp/task4_modular_lora/training/frozen_head_vit_lora.pt"
model_path = "/root/autodl-tmp/Qwen2.5-VL-7B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# 1️⃣ 加载基础模型和 LoRA 权重
# ============================================================
print("🚀 Loading Qwen2.5-VL base model...")
qwen = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
)
visual_module = qwen.visual

print("📂 Loading LoRA checkpoint:", ckpt_path)
ckpt = torch.load(ckpt_path, map_location="cpu")
visual_lora = ckpt.get("visual_lora", None)

if visual_lora is None:
    raise ValueError("❌ visual_lora not found in checkpoint keys! Available:", ckpt.keys())
print(f"✅ visual_lora loaded with {len(visual_lora)} tensors")

# ============================================================
# 2️⃣ 检查 LoRA 参数名与模型结构是否对齐
# ============================================================
model_keys = [n for n, _ in visual_module.named_parameters()]
lora_keys = [k for k in visual_lora.keys() if "lora" in k]

print("\n🔍 Checking alignment between LoRA and model...")
aligned, missing = [], []
for k in lora_keys:
    if any(k.endswith(mk) or k == mk for mk in model_keys):
        aligned.append(k)
    else:
        missing.append(k)

print(f"✅ Aligned LoRA params: {len(aligned)}")
print(f"⚠️ Unmatched params: {len(missing)}")
if missing:
    print("⚠️ Example unmatched keys:")
    for k in missing[:10]:
        print("   ", k)

# ============================================================
# 3️⃣ 统计 LoRA 参数的均值 / 方差
# ============================================================
print("\n📊 LoRA weight statistics:")
means, stds = [], []
for name, w in visual_lora.items():
    if "lora" in name:
        m, s = w.mean().item(), w.std().item()
        means.append(abs(m))
        stds.append(s)
        if np.random.rand() < 0.02:  # 随机展示少量层
            print(f"  {name:<70} mean={m:.5f} std={s:.5f}")

print(f"\n📈 Average |mean| across LoRA weights: {np.mean(means):.5f}")
print(f"📈 Average std across LoRA weights:     {np.mean(stds):.5f}")

# ============================================================
# 4️⃣ 与原模型参数对比：检测是否更新
# ============================================================
print("\n🔬 Comparing LoRA weights to base model parameters...")
diffs = []
for name, lora_tensor in visual_lora.items():
    if name in visual_module.state_dict():
        base_tensor = visual_module.state_dict()[name].cpu().to(dtype=lora_tensor.dtype)
        delta = torch.norm(lora_tensor - base_tensor).item()
        diffs.append(delta)

if len(diffs) == 0:
    print("⚠️ No directly matched parameters for numeric comparison.")
else:
    mean_diff = np.mean(diffs)
    print(f"📊 Mean absolute parameter diff vs base: {mean_diff:.6f}")
    if mean_diff < 1e-6:
        print("❌ LoRA appears identical to base model (not trained or not injected)")
    elif mean_diff < 1e-3:
        print("⚠️ Very small change — training effect might be weak")
    else:
        print("✅ Significant parameter change detected — LoRA updated successfully")

# ============================================================
# 5️⃣ 总结报告
# ============================================================
total_params = sum(p.numel() for p in visual_lora.values())
print("\n================== SUMMARY ==================")
print(f"📦 Total LoRA tensors: {len(visual_lora)}")
print(f"🧩 Trainable LoRA tensors: {len(lora_keys)}")
print(f"📊 Total parameters (LoRA state_dict): {total_params / 1e6:.2f} M")
print("=============================================")

