# -*- coding: utf-8 -*-
import argparse
import os
import json
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from pathlib import Path
from PIL import Image
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from test import (
    ForgeryJointValDataset,
    collate_joint_test,
    QwenVisualTap,
    ForensicJoint,
    set_seed
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen2.5-VL LoRA heads using configuration file",
    )
    default_cfg = Path(__file__).with_name("config_eval.json")
    parser.add_argument(
        "--config",
        type=Path,
        default=default_cfg,
        help="Path to the JSON configuration file.",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def prepare_environment(cfg: dict):
    working_dir = cfg.get("working_dir")
    if working_dir:
        os.chdir(working_dir)
    print("Current working directory:", os.getcwd())

# ---------------------------
# 评估分类性能
# ---------------------------
@torch.no_grad()
def eval_classification(model, visual_tap, dl, device, thr_cls=0.5):
    model.eval(); visual_tap.eval()
    all_y, all_p = [], []
    for batch in tqdm(dl, desc="Eval/Cls"):
        if len(batch) == 5:
            (inputs, labels, masks, paths, _) = batch
        else:
            (inputs, labels, masks, paths) = batch
        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = labels.to(device)

        grids = visual_tap(inputs["pixel_values"], inputs["image_grid_thw"])
        grids = {k: v.float() for k, v in grids.items()}
        logits, _ = model(grids)
        prob = torch.sigmoid(logits).cpu().numpy()
        all_y.append(labels.cpu().numpy())
        all_p.append(prob)
    y_true = np.concatenate(all_y).astype(np.int64)
    y_prob = np.concatenate(all_p)
    y_pred = (y_prob >= thr_cls).astype(int)

    return {
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "acc": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
    }

# ---------------------------
# 评估伪造区域 IoU / Dice
# ---------------------------
@torch.no_grad()
def eval_evidence(model, visual_tap, dl, device, thr_map=0.3, only_fake=True):
    model.eval(); visual_tap.eval()
    inter_sum = union_sum = 0.0
    dice_n = dice_d = 0.0
    for batch in tqdm(dl, desc="Eval/Evi"):
        if len(batch) == 5:
            (inputs, labels, masks, paths, _) = batch
        else:
            (inputs, labels, masks, paths) = batch
        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = labels.to(device)
        masks = masks.to(device)

        grids = visual_tap(inputs["pixel_values"], inputs["image_grid_thw"])
        grids = {k: v.float() for k, v in grids.items()}
        _, hm_logits = model(grids)
        if hm_logits.dim() == 3:
            hm_logits = hm_logits.unsqueeze(1)
        prob32 = torch.sigmoid(hm_logits)
        Ht, Wt = prob32.shape[-2:]
        prob64 = F.interpolate(prob32, size=(Ht*2, Wt*2), mode="bilinear", align_corners=False)
        gt64 = F.interpolate(masks.unsqueeze(1).float(), size=(Ht*2, Wt*2), mode="bilinear", align_corners=False)

        if only_fake:
            idx = (labels > 0.5).view(-1)
            if not idx.any():
                continue
            prob64, gt64 = prob64[idx], gt64[idx]

        pred = (prob64 >= thr_map).float()
        gt = (gt64 >= thr_map).float()

        inter = (pred * gt).sum().item()
        union = (pred + gt - pred * gt).sum().item() + 1e-6
        dice_num = 2 * inter
        dice_den = pred.sum().item() + gt.sum().item() + 1e-6

        inter_sum += inter
        union_sum += union
        dice_n += dice_num
        dice_d += dice_den

    return {
        "mean_iou": inter_sum / union_sum,
        "mean_dice": dice_n / dice_d
    }

# ---------------------------
# 热图可视化（原图+预测+GT）
# ---------------------------
# @torch.no_grad()
# def visualize_samples(model, visual_tap, dl, device, out_dir, n_vis=5):
#     os.makedirs(out_dir, exist_ok=True)
#     saved = 0
#     for batch in dl:
#         (inputs, labels, masks, paths, vis_imgs) = batch
#         inputs = {k: v.to(device) for k,v in inputs.items()}
#         grids = visual_tap(inputs["pixel_values"], inputs["image_grid_thw"])
#         grids = {k: v.float() for k,v in grids.items()}
#         _, hm_logits = model(grids)
#         if hm_logits.dim() == 3:
#             hm_logits = hm_logits.unsqueeze(1)
#         prob = torch.sigmoid(F.interpolate(hm_logits, size=(448,448), mode="bilinear", align_corners=False))[:,0]
#         for i in range(len(paths)):
#             if saved >= n_vis:
#                 return
#             img = vis_imgs[i]
#             heat = (prob[i].cpu().numpy() * 255).clip(0,255).astype(np.uint8)
#             overlay = np.array(img).astype(np.float32)
#             overlay[...,0] = np.clip(overlay[...,0]*0.6 + heat*0.4, 0,255)
#             overlay_img = Image.fromarray(overlay.astype(np.uint8))
#             gt = (masks[i].numpy() * 255).astype(np.uint8)
#             gt_img = Image.fromarray(gt).convert("RGB")

#             canvas = Image.new("RGB", (448*3, 448))
#             canvas.paste(img, (0,0))
#             canvas.paste(overlay_img, (448,0))
#             canvas.paste(gt_img, (896,0))
#             canvas.save(Path(out_dir) / f"{saved:02d}_{Path(paths[i]).stem}.jpg")
#             saved += 1

# ---------------------------
# Main entry
# ---------------------------
def main():
    cli_args = parse_args()
    cfg = load_config(cli_args.config)
    prepare_environment(cfg)

    ckpt_path = cfg["ckpt_path"]
    model_path = cfg["model_path"]
    ann_test = cfg["ann_test"]
    data_root = cfg["data_root"]
    out_dir = Path(cfg.get("out_dir", "./outputs"))
    batch_size = cfg.get("batch_size", 8)
    num_workers = cfg.get("num_workers", 4)
    seed = cfg.get("seed", 42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = Path(cfg.get("csv_path", out_dir / "eval_results.csv"))
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loaded config from: {cli_args.config}")
    print(f"[INFO] Testing on: {ann_test}")

    # ---------------------------
    # 2️⃣ 数据
    # ---------------------------
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    ds_test = ForgeryJointValDataset(ann_test, data_root=data_root)
    dl_test = torch.utils.data.DataLoader(
        ds_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=lambda b: collate_joint_test(b, processor, 448)
    )

    # ---------------------------
    # 3️⃣ 模型
    # ---------------------------
    print("Loading Qwen and LoRA...")
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    dtype_name = cfg.get("torch_dtype")
    if dtype_name:
        dtype = getattr(torch, dtype_name)
    qwen = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=dtype,
    )
    for p in qwen.parameters():
        p.requires_grad = False
    qwen.eval()

    # ✅ 先注入 LoRA 模块（与训练时相同方式）
    ckpt = torch.load(ckpt_path, map_location="cpu")
    ckpt_args = ckpt.get("args", {})  # 从 ckpt 中恢复超参数（如果有存）

    from lora_train import build_lora_on_qwen_visual  
    qwen.visual = build_lora_on_qwen_visual(
        qwen.visual,
        r=ckpt_args.get("lora_r", cfg.get("lora_r", 8)),
        alpha=ckpt_args.get("lora_alpha", cfg.get("lora_alpha", 16)),
        dropout=ckpt_args.get("lora_dropout", cfg.get("lora_dropout", 0.05)),
        target_layers=ckpt_args.get("lora_target_layers", cfg.get("lora_target_layers", [7, 15, 23, 31])),
    )

    # ✅ 再加载 LoRA 权重
    qwen.visual.load_state_dict(ckpt["visual_lora"], strict=False)

    # 头部网络
    visual_layers = tuple(cfg.get("visual_layers", [7, 15, 23, 31]))
    visual_tap = QwenVisualTap(qwen.visual, layers=visual_layers).to(device)
    heads = ForensicJoint(fuse_in_ch=1280, fuse_out_ch=512, layers=visual_layers).to(device)
    heads.load_state_dict(ckpt["state_dict"], strict=False)

    print("✅ LoRA weights loaded successfully!\n")

    # ---------------------------
    # 4️⃣ 评估
    # ---------------------------
    cls_metrics = eval_classification(heads, visual_tap, dl_test, device)
    evi_metrics = eval_evidence(heads, visual_tap, dl_test, device)
    print("[CLS]", cls_metrics)
    print("[EVI]", evi_metrics)

    # ---------------------------
    # 5️⃣ 写入 CSV 文件
    # ---------------------------
    import csv
    from datetime import datetime

    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": Path(ann_test).parent.name,  # 自动记录是哪个数据集
        "auroc": cls_metrics["auroc"],
        "acc": cls_metrics["acc"],
        "f1": cls_metrics["f1"],
        "mean_iou": evi_metrics["mean_iou"],
        "mean_dice": evi_metrics["mean_dice"]
    }

    file_exists = csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"[LOG] 评估结果已写入: {csv_path}")

    # ---------------------------
    # 可选：可视化（当前注释掉）
    # ---------------------------
    # visualize_samples(heads, visual_tap, dl_test, device, Path(out_dir), n_vis=5)
    # print(f"可视化结果已保存到: {out_dir}")

# def main():
#     # ---------------------------
#     # 1️⃣ 读取配置文件
#     # ---------------------------
#     config_path = "/root/autodl-tmp/Qwen_code/config_eval.json"
#     with open(config_path, "r", encoding="utf-8") as f:
#         cfg = json.load(f)

#     ckpt_path = cfg["ckpt_path"]
#     model_path = cfg["model_path"]
#     ann_test = cfg["ann_test"]
#     data_root = cfg["data_root"]
#     out_dir = cfg["out_dir"]
#     batch_size = cfg.get("batch_size", 8)
#     num_workers = cfg.get("num_workers", 4)

#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     set_seed(42)
#     os.makedirs(out_dir, exist_ok=True)

#     print(f"[INFO] Loaded config from: {config_path}")
#     print(f"[INFO] Testing on: {ann_test}")

#     # ---------------------------
#     # 2️⃣ 数据
#     # ---------------------------
#     processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
#     ds_test = ForgeryJointValDataset(ann_test, data_root=data_root)
#     dl_test = torch.utils.data.DataLoader(
#         ds_test,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=num_workers,
#         pin_memory=True,
#         collate_fn=lambda b: collate_joint_test(b, processor, 448)
#     )

#     # ---------------------------
#     # 3️⃣ 模型
#     # ---------------------------
#     print("Loading Qwen and LoRA...")
#     qwen = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#         model_path, device_map="auto", torch_dtype=torch.bfloat16
#     )
#     for p in qwen.parameters():
#         p.requires_grad = False
#     qwen.eval()

#     # ✅ 先注入 LoRA 模块（与训练时相同方式）
#     ckpt = torch.load(ckpt_path, map_location="cpu")
#     args = ckpt.get("args", {})  # 从 ckpt 中恢复超参数（如果有存）

#     from lora_train import build_lora_on_qwen_visual  
#     qwen.visual = build_lora_on_qwen_visual(
#         qwen.visual,
#         r=args.get("lora_r", 8),
#         alpha=args.get("lora_alpha", 16),
#         dropout=args.get("lora_dropout", 0.05)
#     )

#     # ✅ 再加载 LoRA 权重
#     qwen.visual.load_state_dict(ckpt["visual_lora"], strict=False)

#     # ---------------------------
#     # ✅ 头部网络权重（单独从 best_by_IoU_joint.pt 读取）
#     # ---------------------------
#     head_ckpt_path = "/root/autodl-tmp/outputs_lora_joint/best_by_IoU_joint.pt"
#     print(f"[INFO] Loading head weights from: {head_ckpt_path}")

#     heads = ForensicJoint(fuse_in_ch=1280, fuse_out_ch=512, layers=(7, 15, 23, 31)).to(device)
#     head_ckpt = torch.load(head_ckpt_path, map_location="cpu")
#     heads.load_state_dict(head_ckpt["state_dict"], strict=False)

#     # ✅ 重新绑定视觉特征提取器
#     visual_tap = QwenVisualTap(qwen.visual, layers=(7, 15, 23, 31)).to(device)

#     print("✅ LoRA + head weights loaded successfully!\n")

#     # ---------------------------
#     # 4️⃣ 评估
#     # ---------------------------
#     cls_metrics = eval_classification(heads, visual_tap, dl_test, device)
#     evi_metrics = eval_evidence(heads, visual_tap, dl_test, device)
#     print("[CLS]", cls_metrics)
#     print("[EVI]", evi_metrics)

#     # ---------------------------
#     # 5️⃣ 写入 CSV 文件
#     # ---------------------------
#     import csv
#     from datetime import datetime

#     csv_path = Path(out_dir) / "eval_results.csv"
#     row = {
#         "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#         "dataset": Path(ann_test).parent.name,
#         "auroc": cls_metrics["auroc"],
#         "acc": cls_metrics["acc"],
#         "f1": cls_metrics["f1"],
#         "mean_iou": evi_metrics["mean_iou"],
#         "mean_dice": evi_metrics["mean_dice"]
#     }

#     file_exists = csv_path.exists()
#     with open(csv_path, "a", newline="", encoding="utf-8") as f:
#         writer = csv.DictWriter(f, fieldnames=list(row.keys()))
#         if not file_exists:
#             writer.writeheader()
#         writer.writerow(row)

#     print(f"[LOG] 评估结果已写入: {csv_path}")

#     # ---------------------------
#     # 可选：可视化（当前注释掉）
#     # ---------------------------
#     # visualize_samples(heads, visual_tap, dl_test, device, Path(out_dir), n_vis=5)
#     # print(f"可视化结果已保存到: {out_dir}")


if __name__ == "__main__":
    main()



