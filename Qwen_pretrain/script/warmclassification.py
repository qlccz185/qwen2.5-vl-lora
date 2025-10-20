#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage-A: Train classification head on top of Qwen2.5-VL visual features.
- Freeze Qwen backbone (visual + LLM)
- Train only fuse neck + cls head
- Inputs: images + binary labels (1=fake, 0=real)
"""

import os, json, math, argparse, random, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
from PIL import Image
from tqdm import tqdm
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

from sklearn.metrics import roc_auc_score, average_precision_score

# -------------------------
# Utils
# -------------------------
def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def is_image_ok(p: Path) -> bool:
    try:
        with Image.open(p) as im:
            im.verify()
        return True
    except Exception:
        return False

# -------------------------
# Dataset
# -------------------------
class ForgeryClsDataset(Dataset):
    def __init__(self, transform=None):
        self.root_img = Path("/root/data/trainingset2/trainingset2/image")
        ann_path = Path("/root/data/trainingset2/trainingset2/annotation.json")
        with open(ann_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.items = []
        for rec in data:
            img_rel = rec.get("image_path")
            if img_rel is None:
                continue
            # 只取文件名来定位
            img_p = self.root_img / Path(img_rel).name
            lbl = int(rec.get("label", 0))
            if img_p.exists():
                self.items.append({"img": img_p, "label": lbl})

        if len(self.items) == 0:
            raise RuntimeError("No valid images found in fixed path!")

        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        rec = self.items[i]
        img = Image.open(rec["img"]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return {"image": img, "label": rec["label"], "path": str(rec["img"])}

# -------------------------
# Qwen visual tap (explicit multi-layer feature return)
# -------------------------
class QwenVisualTap(nn.Module):
    """
    用 forward hooks 抓取 7/15/23/31 层 token。 显存爆了先调7/15
    处理方式：按样本切分，再 pad 到本批最大网格 [C, H_max, W_max]，最后 stack 成 [B, C, H_max, W_max]。
    """
    def __init__(self, visual, layers=(7, 15)):
        super().__init__()
        self.visual = visual
        self.layers = list(layers)
        self._feat_cache = {}
        self._hooks = []
        for i in self.layers:
            self._hooks.append(self.visual.blocks[i].register_forward_hook(self._make_hook(i)))

    def _make_hook(self, idx):
        def _hook(module, inp, out):
            # out 可能是 [Total_L, C] 或 [B, L_pad, C]
            self._feat_cache[idx] = out
        return _hook

    def forward(self, pixel_values, thw):
        """
        pixel_values: processor 产物（形如 [B, T, C, H, W]）
        thw: [B,3]，每行 [T, H_i, W_i]；这里只用 H_i, W_i
        返回：dict {layer_idx: [B, C, H_max, W_max]}
        """
        # 1) 触发一次visual前向，偷看一眼结构
        self._feat_cache.clear()
        _ = self.visual(pixel_values, thw)

        # 2) 取出每个样本的 H_i, W_i 与长度 L_i
        if isinstance(thw, torch.Tensor):
            thw_list = thw.tolist()
        else:
            thw_list = thw
        Hs = [int(v[1]) for v in thw_list]
        Ws = [int(v[2]) for v in thw_list]
        lengths = [h * w for h, w in zip(Hs, Ws)]
        H_max, W_max = max(Hs), max(Ws)

        grids = {}
        for i in self.layers:
            feat = self._feat_cache[i]          # [Total_L, C] 或 [B, L_pad, C]
            if feat.dim() == 3:
                # 把 [B, L_pad, C] 展成 [Total_L_pad, C]，再按真实 lengths 切
                feat = feat.reshape(-1, feat.size(-1))
            # 3) 按每张图的 token 数切分
            chunks = torch.split(feat, lengths, dim=0)  # list of [L_i, C]

            per_samples = []
            for seg, H, W in zip(chunks, Hs, Ws):
                # [L_i, C] -> [C, H, W]
                C = seg.size(1)
                seg = seg.transpose(0, 1).contiguous().reshape(C, H, W)  # [C,H,W]
                # 4) pad 到 [C, H_max, W_max]（右、下方向补 0）
                if H != H_max or W != W_max:
                    pad_h = H_max - H
                    pad_w = W_max - W
                    # F.pad 的 2D pad 顺序是 (w_left, w_right, h_top, h_bottom)
                    seg = F.pad(seg, (0, pad_w, 0, pad_h), mode="constant", value=0.0)
                per_samples.append(seg)

            layer_grid = torch.stack(per_samples, dim=0)  # [B, C, H_max, W_max]
            grids[i] = layer_grid

        return grids

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

# -------------------------
# Heads (Fuse + Classification)
# -------------------------
class MiniFuse(nn.Module):
    def __init__(self, in_ch=1280, layers=4, mid=512, out=512):
        super().__init__()
        self.proj = nn.ModuleList([nn.Conv2d(in_ch, mid, 1) for _ in range(layers)])
        self.fuse = nn.Conv2d(mid * layers, out, 1)

    def forward(self, grids):  # list of [B,in_ch,H,W]
        zs = [p(g) for p, g in zip(self.proj, grids)]   # -> [B, mid, H, W] * layers
        z = torch.cat(zs, dim=1)                        # -> [B, mid*layers, H, W]
        return self.fuse(z)                             # -> [B, out, H, W]

class ClsHead(nn.Module):
    def __init__(self, in_ch, hidden=256):
        super().__init__()
        self.fc1 = nn.Linear(in_ch, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, x):  # x: [B,in_ch,H,W]
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)  # [B,in_ch]
        x = F.relu(self.fc1(x))
        return self.fc2(x).squeeze(1)               # [B] logit

class ForensicCls(nn.Module):
    def __init__(self, fuse_in_ch=1280, fuse_out_ch=512, layers=(7,15,23,31)):
        super().__init__()
        self.layers = layers
        self.fuser = MiniFuse(in_ch=fuse_in_ch, layers=len(layers), mid=512, out=fuse_out_ch)
        self.cls   = ClsHead(fuse_out_ch)

    def forward(self, grid_dict):
        # Expect keys like {7:..., 15:..., 23:..., 31:...}
        grids = [grid_dict[i] for i in self.layers]
        fused = self.fuser(grids)   # [B, 512, H, W]
        logit = self.cls(fused)     # [B]
        return logit

# -------------------------
# Collate: build Qwen processor inputs for a batch
# -------------------------
def collate_and_process(batch, processor):
    """
    batch: list of dicts with keys image(PIL), label(int), path(str)
    returns dict:
      pixel_values: [B,3,H,W]
      image_grid_thw: [B,3]
      labels: [B]
      paths: list[str]
    """
    messages = []
    images_for_proc = []
    for rec in batch:
        # per-sample chat message: image + placeholder text
        msg = [{"role": "user", "content": [{"type": "image", "image": rec["image"]},
                                            {"type": "text", "text": "."}]}]
        messages.append(msg[0])  # processor wants list of dicts; we'll pass a list later
        # process_vision_info will read images from messages again; we still keep symmetry
        images_for_proc.append(rec["image"])

    # Use qwen_vl_utils to get vision inputs
    image_inputs_list = []
    for m in messages:
        ii, _ = process_vision_info([m])  # returns list for one sample
        # ii is a list of image-like items; we just need placeholder to keep alignment
        image_inputs_list.append(ii)

    # processor can take batched images & texts
    texts = [processor.apply_chat_template([m], tokenize=False, add_generation_prompt=True)
             for m in messages]

    # Flatten image_inputs_list (each is a list with single element)
    images_flat = [ii for ii in image_inputs_list]  # still nested; processor expects same nesting
    # processor handles nested images structure for Qwen2.5-VL
    inputs = processor(text=texts, images=images_flat, return_tensors="pt", padding=True)

    labels = torch.tensor([rec["label"] for rec in batch], dtype=torch.float32)
    paths  = [rec["path"] for rec in batch]
    return inputs, labels, paths

# -------------------------
# Train / Eval
# -------------------------
@torch.no_grad()
def evaluate(model, visual_tap, data_loader, device):
    model.eval()
    ys, ps = [], []
    for batch in data_loader:
        (inputs, labels, _) = batch
        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = labels.to(device)

        grids = visual_tap(inputs["pixel_values"], inputs["image_grid_thw"])
        grids = {k: v.float() for k, v in grids.items()}
        logits = model(grids)
        prob = torch.sigmoid(logits)

        ys.append(labels.detach().cpu().numpy())
        ps.append(prob.detach().cpu().numpy())

    y_true = np.concatenate(ys).astype(np.int64)
    y_prob = np.concatenate(ps)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = float("nan")
    try:
        ap = average_precision_score(y_true, y_prob)  # PR-AUC
    except Exception:
        ap = float("nan")
    return {"auroc": auc, "pr_auc": ap}

def train_one_epoch(model, visual_tap, data_loader, optimizer, loss_fn, device, scaler=None, log_interval=50):
    model.train()
    total_loss = 0.0
    n_samples = 0

    for step, batch in enumerate(data_loader, 1):
        (inputs, labels, _) = batch
        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = labels.to(device)

        # Forward through frozen visual backbone (needs grad=False for speed, but we allow autograd on heads)
        with torch.no_grad():
            grids = visual_tap(inputs["pixel_values"], inputs["image_grid_thw"])
            grids = {k: v.float() for k, v in grids.items()}

        # Heads forward + loss
        logits = model(grids)
        loss = loss_fn(logits, labels)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        bs = labels.size(0)
        total_loss += loss.item() * bs
        n_samples += bs

        if step % log_interval == 0:
            print(f"  step {step:5d} | loss {total_loss/n_samples:.4f}")

    return total_loss / max(1, n_samples)

# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    # 不再传 root/ann/model_path
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--out_dir", default="outputs_clsA")
    parser.add_argument("--use_sampler", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # 固定路径
    model_path = "/root/models/Qwen2.5-VL-7B-Instruct/"

    # ===== 1) 数据集与分层划分 =====
    ds = ForgeryClsDataset(transform=None)
    print(f"Loaded {len(ds)} samples.")

    labels = np.array([ds[i]["label"] for i in range(len(ds))])
    idx_all = np.arange(len(ds))
    idx_pos = idx_all[labels == 1]; idx_neg = idx_all[labels == 0]
    np.random.shuffle(idx_pos); np.random.shuffle(idx_neg)

    n_pos_val = int(len(idx_pos) * args.val_ratio)
    n_neg_val = int(len(idx_neg) * args.val_ratio)
    val_idx = np.concatenate([idx_pos[:n_pos_val], idx_neg[:n_neg_val]])
    train_idx = np.concatenate([idx_pos[n_pos_val:], idx_neg[n_neg_val:]])
    np.random.shuffle(train_idx); np.random.shuffle(val_idx)

    ds_train = Subset(ds, train_idx.tolist())
    ds_val   = Subset(ds, val_idx.tolist())
    print(f"Train: {len(ds_train)} | Val: {len(ds_val)}")

    # ===== 2) 模型与 processor =====
    processor = AutoProcessor.from_pretrained(model_path)
    qwen = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    for p in qwen.parameters():
        p.requires_grad = False
    qwen.eval()

    # 抽两层降显存气死我了
    visual_tap = QwenVisualTap(qwen.visual, layers=(15, 31)).to(device)
    heads = ForensicCls(fuse_in_ch=1280, fuse_out_ch=512, layers=(15, 31)).to(device)

    def collate_fn(batch):
        return collate_and_process(batch, processor)

    # Dataloaders
    if args.use_sampler:
        y_train = labels[train_idx]
        class_sample_count = np.array([np.sum(y_train == t) for t in [0,1]], dtype=np.float32)
        weight = 1.0 / (class_sample_count + 1e-8)
        samples_weight = np.array([weight[int(t)] for t in y_train])
        sampler = WeightedRandomSampler(samples_weight, num_samples=len(samples_weight), replacement=True)
        dl_train = DataLoader(ds_train, batch_size=args.batch_size, sampler=sampler,
                              num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)
    else:
        dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)

    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)

    # 损失/优化器 
    y_train = labels[train_idx]
    n_pos = max(1, int((y_train == 1).sum())); n_neg = max(1, int((y_train == 0).sum()))
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32, device=device)
    bce_logits = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(heads.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = None  # 头部 fp32 即可

    # 训练循环 + 可视化
    history = []
    best_auc = -1.0

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        # --- Train ---
        heads.train()
        total_loss, n_samples = 0.0, 0
        for step, batch in enumerate(dl_train, 1):
            inputs, labels_b, _ = batch
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels_b = labels_b.to(device)

            with torch.no_grad():
                grids = visual_tap(inputs["pixel_values"], inputs["image_grid_thw"])
            # 关键：把 bf16 的视觉特征转成 fp32 给头部
            grids = {k: v.float() for k, v in grids.items()}

            logits = heads(grids)
            loss = bce_logits(logits, labels_b)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            bs = labels_b.size(0)
            total_loss += loss.item() * bs
            n_samples += bs
            if step % 50 == 0:
                print(f"  step {step:5d} | loss {total_loss/max(1,n_samples):.4f}")

        train_loss = total_loss / max(1, n_samples)

        # --- Eval ---
        metrics = evaluate(heads, visual_tap, dl_val, device)
        print(f"Train loss: {train_loss:.4f} | Val AUROC: {metrics['auroc']:.4f} | Val PR-AUC: {metrics['pr_auc']:.4f}")

        # 记录历史
        history.append({
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_auroc": float(metrics["auroc"]),
            "val_pr_auc": float(metrics["pr_auc"]),
        })

        # 保存最好,考虑要不要加一个断点重新跑？
        if metrics["auroc"] > best_auc:
            best_auc = metrics["auroc"]
            ckpt = {
                "epoch": epoch,
                "state_dict": heads.state_dict(),
                "best_val_auc": best_auc,
                "args": vars(args),
            }
            torch.save(ckpt, os.path.join(args.out_dir, "best_val_auc.pt"))
            print(f"Saved checkpoint with AUROC {best_auc:.4f}")

    print("\nDone. Best Val AUROC:", best_auc)

    # 保存 CSV
    import pandas as pd
    pd.DataFrame(history).to_csv(os.path.join(args.out_dir, "training_log.csv"), index=False)
    print("Saved log to training_log.csv")
if __name__ == "__main__":
    main()