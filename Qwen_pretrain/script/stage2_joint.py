import os, json, math, argparse, random, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import pandas as pd, os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, average_precision_score
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

from peft import PeftModel
import math, copy, torch, torch.nn as nn, torch.nn.functional as F

# Utils
def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def resize_square_pad(img: Image.Image, size=448, pad_color=(128,128,128)):
    w, h = img.size; s = size / max(w, h)
    nw, nh = int(round(w*s)), int(round(h*s))
    img = img.resize((nw, nh), Image.BICUBIC)
    canvas = Image.new("RGB", (size, size), pad_color)
    canvas.paste(img, ((size-nw)//2, (size-nh)//2))
    return canvas

def _logit(p, eps=1e-6):
    p = float(np.clip(p, eps, 1 - eps))
    return math.log(p / (1 - p))


def build_warmup_cosine(total_steps, warmup_ratio=0.05, min_lr_scale=0.1):
    warmup_steps = int(total_steps * warmup_ratio)
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_lr_scale + (1 - min_lr_scale) * 0.5 * (1 + math.cos(math.pi * progress))
    return lr_lambda

# Dataset
def _parse_size_str(s: str):
    if isinstance(s, str) and "x" in s.lower():
        try:
            h, w = s.lower().split("x")
            return int(h), int(w)
        except: return None
    return None

class ForgeryJointDataset(Dataset):
    def __init__(self, ann_path, data_root="/root/data"):
        self.root = Path(data_root).resolve()
        ann_path = Path(ann_path).resolve()
        assert ann_path.exists(), f"ann not found: {ann_path}"
        arr = json.loads(ann_path.read_text(encoding="utf-8"))
        assert isinstance(arr, list) and len(arr) > 0, f"Empty ann: {ann_path}"

        # 常见根目录
        self.img_dirs  = [
            self.root / "trainingsetbig/image",
            self.root / "trainingset2/image",
            self.root / "trainingset2/trainingset2/image",  # 兼容旧奇怪层级
        ]
        self.mask_dirs = [
            self.root / "trainingsetbig/spatial_localize",
            self.root / "trainingset2/spatial_localize",
        ]

        def _normalize_rel(p: str) -> str:
            if not isinstance(p, str): return ""
            p = p.replace("trainingset2/trainingset2/", "trainingset2/")
            p = p.replace("\\", "/")
            return p

        def _resolve_any(rel_or_name: str, kind: str = "image") -> Path | None:

            if not rel_or_name: return None
            rel_or_name = _normalize_rel(rel_or_name)
            p = Path(rel_or_name)
            if p.is_absolute() and p.exists():
                return p.resolve()
            cand = (self.root / rel_or_name).resolve()
            if cand.exists():
                return cand
            name = p.name
            dirs = self.img_dirs if kind == "image" else self.mask_dirs
            for d in dirs:
                c = (d / name).resolve()
                if c.exists(): 
                    return c
            return None

        def _parse_size_str(s: str):
            if isinstance(s, str) and "x" in s.lower():
                try:
                    h, w = s.lower().split("x")
                    return int(h), int(w)
                except:
                    return None
            return None

        self.items = []
        miss_img, miss_mask = 0, 0
        for rec in arr:
            img_rel = rec.get("image_path", "")
            img_p = _resolve_any(img_rel, kind="image")
            if img_p is None:
                miss_img += 1
                continue

            y = int(rec.get("label", 0))
            mp = rec.get("mask_path", "")

            if y == 1:
                mask_p = None
                if not (isinstance(mp, str) and "x" in mp.lower()):
                    mask_p = _resolve_any(mp, kind="mask")
                if mask_p is None:
                    miss_mask += 1
                self.items.append({"img": img_p, "label": 1, "mask": mask_p})
            else:
                size = _parse_size_str(mp)
                if size is None:
                    W, H = Image.open(img_p).size
                    size = (H, W)
                self.items.append({"img": img_p, "label": 0, "mask": size})

        if miss_img:
            print(f"[WARN][Dataset] missing images ignored: {miss_img}")
        if miss_mask:
            print(f"[WARN][Dataset] fake samples without mask file: {miss_mask}")
        if not self.items:
            raise RuntimeError("No valid items after resolving paths.")

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        rec = self.items[i]
        img = Image.open(rec["img"]).convert("RGB")
        y   = rec["label"]
        m   = rec["mask"]
        if y == 1:
            if isinstance(m, Path):
                mask = Image.open(m).convert("L")
            else:
                mask = None  # 假图但没掩码
        else:
            H, W = m
            mask = Image.new("L", (W, H), 0)
        return {"image": img, "label": y, "mask": mask, "path": str(rec["img"])}

# Collate（与 Qwen processor 对齐）+ mask letterbox

def resize_square_pad_mask(img: Image.Image, size=448):
    # 灰度掩码专用：保持单通道
    if img.mode != "L":
        img = img.convert("L")
    w, h = img.size
    s = size / max(w, h)
    nw, nh = int(round(w * s)), int(round(h * s))
    img = img.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("L", (size, size), 0)
    canvas.paste(img, ((size - nw) // 2, (size - nh) // 2))
    return canvas

def collate_joint(batch, processor, fixed_res=448):
    messages, masks = [], []
    for rec in batch:
        im = resize_square_pad(rec["image"], fixed_res)
        messages.append({"role":"user","content":[{"type":"image","image":im},{"type":"text","text":"."}]})

        m = rec["mask"]
        if m is None:
            arr = np.zeros((fixed_res, fixed_res), dtype=np.float32)
        else:
            m_res = resize_square_pad_mask(m, fixed_res)      # 单通道
            arr = np.array(m_res, dtype=np.float32)
            if arr.max() > 1.0:
                arr /= 255.0

        # 先二值化，再 append
        if rec["label"] == 1:
            p99 = np.percentile(arr, 99)
            thr = 0.2 * float(p99)
            arr = (arr >= thr).astype(np.float32)
        else:
            arr = np.zeros_like(arr, dtype=np.float32)

        masks.append(arr)

    texts  = [processor.apply_chat_template([m], tokenize=False, add_generation_prompt=True) for m in messages]
    images = [m["content"][0]["image"] for m in messages]
    inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)
    labels = torch.tensor([rec["label"] for rec in batch], dtype=torch.float32)
    masks  = torch.from_numpy(np.stack(masks, axis=0))  # [B,H,W] ∈ {0,1}
    paths  = [rec["path"] for rec in batch]
    return inputs, labels, masks, paths
    
class QwenVisualTap(nn.Module):
    """
    从视觉塔 blocks 取特征；返回 {layer_idx: [B,C,Hmax,Wmax]}
    hooks 挂 core（visual.base_model），forward 也走 core
    """
    def __init__(self, visual, layers=(15,23,31)):
        super().__init__()
        self.layers = list(layers)
        self._feat_cache = {}
        self._hooks = []
        self.rebind(visual)

    def _make_hook(self, idx):
        def _hook(module, inp, out):
            if isinstance(out, (tuple, list)): out = out[0]
            self._feat_cache[idx] = out
        return _hook

    def _clear_hooks(self):
        for h in self._hooks:
            try: h.remove()
            except: pass
        self._hooks = []

    def rebind(self, visual):
        self.visual = visual
        self.core = getattr(visual, "base_model", visual)
        self._clear_hooks()
        for i in self.layers:
            self._hooks.append(self.core.blocks[i].register_forward_hook(self._make_hook(i)))

    def forward(self, pixel_values, thw):
        self._feat_cache.clear()
        dev   = next(self.core.parameters()).device
        dtype = next(self.core.parameters()).dtype
        pv  = pixel_values.to(device=dev, dtype=dtype)
        thw = thw.to(dev) if isinstance(thw, torch.Tensor) else thw

        _ = self.core(pv, thw)  # 触发 hooks

        thw_list = thw.tolist() if isinstance(thw, torch.Tensor) else thw
        Hs = [int(v[1]) for v in thw_list]; Ws = [int(v[2]) for v in thw_list]
        Hmax, Wmax = max(Hs), max(Ws)
        lengths = [h*w for h,w in zip(Hs, Ws)]

        missing = [i for i in self.layers if i not in self._feat_cache]
        if missing:
            raise RuntimeError(f"[VIS TAP] hooks not triggered for layers: {missing}.")

        out = {}
        for i in self.layers:
            feat = self._feat_cache[i]  # [Total_L,C] or [B,L,C]
            if feat.dim() == 3:
                feat = feat.reshape(-1, feat.size(-1))
            chunks = torch.split(feat, lengths, dim=0)  # N*[L_i,C]

            per = []
            for seg, H, W in zip(chunks, Hs, Ws):
                C = seg.size(1)
                seg = seg.transpose(0,1).reshape(C, H, W)  # [C,H,W]
                if H != Hmax or W != Wmax:
                    seg = F.pad(seg, (0, Wmax - W, 0, Hmax - H), value=0.0)
                per.append(seg)
            out[i] = torch.stack(per, dim=0).float()       # [B,C,Hmax,Wmax]
        return out

#visual_tap = QwenVisualTap(qwen.visual, layers=(15,23,31)).to(device)

# ===== 头结构（与 LoRA 期一致：MiniFuse(3路)+ClsHead+EvidenceHead64）=====
# ========== Shared Components ==========
class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(p, dtype=torch.float32))
        self.eps = eps
    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)
        x = F.adaptive_avg_pool2d(x, 1)
        return x.pow(1.0 / self.p).flatten(1)

class MiniFuse(nn.Module):
    def __init__(self, in_ch=1280, layers=3, mid=512, out=512):
        super().__init__()
        self.proj = nn.ModuleList([nn.Conv2d(in_ch, mid, 1) for _ in range(layers)])
        self.dw   = nn.Conv2d(mid * layers, mid * layers, 3, padding=1, groups=mid * layers)
        self.pw   = nn.Conv2d(mid * layers, out, 1)
    def forward(self, grids):
        zs = [p(g) for p, g in zip(self.proj, grids)]
        z  = torch.cat(zs, dim=1)
        z  = self.dw(z)
        return self.pw(z)

class ClsHead(nn.Module):
    """完全与 Stage 2.1 一致"""
    def __init__(self, in_ch, hidden=256, use_l2norm=True, dropout=0.0):
        super().__init__()
        self.pool   = GeM()
        self.use_l2 = use_l2norm
        self.fc1    = nn.Linear(in_ch, hidden)
        self.drop   = nn.Dropout(p=dropout) if dropout and dropout > 0 else nn.Identity()
        self.fc2    = nn.Linear(hidden, 1)
    def forward(self, feat_map):
        x = self.pool(feat_map)
        if self.use_l2:
            x = F.normalize(x, p=2, dim=1)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x)[:, 0]

class EvidenceHead64(nn.Module):
    def __init__(self, in_ch=512):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, 256, 3, padding=1), nn.GELU(),
            nn.Conv2d(256, 128, 3, padding=1), nn.GELU(),
        )
        self.up  = nn.Sequential(
            nn.Conv2d(128, 128*4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.Conv2d(128, 64, 3, padding=1), nn.GELU(),
        )
        self.out32 = nn.Conv2d(128, 1, 1)
        self.out64 = nn.Conv2d(64,  1, 1)
    def forward(self, fmap):
        h  = self.enc(fmap)
        c32 = self.out32(h).squeeze(1)
        u  = self.up(h)
        f64 = self.out64(u).squeeze(1)
        return c32, f64

# ========== Joint Model ==========
class ForensicJoint(nn.Module):
    def __init__(self, fuse_in_ch=1280, fuse_out_ch=512, layers=(15,23,31)):
        super().__init__()
        self.layers = tuple(layers)
        self.fuser  = MiniFuse(in_ch=fuse_in_ch, layers=len(self.layers), mid=512, out=fuse_out_ch)
        self.cls    = ClsHead(fuse_out_ch, hidden=256, use_l2norm=True)
        self.evi    = EvidenceHead64(in_ch=fuse_out_ch)

    def forward(self, grid_dict, detach_evi=False):
        grids = [grid_dict[i] for i in self.layers]
        fused = self.fuser(grids)
        logits = self.cls(fused)
        if detach_evi:
            hm32, hm64 = self.evi(fused.detach())
        else:
            hm32, hm64 = self.evi(fused)
        return logits, (hm32, hm64)

@torch.no_grad()
def init_joint_heads_with_priors(heads, p1_prior=0.5, pix_prior=0.03):
    if hasattr(heads, "cls") and hasattr(heads.cls, "fc2") and heads.cls.fc2.bias is not None:
        heads.cls.fc2.bias.fill_(_logit(p1_prior))
    evi = getattr(heads, "evi", None)
    if evi is not None:
        for m in evi.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None: nn.init.zeros_(m.bias)
        if hasattr(evi, "out64") and evi.out64.bias is not None:
            evi.out64.bias.fill_(_logit(pix_prior))

# Losses
def dice_loss(pred_prob, target, eps=1e-6):
    # pred_prob/target: (B,H,W) in [0,1]
    num = 2.0 * (pred_prob * target).sum(dim=(1,2))
    den = (pred_prob*pred_prob).sum(dim=(1,2)) + (target*target).sum(dim=(1,2)) + eps
    return 1.0 - (num/den).mean()

def focal_bce_with_logits(logits, targets, alpha=0.25, gamma=2.0):
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    p   = torch.sigmoid(logits)
    pt  = targets*p + (1-targets)*(1-p)
    return (alpha * (1-pt).pow(gamma) * bce).mean()

# Train
def _unpack_heatmap(hm_out):
    """
    统一把证据头输出拆成 (hm32_opt, hm64_req)
    - 若是 tuple/list，形如 (hm32, hm64)
    - 若是单张 heatmap，就当作 hm64
    返回：(hm32 or None, hm64)；hm64 保证是 [B,1,H,W]
    """
    if isinstance(hm_out, (tuple, list)):
        hm32, hm64 = hm_out
    else:
        hm32, hm64 = None, hm_out

    if hm64.dim() == 3:          # [B,H,W]
        hm64 = hm64.unsqueeze(1) # -> [B,1,H,W]
    elif hm64.dim() == 4:
        if hm64.size(1) != 1:    
            hm64 = hm64[:, :1, ...]
    else:
        raise RuntimeError(f"Unexpected heatmap shape: {tuple(hm64.shape)}")

    return hm32, hm64


def train_one_epoch_joint(model, visual_tap, loader, optimizer, device,
                          grad_accum=1, scheduler=None, log_interval=50,
                          evi_alpha=0.7, λ_e=0.5, use_focal=True, λ_sparse=5e-4, λ_contrast=5e-3, diagnose_grad=False):
    model.train()
    bce_cls = nn.BCEWithLogitsLoss()
    optimizer.zero_grad(set_to_none=True)
    total, n = 0.0, 0
    def compute_grad_cosine(model, loss_cls, loss_evi):
        model.zero_grad(set_to_none=True)
    
        # 分类梯度
        loss_cls.backward(retain_graph=True)
        grad_cls = []
        for p in model.fuser.parameters():
            if p.grad is not None:
                grad_cls.append(p.grad.detach().flatten())
        grad_cls = torch.cat(grad_cls)
        model.zero_grad(set_to_none=True)
    
        # 证据梯度
        loss_evi.backward(retain_graph=True)
        grad_evi = []
        for p in model.fuser.parameters():
            if p.grad is not None:
                grad_evi.append(p.grad.detach().flatten())
        grad_evi = torch.cat(grad_evi)
    
        cosine = torch.dot(grad_cls, grad_evi) / (grad_cls.norm() * grad_evi.norm() + 1e-8)
        model.zero_grad(set_to_none=True)
        return cosine.item()

    for step, (inputs, labels, masks, _) in enumerate(loader, 1):
        inputs = {k: v.to(device) for k,v in inputs.items()}
        labels = labels.to(device)                 # (B,)
        masks  = masks.to(device)                  # (B,448,448)

        with torch.no_grad():
            grids = visual_tap(inputs["pixel_values"], inputs["image_grid_thw"])

        logits, hm_out = model(grids)              # logits: [B], hm_out: (hm32, hm64) 或 hm64
        hm32, hm64 = _unpack_heatmap(hm_out)       # hm64: [B,1,H,W]

        # 分类损失
        L_cls = bce_cls(logits, labels)

        # 像素监督 (直接在 64×64 上对齐)
        Ht, Wt = hm64.shape[-2:]
        prob64 = torch.sigmoid(hm64)                               # [B,1,H,W]
        mask64 = F.interpolate(masks.unsqueeze(1).float(), size=(Ht, Wt),
                               mode="bilinear", align_corners=False).clamp(0,1)  # [B,1,H,W]

        # 正负像素不均衡权重
        with torch.no_grad():
            pos_pix = mask64.sum()
            tot_pix = mask64.numel()
            neg_pix = tot_pix - pos_pix
            pos_w   = (neg_pix / (pos_pix + 1e-6)).clamp(1.0, 50.0)
        bce_pix = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        L_bce   = bce_pix(hm64, mask64)

        # Dice 只在假图上计算
        pos_mask = (labels.view(-1,1,1,1) > 0.5).float()
        pp = prob64 * pos_mask
        gg = mask64 * pos_mask
        inter  = (pp * gg).sum(dim=(1,2,3))
        denom  = pp.sum(dim=(1,2,3)) + gg.sum(dim=(1,2,3)) + 1e-6
        dice_b = 1. - (2*inter + 1e-6) / denom
        valid  = (pos_mask.view(pos_mask.size(0), -1).sum(dim=1) > 0).float()
        L_dice = (dice_b * valid).sum() / (valid.sum() + 1e-6)

        # 主证据损失
        L_evi = evi_alpha * L_bce + (1. - evi_alpha) * L_dice

        # 稀疏 & 对比（可选）
        L_sparse = prob64.mean()
        real_idx = (labels < 0.5)
        fake_idx = (labels > 0.5)
        if λ_contrast > 0 and real_idx.any() and fake_idx.any():
            L_contrast = prob64[real_idx].mean() - prob64[fake_idx].mean()
        else:
            L_contrast = torch.zeros((), device=device)

        loss = L_cls + λ_e*L_evi + λ_sparse*L_sparse + λ_contrast*L_contrast
        if diagnose_grad:
            cosine = compute_grad_cosine(model, L_cls, L_evi)
            print(f"[DEBUG] Grad cosine (cls vs evi): {cosine:.3f}")
            # 不更新权重，只观测方向
            optimizer.zero_grad(set_to_none=True)
            continue
        # ====== 梯度夹角分析 ======
        if step % 50 == 0:  # 每50步打印一次
            cosine = compute_grad_cosine(model, L_cls, L_evi)
            print(f"[DEBUG] Gradient cosine (cls vs evi): {cosine:.3f}")
        # ========================
        (loss / grad_accum).backward()

        if step % grad_accum == 0:
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None: scheduler.step()

        bs = labels.size(0)
        total += loss.item() * bs; n += bs

        if step == 1:
            with torch.no_grad():
                p = torch.sigmoid(logits); pos=(labels>0.5)
                print(f"[DEBUG] pos={pos.float().mean():.3f} | "
                      f"logit y0={logits[~pos].mean():.4f} y1={logits[pos].mean() if pos.any() else float('nan'):.4f} | "
                      f"hm_mean={prob64.mean():.4f}")
        if step % log_interval == 0:
            print(f"  step {step:5d} | loss {total/max(1,n):.4f}")

    return total / max(1,n)

def schedule_joint_weights(epoch, total_epochs):
    """
    动态调度 λₑ（证据损失权重）和 evi_alpha（BCE:Dice 比例）
    按阶段递增强化联合训练。
    """
    if epoch <= total_epochs * 0.2:       # 前 20% epoch：分类主导
        λ_e = 0.4
        evi_alpha = 0.8
    elif epoch <= total_epochs * 0.5:     # 中期平衡
        λ_e = 0.6
        evi_alpha = 0.75
    elif epoch <= total_epochs * 0.8:     # 后期强化证据
        λ_e = 0.7
        evi_alpha = 0.7
    else:                                 # 收尾：微调阶段
        λ_e = 0.5
        evi_alpha = 0.8
    return λ_e, evi_alpha


import os, csv

def train_joint_v2(model, visual_tap, dl_train, dl_val, optimizer, scheduler,
                   device, total_epochs=20, grad_accum=8, log_interval=100,
                   out_dir="outputs_lora_joint", csv_name="train_history_joint.csv",
                   early_stop_patience=3):
    """
    主训练循环：动态调整 λₑ / evi_alpha，自动保存最佳 AUROC / IoU，按 epoch 追加写入 CSV，
    并启用早停（AUROC 与 IoU 都无提升计 1 次，连续 early_stop_patience 次则停止）。
    """
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, csv_name)

    # 若 CSV 不存在，先写表头
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "epoch", "lambda_e", "evi_alpha", "train_loss",
                "val_auroc", "val_acc_best", "val_f1_best",
                "val_thr_acc", "val_thr_f1",
                "val_iou", "val_dice",
                "lr"
            ])
            writer.writeheader()

    best_auc, best_iou = -1.0, -1.0
    best_auc_ep, best_iou_ep = -1, -1
    no_improve_counter = 0
    history = []

    for epoch in range(1, total_epochs + 1):
        # 动态权重调度（你已有的函数）
        λ_e, evi_alpha = schedule_joint_weights(epoch, total_epochs)
        print(f"\nEpoch {epoch}/{total_epochs} | λₑ={λ_e:.2f} | evi_alpha={evi_alpha:.2f} | lr={optimizer.param_groups[0]['lr']:.2e}")

        # === Train ===
        train_loss = train_one_epoch_joint(
            model, visual_tap, dl_train, optimizer, device,
            grad_accum=grad_accum, scheduler=scheduler,
            evi_alpha=evi_alpha, λ_e=λ_e,
            λ_sparse=1e-4, λ_contrast=5e-3
        )

        # === Validation ===
        m_cls = evaluate(model, visual_tap, dl_val, device)
        m_evi = evaluate_evidence_iou(model, visual_tap, dl_val, device)

        print(f"Epoch {epoch}: AUROC={m_cls['auroc']:.4f} | "
              f"ACC@best={m_cls['acc']:.4f}@thr={m_cls['thr_acc']:.2f} | "
              f"F1@best={m_cls['f1']:.4f}@thr={m_cls['thr_f1']:.2f} | "
              f"IoU={m_evi['mean_iou']:.4f} | Dice={m_evi['mean_dice']:.4f}")

        # === 保存到 history（内存） ===
        row = {
            "epoch": int(epoch),
            "lambda_e": float(λ_e),
            "evi_alpha": float(evi_alpha),
            "train_loss": float(train_loss),
            "val_auroc": float(m_cls["auroc"]),
            "val_acc_best": float(m_cls["acc"]),
            "val_f1_best": float(m_cls["f1"]),
            "val_thr_acc": float(m_cls["thr_acc"]),
            "val_thr_f1": float(m_cls["thr_f1"]),
            "val_iou": float(m_evi["mean_iou"]),
            "val_dice": float(m_evi["mean_dice"]),
            "lr": float(optimizer.param_groups[0]['lr']),
        }
        history.append(row)

        # === 追加写入 CSV（每个 epoch 写一次） ===
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writerow(row)

        # === 保存最佳模型 ===
        improved = False
        if m_cls["auroc"] > best_auc + 1e-6:
            best_auc = m_cls["auroc"]; best_auc_ep = epoch
            torch.save(model.state_dict(), os.path.join(out_dir, "best_joint_by_AUROC.pt"))
            print(f"✔ Saved new best AUROC={best_auc:.4f} @epoch {best_auc_ep}")
            improved = True

        if m_evi["mean_iou"] > best_iou + 1e-6:
            best_iou = m_evi["mean_iou"]; best_iou_ep = epoch
            torch.save(model.state_dict(), os.path.join(out_dir, "best_joint_by_IoU.pt"))
            print(f"✔ Saved new best IoU={best_iou:.4f} @epoch {best_iou_ep}")
            improved = True

        # === 早停逻辑：两个指标都没提升才 +1 ===
        if not improved:
            no_improve_counter += 1
            print(f"[EarlyStop] No improvement this epoch. Patience {no_improve_counter}/{early_stop_patience}")
            if no_improve_counter >= early_stop_patience:
                print(f"\n⏹ Early stopped at epoch {epoch}. "
                      f"Best AUROC={best_auc:.4f} @epoch {best_auc_ep}, "
                      f"Best IoU={best_iou:.4f} @epoch {best_iou_ep}")
                break
        else:
            no_improve_counter = 0

    print(f"\n🏁 Training done. Best AUROC={best_auc:.4f} @epoch {best_auc_ep}, Best IoU={best_iou:.4f} @epoch {best_iou_ep}")
    return history


@torch.no_grad()
def evaluate(model, visual_tap, data_loader, device):
    model.eval()
    ys, ps = [], []
    for (inputs, labels, masks, _) in data_loader:
        inputs = {k: v.to(device) for k,v in inputs.items()}
        labels = labels.to(device)
        grids  = visual_tap(inputs["pixel_values"], inputs["image_grid_thw"])
        logit_cls, _ = model(grids)
        ys.append(labels.detach().cpu().numpy())
        ps.append(torch.sigmoid(logit_cls).detach().cpu().numpy())

    y_true = np.concatenate(ys).astype(np.int64)
    y_prob = np.concatenate(ps)

    auroc = roc_auc_score(y_true, y_prob)

    # 扫阈值
    thrs = np.linspace(0.05, 0.95, 19)
    best_acc = best_f1 = -1.0
    thr_acc = thr_f1 = 0.5
    for t in thrs:
        pred = (y_prob >= t).astype(int)
        acc = accuracy_score(y_true, pred)
        f1  = f1_score(y_true, pred)
        if acc > best_acc: best_acc, thr_acc = acc, t
        if f1  > best_f1:  best_f1,  thr_f1  = f1,  t

    return {
        "auroc": float(auroc),
        "acc": float(best_acc),
        "f1": float(best_f1),
        "thr_acc": float(thr_acc),
        "thr_f1": float(thr_f1),
    }


@torch.no_grad()
def evaluate_evidence_iou(model, visual_tap, data_loader, device, thr=0.3, only_fake=True):
    model.eval()
    inter_sum = union_sum = 0.0
    dice_num = dice_den = 0.0
    any_valid = False

    for (inputs, labels, masks, _) in data_loader:
        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = labels.to(device)
        grids  = visual_tap(inputs["pixel_values"], inputs["image_grid_thw"])
        grids  = {k: v.float() for k,v in grids.items()}

        _, hm_out = model(grids)                  # 可能是 (hm32, hm64) 或 hm64
        _, hm64 = _unpack_heatmap(hm_out)         # [B,1,H,W]

        prob64 = torch.sigmoid(hm64)              # [B,1,H,W]
        if isinstance(masks, list):
            masks = torch.stack(masks, dim=0)
        masks = masks.to(device)
        gt64 = F.interpolate(masks.unsqueeze(1).float(), size=prob64.shape[-2:],
                             mode="bilinear", align_corners=False).clamp(0,1)

        if only_fake:
            idx = (labels > 0.5).view(-1)
            if not idx.any():
                continue
            prob64 = prob64[idx]
            gt64   = gt64[idx]

        pred = (prob64 >= thr).float()
        gt   = (gt64   >= thr).float()

        inter  = (pred * gt).sum()
        union  = (pred + gt - pred * gt).sum() + 1e-6
        dice_n = 2 * inter
        dice_d = pred.sum() + gt.sum() + 1e-6

        inter_sum += inter.item()
        union_sum += union.item()
        dice_num  += dice_n.item()
        dice_den  += dice_d.item()
        any_valid = True

    if not any_valid:
        return {"mean_iou": 0.0, "mean_dice": 0.0}

    return {
        "mean_iou":  float(inter_sum / union_sum),
        "mean_dice": float(dice_num / dice_den),
    }
# Main
def main():
    ap = argparse.ArgumentParser()
    # 数据
    ap.add_argument("--data_root", default="//root/autodl-tmp/data")
    ap.add_argument("--train_ann", default="/root/autodl-tmp/data/trainval/train_idx.json")
    ap.add_argument("--val_ann",   default="/root/autodl-tmp/data/trainval/val_idx.json")
    # 模型与路径
    ap.add_argument("--model_path", default="/root/autodl-tmp/models/Qwen2.5-VL-7B-Instruct/")
    ap.add_argument("--clsA_dir",   default="outputs_lora_cls")
    ap.add_argument("--out_dir",    default="outputs_lora_joint")
    # 训练
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--base_lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=3e-4)
    ap.add_argument("--warmup_ratio", type=float, default=0.06)
    ap.add_argument("--min_lr_scale", type=float, default=0.1)
    # 证据损失权重
    ap.add_argument("--evi_weight", type=float, default=1.0)
    ap.add_argument("--evi_alpha",  type=float, default=0.5, help="BCE:Dice mixing for evidence loss")
    ap.add_argument("--no_focal",   action="store_true")
    ap.add_argument("--sparse_w",   type=float, default=1e-4)
    ap.add_argument("--contrast_w", type=float, default=5e-3)
    # 评估/可视化
    ap.add_argument("--diagnose_grad", action="store_true", help="Enable gradient conflict diagnosis mode")
    
    ap.add_argument("--eval_vis",   type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    history = []
    best_auc, best_iou = -1.0, -1.0
    best_auc_ep, best_iou_ep = -1, -1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # data
    ds_train = ForgeryJointDataset(args.train_ann, data_root=args.data_root)
    ds_val   = ForgeryJointDataset(args.val_ann,   data_root=args.data_root)
    print(f"Train: {len(ds_train)} | Val: {len(ds_val)}")
    
    # 快速自检
    for k in [0, len(ds_train)//2, len(ds_train)-1]:
        r = ds_train[k]
        print("TRAIN sample:", k, "| y=", r["label"], "| path=", r["path"], "| mask=", type(r["mask"]).__name__)
    for k in [0, len(ds_val)//2, len(ds_val)-1]:
        r = ds_val[k]
        print("VAL   sample:", k, "| y=", r["label"], "| path=", r["path"], "| mask=", type(r["mask"]).__name__)
    
    # 只初始化一次 processor
    processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=True)
    def collate_fn(batch): 
        return collate_joint(batch, processor, fixed_res=448)
    
    # DataLoader
    dl_train = DataLoader(
        ds_train, batch_size=args.batch_size, shuffle=True,
        num_workers=8, pin_memory=True, persistent_workers=True,
        collate_fn=collate_fn
    )
    dl_val = DataLoader(
        ds_val, batch_size=args.batch_size, shuffle=False,
        num_workers=8, pin_memory=True, persistent_workers=True,
        collate_fn=collate_fn
    )

    def sanity_count_nonzero_masks(dl, max_batches=20):
        n_fake, n_fake_with_mask = 0, 0
        for b, (inputs, labels, masks, _) in enumerate(dl):
            y = labels.numpy().astype(int)     # [B]
            if isinstance(masks, list):
                m = torch.stack(masks, dim=0)  # 兼容老版
            else:
                m = masks                      # [B,H,W]
            # 非零判定（原分辨率）
            nz = (m.view(m.size(0), -1).sum(dim=1) > 0).numpy()
            n_fake         += int((y == 1).sum())
            n_fake_with_mask += int(((y == 1) & nz).sum())
            if b + 1 >= max_batches: break
        return n_fake, n_fake_with_mask
    
    nf, nfm = sanity_count_nonzero_masks(dl_val, max_batches=50)
    print(f"[SANITY] val fake total={nf}, fake-with-nonzero-mask={nfm}, ratio={0 if nf==0 else nfm/nf:.3f}")
    
    # === 构建模型 ===
    # (1) 加载 Qwen + LoRA
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation="flash_attention_2"
    )
    qwen = PeftModel.from_pretrained(base, "/root/autodl-tmp/outputs_lora_stage2/best_lora_by_IoU")
    qwen.set_adapter("default")
    for p in qwen.parameters(): 
        p.requires_grad = False
    qwen.eval()
    
    # (2) 初始化 visual_tap —— 注意现在 qwen 已经定义！
    visual_tap = QwenVisualTap(qwen.visual, layers=(15,23,31)).to(device)
    
    # (3) 构建 forensic heads
    heads = ForensicJoint(fuse_in_ch=1280, fuse_out_ch=512, layers=(15,23,31)).to(device)
    init_joint_heads_with_priors(heads, p1_prior=0.5, pix_prior=0.03)
    
    # 直接从分类阶段 ckpt 加载 fuser/cls（因 layers 相同=3路，形状能对上）
    cls_ckpt = "/root/script/outputs_clsA/best_by_AUROC.pt"
    sd_all = torch.load(cls_ckpt, map_location="cpu"); sd_all = sd_all.get("state_dict", sd_all)
    sd = heads.state_dict(); loaded = 0
    for k, v in sd_all.items():
        if (k.startswith("fuser.") or k.startswith("cls.")) and k in sd and sd[k].shape == v.shape:
            sd[k] = v; loaded += 1
    heads.load_state_dict(sd, strict=False)
    print(f"[LOAD] Warmed fuser/cls from {cls_ckpt} | loaded={loaded} tensors")
    
    # === 训练（保留你现有的损失与超参）===
    optimizer = torch.optim.AdamW(heads.parameters(), lr=args.base_lr, weight_decay=args.weight_decay)
    steps_per_epoch = math.ceil(len(dl_train) / max(1, args.grad_accum))
    total_steps = args.epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, build_warmup_cosine(total_steps, warmup_ratio=args.warmup_ratio, min_lr_scale=args.min_lr_scale)
    )
    
    history = train_joint_v2(
        heads, visual_tap, dl_train, dl_val, optimizer, scheduler,
        device, total_epochs=args.epochs,
        grad_accum=args.grad_accum, log_interval=50,
        out_dir=args.out_dir,                     # ✅ 保存模型与CSV到这个目录
        csv_name="train_history_joint.csv",       # ✅ CSV 文件名
        early_stop_patience=3                     # ✅ 连续3个epoch无提升则早停
    )

if __name__ == "__main__":
    main()