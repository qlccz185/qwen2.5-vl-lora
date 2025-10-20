import os, json, math, argparse, random, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, average_precision_score
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

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
    """
    - 假图(label=1): mask_path 为真实掩码路径
    - 真图(label=0): mask_path 是 "HxW" 
    """
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

        masks.append(arr)   # 挪到这里

    texts  = [processor.apply_chat_template([m], tokenize=False, add_generation_prompt=True) for m in messages]
    images = [m["content"][0]["image"] for m in messages]
    inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)
    labels = torch.tensor([rec["label"] for rec in batch], dtype=torch.float32)
    masks  = torch.from_numpy(np.stack(masks, axis=0))  # [B,H,W] ∈ {0,1}
    paths  = [rec["path"] for rec in batch]
    return inputs, labels, masks, paths
    
# Visual Tap（只信 thw）
class QwenVisualTap(nn.Module):
    def __init__(self, visual, layers=(7,15,23,31)):
        super().__init__()
        self.visual = visual
        self.layers = list(layers)
        self.cache = {}
        self.hooks = [self.visual.blocks[i].register_forward_hook(self._hook(i)) for i in self.layers]

    def _hook(self, idx):
        def _fn(module, inp, out):
            self.cache[idx] = out
        return _fn

    def forward(self, pixel_values, thw):
        self.cache.clear()
        _ = self.visual(pixel_values, thw)  # 触发 hooks
    
        # 用 thw 的行数作为样本数
        if isinstance(thw, torch.Tensor):
            N = thw.size(0)
            thw_list = thw.tolist()
        else:
            thw_list = thw
            N = len(thw_list)
    
        Hs = [int(t[1]) for t in thw_list]
        Ws = [int(t[2]) for t in thw_list]
        lengths = [h*w for h, w in zip(Hs, Ws)]
        Hmax, Wmax = max(Hs), max(Ws)
    
        grid_dict = {}
        for i in self.layers:
            feat = self.cache[i]                      # [Total_L, C] 或 [B,L,C]
            if feat.dim() == 3:                       # 统一拍扁成 [Total_L, C]
                feat = feat.reshape(-1, feat.size(-1))
            chunks = torch.split(feat, lengths, dim=0)
    
            xs = []
            for seg, (H, W) in zip(chunks, zip(Hs, Ws)):
                C = seg.size(1)
                seg = seg.transpose(0, 1).reshape(C, H, W)     # [C,H,W]
                if H != Hmax or W != Wmax:
                    seg = F.pad(seg, (0, Wmax - W, 0, Hmax - H))
                xs.append(seg)
            grid = torch.stack(xs, dim=0)                      # [N,C,Hmax,Wmax]
            grid_dict[i] = grid.float()
    
        return grid_dict
        
# Heads：融合 + 分类头 + 证据头
class MiniFuse(nn.Module):
    def __init__(self, in_ch=1280, n_layers=4, mid=512, out=512):
        super().__init__()
        self.proj = nn.ModuleList([nn.Conv2d(in_ch, mid, 1) for _ in range(n_layers)])
        self.dw   = nn.Conv2d(mid*n_layers, mid*n_layers, 3, padding=1, groups=mid*n_layers)
        self.pw   = nn.Conv2d(mid*n_layers, out, 1)
    def forward(self, grids):
        zs = [p(g) for p,g in zip(self.proj, grids)]
        z  = torch.cat(zs, dim=1)
        z  = self.dw(z)
        return self.pw(z)

class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__(); self.p = nn.Parameter(torch.tensor(p)); self.eps = eps
    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)
        x = F.adaptive_avg_pool2d(x, 1)
        return x.pow(1.0/self.p).flatten(1)

class ClsHead(nn.Module):
    def __init__(self, in_ch, hidden=256):
        super().__init__()
        self.pool = GeM()
        self.fc1  = nn.Linear(in_ch, hidden)
        self.fc2  = nn.Linear(hidden, 1)
    def forward(self, fmap):
        g = self.pool(fmap)
        g = F.relu(self.fc1(g))
        return self.fc2(g)[:,0]  # logits

class EvidenceHead(nn.Module):
    def __init__(self, in_ch=512):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 256, 3, padding=1)
        self.conv2 = nn.Conv2d(256, 1, 1)
    def forward(self, fmap):
        x = F.gelu(self.conv1(fmap))
        return self.conv2(x).squeeze(1)  # (B,H,W) logits

class ForensicJoint(nn.Module):
    def __init__(self, fuse_in_ch=1280, fuse_out_ch=512, layers=(7,15,23,31)):
        super().__init__()
        self.layers = layers
        self.fuser  = MiniFuse(in_ch=fuse_in_ch, n_layers=len(layers), mid=512, out=fuse_out_ch)
        self.cls    = ClsHead(fuse_out_ch)
        self.evi    = EvidenceHead(fuse_out_ch)
    def forward(self, grid_dict):
        grids = [grid_dict[i] for i in self.layers]
        fused = self.fuser(grids)
        logits = self.cls(fused)
        heatmap_logits = self.evi(fused)
        return logits, heatmap_logits

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
def train_one_epoch_joint(model, visual_tap, loader, optimizer, device,
                          grad_accum=1, scheduler=None, log_interval=50,
                          evi_alpha=0.5, λ_e=2.0, use_focal=True, λ_sparse=1e-3, λ_contrast=1e-2):
    model.train()
    bce_cls = nn.BCEWithLogitsLoss()
    optimizer.zero_grad(set_to_none=True)
    total, n = 0.0, 0

    for step, (inputs, labels, masks, _) in enumerate(loader, 1):
        inputs = {k: v.to(device) for k,v in inputs.items()}
        labels = labels.to(device)                 # (B,)
        masks  = masks.to(device)                  # (B,448,448)

        with torch.no_grad():
            grids = visual_tap(inputs["pixel_values"], inputs["image_grid_thw"])
        logits, hm_logits = model(grids)           # (B,), (B,Ht,Wt)

        # 分类损失（不加温度）
        L_cls = bce_cls(logits, labels)

        # 证据监督提升到 64×64, 可以更高精度其实
        Ht, Wt = hm_logits.shape[-2:]
        if hm_logits.dim() == 3:
            hm_logits = hm_logits.unsqueeze(1)                 # [B,1,Ht,Wt]
        
        # 先得到 32×32 概率
        prob32 = torch.sigmoid(hm_logits)                      # [B,1,Ht,Wt]
        
        # 上采样到 64×64 的概率/对齐的 GT
        prob64 = F.interpolate(prob32, size=(Ht*2, Wt*2), mode="bilinear", align_corners=False)  # [B,1,2Ht,2Wt]
        mask64 = F.interpolate(masks.unsqueeze(1).float(), size=(Ht*2, Wt*2),
                               mode="bilinear", align_corners=False).clamp(0,1)                  # [B,1,2Ht,2Wt]
        
        # 用概率反算回 logit 以便用 BCE
        hm64 = torch.logit(prob64.clamp(1e-6, 1-1e-6))        # [B,1,2Ht,2Wt]
        
        # 像素 BCE：带正例重权（前景稀疏）
        with torch.no_grad():
            pos_pix = mask64.sum()
            tot_pix = mask64.numel()
            neg_pix = tot_pix - pos_pix
            pos_w   = (neg_pix / (pos_pix + 1e-6)).clamp(1.0, 100.0)   # 限幅防爆
        bce_pix = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        L_bce   = bce_pix(hm64, mask64)
        
        # Dice：只在正样本上算（避免全零 GT 压黑）
        pos_mask = (labels.view(-1,1,1,1) > 0.5).float()      # 1=假图
        pp = prob64 * pos_mask
        gg = mask64 * pos_mask
        inter  = (pp * gg).sum(dim=(1,2,3))
        denom  = pp.sum(dim=(1,2,3)) + gg.sum(dim=(1,2,3)) + 1e-6
        dice_b = 1. - (2*inter + 1e-6) / denom
        valid  = (pos_mask.view(pos_mask.size(0), -1).sum(dim=1) > 0).float()
        L_dice = (dice_b * valid).sum() / (valid.sum() + 1e-6)
        
        # 合成证据损失（比例由 evi_alpha 控制，现在用2，不行继续调高，分类不太动了）
        L_evi = evi_alpha * L_bce + (1. - evi_alpha) * L_dice
        
        # 稀疏 & 对比
        L_sparse = prob64.mean()
        real_idx = (labels < 0.5)
        fake_idx = (labels > 0.5)
        if real_idx.any() and fake_idx.any():
            L_contrast = prob64[real_idx].mean() - prob64[fake_idx].mean()
        else:
            L_contrast = torch.zeros((), device=device)
        
        loss = L_cls + λ_e*L_evi + λ_sparse*L_sparse + λ_contrast*L_contrast
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

@torch.no_grad()
def evaluate(model, visual_tap, data_loader, device):
    model.eval()
    ys, ps = [], []
    for (inputs, labels, masks, _) in data_loader:
        inputs = {k: v.to(device) for k,v in inputs.items()}
        labels = labels.to(device)
        grids = visual_tap(inputs["pixel_values"], inputs["image_grid_thw"])
        grids = {k: v.float() for k,v in grids.items()}

        logit_cls, _ = model(grids)
        prob = torch.sigmoid(logit_cls)
        ys.append(labels.detach().cpu().numpy())
        ps.append(prob.detach().cpu().numpy())

    y_true = np.concatenate(ys).astype(np.int64)
    y_prob = np.concatenate(ps)

    from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

    thr = 0.5
    y_pred = (y_prob >= thr).astype(int)
    auroc = roc_auc_score(y_true, y_prob)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    return {"auroc": float(auroc), "acc": float(acc), "f1": float(f1)}

@torch.no_grad()
def evaluate_evidence_iou(model, visual_tap, data_loader, device, thr=0.35, only_fake=True):
    model.eval()
    inter_sum = union_sum = 0.0
    dice_num = dice_den = 0.0
    any_valid = False

    for (inputs, labels, masks, _) in data_loader:
        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = labels.to(device)                           # [B]
        grids  = visual_tap(inputs["pixel_values"], inputs["image_grid_thw"])
        grids  = {k: v.float() for k,v in grids.items()}

        logit_cls, hm_logits = model(grids)                 # hm_logits: [B,1,Ht,Wt] or [B,Ht,Wt]
        if hm_logits.dim() == 3:
            hm_logits = hm_logits.unsqueeze(1)              # [B,1,Ht,Wt]
        Ht, Wt = hm_logits.shape[-2:]

        # 32->64 概率
        prob32 = torch.sigmoid(hm_logits)                   # [B,1,Ht,Wt]
        prob64 = F.interpolate(prob32, size=(Ht*2, Wt*2), mode="bilinear", align_corners=False)
        
        if isinstance(masks, list):
            masks = torch.stack(masks, dim=0)               # [B,H,W]
        masks = masks.to(device)                            # <<< 这行是修复点
        gt64 = F.interpolate(masks.unsqueeze(1).float(), size=(Ht*2, Wt*2),
                             mode="bilinear", align_corners=False).clamp(0,1)

        if only_fake:
            idx = (labels > 0.5).view(-1)                  # bool 索引
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

    miou  = inter_sum / union_sum
    mdice = dice_num / dice_den
    return {"mean_iou": float(miou), "mean_dice": float(mdice)}

# Main
def main():
    ap = argparse.ArgumentParser()
    # 数据
    ap.add_argument("--data_root", default="/root/data")
    ap.add_argument("--train_ann", default="/root/data/trainval/train_idx.json")
    ap.add_argument("--val_ann",   default="/root/data/trainval/val_idx.json")
    # 模型与路径
    ap.add_argument("--model_path", default="/root/models/Qwen2.5-VL-7B-Instruct/")
    ap.add_argument("--clsA_dir",   default="outputs_clsA", help="分类阶段目录（含 best_by_AUROC.pt 与 calibration.json）")
    ap.add_argument("--out_dir",    default="outputs_joint")
    # 训练
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--base_lr", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=3e-4)
    ap.add_argument("--warmup_ratio", type=float, default=0.06)
    ap.add_argument("--min_lr_scale", type=float, default=0.07)
    # 证据损失权重
    ap.add_argument("--evi_weight", type=float, default=1.0)
    ap.add_argument("--evi_alpha",  type=float, default=0.5, help="BCE:Dice mixing for evidence loss")
    ap.add_argument("--no_focal",   action="store_true")
    ap.add_argument("--sparse_w",   type=float, default=5e-3)
    ap.add_argument("--contrast_w", type=float, default=1e-2)
    # 评估/可视化
    ap.add_argument("--eval_vis",   type=int, default=16, help="每次验证保存多少张热力图")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # data
    ds_train = ForgeryJointDataset(args.train_ann, data_root=args.data_root)
    ds_val   = ForgeryJointDataset(args.val_ann,   data_root=args.data_root)
    print(f"Train: {len(ds_train)} | Val: {len(ds_val)}")
    assert len(ds_train) > 0 and len(ds_val) > 0, "空数据集，请检查 --train_ann/--val_ann 与 --data_root。"
    
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
            # “非零”判定（原分辨率）
            nz = (m.view(m.size(0), -1).sum(dim=1) > 0).numpy()
            n_fake         += int((y == 1).sum())
            n_fake_with_mask += int(((y == 1) & nz).sum())
            if b + 1 >= max_batches: break
        return n_fake, n_fake_with_mask
    
    nf, nfm = sanity_count_nonzero_masks(dl_val, max_batches=50)
    print(f"[SANITY] val fake total={nf}, fake-with-nonzero-mask={nfm}, ratio={0 if nf==0 else nfm/nf:.3f}")
    
    # 模型冻结 Qwen，仅挂头
    qwen = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="flash_attention_2"
    )
    for p in qwen.parameters(): p.requires_grad = False
    qwen.eval()
    visual_tap = QwenVisualTap(qwen.visual, layers=(7,15,23,31)).to(device)
    heads = ForensicJoint(fuse_in_ch=1280, fuse_out_ch=512, layers=(7,15,23,31)).to(device)

    # 从分类阶段加载分类子头权重（不覆盖证据头）
    cls_ckpt = Path(args.clsA_dir) / "best_by_AUROC.pt"
    if cls_ckpt.exists():
        ckpt = torch.load(str(cls_ckpt), map_location="cpu")
        sd = ckpt.get("state_dict", ckpt)
        # 只挑选分类子头与融合颈的权重
        new_sd = heads.state_dict()
        for k,v in sd.items():
            if k.startswith("fuser.") or k.startswith("cls."):
                if k in new_sd and new_sd[k].shape == v.shape:
                    new_sd[k] = v
        heads.load_state_dict(new_sd, strict=False)
        print(f"[LOAD] Warmed from {cls_ckpt} (fuser/cls).")
    else:
        print(f"[WARN] {cls_ckpt} not found. Training both heads from scratch.")

    # 评估温度：若 A 阶段有 calibration.json，则用它的 T* 供评估（训练仍用原始 logits）
    T_eval = 1.0
    calib_json = Path(args.clsA_dir) / "calibration.json"
    if calib_json.exists():
        try:
            c = json.loads(calib_json.read_text())
            T_eval = float(c.get("temperature", 1.0))
            print(f"[CALIB] Use temperature T*={T_eval:.3f} for evaluation.")
        except Exception as e:
            print(f"[WARN] Failed to read calibration.json: {e}")

    # 优化器
    optimizer = torch.optim.AdamW(heads.parameters(), lr=args.base_lr, weight_decay=args.weight_decay)
    steps_per_epoch = math.ceil(len(dl_train) / max(1, args.grad_accum))
    total_steps = args.epochs * steps_per_epoch
    lr_lambda = build_warmup_cosine(total_steps, warmup_ratio=args.warmup_ratio, min_lr_scale=args.min_lr_scale)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # 训练循环
    best_auc, best_iou = -1.0, -1.0
    best_auc_ep, best_iou_ep = -1, -1
    
    for epoch in range(1, args.epochs+1):
        print(f"\nEpoch {epoch}/{args.epochs} | lr={optimizer.param_groups[0]['lr']:.2e}")
    
        # 根据 epoch 退火证据权重 + 延后开启稀疏
        ep = epoch
        λe_now   = args.evi_weight if ep <= 3 else max(1.0, args.evi_weight * (1 - (ep - 3)/7))
        λ_sparse = 5e-4 if ep >= 10 else 0.0   # 太亮时再开；你也可用 1e-4 起步
    
        tr_loss = train_one_epoch_joint(
            heads, visual_tap, dl_train, optimizer, device,
            grad_accum=args.grad_accum, scheduler=scheduler,
            evi_alpha=args.evi_alpha,
            λ_e=λe_now,                 # 用退火后的证据权重
            use_focal=(not args.no_focal),
            λ_sparse=λ_sparse,          # 用按 epoch 打开的稀疏项
            λ_contrast=args.contrast_w
        )

        vis_dir = os.path.join(args.out_dir, f"vis_epoch{epoch:03d}") if args.eval_vis>0 else None
        metrics_cls = evaluate(heads, visual_tap, dl_val, device)
        metrics_evi = evaluate_evidence_iou(heads, visual_tap, dl_val, device, thr=0.3, only_fake=True)
        print(
            f"Train loss: {tr_loss:.4f} | "
            f"Val AUROC: {metrics_cls['auroc']:.4f} | "
            f"ACC@0.5: {metrics_cls['acc']:.4f} | "
            f"F1@0.5: {metrics_cls['f1']:.4f} | "
            f"IoU: {metrics_evi['mean_iou']:.4f} | "
            f"Dice: {metrics_evi['mean_dice']:.4f}"
        )

        # 保存最佳（按 AUROC）
        if metrics_cls["auroc"] > best_auc + 1e-6:
            best_auc = metrics_cls["auroc"]; best_auc_ep = epoch
            torch.save({
                "epoch": epoch,
                "state_dict": heads.state_dict(),
                "metric": {
                    "auroc": metrics_cls["auroc"],
                    "acc":   metrics_cls["acc"],
                    "f1":    metrics_cls["f1"],
                    "iou":   metrics_evi["mean_iou"],
                    "dice":  metrics_evi["mean_dice"],
                },
                "args": vars(args)
            }, os.path.join(args.out_dir, "best_by_AUROC_joint.pt"))
            print(f"✔ Saved best_by_AUROC_joint.pt (AUROC={best_auc:.4f})")
        
        # 保存最佳（按 IoU）
        curr_iou = float(metrics_evi["mean_iou"])
        if curr_iou > best_iou + 1e-6:
            best_iou = curr_iou; best_iou_ep = epoch
            torch.save({
                "epoch": epoch,
                "state_dict": heads.state_dict(),
                "metric": {
                    "auroc": metrics_cls["auroc"],
                    "acc":   metrics_cls["acc"],
                    "f1":    metrics_cls["f1"],
                    "iou":   metrics_evi["mean_iou"],
                    "dice":  metrics_evi["mean_dice"],
                },
                "args": vars(args)
            }, os.path.join(args.out_dir, "best_by_IoU_joint.pt"))
            print(f"✔ Saved best_by_IoU_joint.pt (IoU={best_iou:.4f})")

        # 最近 checkpoint
        torch.save({
            "epoch": epoch,
            "state_dict": heads.state_dict(),
            "metric": {
                "auroc": metrics_cls["auroc"],
                "acc":   metrics_cls["acc"],
                "f1":    metrics_cls["f1"],
                "iou":   metrics_evi["mean_iou"],
                "dice":  metrics_evi["mean_dice"],
                },
            "args": vars(args)
        }, os.path.join(args.out_dir, f"last_epoch_{epoch:03d}.pt"))

    print(f"\nDone. Best Val AUROC (joint) = {best_auc:.4f}")
    print(f"[BEST] AUROC={best_auc:.4f} @epoch {best_auc_ep} | IoU={best_iou:.4f} @epoch {best_iou_ep}")
if __name__ == "__main__":
    main()