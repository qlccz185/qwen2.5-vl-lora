import os, json, math, argparse, random, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             accuracy_score, balanced_accuracy_score,
                             f1_score, precision_recall_curve)

# -------------------------
# Utils
# -------------------------
def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

# -------------------------
# Dataset
# -------------------------
class ForgeryClsDataset(Dataset):
    # 0=real,1=fake
    def __init__(self, ann_path, data_root="/root/data", invert_label=False):
        self.data_root = Path(data_root)
        self.records = json.loads(Path(ann_path).read_text(encoding="utf-8"))
        if not isinstance(self.records, list) or len(self.records) == 0:
            raise RuntimeError(f"Empty or invalid ann file: {ann_path}")
        self.invert_label = invert_label

        def _resolve(img_rel: str):
            if not img_rel: return None
            name = Path(img_rel).name
            cands = [
                (self.data_root / img_rel).resolve(),
                (self.data_root / "trainingset2"  / "image" / name).resolve(),
                (self.data_root / "trainingsetbig" / "image" / name).resolve(),
            ]
            for p in cands:
                if p.exists(): return p
            return None

        self.items = []
        miss = 0
        for rec in self.records:
            p = _resolve(rec.get("image_path", ""))
            if p is None:
                miss += 1
                continue
            lbl = int(rec.get("label", 0))
            if self.invert_label:
                lbl = 1 - lbl
            self.items.append({"img": p, "label": lbl})
        if miss > 0:
            print(f"[WARN][Dataset] missing images ignored: {miss}")
        if not self.items:
            raise RuntimeError("No valid images after resolving paths.")

    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        rec = self.items[i]
        img = Image.open(rec["img"]).convert("RGB")
        return {"image": img, "label": rec["label"], "path": str(rec["img"])}

# -------------------------
# Qwen visual tap
# -------------------------
class QwenVisualTap(nn.Module):
    def __init__(self, visual, layers=(7,15,23,31)):
        super().__init__()
        self.visual = visual
        self.layers = list(layers)
        self._feat_cache = {}
        self._hooks = [self.visual.blocks[i].register_forward_hook(self._make_hook(i)) for i in self.layers]
    def _make_hook(self, idx):
        def _hook(module, inp, out):
            self._feat_cache[idx] = out
        return _hook
    def forward(self, pixel_values, thw):
        self._feat_cache.clear()
        _ = self.visual(pixel_values, thw)

        thw_list = thw.tolist() if isinstance(thw, torch.Tensor) else thw
        Hs = [int(v[1]) for v in thw_list]
        Ws = [int(v[2]) for v in thw_list]
        B = pixel_values.shape[0]
        H_max, W_max = max(Hs), max(Ws)

        grids = {}
        for i in self.layers:
            feat = self._feat_cache[i]  # [B, L_pad, C] or [Total_L, C]
            if feat.dim() == 3:  # [B, L_pad, C]
                per_samples = []
                for b in range(B):
                    H, W = int(thw[b,1]), int(thw[b,2])
                    L = H*W
                    seg = feat[b, 1:1+L, :]           # skip the CLS token
                    C = seg.size(-1)
                    seg = seg.transpose(0,1).contiguous().reshape(C, H, W)
                    if H != H_max or W != W_max:
                        seg = F.pad(seg, (0, W_max-W, 0, H_max-H), value=0.0)
                    per_samples.append(seg)
                layer_grid = torch.stack(per_samples, dim=0)
            else:  # [Total_L, C]
                lengths = [h*w for h,w in zip(Hs, Ws)]
                chunks = torch.split(feat, lengths, dim=0)
                per_samples = []
                for seg, H, W in zip(chunks, Hs, Ws):
                    C = seg.size(1)
                    seg = seg.transpose(0,1).reshape(C, H, W)
                    if H != H_max or W != W_max:
                        seg = F.pad(seg, (0, W_max-W, 0, H_max-H), value=0.0)
                    per_samples.append(seg)
                layer_grid = torch.stack(per_samples, dim=0)
            grids[i] = layer_grid
        return grids

# -------------------------
# Heads
# -------------------------
class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__(); self.p = nn.Parameter(torch.tensor(p)); self.eps = eps
    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)
        x = F.adaptive_avg_pool2d(x, 1)
        return x.pow(1.0/self.p).flatten(1)

class MiniFuse(nn.Module):
    def __init__(self, in_ch=1280, layers=3, mid=512, out=512):
        super().__init__()
        self.proj = nn.ModuleList([nn.Conv2d(in_ch, mid, 1) for _ in range(layers)])
        self.dw = nn.Conv2d(mid*layers, mid*layers, 3, padding=1, groups=mid*layers)
        self.pw = nn.Conv2d(mid*layers, out, 1)
    def forward(self, grids):
        zs = [p(g) for p,g in zip(self.proj, grids)]
        z = torch.cat(zs, dim=1)
        z = self.dw(z)
        return self.pw(z)

class ClsHead(nn.Module):
    def __init__(self, in_ch, hidden=256):
        super().__init__()
        self.pool = GeM()
        self.fc1 = nn.Linear(in_ch, hidden)
        self.fc2 = nn.Linear(hidden, 1)
    def forward(self, x):
        x = self.pool(x)
        x = F.relu(self.fc1(x))
        return self.fc2(x)[:, 0]

class ForensicCls(nn.Module):
    def __init__(self, fuse_in_ch=1280, fuse_out_ch=512, layers=(7,15,23,31)):
        super().__init__()
        self.layers = layers
        self.fuser = MiniFuse(in_ch=fuse_in_ch, layers=len(layers), mid=512, out=fuse_out_ch)
        self.cls   = ClsHead(fuse_out_ch)
    def forward(self, grid_dict):
        grids = [grid_dict[i] for i in self.layers]
        fused = self.fuser(grids)
        return self.cls(fused)


class LinearProbe(nn.Module):
    def __init__(self, in_ch=1280, layer=31):
        super().__init__()
        self.layer = layer
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_ch, 1)
    def forward(self, grid_dict):
        x = self.pool(grid_dict[self.layer]).flatten(1)
        return self.fc(x)[:, 0]

def resize_square_pad(img: Image.Image, size=448, pad_color=(128,128,128)):
    w, h = img.size
    s = size / max(w, h)
    nw, nh = int(round(w * s)), int(round(h * s))
    img = img.resize((nw, nh), Image.BICUBIC)
    canvas = Image.new("RGB", (size, size), pad_color)
    canvas.paste(img, ((size - nw) // 2, (size - nh) // 2))
    return canvas


def collate_and_process(batch, processor, fixed_res=448):
    messages = []
    for rec in batch:
        img = resize_square_pad(rec["image"], fixed_res)
        messages.append({
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": "."}
            ]
        })
    texts = [processor.apply_chat_template([m], tokenize=False, add_generation_prompt=True)
             for m in messages]
    images = [m["content"][0]["image"] for m in messages]
    inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)
    labels = torch.tensor([rec["label"] for rec in batch], dtype=torch.float32)
    paths = [rec["path"] for rec in batch]
    return inputs, labels, paths

# -------------------------

# -------------------------
@torch.no_grad()
def evaluate(model, visual_tap, data_loader, device):
    model.eval()
    ys, ps = [], []
    for inputs, labels, _ in data_loader:
        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = labels.to(device)
        grids = visual_tap(inputs["pixel_values"], inputs["image_grid_thw"])
        grids = {k: v.float() for k,v in grids.items()}
        prob  = torch.sigmoid(model(grids))
        ys.append(labels.detach().cpu().numpy())
        ps.append(prob.detach().cpu().numpy())

    y_true = np.concatenate(ys).astype(np.int64)
    y_prob = np.concatenate(ps)            # shape [N]


    try:
        auroc = roc_auc_score(y_true, y_prob)
    except Exception:
        auroc = float("nan")

    # --- ACC@0.5
    y_pred05 = (y_prob >= 0.5).astype(np.int32)
    acc05 = accuracy_score(y_true, y_pred05)


    uniq = np.unique(y_prob[~np.isnan(y_prob)])
    if uniq.size == 0:
        thr_accopt = float("nan")
        acc_star   = float("nan")
    else:
     
        thr_cands = []
        thr_cands.append(uniq[0] - 1e-12)
        thr_cands.extend((uniq[:-1] + uniq[1:]) * 0.5)
        thr_cands.append(uniq[-1] + 1e-12)
        thr_cands = np.array(thr_cands, dtype=np.float64)

        best_acc = -1.0
        best_thr = 0.5
        for t in thr_cands:
            acc = accuracy_score(y_true, (y_prob >= t).astype(np.int32))
            if acc > best_acc:
                best_acc, best_thr = acc, t
        thr_accopt = float(best_thr)
        acc_star   = float(best_acc)


    if uniq.size == 0:
        thr_f1opt = float("nan")
        f1_opt    = float("nan")
    else:
        best_f1 = -1.0
        best_thr_f1 = 0.5
        for t in thr_cands:
            f1 = f1_score(y_true, (y_prob >= t).astype(np.int32))
            if f1 > best_f1:
                best_f1, best_thr_f1 = f1, t
        thr_f1opt = float(best_thr_f1)
        f1_opt    = float(best_f1)


    try:
        auc_flip = roc_auc_score(y_true, 1.0 - y_prob)
        if auc_flip > auroc + 1e-4:
            print(f"[WARN] AUC improves when flipping scores: auc={auroc:.4f} vs 1-auc={auc_flip:.4f}")
    except Exception:
        pass

    return {
        "auroc": auroc,
        "acc_05": acc05,
        "thr_accopt": thr_accopt, "acc_star": acc_star,
        "thr_f1opt": thr_f1opt,   "f1_opt": f1_opt,
    }

@torch.no_grad()
def calibrate_temperature(model, visual_tap, data_loader, device):

    model.eval()
    logits_all, y_all = [], []
    for inputs, labels, _ in data_loader:
        inputs = {k: v.to(device) for k, v in inputs.items()}
        grids = visual_tap(inputs["pixel_values"], inputs["image_grid_thw"])
        grids = {k: v.float() for k, v in grids.items()}
        logits = model(grids).float()
        logits_all.append(logits.cpu())
        y_all.append(labels.float().cpu())
    logits = torch.cat(logits_all)        # [N]
    y = torch.cat(y_all)                  # [N]

    T = torch.tensor(1.0, requires_grad=True)
    opt = torch.optim.LBFGS([T], lr=0.1, max_iter=50)

    def closure():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(logits / T, y)
        loss.backward()
        return loss

    opt.step(closure)
    T_star = float(T.detach().clamp(1e-3, 10.0))
    print(f"[CALIB] Temperature* = {T_star:.3f}")
    return T_star


@torch.no_grad()
def evaluate_calibrated(model, visual_tap, data_loader, device, temperature: float = 1.0, fixed_thr: float = None):

    model.eval()
    ys, ps = [], []
    for inputs, labels, _ in data_loader:
        inputs = {k: v.to(device) for k, v in inputs.items()}
        grids = visual_tap(inputs["pixel_values"], inputs["image_grid_thw"])
        grids = {k: v.float() for k, v in grids.items()}
        logits = model(grids).float()
        prob  = torch.sigmoid(logits / temperature)
        ys.append(labels.detach().cpu().numpy())
        ps.append(prob.detach().cpu().numpy())

    y_true = np.concatenate(ys).astype(np.int64)
    y_prob = np.concatenate(ps)  # [N]


    try:
        auroc = roc_auc_score(y_true, y_prob)
    except Exception:
        auroc = float("nan")

    y_pred05 = (y_prob >= 0.5).astype(np.int32)
    acc05 = accuracy_score(y_true, y_pred05)

    uniq = np.unique(y_prob[~np.isnan(y_prob)])
    if uniq.size == 0:
        thr_accopt, acc_star = float("nan"), float("nan")
        thr_f1opt, f1_opt    = float("nan"), float("nan")
    else:
        thr_cands = []
        thr_cands.append(uniq[0] - 1e-12)
        thr_cands.extend((uniq[:-1] + uniq[1:]) * 0.5)
        thr_cands.append(uniq[-1] + 1e-12)
        thr_cands = np.array(thr_cands, dtype=np.float64)


        best_acc, best_thr = -1.0, 0.5

        best_f1, best_thr_f1 = -1.0, 0.5
        for t in thr_cands:
            pred = (y_prob >= t).astype(np.int32)
            acc = accuracy_score(y_true, pred)
            if acc > best_acc:
                best_acc, best_thr = acc, t
            f1 = f1_score(y_true, pred)
            if f1 > best_f1:
                best_f1, best_thr_f1 = f1, t

        thr_accopt, acc_star = float(best_thr), float(best_acc)
        thr_f1opt, f1_opt    = float(best_thr_f1), float(best_f1)

    out = {
        "auroc": auroc,
        "acc_05": acc05,
        "thr_accopt": thr_accopt, "acc_star": acc_star,
        "thr_f1opt": thr_f1opt,   "f1_opt": f1_opt,
        "temperature": float(temperature),
    }


    if fixed_thr is not None and not np.isnan(fixed_thr):
        pred_fixed = (y_prob >= fixed_thr).astype(np.int32)
        out["acc_fixed"] = accuracy_score(y_true, pred_fixed)
        out["f1_fixed"]  = f1_score(y_true, pred_fixed)
        out["thr_fixed"] = float(fixed_thr)

    return out

def build_warmup_cosine(total_steps, warmup_ratio=0.05, min_lr_scale=0.0):
    warmup_steps = int(total_steps * warmup_ratio)
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_lr_scale + (1 - min_lr_scale) * 0.5 * (1 + math.cos(math.pi * progress))
    return lr_lambda

def train_one_epoch(model, visual_tap, data_loader, optimizer, loss_fn, device,
                    grad_clip=1.0, grad_accum=1, scheduler=None, log_interval=50):
    model.train()
    total_loss, n_samples = 0.0, 0
    optimizer.zero_grad(set_to_none=True)

    for step, (inputs, labels, _) in enumerate(data_loader, 1):
        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = labels.to(device)

        with torch.no_grad():
            grids = visual_tap(inputs["pixel_values"], inputs["image_grid_thw"])
        grids = {k: v.float() for k, v in grids.items()}

        logits = model(grids)
        loss = loss_fn(logits, labels) / grad_accum


        if step == 1:
            with torch.no_grad():
                p = torch.sigmoid(logits); pos = (labels == 1)
                print(f"[DEBUG] batch pos={pos.float().mean().item():.3f}, "
                      f"logit(mean) y=0:{logits[~pos].mean().item():.4f} y=1:{logits[pos].mean().item():.4f}, "
                      f"prob(mean) y=0:{p[~pos].mean().item():.4f} y=1:{p[pos].mean().item():.4f}")
                any_key = list(grids.keys())[0]; g = grids[any_key]
                print(f"[DEBUG] grid[{any_key}] mean/std = {g.mean().item():.6f} / {g.std().item():.6f}")

        loss.backward()
        if step % grad_accum == 0:
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()

        bs = labels.size(0)
        total_loss += loss.item() * bs * grad_accum
        n_samples += bs
        if step % log_interval == 0:
            print(f"  step {step:5d} | loss {total_loss / max(1, n_samples):.4f}")
    return total_loss / max(1, n_samples)

def build_tiny_balanced_subset(ds, per_class=32):
    idx0, idx1 = [], []
    for i in range(len(ds)):
        y = ds[i]["label"]
        if y == 0 and len(idx0) < per_class: idx0.append(i)
        if y == 1 and len(idx1) < per_class: idx1.append(i)
        if len(idx0) >= per_class and len(idx1) >= per_class: break
    tiny_idx = idx0 + idx1
    print(f"[SMOKE] tiny subset -> real(0)={len(idx0)}, fake(1)={len(idx1)}, total={len(tiny_idx)}")
    return torch.utils.data.Subset(ds, tiny_idx)

# Main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--base_lr", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=3e-4)
    ap.add_argument("--warmup_ratio", type=float, default=0.06)
    ap.add_argument("--min_lr_scale", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--out_dir", default="outputs_clsA")
    ap.add_argument("--split_dir", default="/root/data/trainval")
    ap.add_argument("--model_path", default="/root/models/Qwen2.5-VL-7B-Instruct/")
    ap.add_argument("--smoke", action="store_true", help="run 64-sample overfit test")
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--min_delta", type=float, default=1e-4)
    ap.add_argument("--calibrate_only", action="store_true", help="Run temperature calibration only (skip training)

    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


    train_ann = str(Path(args.split_dir) / "train_idx.json")
    val_ann   = str(Path(args.split_dir) / "val_idx.json")
    ds_train = ForgeryClsDataset(train_ann, data_root="/root/data", invert_label=False)
    ds_val   = ForgeryClsDataset(val_ann,   data_root="/root/data", invert_label=False)
    print(f"Train: {len(ds_train)} | Val: {len(ds_val)}")

    cnt0 = sum(ds_train[i]["label"] == 0 for i in range(len(ds_train)))
    cnt1 = len(ds_train) - cnt0
    print(f"[CHECK] train labels -> real(0)={cnt0}, fake(1)={cnt1}")


    processor = AutoProcessor.from_pretrained(args.model_path)
    qwen = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="flash_attention_2"
    )
    for p in qwen.parameters(): p.requires_grad = False
    qwen.eval()
    visual_tap = QwenVisualTap(qwen.visual, layers=(7,15,23,31)).to(device)


    if args.smoke:
        ds_train = build_tiny_balanced_subset(ds_train, per_class=32)
        ds_val   = ds_train

        args.epochs = 20
        args.batch_size = min(args.batch_size, 32)
        args.grad_accum = 1

        heads = LinearProbe(in_ch=1280, layer=31).to(device)
    else:
        heads = ForensicCls(fuse_in_ch=1280, fuse_out_ch=512, layers=(7,15,23,31)).to(device)

    def collate_fn(batch): return collate_and_process(batch, processor)
    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=True, persistent_workers=True,
                          collate_fn=collate_fn)
    dl_val   = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=True, persistent_workers=True,
                          collate_fn=collate_fn)


    if args.smoke:
       
        lr = 3e-4
        optimizer = torch.optim.AdamW(heads.parameters(), lr=lr, weight_decay=1e-2)
        scheduler = None
        print(f"[SMOKE] using constant lr={lr:.1e}, wd=1e-2, no scheduler")
    else:
        #B_eff = args.batch_size * args.grad_accum
        #lr = args.base_lr * (B_eff / 128.0)
        lr = args.base_lr
        optimizer = torch.optim.AdamW(heads.parameters(), lr=lr, weight_decay=args.weight_decay)
        steps_per_epoch = math.ceil(len(dl_train) / max(1, args.grad_accum))
        total_steps = args.epochs * steps_per_epoch
        lr_lambda = build_warmup_cosine(total_steps, warmup_ratio=args.warmup_ratio, min_lr_scale=args.min_lr_scale)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


    pos_cnt = sum(ds_train[i]["label"] == 1 for i in range(len(ds_train)))
    p = max(1e-4, min(1 - 1e-4, pos_cnt / max(1, len(ds_train))))
    with torch.no_grad():
        if isinstance(heads, LinearProbe):
            heads.fc.bias.fill_(math.log(p / (1 - p)))
        else:
            heads.cls.fc2.bias.fill_(math.log(p / (1 - p)))
    print(f"[INIT] set output bias with prior p1={p:.4f}")

    bce_logits = nn.BCEWithLogitsLoss()

    bests = {"auroc": (-1.0, None), "acc05": (-1.0, None), "accstar": (-1.0, None), "f1star": (-1.0, None)}
    history = []
    
    best_metric = -float("inf")
    best_epoch  = 0
    patience_counter = 0
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs} | lr={optimizer.param_groups[0]['lr']:.2e}")
        train_loss = train_one_epoch(
            heads, visual_tap, dl_train, optimizer, bce_logits, device,
            grad_clip=1.0, grad_accum=args.grad_accum, scheduler=scheduler
        )
    

        metrics = evaluate(heads, visual_tap, dl_val, device)
        auroc   = metrics["auroc"]
        acc05   = metrics["acc_05"]
        accstar = metrics.get("acc_star", float("nan"))
        f1star  = metrics.get("f1_opt", float("nan"))
    
        print(
            f"Train loss: {train_loss:.4f} | "
            f"Val AUROC: {metrics['auroc']:.4f} | "
            f"ACC@0.5: {metrics['acc_05']:.4f} | "
            f"ACC@thr_accopt: {metrics['acc_star']:.4f} @thr={metrics['thr_accopt']:.3f} | "
            f"F1*: {metrics['f1_opt']:.4f} @thr={metrics['thr_f1opt']:.3f}"
        )
    
        history.append({
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_auroc": float(auroc),
            "val_acc@0.5": float(acc05),
            "val_acc@thr_accopt": float(accstar),
            "val_f1_opt": float(f1star),
            "val_thr_acc_opt": float(metrics.get("thr_accopt", float("nan"))),
            "lr": float(optimizer.param_groups[0]['lr']),
        })
    

        def _save_best(key, score, fname):
            nonlocal bests
            if score > bests[key][0] + 1e-12:
                path = os.path.join(args.out_dir, fname)
                torch.save({"epoch": epoch, "state_dict": heads.state_dict(),
                            "metric": metrics, "args": vars(args)}, path)
                bests[key] = (score, path)
                print(f"Saved {path} ({key.upper()} {score:.4f})")
        _save_best("auroc",  metrics["auroc"],     "best_by_AUROC.pt")
        _save_best("acc05",  metrics["acc_05"],    "best_by_ACC05.pt")
        _save_best("accstar",metrics["acc_star"],  "best_by_ACCstar.pt")
        _save_best("f1star", metrics["f1_opt"],    "best_by_F1star.pt")

        torch.save({"epoch": epoch, "state_dict": heads.state_dict(),
                    "metric": metrics, "args": vars(args)},
                   os.path.join(args.out_dir, f"last_epoch_{epoch:03d}.pt"))
    

        if auroc > best_metric + args.min_delta:
            best_metric = auroc
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= args.patience:
            print(f"[Early Stop] No improvement in {args.patience} epochs "
                  f"(best AUROC={best_metric:.4f} @ epoch {best_epoch}).")
            break
    

    best_auroc_ckpt = os.path.join(args.out_dir, "best_by_AUROC.pt")
    if os.path.exists(best_auroc_ckpt):
        ckpt = torch.load(best_auroc_ckpt, map_location="cpu")
        heads.load_state_dict(ckpt["state_dict"])
        heads.to(device).eval()
        print(f"[LOAD] Loaded best_by_AUROC from {best_auroc_ckpt}")
    
 
        T_star = calibrate_temperature(heads, visual_tap, dl_val, device)
    
 
        metrics_cal = evaluate_calibrated(heads, visual_tap, dl_val, device,
                                          temperature=T_star, fixed_thr=None)
        thr_star = float(metrics_cal["thr_accopt"])
        print(f"[CALIB] After temperature scaling: "
              f"AUROC={metrics_cal['auroc']:.4f}, "
              f"ACC@thr*={metrics_cal['acc_star']:.4f} @thr*={thr_star:.3f}")
    

        calib_path = os.path.join(args.out_dir, "calibration.json")
        with open(calib_path, "w") as f:
            json.dump({"temperature": T_star, "threshold": thr_star}, f, indent=2)
        print(f"[CALIB] Saved calibration to {calib_path}")
    else:
        print(f"[WARN] best_by_AUROC.pt not found in {args.out_dir}; skip post-training calibration.")


    T_star = calibrate_temperature(heads, visual_tap, dl_val, device)
    

    metrics_cal = evaluate_calibrated(heads, visual_tap, dl_val, device,
                                      temperature=T_star, fixed_thr=None)
    thr_star = float(metrics_cal["thr_accopt"])
    print(f"[CALIB] After temperature scaling: AUROC={metrics_cal['auroc']:.4f}, "
          f"ACC@thr*={metrics_cal['acc_star']:.4f} @thr*={thr_star:.3f}")
    

    calib_path = os.path.join(args.out_dir, "calibration.json")
    with open(calib_path, "w") as f:
        json.dump({"temperature": T_star, "threshold": thr_star}, f, indent=2)
    print(f"[CALIB] Saved calibration to {calib_path}")
    

    import pandas as pd
    pd.DataFrame(history).to_csv(os.path.join(args.out_dir, "training_log.csv"), index=False)
    print("Saved log to training_log.csv")

if __name__ == "__main__":
    main()
