# -*- coding: utf-8 -*-
import os, json, math, argparse, random
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

# 基础工具
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def resize_square_pad_rgb(img: Image.Image, size=448, pad_color=(128,128,128)):
    w, h = img.size
    s = size / max(w, h)
    nw, nh = int(round(w*s)), int(round(h*s))
    img = img.resize((nw, nh), Image.BICUBIC)
    canvas = Image.new("RGB", (size, size), pad_color)
    canvas.paste(img, ((size-nw)//2, (size-nh)//2))
    return canvas

def resize_square_pad_mask(img: Image.Image | None, size=448):
    if img is None:
        return Image.new("L", (size, size), 0)
    if img.mode != "L":
        img = img.convert("L")
    w, h = img.size
    s = size / max(w, h)
    nw, nh = int(round(w*s)), int(round(h*s))
    img = img.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("L", (size, size), 0)
    canvas.paste(img, ((size-nw)//2, (size-nh)//2))
    return canvas

def _parse_size_str(s: str):
    if isinstance(s, str) and "x" in s.lower():
        try:
            h, w = s.lower().split("x")
            return int(h), int(w)
        except:
            return None
    return None

# Dataset（image + label + mask）
class ForgeryJointValDataset(Dataset):
    def __init__(self, ann_path, data_root="/root/data"):
        self.root = Path(data_root).resolve()
        items = json.loads(Path(ann_path).read_text(encoding="utf-8"))
        self.items = []
        for rec in items:
            img_rel = rec.get("image_path", "")
            y = int(rec.get("label", 0))
            mp = rec.get("mask_path", "")

            # 解析图片
            p = Path(img_rel)
            if not p.is_absolute():
                p = (self.root / img_rel).resolve()
            if not p.exists():
                name = Path(img_rel).name
                cands = [
                    self.root / "trainingsetbig/image" / name,
                    self.root / "trainingset2/image" / name,
                    self.root / name,
                ]
                p = next((q for q in cands if q.exists()), None)
            if p is None or not p.exists():
                continue

            # 掩码（假图：路径；真图：尺寸，全0就行）
            if y == 1:
                mask_p = None
                if not (isinstance(mp, str) and "x" in mp.lower()):
                    q = Path(mp)
                    if not q.is_absolute():
                        q = (self.root / mp).resolve()
                    if not q.exists():
                        name = Path(mp).name
                        cands = [
                            self.root / "trainingsetbig/spatial_localize" / name,
                            self.root / "trainingset2/spatial_localize" / name,
                            self.root / name,
                        ]
                        q = next((t for t in cands if t.exists()), None)
                    mask_p = q
                self.items.append({"path": str(p), "label": 1, "mask_path": (str(mask_p) if mask_p else None)})
            else:
                size = _parse_size_str(mp)
                self.items.append({"path": str(p), "label": 0, "mask_size": size})

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        rec = self.items[i]
        img = Image.open(rec["path"]).convert("RGB")
        y   = rec["label"]
        if y == 1:
            if rec.get("mask_path", None):
                m = Image.open(rec["mask_path"]).convert("L")
            else:
                m = None
        else:
            size = rec.get("mask_size", None)
            if size is None:
                W, H = img.size
                size = (H, W)
            H, W = size
            m = Image.new("L", (W, H), 0)
        return {"image": img, "label": y, "mask": m, "path": rec["path"]}

# Collate（固定 448）
def collate_joint_test(batch, processor, fixed_res=448):
    messages, masks, labels, paths, vis_imgs = [], [], [], [], []
    for rec in batch:
        im448 = resize_square_pad_rgb(rec["image"], fixed_res)
        vis_imgs.append(im448)  # 可视化
        messages.append({"role":"user","content":[{"type":"image","image":im448},{"type":"text","text":"."}]})

        m = rec["mask"]
        m_res = resize_square_pad_mask(m, fixed_res)
        arr = np.array(m_res, dtype=np.float32)
        if arr.max() > 1.0: arr /= 255.0
        if rec["label"] == 1:
            p99 = np.percentile(arr, 99)
            thr = 0.2 * float(p99)
            arr = (arr >= thr).astype(np.float32)
        else:
            arr = np.zeros_like(arr, dtype=np.float32)
        masks.append(arr)

        labels.append(rec["label"])
        paths.append(rec["path"])

    texts  = [processor.apply_chat_template([m], tokenize=False, add_generation_prompt=True) for m in messages]
    images = [m["content"][0]["image"] for m in messages]
    inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)

    labels = torch.tensor(labels, dtype=torch.float32)
    masks  = torch.from_numpy(np.stack(masks, axis=0))  # [B,H,W] in {0,1}
    return inputs, labels, masks, paths, vis_imgs

# Visual Tap（与训练一致）
class QwenVisualTap(nn.Module):
    def __init__(self, visual, layers=(7,15,23,31)):
        super().__init__()
        self.visual = visual
        self.layers = list(layers)
        self.cache = {}
        self.hooks = [self.visual.blocks[i].register_forward_hook(self._hook(i)) for i in self.layers]

    def _hook(self, idx):
        def _fn(module, inp, out): self.cache[idx] = out
        return _fn

    def forward(self, pixel_values, thw):
        self.cache.clear()
        _ = self.visual(pixel_values, thw)
        thw_list = thw.tolist() if isinstance(thw, torch.Tensor) else thw
        Hs = [int(t[1]) for t in thw_list]
        Ws = [int(t[2]) for t in thw_list]
        lengths = [h*w for h,w in zip(Hs,Ws)]
        Hmax, Wmax = max(Hs), max(Ws)

        grid_dict = {}
        for i in self.layers:
            feat = self.cache[i]
            if feat.dim() == 3:
                feat = feat.reshape(-1, feat.size(-1))
            chunks = torch.split(feat, lengths, dim=0)

            xs = []
            for seg, H, W in zip(chunks, Hs, Ws):
                C = seg.size(1)
                seg = seg.transpose(0,1).reshape(C, H, W)
                if H != Hmax or W != Wmax:
                    seg = F.pad(seg, (0, Wmax-W, 0, Hmax-H))
                xs.append(seg)
            grid = torch.stack(xs, dim=0)                      # [B,C,Hmax,Wmax]
            grid_dict[i] = grid.float()
        return grid_dict

# Heads 与联训一致
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
        return self.fc2(g)[:,0]

class EvidenceHead(nn.Module):
    def __init__(self, in_ch=512):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 256, 3, padding=1)
        self.conv2 = nn.Conv2d(256, 1, 1)
    def forward(self, fmap):
        x = F.gelu(self.conv1(fmap))
        return self.conv2(x).squeeze(1)  # [B,H,W] logits

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
        heatmap_logits = self.evi(fused)  # [B,H,W]
        return logits, heatmap_logits

# 测分类
@torch.no_grad()
def eval_classification(model, visual_tap, dl, device, thr_cls=0.5):
    model.eval()
    all_prob, all_y = [], []
    for batch in tqdm(dl, desc="Eval/Cls"):
        # 兼容 4/5 元
        if len(batch) == 5:
            (inputs, labels, masks, paths, vis_imgs) = batch
        else:
            (inputs, labels, masks, paths) = batch
        inputs = {k: v.to(device) for k,v in inputs.items()}
        grids  = visual_tap(inputs["pixel_values"], inputs["image_grid_thw"])
        grids  = {k: v.float() for k,v in grids.items()}
        logits, _ = model(grids)
        prob   = torch.sigmoid(logits).cpu().numpy()
        all_prob.append(prob); all_y.append(labels.numpy())

    y_true = np.concatenate(all_y).astype(np.int64)
    y_prob = np.concatenate(all_prob)
    from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
    auroc = roc_auc_score(y_true, y_prob)
    y_pred = (y_prob >= thr_cls).astype(int)
    return {
        "auroc": float(auroc),
        "acc":   float(accuracy_score(y_true, y_pred)),
        "f1":    float(f1_score(y_true, y_pred)),
    }

# 测证据 
@torch.no_grad()
def eval_evidence(model, visual_tap, dl, device, thr_map=0.5, only_fake=True, skip_empty_gt=True):
    model.eval()
    inter_sum = union_sum = 0.0
    dice_num = dice_den = 0.0
    any_valid = False

    for batch in tqdm(dl, desc="Eval/Evi"):
        if len(batch) == 5:
            (inputs, labels, masks, paths, vis_imgs) = batch
        else:
            (inputs, labels, masks, paths) = batch
        inputs = {k: v.to(device) for k,v in inputs.items()}
        labels = labels.to(device)
        masks  = masks.to(device)  # [B,448,448] (已二值化或接近二值)

        grids  = visual_tap(inputs["pixel_values"], inputs["image_grid_thw"])
        grids  = {k: v.float() for k,v in grids.items()}
        _, hm_logits = model(grids)
        if hm_logits.dim() == 3: hm_logits = hm_logits.unsqueeze(1)      # [B,1,Ht,Wt]

        prob32 = torch.sigmoid(hm_logits)                                 # [B,1,Ht,Wt]
        Ht, Wt = prob32.shape[-2:]
        prob64 = F.interpolate(prob32, size=(Ht*2, Wt*2), mode="bilinear", align_corners=False)
        gt64   = F.interpolate(masks.unsqueeze(1).float(), size=(Ht*2, Wt*2), mode="bilinear", align_corners=False)
        prob64 = prob64.clamp(0,1); gt64 = gt64.clamp(0,1)

        if only_fake:
            idx = (labels > 0.5).view(-1)
            if not idx.any():
                continue
            prob64 = prob64[idx]
            gt64   = gt64[idx]

        if skip_empty_gt:
            # 过滤掉 GT 全零的样本
            keep = (gt64.view(gt64.size(0), -1).sum(dim=1) > 0)
            if not keep.any():
                continue
            prob64 = prob64[keep]
            gt64   = gt64[keep]

        pred = (prob64 >= thr_map).float()
        gt   = (gt64   >= thr_map).float()

        inter = (pred * gt).sum()
        union = (pred + gt - pred * gt).sum() + 1e-6
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
        "mean_iou": float(inter_sum / union_sum),
        "mean_dice": float(dice_num / dice_den),
    }

# 可视化：随机抽 k 张（原图叠热图，红色）
@torch.no_grad()
def visualize_random(model, visual_tap, dl, device, out_dir, k=3):
    os.makedirs(out_dir, exist_ok=True)
    saved = 0
    for (inputs, labels, masks, paths, vis_imgs) in dl:
        inputs = {k: v.to(device) for k, v in inputs.items()}
        grids  = visual_tap(inputs["pixel_values"], inputs["image_grid_thw"])
        grids  = {k: v.float() for k,v in grids.items()}
        _, hm_logits = model(grids)                # [B,Ht,Wt] or [B,1,Ht,Wt]
        if hm_logits.dim() == 3:
            hm_logits = hm_logits.unsqueeze(1)
        prob32 = torch.sigmoid(hm_logits)
        prob   = F.interpolate(prob32, size=(448, 448), mode='bilinear', align_corners=False)[:,0] # [B,448,448]

        labels_np = labels.numpy().astype(int)
        masks_np  = masks.numpy()                  # [B,448,448], {0,1}
        for i in range(len(paths)):
            if saved >= k: return
            img = vis_imgs[i]                      # 448×448 RGB
            pm  = prob[i].cpu().numpy()
            gt  = masks_np[i]

            # 红色叠加
            heat = (pm * 255.0).clip(0,255).astype(np.uint8)
            overlay = np.array(img).astype(np.float32)
            overlay[...,0] = np.clip(overlay[...,0] * 0.65 + heat*0.35, 0, 255)
            overlay = Image.fromarray(overlay.astype(np.uint8))

            gt_vis = (gt*255).clip(0,255).astype(np.uint8)
            gt_img = Image.fromarray(gt_vis, mode="L").convert("RGB")

            canvas = Image.new("RGB", (448*3, 448))
            canvas.paste(img, (0,0))
            canvas.paste(overlay, (448,0))
            canvas.paste(gt_img, (896,0))

            name = f"{saved:02d}_y{labels_np[i]}_{Path(paths[i]).name}"
            canvas.save(Path(out_dir)/name)
            saved += 1
            if saved >= k: return

# Main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="/root/models/Qwen2.5-VL-7B-Instruct/")
    ap.add_argument("--ckpt",       default="/root/script/outputs_joint/best_by_IoU_joint.pt")
    ap.add_argument("--ann",        default="/root/data/trainval/val_idx.json")  # 用有掩码的 val
    ap.add_argument("--data_root",  default="/root/data")
    ap.add_argument("--out_dir",    default="/root/script/test_out")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--seed",       type=int, default=42)

    ap.add_argument("--thr_cls",    type=float, default=0.5)  # 分类阈值
    ap.add_argument("--thr_map",    type=float, default=0.3)  # 热力图二值化阈值
    ap.add_argument("--only_fake",  action="store_true")      # 只在假图上评 IoU/Dice
    ap.add_argument("--vis_n",      type=int, default=3)      # 可视化数量
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # 数据
    processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=True)
    ds_test   = ForgeryJointValDataset(args.ann, data_root=args.data_root)
    dl_test   = DataLoader(
        ds_test, batch_size=args.batch_size, shuffle=False, num_workers=4,
        pin_memory=True, collate_fn=lambda b: collate_joint_test(b, processor, 448)
    )

    # 模型（冻结 Qwen，只跑头）
    qwen = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation="flash_attention_2"
    )
    for p in qwen.parameters(): p.requires_grad = False
    qwen.eval()
    visual_tap = QwenVisualTap(qwen.visual, layers=(7,15,23,31)).to(device)

    heads = ForensicJoint(fuse_in_ch=1280, fuse_out_ch=512, layers=(7,15,23,31)).to(device)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    sd   = ckpt.get("state_dict", ckpt)
    heads.load_state_dict(sd, strict=False)  # 用 strict=False 更稳妥
    heads.eval().to(device)

    # 分类评估（无温度）
    cls_metrics = eval_classification(heads, visual_tap, dl_test, device, thr_cls=args.thr_cls)
    # 证据评估（IoU/Dice）
    evi_metrics = eval_evidence(heads, visual_tap, dl_test, device, thr_map=args.thr_map, only_fake=args.only_fake)

    # 可视化
    visualize_random(heads, visual_tap, dl_test, device, Path(args.out_dir)/"vis", k=args.vis_n)

    # 保存结果
    out_json = {
        "cls": cls_metrics,
        "evidence": evi_metrics,
        "thr_cls": args.thr_cls,
        "thr_map": args.thr_map,
        "only_fake_for_iou": bool(args.only_fake)
    }
    with open(Path(args.out_dir)/"test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2, ensure_ascii=False)

    print(json.dumps(out_json, indent=2, ensure_ascii=False))
    print(f"[DONE] Saved metrics & visualizations to: {args.out_dir}")

if __name__ == "__main__":
    main()