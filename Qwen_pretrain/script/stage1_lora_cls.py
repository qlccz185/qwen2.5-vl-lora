# stage1_lora_cls.py
import os, re, json, math, argparse, random, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
from PIL import Image
from peft import PeftModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from peft import LoraConfig, get_peft_model, TaskType

# =========================================
# Utils
# =========================================
def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def resize_square_pad(img: Image.Image, size=448, pad_color=(128,128,128)):
    w, h = img.size
    s = size / max(w, h)
    nw, nh = int(round(w * s)), int(round(h * s))
    img = img.resize((nw, nh), Image.BICUBIC)
    canvas = Image.new("RGB", (size, size), pad_color)
    canvas.paste(img, ((size - nw) // 2, (size - nh) // 2))
    return canvas

# =========================================
# Dataset
# =========================================
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

# =========================================
# Collate（固定 448）
# =========================================
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

# =========================================
# Qwen visual tap（修复：PEFT 下调用 base_model）
# =========================================
class QwenVisualTap(nn.Module):
    """
    只 hook 视觉 blocks 若干层输出；返回对齐到同一网格的特征：
    grids: {layer_idx: [B, C, H_max, W_max]}
    """
    def __init__(self, visual, layers=(7, 15, 23, 31)):
        super().__init__()
        self.layers = list(layers)
        self._feat_cache = {}
        self._hooks = []
        self.rebind(visual)  # 初始化时就完成绑定

    def _make_hook(self, idx):
        def _hook(module, inp, out):
            # out: [B, L_pad, C] 或 [Total_L, C]
            self._feat_cache[idx] = out
        return _hook

    def _pad_to(self, seg, H, W, H_max, W_max):
        # seg: [C,H,W]  —— 注意：这里不要 no_grad，Stage-2 需要回传梯度
        if H != H_max or W != W_max:
            seg = F.pad(seg, (0, W_max - W, 0, H_max - H), value=0.0)
        return seg

    def _clear_hooks(self):
        if self._hooks:
            for h in self._hooks:
                try:
                    h.remove()
                except Exception:
                    pass
        self._hooks = []

    def rebind(self, visual):
        """
        当 visual 被 PEFT 包装/替换后，调用一次以重挂 hooks。
        """
        self.visual = visual
        self.core = getattr(visual, "base_model", visual)  # 真正执行 forward 的对象
        self._clear_hooks()
        for i in self.layers:
            self._hooks.append(self.core.blocks[i].register_forward_hook(self._make_hook(i)))

    def forward(self, pixel_values, thw):
        self._feat_cache.clear()

        # 与 core 参数 dtype 对齐，避免硬编码 bfloat16
        dtype = next(self.core.parameters()).dtype
        pv = pixel_values.to(dtype=dtype)
        _ = self.core(pv, thw)  # 必须走 core，确保触发 hooks

        # 形状还原
        thw_list = thw.tolist() if isinstance(thw, torch.Tensor) else thw
        Hs = [int(v[1]) for v in thw_list]
        Ws = [int(v[2]) for v in thw_list]
        B = pv.shape[0]
        H_max, W_max = max(Hs), max(Ws)

        # 自检
        missing = [i for i in self.layers if i not in self._feat_cache]
        if missing:
            raise RuntimeError(f"[VIS TAP] hooks not triggered for layers: {missing}. "
                               f"Check that forward uses the same core you hooked.")

        grids = {}
        for i in self.layers:
            feat = self._feat_cache[i]  # [B,L_pad,C] 或 [Total_L,C]
            if feat.dim() == 3:  # [B, L_pad, C]
                per_samples = []
                for b in range(B):
                    H, W = int(Hs[b]), int(Ws[b])
                    L = H * W
                    seg = feat[b, 1:1 + L, :].transpose(0, 1).contiguous().reshape(-1, H, W)  # 跳过 CLS
                    seg = self._pad_to(seg, H, W, H_max, W_max)
                    per_samples.append(seg)
                layer_grid = torch.stack(per_samples, dim=0)
            else:  # [Total_L, C]
                lengths = [h * w for h, w in zip(Hs, Ws)]
                chunks = torch.split(feat, lengths, dim=0)
                per_samples = []
                for seg, H, W in zip(chunks, Hs, Ws):
                    seg = seg.transpose(0, 1).reshape(-1, H, W)
                    seg = self._pad_to(seg, H, W, H_max, W_max)
                    per_samples.append(seg)
                layer_grid = torch.stack(per_samples, dim=0)

            grids[i] = layer_grid.float()  # [B, C, H_max, W_max]
        return grids

    def __del__(self):
        self._clear_hooks()

# =========================================
# Heads（ForensicCls：多层融合 + 分类）
# =========================================
class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(p, dtype=torch.float32))
        self.eps = eps
    def forward(self, x):
        # x: [B,C,H,W]
        x = x.clamp(min=self.eps).pow(self.p)
        x = F.adaptive_avg_pool2d(x, 1)
        return x.pow(1.0 / self.p).flatten(1)

class MiniFuse(nn.Module):
    """
    对 (7,15,23,31) 四层 grid 做 1x1 通道对齐 → 深度卷积混合 → 1x1 压到 out
    输入每层通道数都是 1280（Qwen2.5-VL 视觉塔宽度）
    """
    def __init__(self, in_ch=1280, layers=4, mid=512, out=512):
        super().__init__()
        self.proj = nn.ModuleList([nn.Conv2d(in_ch, mid, 1) for _ in range(layers)])
        self.dw   = nn.Conv2d(mid * layers, mid * layers, 3, padding=1, groups=mid * layers)
        self.pw   = nn.Conv2d(mid * layers, out, 1)
    def forward(self, grids):
        # grids: List[[B,C,H,W]]，长度=layers
        zs = [p(g) for p, g in zip(self.proj, grids)]
        z  = torch.cat(zs, dim=1)
        z  = self.dw(z)
        return self.pw(z)  # [B,out,H,W]

class ClsHead(nn.Module):
    def __init__(self, in_ch, hidden=256, use_l2norm=True):
        super().__init__()
        self.pool   = GeM()
        self.use_l2 = use_l2norm
        self.fc1    = nn.Linear(in_ch, hidden)
        self.fc2    = nn.Linear(hidden, 1)
    def forward(self, feat_map):  # feat_map: [B,C,H,W]
        x = self.pool(feat_map)   # [B,C]
        if self.use_l2:
            x = F.normalize(x, p=2, dim=1)  # 轻量稳态化，抑制初期 logit 抖动
        x = F.relu(self.fc1(x))
        return self.fc2(x)[:, 0]  # [B]

class ForensicCls(nn.Module):
    """
    从多层视觉特征融合 → 分类分数
    """
    def __init__(self, fuse_in_ch=1280, fuse_out_ch=512, layers=(7,15,23,31), head_hidden=256):
        super().__init__()
        self.layers = tuple(layers)
        self.fuser  = MiniFuse(in_ch=fuse_in_ch, layers=len(self.layers), mid=512, out=fuse_out_ch)
        self.cls    = ClsHead(fuse_out_ch, hidden=head_hidden, use_l2norm=True)
    def forward(self, grid_dict: dict):
        # grid_dict[k]: [B,C,H_max,W_max]
        grids = [grid_dict[i] for i in self.layers]
        fused = self.fuser(grids)        # [B, fuse_out_ch, H, W]
        return self.cls(fused)           # [B]


# =========================================
# Evaluate
# =========================================
@torch.no_grad()
def evaluate(qwen, head, visual_tap, data_loader, device):
    qwen.eval(); head.eval()
    ys, ps = [], []
    for inputs, labels, _ in data_loader:
        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = labels.to(device)
        grids = visual_tap(inputs["pixel_values"], inputs["image_grid_thw"])
        prob  = torch.sigmoid(head(grids))
        ys.append(labels.detach().cpu().numpy())
        ps.append(prob.detach().cpu().numpy())

    y_true = np.concatenate(ys).astype(np.int64)
    y_prob = np.concatenate(ps)

    try:
        auroc = roc_auc_score(y_true, y_prob)
    except Exception:
        auroc = float("nan")

    y_pred05 = (y_prob >= 0.5).astype(np.int32)
    acc05 = accuracy_score(y_true, y_pred05)

    uniq = np.unique(y_prob[~np.isnan(y_prob)])
    if uniq.size == 0:
        thr_accopt = float("nan"); acc_star = float("nan")
        thr_f1opt  = float("nan"); f1_opt   = float("nan")
    else:
        thr_cands = []
        thr_cands.append(uniq[0] - 1e-12)
        thr_cands.extend((uniq[:-1] + uniq[1:]) * 0.5)
        thr_cands.append(uniq[-1] + 1e-12)
        thr_cands = np.array(thr_cands, dtype=np.float64)

        best_acc = -1.0; best_thr = 0.5
        best_f1  = -1.0; best_thr_f1 = 0.5
        for t in thr_cands:
            pred = (y_prob >= t).astype(np.int32)
            acc  = accuracy_score(y_true, pred)
            f1   = f1_score(y_true, pred)
            if acc > best_acc:
                best_acc, best_thr = acc, t
            if f1 > best_f1:
                best_f1, best_thr_f1 = f1, t

        thr_accopt, acc_star = float(best_thr), float(best_acc)
        thr_f1opt,  f1_opt   = float(best_thr_f1), float(best_f1)

    return {
        "auroc": auroc,
        "acc_05": acc05,
        "thr_accopt": thr_accopt, "acc_star": acc_star,
        "thr_f1opt": thr_f1opt,   "f1_opt": f1_opt,
    }

# =========================================
# LR schedule
# =========================================
def build_warmup_cosine(total_steps, warmup_ratio=0.06, min_lr_scale=0.1):
    warmup_steps = int(total_steps * warmup_ratio)
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_lr_scale + (1 - min_lr_scale) * 0.5 * (1 + math.cos(math.pi * progress))
    return lr_lambda

# =========================================
# Train one epoch
# =========================================
def train_one_epoch(qwen, head, visual_tap, data_loader, optimizer, loss_fn, device,
                    grad_clip=1.0, grad_accum=8, scheduler=None, log_interval=50,
                    epoch_idx=1, freeze_epochs_head=2, cls_loss_scale_after=1.3):
    qwen.train(); head.train()
    total_loss, n_samples = 0.0, 0
    optimizer.zero_grad(set_to_none=True)

    # 阶段判断：epoch 基准足够稳妥（避免与 step/accum 细节耦合）
    in_stage2 = (epoch_idx > freeze_epochs_head)
    loss_scale = (cls_loss_scale_after if in_stage2 and cls_loss_scale_after > 1.0 else 1.0)

    for step, (inputs, labels, _) in enumerate(data_loader, 1):
        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = labels.to(device)

        grids = visual_tap(inputs["pixel_values"], inputs["image_grid_thw"])
        logits = head(grids)
        # —— 关键：阶段2按系数放大 BCE，增强反传到 LoRA 的梯度强度 ——
        loss = (loss_fn(logits, labels) * loss_scale) / grad_accum

        if step == 1:
            with torch.no_grad():
                p = torch.sigmoid(logits); pos = (labels == 1)
                print(f"[DEBUG] stage2={in_stage2} loss_scale={loss_scale:.2f} | "
                      f"pos={pos.float().mean().item():.3f} | "
                      f"logit y0={logits[~pos].mean().item():.4f} y1={logits[pos].mean().item() if pos.any() else float('nan'):.4f}")

        loss.backward()

        if step == 1:
            g_sum = 0.0; cnt = 0
            for n, p in qwen.visual.named_parameters():
                if p.requires_grad and "lora_" in n and p.grad is not None:
                    g_sum += p.grad.detach().abs().mean().item(); cnt += 1
            print(f"[GRAD] mean |grad| on LoRA = {0 if cnt==0 else g_sum/cnt:.3e}")

        if step % grad_accum == 0:
            if grad_clip is not None:
                params = []
                for g in optimizer.param_groups:
                    params += list(g["params"])
                nn.utils.clip_grad_norm_(params, grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()

        bs = labels.size(0)
        total_loss += loss.item() * bs * grad_accum
        n_samples += bs
        if step % log_interval == 0:
            lrs = [pg["lr"] for pg in optimizer.param_groups]
            print(f"  step {step:5d} | loss {total_loss / max(1, n_samples):.4f} | "
                  f"lr_lora={lrs[0]:.2e} lr_head={lrs[1]:.2e}")
    return total_loss / max(1, n_samples)
    
# =========================================
# Tiny subset (smoke)
# =========================================
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

# =========================================
# LoRA 注入（只打 visual，Attn-only），并仅启用指定 blocks 的 LoRA 参数
# =========================================
def inject_visual_lora_attn_only(visual, layers=(23,31), r=8, alpha=16, dropout=0.05):
    target = ["attn.qkv", "attn.proj"]  # 只打注意力
    cfg = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=r, lora_alpha=alpha, lora_dropout=dropout,
        target_modules=target, bias="none"
    )
    visual = get_peft_model(visual, cfg)

    keep = set(layers)
    pat = re.compile(
        r"blocks\.(\d+)\.(attn)\.(qkv|proj)\.(lora_[AB]|lora_embedding_A|lora_embedding_B)"
    )
    on_cnt, off_cnt = 0, 0
    for n, p in visual.named_parameters():
        if "lora_" in n:
            m = pat.search(n)
            if m and int(m.group(1)) in keep:
                p.requires_grad = True;  on_cnt += p.numel()
            else:
                p.requires_grad = False; off_cnt += p.numel()
    print(f"[LoRA/ATTN] enable: {on_cnt} | disable: {off_cnt} params on blocks={sorted(keep)}")
    return visual

# =========================================
# Main
# =========================================
def main():
    ap = argparse.ArgumentParser()
    # ======= 基础训练参数 =======
    ap.add_argument("--epochs", type=int, default=22)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--base_lr", type=float, default=3e-4, help="head learning rate")
    ap.add_argument("--lora_lr", type=float, default=3e-5, help="LoRA learning rate")
    ap.add_argument("--weight_decay", type=float, default=3e-4)
    ap.add_argument("--warmup_ratio", type=float, default=0.06)
    ap.add_argument("--min_lr_scale", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--out_dir", default="/root/autodl-tmp/outputs_lora_stage1")
    ap.add_argument("--split_dir", default="/root/autodl-tmp/data/trainval")
    ap.add_argument("--data_root", default="/root/autodl-tmp/data")
    ap.add_argument("--model_path", default="/root/autodl-tmp/models/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--smoke", action="store_true", help="run 64-sample overfit test")
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--min_delta", type=float, default=1e-4)
    
    ap.add_argument("--freeze_epochs_head", type=int, default=3, help="头部热身的 epoch 数；热身结束后就把头冻住")
    ap.add_argument("--lora_after_lr", type=float, default=2e-5, help="进入第二阶段后 LoRA 的有效学习率（第一阶段为0）")
    ap.add_argument("--head_after_scale", type=float, default=0.015,
                help="阶段2中Head LR相对于base_lr的缩放")
    ap.add_argument("--cls_loss_scale_after", type=float, default=1.3,
                help="阶段2分类损失放大系数（增强LoRA梯度），1.0表示不放大")

    # ======= LoRA 专属参数 =======
    ap.add_argument("--lora_layers", type=str, default="15,23,31",
                    help="LoRA 层编号，用逗号分隔，例如 '31' 或 '23,31'")
    ap.add_argument("--lora_rank", type=int, default=16, help="LoRA rank r")
    ap.add_argument("--lora_alpha", type=int, default=32, help="LoRA scaling alpha")
    ap.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    # ======= 消融 =======
    ap.add_argument("--do_ablation", action="store_true",
                help="训练结束后立刻做 A/B 消融（RandHead+Frozen LoRA vs RandHead+No-LoRA）")
    ap.add_argument("--ablate_ckpt", type=str, default="best_by_AUROC.pt",
                    help="用哪个组合 ckpt 做消融（相对 out_dir 的路径或绝对路径）")
    ap.add_argument("--ablate_head_epochs", type=int, default=1,
                    help="消融时随机头微调的 epoch 数（0 表示只做 0-shot）")

    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # --- 数据集 ---
    train_ann = str(Path(args.split_dir) / "train_idx.json")
    val_ann   = str(Path(args.split_dir) / "val_idx.json")
    ds_train = ForgeryClsDataset(train_ann, data_root=args.data_root, invert_label=False)
    ds_val   = ForgeryClsDataset(val_ann,   data_root=args.data_root, invert_label=False)
    print(f"Train: {len(ds_train)} | Val: {len(ds_val)}")

    # --- 模型/processor ---
    processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=True)
    qwen = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="flash_attention_2"
    )
    for p in qwen.parameters(): p.requires_grad = False
    qwen.eval()

    # --- 仅给视觉塔注入 LoRA（Attn-only），并只启用指定 blocks ---
    layers = [int(x) for x in args.lora_layers.split(",") if x.strip()]
    qwen.visual = inject_visual_lora_attn_only(
        qwen.visual, layers=layers, r=args.lora_rank,
        alpha=args.lora_alpha, dropout=args.lora_dropout
    )
    qwen.visual.train()   # LoRA 子模块需要 train；语言塔保持 eval()

    # sanity：统计可训练 LoRA 参数
    n_lora = sum(p.numel() for n,p in qwen.visual.named_parameters()
                 if p.requires_grad and "lora_" in n)
    print(f"[LoRA] trainable lora params: {n_lora/1e6:.3f}M on blocks={layers}")
    assert n_lora > 0, "LoRA 未正确注入（检查匹配与层号）"

    # --- taps & head ---
    visual_tap = QwenVisualTap(qwen.visual, layers=(7,15,23,31)).to(device)
    head = ForensicCls(
        fuse_in_ch=1280,    # Qwen 视觉通道
        fuse_out_ch=512,    # 融合后通道
        layers=(7,15,23,31),
        head_hidden=256
    ).to(device)

    # --- 烟雾测试（小样本过拟合检查）---
    if args.smoke:
        ds_train = build_tiny_balanced_subset(ds_train, per_class=32)
        ds_val   = ds_train
        args.epochs = 20
        args.batch_size = min(args.batch_size, 32)
        args.grad_accum = 1

    def collate_fn(batch): return collate_and_process(batch, processor)
    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=True, persistent_workers=True,
                          collate_fn=collate_fn)
    dl_val   = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=True, persistent_workers=True,
                          collate_fn=collate_fn)

    # ---- 优化器 / 调度（两组参数：LoRA + 头）----
    lora_params = [p for n,p in qwen.visual.named_parameters() if p.requires_grad and "lora_" in n]
    head_params = list(head.parameters())
    
    optimizer = torch.optim.AdamW(
        [
            {"params": lora_params, "lr": args.lora_after_lr, "weight_decay": 0.0},                 # LoRA
            {"params": head_params, "lr": args.base_lr,        "weight_decay": args.weight_decay},  # Head
        ]
    )
    
    steps_per_epoch = math.ceil(len(dl_train) / max(1, args.grad_accum))
    total_steps     = args.epochs * steps_per_epoch
    steps_freeze    = args.freeze_epochs_head * steps_per_epoch  # “阶段切换”步
    
    # Head：阶段1用 warmup→恒1.0；阶段2改为一个很小的常数比例（不为0）
    def lr_lambda_head(global_step: int):
        if global_step < steps_freeze:
            warmup = int(steps_freeze * args.warmup_ratio)
            if global_step < warmup:
                return global_step / max(1, warmup)
            return 1.0
        else:
            # 阶段2：给个很小但非零的比例，保证LoRA端有可观梯度
            return args.head_after_scale
    
    # LoRA：阶段1关闭，阶段2开启（恒1.0；如需可再加轻微warmup）
    def lr_lambda_lora(global_step: int):
        if global_step < steps_freeze:
            return 0.0
        else:
            return 1.0
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=[lr_lambda_lora, lr_lambda_head]  # 顺序对应 param_groups
    )
    
    print(f"[SCHED] steps_per_epoch={steps_per_epoch}, total_steps={total_steps}, steps_freeze={steps_freeze}")
    print(f"[SCHED] lora_after_lr={args.lora_after_lr}, head_after_scale={args.head_after_scale}")

    # ---- 输出层 bias 先验（按训练集正例率）----
    pos_cnt = sum(ds_train[i]["label"] == 1 for i in range(len(ds_train)))
    p1 = max(1e-4, min(1 - 1e-4, pos_cnt / max(1, len(ds_train))))
    logit_p1 = math.log(p1 / (1 - p1))
    
    with torch.no_grad():
        head.cls.fc2.bias.fill_(logit_p1)
    print(f"[INIT] set output bias with prior p1={p1:.4f}")
    bce_logits = nn.BCEWithLogitsLoss()

    # ---- 训练循环 ----
    bests = {"auroc": (-1.0, None), "acc05": (-1.0, None), "accstar": (-1.0, None), "f1star": (-1.0, None)}
    history = []
    best_metric = -float("inf")
    best_epoch  = 0
    patience_counter = 0

    def _save_combo_ckpt(dst_path, qwen, head, metrics, epoch, args):
        """
        统一保存三份：
        1) 组合 ckpt（含 head_state + visual_lora）→ dst_path
        2) 仅 head → <stem>_head_only.pt
        3) 仅 lora → <stem>_lora_only.pt （若视觉为 PeftModel 才会导出）
        """
        payload = {
            "epoch": epoch,
            "metric": metrics,
            "args": vars(args),
            "head_state": head.state_dict(),
        }
        if isinstance(qwen.visual, PeftModel):
            payload["visual_lora"] = qwen.visual.state_dict()
    
        torch.save(payload, dst_path)
    
        out_dir = Path(dst_path).parent
        stem = Path(dst_path).stem
        # 分别落盘 head / lora
        torch.save(head.state_dict(), out_dir / f"{stem}_head_only.pt")
        if isinstance(qwen.visual, PeftModel):
            torch.save(qwen.visual.state_dict(), out_dir / f"{stem}_lora_only.pt")


    for epoch in range(1, args.epochs + 1):
        lrs = [pg["lr"] for pg in optimizer.param_groups]
        lr_lora, lr_head = (lrs + [None, None])[:2]
        print(f"\nEpoch {epoch}/{args.epochs} | lr_lora={lr_lora:.2e} | lr_head={lr_head:.2e}")

        train_loss = train_one_epoch(
            qwen, head, visual_tap, dl_train, optimizer, bce_logits, device,
            grad_clip=1.0, grad_accum=args.grad_accum, scheduler=scheduler,
            epoch_idx=epoch, freeze_epochs_head=args.freeze_epochs_head,
            cls_loss_scale_after=args.cls_loss_scale_after
        )

        metrics = evaluate(qwen, head, visual_tap, dl_val, device)
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
            "lr_lora": float(lr_lora),
            "lr_head": float(lr_head),
        })

        # ---- 保存 best（按不同指标）----
        def _save_best(key, score, fname):
            nonlocal bests
            if score > bests[key][0] + 1e-12:
                path = os.path.join(args.out_dir, fname)
                _save_combo_ckpt(path, qwen, head, metrics, epoch, args)
                bests[key] = (score, path)
                print(f"Saved {path} ({key.upper()} {score:.4f})")
        
        _save_best("auroc",   metrics["auroc"],   "best_by_AUROC.pt")
        _save_best("acc05",   metrics["acc_05"],  "best_by_ACC05.pt")
        _save_best("accstar", metrics["acc_star"],"best_by_ACCstar.pt")
        _save_best("f1star",  metrics["f1_opt"],  "best_by_F1star.pt")
        
        # ---- 保存最近 checkpoint ----
        #_last_path = os.path.join(args.out_dir, f"last_epoch_{epoch:03d}.pt")
        #_save_combo_ckpt(_last_path, qwen, head, metrics, epoch, args)

        # ---- Early Stopping（以 AUROC 为准）----
        if auroc > best_metric + args.min_delta:
            best_metric = auroc
            best_epoch  = epoch
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= args.patience:
            print(f"[Early Stop] No improvement in {args.patience} epochs "
                  f"(best AUROC={best_metric:.4f} @ epoch {best_epoch}).")
            break

    # ---- 保存训练日志 ----
    try:
        import pandas as pd
        pd.DataFrame(history).to_csv(os.path.join(args.out_dir, "training_log.csv"), index=False)
        print("Saved log to training_log.csv")
    except Exception as e:
        print(f"[WARN] failed to save training_log.csv: {e}")

if __name__ == "__main__":
    main()
