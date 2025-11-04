# -*- coding: utf-8 -*-
import os, math, argparse, json, re
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model, PeftModel, TaskType

# 你已有的工具/组件（按你的工程结构复用）
from classanddetect import (
    ForgeryJointDataset, collate_joint,
    ForensicJoint,                # fuser + cls + evi （分类头+证据头）
    evaluate, 
    build_warmup_cosine, set_seed
)

# =========================
# Visual Tap（hooks 绑 core）
# =========================
class QwenVisualTap(nn.Module):
    """
    从视觉塔指定 blocks 取特征；返回：{layer_idx: [B,C,H_max,W_max]}
    重要：hooks 挂在 base_model（core）上，forward 也走 core
    """
    def __init__(self, visual, layers=(7,15,23,31)):
        super().__init__()
        self.layers = list(layers)
        self._feat_cache = {}
        self._hooks = []
        self.rebind(visual)

    def _make_hook(self, idx):
        def _hook(module, inp, out):
            # 有些实现可能 tuple 返回
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
        self.core = getattr(visual, "base_model", visual)  # 真正执行 forward 的对象
        self._clear_hooks()
        for i in self.layers:
            self._hooks.append(self.core.blocks[i].register_forward_hook(self._make_hook(i)))

    def forward(self, pixel_values, thw):
        self._feat_cache.clear()
        dev   = next(self.core.parameters()).device
        dtype = next(self.core.parameters()).dtype

        pv  = pixel_values.to(device=dev, dtype=dtype)
        thw = thw.to(dev) if isinstance(thw, torch.Tensor) else thw

        # 关键：走 core 保证 hooks 触发
        _ = self.core(pv, thw)

        # 形状还原
        thw_list = thw.tolist() if isinstance(thw, torch.Tensor) else thw
        Hs = [int(v[1]) for v in thw_list]
        Ws = [int(v[2]) for v in thw_list]
        N  = len(Hs)
        Hmax, Wmax = max(Hs), max(Ws)
        lengths = [h*w for h,w in zip(Hs, Ws)]

        # 自检
        missing = [i for i in self.layers if i not in self._feat_cache]
        if missing:
            raise RuntimeError(f"[VIS TAP] hooks not triggered for layers: {missing}.")

        out = {}
        for i in self.layers:
            feat = self._feat_cache[i]              # [Total_L,C] 或 [B,L,C]
            if feat.dim() == 3:
                feat = feat.reshape(-1, feat.size(-1))
            chunks = torch.split(feat, lengths, dim=0)  # N * [L_i,C]

            sz = []
            for seg, H, W in zip(chunks, Hs, Ws):
                C = seg.size(1)
                seg = seg.transpose(0,1).reshape(C, H, W)
                if H != Hmax or W != Wmax:
                    seg = F.pad(seg, (0, Wmax - W, 0, Hmax - H), value=0.0)
                sz.append(seg)
            grid = torch.stack(sz, dim=0).float()       # [N,C,Hmax,Wmax]
            out[i] = grid
        return out

class EvidenceHead64(nn.Module):
    """输入: [B,C,H,W]（通常 H=W=32），输出: coarse32 & fine64 logits"""
    def __init__(self, in_ch=512):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, 256, 3, padding=1), nn.GELU(),
            nn.Conv2d(256, 128, 3, padding=1),   nn.GELU(),
        )
        self.up  = nn.Sequential(
            nn.Conv2d(128, 128*4, 3, padding=1),  # 为 PixelShuffle(2) 准备 4x 通道
            nn.PixelShuffle(2),                   # 32->64
            nn.Conv2d(128, 64, 3, padding=1), nn.GELU(),
        )
        self.out32 = nn.Conv2d(128, 1, 1)  # 32×32 logits（可选 aux）
        self.out64 = nn.Conv2d(64,  1, 1)  # 64×64 logits（主 supervision）

    def forward(self, fmap):
        h  = self.enc(fmap)                 # [B,128,32,32]
        c32 = self.out32(h).squeeze(1)      # [B,32,32]
        u  = self.up(h)                     # [B,64,64]
        f64 = self.out64(u).squeeze(1)      # [B,64,64]
        return c32, f64
        
def _logit(p, eps=1e-6):
    p = float(np.clip(p, eps, 1-eps))
    return math.log(p/(1-p))

@torch.no_grad()
def init_joint_heads_with_priors(heads, p1_prior=0.5, pix_prior=0.03):
    # 分类头
    if hasattr(heads, "cls") and hasattr(heads.cls, "fc2") and heads.cls.fc2.bias is not None:
        heads.cls.fc2.bias.fill_(_logit(p1_prior))

    # 证据头
    evi = getattr(heads, "evi", None)
    if evi is not None:
        # 先统一初始化
        for m in evi.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # 再覆盖最终输出层 bias 先验
        if hasattr(evi, "out64") and evi.out64.bias is not None:
            evi.out64.bias.fill_(_logit(pix_prior))
        # 兼容旧实现
        if hasattr(evi, "conv2") and evi.conv2.bias is not None:
            evi.conv2.bias.fill_(_logit(pix_prior))

@torch.no_grad()
def evaluate_evidence_iou(model, visual_tap, data_loader, device, thr=0.4, only_fake=True):
    model.eval()
    inter_sum = union_sum = 0.0
    dice_num = dice_den = 0.0
    any_valid = False

    for (inputs, labels, masks, _) in data_loader:
        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = labels.to(device)
        grids  = visual_tap(inputs["pixel_values"], inputs["image_grid_thw"])
        grids  = {k: v.float() for k,v in grids.items()}

        _, hm = model(grids)                 # hm: (hm32, hm64) or hm64
        if isinstance(hm, (tuple, list)):
            _, hm64 = hm
        else:
            hm64 = hm                         # 兼容旧返回

        prob64 = torch.sigmoid(hm64.unsqueeze(1))  # [B,1,64,64]
        if isinstance(masks, list):
            masks = torch.stack(masks, dim=0)
        masks = masks.to(device)  # <<< 关键修复：把掩码搬到同一设备
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

# =========================
# LoRA 注入（注意力层，仅 15/23/31）
# =========================
def inject_visual_lora_attn_only(visual, layers=(15,23,31), r=16, alpha=16, dropout=0.05):
    target = ["attn.qkv", "attn.proj"]   # 仅注意力，不打 MLP
    cfg = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=r, lora_alpha=alpha, lora_dropout=dropout,
        target_modules=target, bias="none"
    )
    visual = get_peft_model(visual, cfg)

    keep = set(layers)
    pat = re.compile(r"blocks\.(\d+)\.(attn)\.(qkv|proj)\.(lora_[AB]|lora_embedding_A|lora_embedding_B)")
    on_cnt, off_cnt = 0, 0
    for n, p in visual.named_parameters():
        if "lora_" in n:
            m = pat.search(n)
            if m and int(m.group(1)) in keep:
                p.requires_grad = True;  on_cnt += p.numel()
            else:
                p.requires_grad = False; off_cnt += p.numel()
    print(f"[LoRA/ATTN] enable={on_cnt} | disable={off_cnt} params on blocks={sorted(keep)}")
    return visual

def load_visual_lora_state_dict(visual, ckpt_path: str):
    """
    将 best_by_AUROC_lora_only.pt 加载到已注入LoRA的 visual（PeftModel）中。
    该 ckpt 预计是 `qwen.visual.state_dict()`（只包含 lora_* 权重）。
    """
    sd = torch.load(ckpt_path, map_location="cpu")
    # 兼容：如果是打包的组合ckpt，尝试取 "visual_lora"
    if isinstance(sd, dict) and "visual_lora" in sd and isinstance(sd["visual_lora"], dict):
        sd = sd["visual_lora"]

    missing, unexpected = visual.load_state_dict(sd, strict=False)
    # 只要 missing/unexpected 都不是 lora 之外的权重即可接受
    miss_lora = [k for k in missing if "lora_" in k]
    unexp_lora = [k for k in unexpected if "lora_" in k]
    print(f"[LOAD LORA] loaded={len(sd)} tensors | missing_lora={len(miss_lora)} | unexpected_lora={len(unexp_lora)}")
    return len(sd) > 0

# =========================
# 训练一个 epoch（联合）
# =========================
def train_one_epoch_joint(
    heads: nn.Module, visual_tap: nn.Module, loader: DataLoader, optimizer, device,
    epoch_idx: int, total_epochs: int,
    grad_accum=1, scheduler=None, log_interval=50,
    lambda_e=1.0,
    evi_alpha=0.5,
    lambda_sparse=1e-4,
    lambda_contrast=0.0,
    lambda_aux=0.0,
    pos_weight_cap=8.0,
    phase="B",               # "A_EVI" | "A_CLS" | "B" | "C"
    cls_weight=1.0,          # 分段控制分类项权重；证据探针期设为0
):
    heads.train()
    bce_cls = nn.BCEWithLogitsLoss()
    optimizer.zero_grad(set_to_none=True)
    total, seen = 0.0, 0

    for step, (inputs, labels, masks, _) in enumerate(loader, 1):
        # ---- 准备数据 ----
        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = labels.to(device)                     # [B]
        masks  = masks.to(device)                      # [B,H,W]
        bs = labels.size(0)                            # <<< 先定义 bs

        # ---- 取多层特征 ----
        grids = visual_tap(inputs["pixel_values"], inputs["image_grid_thw"])  # dict: layer -> [B,C,H,W]

        # ---- 前向 ----
        logits, hm_logits = heads(grids)               # hm: (hm32, hm64) 或 单个 hm64
        if isinstance(hm_logits, (tuple, list)):
            hm32, hm64 = hm_logits
        else:
            hm32, hm64 = None, hm_logits              # 兼容旧返回
        if hm64.dim() == 3:
            hm64 = hm64                                # [B,64,64]（本实现已 squeeze）
        elif hm64.dim() == 4 and hm64.size(1) == 1:
            hm64 = hm64.squeeze(1)                     # [B,64,64]

        # ---- 分类损失（联合期/分类探针用）----
        L_cls = bce_cls(logits, labels)

        # ---- 证据损失（主分支：64×64）----
        prob64 = torch.sigmoid(hm64.unsqueeze(1))      # [B,1,64,64]
        mask64 = F.interpolate(masks.unsqueeze(1).float(), size=prob64.shape[-2:],
                               mode="bilinear", align_corners=False).clamp(0,1)

        with torch.no_grad():
            pos_pix = mask64.sum()
            tot_pix = mask64.numel()
            neg_pix = tot_pix - pos_pix
            pos_w   = (neg_pix / (pos_pix + 1e-6)).clamp(1.0, pos_weight_cap)
        bce_pix = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        L_bce   = bce_pix(hm64.unsqueeze(1), mask64)

        pos_mask = (labels.view(-1,1,1,1) > 0.5).float()  # 只在假图上算 Dice
        pp = prob64 * pos_mask
        gg = mask64 * pos_mask
        inter  = (pp * gg).sum(dim=(1,2,3))
        denom  = pp.sum(dim=(1,2,3)) + gg.sum(dim=(1,2,3)) + 1e-6
        dice_b = 1. - (2*inter + 1e-6) / denom
        valid  = (pos_mask.view(pos_mask.size(0), -1).sum(dim=1) > 0).float()
        L_dice = (dice_b * valid).sum() / (valid.sum() + 1e-6)

        L_evi_main = evi_alpha * L_bce + (1. - evi_alpha) * L_dice

        # ---- 可选：32×32 辅助监督（默认关）----
        if (lambda_aux > 0.0) and (hm32 is not None):
            if hm32.dim() == 3:
                hm32 = hm32
            elif hm32.dim() == 4 and hm32.size(1) == 1:
                hm32 = hm32.squeeze(1)
            prob32 = torch.sigmoid(hm32.unsqueeze(1))      # [B,1,32,32]
            mask32 = F.interpolate(masks.unsqueeze(1).float(), size=prob32.shape[-2:],
                                   mode="bilinear", align_corners=False).clamp(0,1)
            bce32  = nn.BCEWithLogitsLoss(pos_weight=pos_w)
            L_aux  = bce32(hm32.unsqueeze(1), mask32)
        else:
            L_aux = torch.zeros((), device=device)

        # ---- 稀疏/对比 ----
        L_sparse = prob64.mean() * lambda_sparse
        if lambda_contrast > 0:
            real_idx = (labels < 0.5)
            fake_idx = (labels > 0.5)
            if real_idx.any() and fake_idx.any():
                L_contrast = (prob64.detach()[real_idx].mean() - prob64.detach()[fake_idx].mean()) * lambda_contrast
            else:
                L_contrast = torch.zeros((), device=device)
        else:
            L_contrast = torch.zeros((), device=device)

        # ---- 汇总损失（关键分支）----
        if phase == "A_EVI":
            # 证据探针：不走分类
            loss = (lambda_e * L_evi_main) + L_aux + L_sparse + L_contrast
        elif phase == "A_CLS":
            # 分类探针：不走证据
            loss = L_cls
        else:
            # 联合期
            loss = (cls_weight * L_cls) + (lambda_e * L_evi_main) + L_aux + L_sparse + L_contrast

        (loss / grad_accum).backward()

        if step % grad_accum == 0:
            nn.utils.clip_grad_norm_(heads.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()

        total += loss.item() * bs
        seen  += bs

        if step == 1:
            with torch.no_grad():
                p = torch.sigmoid(logits); pos = (labels > 0.5)
                print(f"[DEBUG] phase={phase} | pos={pos.float().mean():.3f} | "
                      f"logit y0={logits[~pos].mean().item():.4f} y1={logits[pos].mean().item() if pos.any() else float('nan'):.4f} | "
                      f"hm_mean={prob64.mean().item():.4f} | λe={lambda_e:.2f} α={evi_alpha:.2f} λs={lambda_sparse:.1e}")

        if step % log_interval == 0:
            print(f"  step {step:5d} | loss {total/max(1,seen):.4f}")

    return total / max(1, seen)
# =========================
# 主流程（两阶段）
# =========================
def main():
    ap = argparse.ArgumentParser()
    # data
    ap.add_argument("--data_root", default="/root/autodl-tmp/data")
    ap.add_argument("--train_ann", default="/root/autodl-tmp/data/trainval/train_idx.json")
    ap.add_argument("--val_ann",   default="/root/autodl-tmp/data/trainval/val_idx.json")

    # model path & out
    ap.add_argument("--model_path", default="/root/autodl-tmp/models/Qwen2.5-VL-7B-Instruct/")
    ap.add_argument("--out_dir",    default="/root/autodl-tmp/outputs_lora_stage2")
    ap.add_argument("--lora_ckpt", default="/root/autodl-tmp/outputs_lora_stage1/best_by_AUROC_lora_only.pt",
                help="Stage-1 的 LoRA 权重（lora_only.pt）")

    # LoRA
    ap.add_argument("--lora_layers", default="15,23,31")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.05)

    # train
    ap.add_argument("--epochs_head_probe", type=int, default=3, help="Stage-2A: 线性探针（只训头）")
    ap.add_argument("--epochs_joint",      type=int, default=18, help="Stage-2B: 联合训练")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=8)

    # lr 设置：先线性探针（LoRA lr=0, Head lr > 0），再联合（LoRA:Head ≈ 5:1）
    ap.add_argument("--head_lr_probe", type=float, default=3e-4, help="线性探针阶段：头学习率")
    ap.add_argument("--lora_lr_joint", type=float, default=2.5e-5, help="联合阶段：LoRA lr")
    ap.add_argument("--head_lr_joint", type=float, default=5e-6,   help="联合阶段：头 lr（≈LoRA的1/5）")

    ap.add_argument("--weight_decay_head", type=float, default=3e-4)
    ap.add_argument("--warmup_ratio", type=float, default=0.06)
    ap.add_argument("--min_lr_scale", type=float, default=0.06)
    ap.add_argument("--seed", type=int, default=42)

    # 联合期损失权重
    ap.add_argument("--lambda_e", type=float, default=1.0, help="证据损失权重（按你的要求=1）")
    ap.add_argument("--evi_alpha", type=float, default=0.8, help="证据 BCE:Dice 混合权重")
    ap.add_argument("--lambda_sparse", type=float, default=2e-4, help="稀疏项权重（保留）")
    ap.add_argument("--lambda_contrast", type=float, default=0.0, help="对比项权重（先关）")
    ap.add_argument("--pos_weight_cap", type=float, default=18.0, help="像素 BCE 正例权重上限")

    ap.add_argument("--lambda_e_phase1", type=float, default=1.0,
                help="联合期前若干个epoch使用的证据损失权重 λ_e（默认1.0）")
    ap.add_argument("--lambda_e_phase2", type=float, default=1.2,
                help="联合期后续epoch使用的证据损失权重 λ_e（默认1.2）")
    ap.add_argument("--lambda_e_phase1_epochs", type=int, default=3,
                help="联合期phase1持续的epoch数（默认3）")

    args = ap.parse_args()
    history = []  
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # datasets
    ds_tr = ForgeryJointDataset(args.train_ann, data_root=args.data_root)
    ds_va = ForgeryJointDataset(args.val_ann,   data_root=args.data_root)
    processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=True)
    def collate_fn(b): return collate_joint(b, processor, fixed_res=448)
    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True,
                       num_workers=args.num_workers, pin_memory=True, persistent_workers=True,
                       collate_fn=collate_fn)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False,
                       num_workers=args.num_workers, pin_memory=True, persistent_workers=True,
                       collate_fn=collate_fn)

    # model & LoRA
    qwen = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="flash_attention_2"
    )
    for p in qwen.parameters(): p.requires_grad = False
    qwen.eval()

    # 只给视觉塔注入 LoRA（注意力层；不动 MLP）
    layers = [int(x) for x in args.lora_layers.split(",") if x.strip()]
    qwen.visual = inject_visual_lora_attn_only(
        qwen.visual, layers=layers, r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout
    )
    
    # === 在这里加载 Stage-1 的 LoRA 适配器 ===
    ok = load_visual_lora_state_dict(qwen.visual, args.lora_ckpt)
    if not ok:
        raise RuntimeError(f"Failed to load LoRA from {args.lora_ckpt}. 请检查 rank/alpha/target_modules 与 Stage-1 一致。")
    
    # taps & heads（两头随机初始化）
    visual_tap = QwenVisualTap(qwen.visual, layers=(7,15,23,31)).to(device)  # 注意：tap 已绑定到“带LoRA的core”
    heads = ForensicJoint(fuse_in_ch=1280, fuse_out_ch=512, layers=(7,15,23,31)).to(device)

    # 用 64×64 版本的证据头替换
    heads.evi = EvidenceHead64(in_ch=512).to(device)
    
    # 按先验做初始化（π1=正样本比例；πpix=假图像素占比）
    init_joint_heads_with_priors(heads, p1_prior=0.5, pix_prior=0.03)
    # ============ Stage-2A：证据探针（只训 fuser+evi；LoRA 与 cls 冻结） ============
    for n, p in qwen.visual.named_parameters():
        if "lora_" in n:
            p.requires_grad = False
    
    # 冻结分类头，只训练 fuser + evi
    for p in heads.cls.parameters():
        p.requires_grad = False
    for p in heads.fuser.parameters():
        p.requires_grad = True
    for p in heads.evi.parameters():
        p.requires_grad = True
    
    probe_params = []
    probe_params += [p for p in heads.fuser.parameters() if p.requires_grad]
    probe_params += [p for p in heads.evi.parameters()   if p.requires_grad]
    
    opt_probe = torch.optim.AdamW(
        [{"params": probe_params, "lr": args.head_lr_probe, "weight_decay": args.weight_decay_head}]
    )
    
    steps_per_epoch = math.ceil(len(dl_tr) / max(1, args.grad_accum))
    total_steps     = args.epochs_head_probe * steps_per_epoch
    sched_probe = torch.optim.lr_scheduler.LambdaLR(
        opt_probe, build_warmup_cosine(total_steps, warmup_ratio=args.warmup_ratio, min_lr_scale=args.min_lr_scale)
    )
    
    best_auc, best_iou = -1.0, -1.0
    print(f"\n[Stage-2A] Evidence probe for {args.epochs_head_probe} epochs (LoRA & cls frozen)")
    for epoch in range(1, args.epochs_head_probe + 1):
        print(f"Epoch {epoch}/{args.epochs_head_probe} | lr(evi+fuser)={opt_probe.param_groups[0]['lr']:.2e}")
        _ = train_one_epoch_joint(
            heads, visual_tap, dl_tr, opt_probe, device,
            epoch_idx=epoch, total_epochs=args.epochs_head_probe,
            grad_accum=args.grad_accum, scheduler=sched_probe,
            # 证据探针：不算分类
            cls_weight=0.0,
            lambda_e=args.lambda_e,
            evi_alpha=args.evi_alpha,
            lambda_sparse=args.lambda_sparse,
            lambda_contrast=0.0,
            lambda_aux=0.0,
            pos_weight_cap=args.pos_weight_cap,
            phase="A_EVI",
        )
    
        metrics_cls = evaluate(heads, visual_tap, dl_va, device)
        metrics_evi = evaluate_evidence_iou(heads, visual_tap, dl_va, device, thr=0.3, only_fake=True)
        print(f"[Probe/EVI] IoU: {metrics_evi['mean_iou']:.4f} | Dice: {metrics_evi['mean_dice']:.4f} | "
              f"AUROC: {metrics_cls['auroc']:.4f} | ACC@0.5: {metrics_cls['acc']:.4f}")
    
        # 记录 CSV（标记 phase=A_evi）
        rowA = {
            "phase": "A_evi",
            "epoch": int(epoch),
            "lr_lora": 0.0,
            "lr_head": float(opt_probe.param_groups[0]["lr"]),
            "lambda_e": float(args.lambda_e),
            "evi_alpha": float(args.evi_alpha),
            "lambda_sparse": float(args.lambda_sparse),
            "val_auroc": float(metrics_cls.get("auroc", float("nan"))),
            "val_acc@0.5": float(metrics_cls.get("acc", float("nan"))),
            "val_f1@0.5": float(metrics_cls.get("f1", float("nan"))),
            "val_acc@thr_accopt": float(metrics_cls.get("acc_star", float("nan"))),
            "val_thr_accopt": float(metrics_cls.get("thr_accopt", float("nan"))),
            "val_f1_opt": float(metrics_cls.get("f1_opt", float("nan"))),
            "val_thr_f1opt": float(metrics_cls.get("thr_f1opt", float("nan"))),
            "val_mean_iou": float(metrics_evi.get("mean_iou", float("nan"))),
            "val_mean_dice": float(metrics_evi.get("mean_dice", float("nan"))),
        }
        history.append(rowA)
    
        # 保存 best（以 IoU 为主，同时保留 AUROC best）
        if metrics_cls["auroc"] > best_auc + 1e-6:
            best_auc = metrics_cls["auroc"]
            torch.save(heads.state_dict(), os.path.join(args.out_dir, "best_head_by_AUROC_probe_evi.pt"))
        if float(metrics_evi["mean_iou"]) > best_iou + 1e-6:
            best_iou = float(metrics_evi["mean_iou"])
            torch.save(heads.state_dict(), os.path.join(args.out_dir, "best_head_by_IoU_probe_evi.pt"))

    # ============ Stage-2B：联合训练（LoRA:Head ≈ 5:1） ============
    qwen.visual.train()
    for n, p in qwen.visual.named_parameters():
        if "lora_" in n: 
            p.requires_grad = True
    
    fuser_params         = [p for n,p in heads.fuser.named_parameters() if p.requires_grad]
    cls_head_params      = [p for n,p in heads.cls.named_parameters()   if p.requires_grad]
    evidence_head_params = [p for n,p in heads.evi.named_parameters()   if p.requires_grad]
    
    # LoRA 组（确保只拿到 lora_* 且 requires_grad=True 的参数）
    lora_params = [p for n,p in qwen.visual.named_parameters()
                   if p.requires_grad and "lora_" in n]
    #head_params = list(heads.parameters())
    
    optimizer = torch.optim.AdamW([
        {"params": lora_params,            "lr": args.lora_lr_joint, "weight_decay": 0.0},
        {"params": fuser_params,           "lr": args.head_lr_joint * 0.8, "weight_decay": args.weight_decay_head},
        {"params": cls_head_params,        "lr": args.head_lr_joint * 1.0, "weight_decay": args.weight_decay_head},
        {"params": evidence_head_params,   "lr": args.head_lr_joint * 1.2, "weight_decay": args.weight_decay_head},  # ← 1.2~1.5
    ])
    
    steps_per_epoch = math.ceil(len(dl_tr) / max(1, args.grad_accum))
    total_steps     = args.epochs_joint * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        build_warmup_cosine(total_steps, warmup_ratio=args.warmup_ratio, min_lr_scale=args.min_lr_scale)
    )
    
    best_auc, best_iou = -1.0, -1.0
    print(f"\n[Stage-2B] Joint training for {args.epochs_joint} epochs (LoRA:Head ≈ 5:1)")
    
    for epoch in range(1, args.epochs_joint + 1):
        # ---- 先算 lam_e_now，再打印 ----
        lam_e_now = args.lambda_e_phase1 if epoch <= args.lambda_e_phase1_epochs else args.lambda_e_phase2
        lrs = [pg["lr"] for pg in optimizer.param_groups]
        print(f"Epoch {epoch}/{args.epochs_joint} | lr_lora={lrs[0]:.2e} | lr_head={lrs[1]:.2e} | lambda_e={lam_e_now:.2f}")
    
        _ = train_one_epoch_joint(
            heads, visual_tap, dl_tr, optimizer, device,
            epoch_idx=epoch, total_epochs=args.epochs_joint,
            grad_accum=args.grad_accum, scheduler=scheduler,
            lambda_e=lam_e_now,                      # ← 用当前分段值
            evi_alpha=args.evi_alpha,
            lambda_sparse=args.lambda_sparse, lambda_contrast=args.lambda_contrast,
            pos_weight_cap=args.pos_weight_cap, phase="B"
        )
    
        metrics_cls = evaluate(heads, visual_tap, dl_va, device)
        metrics_evi = evaluate_evidence_iou(heads, visual_tap, dl_va, device, thr=0.3, only_fake=True)
        print(f"Joint | AUROC: {metrics_cls['auroc']:.4f} | ACC@0.5: {metrics_cls['acc']:.4f} | "
              f"F1@0.5: {metrics_cls['f1']:.4f} | IoU: {metrics_evi['mean_iou']:.4f} | Dice: {metrics_evi['mean_dice']:.4f}")
    
        # —— CSV 行里也写 lam_e_now —— #
        rowB = {
            "phase": "B",
            "epoch": int(epoch),
            "lr_lora": float(lrs[0]),
            "lr_head": float(lrs[1]),
            "lambda_e": float(lam_e_now),           # ← 这里
            "evi_alpha": float(args.evi_alpha),
            "lambda_sparse": float(args.lambda_sparse),
            "val_auroc": float(metrics_cls.get("auroc", float("nan"))),
            "val_acc@0.5": float(metrics_cls.get("acc", float("nan"))),
            "val_f1@0.5": float(metrics_cls.get("f1", float("nan"))),
            "val_acc@thr_accopt": float(metrics_cls.get("acc_star", float("nan"))),
            "val_thr_accopt": float(metrics_cls.get("thr_accopt", float("nan"))),
            "val_f1_opt": float(metrics_cls.get("f1_opt", float("nan"))),
            "val_thr_f1opt": float(metrics_cls.get("thr_f1opt", float("nan"))),
            "val_mean_iou": float(metrics_evi.get("mean_iou", float("nan"))),
            "val_mean_dice": float(metrics_evi.get("mean_dice", float("nan"))),
        }
        history.append(rowB)
        # ====== 落盘 CSV 日志 ======
        try:
            df = pd.DataFrame(history, columns=[
                "phase","epoch","lr_lora","lr_head","lambda_e","evi_alpha","lambda_sparse",
                "val_auroc","val_acc@0.5","val_f1@0.5",
                "val_acc@thr_accopt","val_thr_accopt","val_f1_opt","val_thr_f1opt",
                "val_mean_iou","val_mean_dice"
            ])
            csv_path = os.path.join(args.out_dir, "training_log_stage2.csv")
            df.to_csv(csv_path, index=False)
            print(f"[LOG] Saved Stage-2 training log to {csv_path}")
        except Exception as e:
            print(f"[WARN] failed to save training_log_stage2.csv: {e}")

        # —— 双 best 策略：分别保存头（state_dict）与 LoRA adapter —— #
        if metrics_cls["auroc"] > best_auc + 1e-6:
            best_auc = metrics_cls["auroc"]
            torch.save({
                "epoch": epoch,
                "metric": {"auroc": best_auc},
                "heads": heads.state_dict()
            }, os.path.join(args.out_dir, "best_by_AUROC_joint_lora.pt"))
            # 单独落盘 heads / LoRA
            torch.save(heads.state_dict(), os.path.join(args.out_dir, "best_heads_by_AUROC.pt"))
            if isinstance(qwen.visual, PeftModel):
                qwen.visual.save_pretrained(os.path.join(args.out_dir, "best_lora_by_AUROC"))
            print("[SAVE] best AUROC heads & LoRA saved.")

        mean_iou = float(metrics_evi["mean_iou"])
        if mean_iou > best_iou + 1e-6:
            best_iou = mean_iou
            torch.save({
                "epoch": epoch,
                "metric": {"iou": best_iou},
                "heads": heads.state_dict()
            }, os.path.join(args.out_dir, "best_by_IoU_joint_lora.pt"))
            torch.save(heads.state_dict(), os.path.join(args.out_dir, "best_heads_by_IoU.pt"))
            if isinstance(qwen.visual, PeftModel):
                qwen.visual.save_pretrained(os.path.join(args.out_dir, "best_lora_by_IoU"))
            print("[SAVE] best IoU heads & LoRA saved.")

    # 统一再落一份“最后 LoRA adapter”
    if isinstance(qwen.visual, PeftModel):
        qwen.visual.save_pretrained(os.path.join(args.out_dir, "lora_visual_adapter_last"))
        print("[SAVE] LoRA adapter (last) saved.")

if __name__ == "__main__":
    main()
