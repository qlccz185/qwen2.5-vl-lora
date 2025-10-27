# -*- coding: utf-8 -*-
"""Task8: Frozen multimodal reasoning pipeline for forensic explanations."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

from peft import PeftModel
from peft.tuners.lora import LoraLayer
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


# ---------------------------------------------------------------------------
# Vision taps and forensic head (copied and simplified from previous tasks)
# ---------------------------------------------------------------------------
class QwenVisualTap(nn.Module):
    """Utility that records intermediate ViT activations at given layers."""

    def __init__(self, visual: nn.Module, layers: Iterable[int] = (7, 15, 23, 31)) -> None:
        super().__init__()
        self.visual = visual
        self.layers = [int(i) for i in layers]
        self.cache: Dict[int, torch.Tensor] = {}
        self._hooks = [self.visual.blocks[i].register_forward_hook(self._make_hook(i)) for i in self.layers]

    def _make_hook(self, idx: int):
        def _hook(module, inp, out):
            self.cache[idx] = out

        return _hook

    def forward(self, pixel_values: torch.Tensor, thw: torch.Tensor) -> Dict[int, torch.Tensor]:
        self.cache.clear()
        _ = self.visual(pixel_values, thw)

        thw_list = thw.tolist() if isinstance(thw, torch.Tensor) else thw
        Hs = [int(t[1]) for t in thw_list]
        Ws = [int(t[2]) for t in thw_list]
        lengths = [h * w for h, w in zip(Hs, Ws)]
        Hmax, Wmax = max(Hs), max(Ws)

        grid_dict: Dict[int, torch.Tensor] = {}
        for layer_idx in self.layers:
            feat = self.cache[layer_idx]
            if feat.dim() == 3:
                feat = feat.reshape(-1, feat.size(-1))
            chunks = torch.split(feat, lengths, dim=0)

            xs: List[torch.Tensor] = []
            for seg, H, W in zip(chunks, Hs, Ws):
                C = seg.size(1)
                seg = seg.transpose(0, 1).reshape(C, H, W)
                if H != Hmax or W != Wmax:
                    seg = F.pad(seg, (0, Wmax - W, 0, Hmax - H))
                xs.append(seg)
            grid = torch.stack(xs, dim=0)
            grid_dict[layer_idx] = grid.float()
        return grid_dict

    def close(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()


class GeM(nn.Module):
    def __init__(self, p: float = 3.0, eps: float = 1e-6) -> None:
        super().__init__()
        self.p = nn.Parameter(torch.tensor(p))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(min=self.eps).pow(self.p)
        x = F.adaptive_avg_pool2d(x, 1)
        return x.pow(1.0 / self.p).flatten(1)


class MiniFuse(nn.Module):
    def __init__(self, in_ch: int = 1280, n_layers: int = 4, mid: int = 512, out: int = 512) -> None:
        super().__init__()
        self.proj = nn.ModuleList([nn.Conv2d(in_ch, mid, kernel_size=1) for _ in range(n_layers)])
        self.dw = nn.Conv2d(mid * n_layers, mid * n_layers, kernel_size=3, padding=1, groups=mid * n_layers)
        self.pw = nn.Conv2d(mid * n_layers, out, kernel_size=1)

    def forward(self, feature_list: List[torch.Tensor]) -> torch.Tensor:
        proj_feats = [proj(feat) for proj, feat in zip(self.proj, feature_list)]
        concat = torch.cat(proj_feats, dim=1)
        fused = self.dw(concat)
        return self.pw(fused)


class ClsHead(nn.Module):
    def __init__(self, in_ch: int, hidden: int = 256) -> None:
        super().__init__()
        self.pool = GeM()
        self.fc1 = nn.Linear(in_ch, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, fmap: torch.Tensor) -> torch.Tensor:
        pooled = self.pool(fmap)
        hidden = F.relu(self.fc1(pooled))
        return self.fc2(hidden)[:, 0]


class EvidenceHead(nn.Module):
    def __init__(self, in_ch: int = 512) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(256, 1, kernel_size=1)

    def forward(self, fmap: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.conv1(fmap))
        return self.conv2(x).squeeze(1)


class ForensicHead(nn.Module):
    def __init__(self, fuse_in_ch: int = 1280, fuse_out_ch: int = 512, layers: Iterable[int] = (7, 15, 23, 31)) -> None:
        super().__init__()
        self.layers = tuple(int(i) for i in layers)
        self.fuser = MiniFuse(in_ch=fuse_in_ch, n_layers=len(self.layers), mid=512, out=fuse_out_ch)
        self.cls = ClsHead(fuse_out_ch)
        self.evidence = EvidenceHead(fuse_out_ch)

    def forward(self, grid_dict: Dict[int, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        features = [grid_dict[i] for i in self.layers]
        fused = self.fuser(features)
        logits = self.cls(fused)
        heatmap_logits = self.evidence(fused)
        return logits, heatmap_logits


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

GLOBAL_RNG = random.Random()

DEFAULT_CHAT_SYSTEM_PROMPT = (
    "You are a professional AI visual forensic analyst. Your task is to analyze an image and its corresponding heatmap, "
    "identify whether the content is authentic or manipulated, and provide a clear, structured, and evidence-based explanation "
    "for your judgment. Use precise and objective language. Avoid repetition or subjective phrases."
)


def set_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    GLOBAL_RNG.seed(seed)


def load_config(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_dtype(name: str | None) -> torch.dtype:
    if not name:
        return torch.bfloat16
    try:
        return getattr(torch, name)
    except AttributeError as exc:
        raise ValueError(f"Unsupported torch dtype: {name}") from exc


def merge_visual_lora(model: nn.Module, cfg: Dict) -> nn.Module:
    visual_cfg = cfg.get("visual_lora", {})
    adapter_path = visual_cfg.get("path")
    if not adapter_path:
        return model

    adapter_dir = Path(adapter_path).expanduser()
    if adapter_dir.is_file():
        adapter_dir = adapter_dir.parent
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Visual LoRA adapter not found: {adapter_dir}")

    peft_model = PeftModel.from_pretrained(model, adapter_dir, is_trainable=False)

    try:
        base_model = peft_model.get_base_model()
    except AttributeError:
        base_model = peft_model

    injected = sorted({name for name, module in base_model.named_modules() if isinstance(module, LoraLayer)})
    if injected:
        print("Merged visual LoRA modules:")
        for name in injected:
            print(" -", name)
    else:
        print("[WARN] No LoRA layers detected before merge.")

    merged = peft_model.merge_and_unload()
    return merged


def load_model_components(cfg: Dict):
    base_model_path = cfg["base_model_path"]
    dtype = get_dtype(cfg.get("torch_dtype"))

    print(f"Loading Qwen2.5-VL model from {base_model_path}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(base_model_path, torch_dtype=dtype)
    model = merge_visual_lora(model, cfg)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    processor = AutoProcessor.from_pretrained(base_model_path)

    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    visual_layers = cfg.get("visual_lora", {}).get("layers") or [7, 15, 23, 31]
    visual_tap = QwenVisualTap(base.visual, layers=visual_layers)

    head = ForensicHead(layers=visual_layers)
    head_ckpt = cfg.get("head_checkpoint")
    if not head_ckpt:
        raise ValueError("Configuration must provide head_checkpoint for the frozen forensic head.")
    head_state = torch.load(Path(head_ckpt).expanduser(), map_location="cpu")
    if isinstance(head_state, dict) and "state_dict" in head_state:
        head_state = head_state["state_dict"]
    missing, unexpected = head.load_state_dict(head_state, strict=False)
    if missing:
        print(f"[WARN] Missing keys when loading head: {missing}")
    if unexpected:
        print(f"[WARN] Unexpected keys when loading head: {unexpected}")
    head.eval()
    for param in head.parameters():
        param.requires_grad = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    head.to(device)

    return model, processor, visual_tap, head, device


def resolve_image_path(path: str | Path, data_root: Optional[str | Path] = None) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    if data_root is None:
        return p.resolve()
    return (Path(data_root).expanduser() / p).resolve()


def load_annotation_samples(cfg: Dict, override_annotation: Optional[Path] = None, override_data_root: Optional[Path] = None) -> List[Dict[str, Any]]:
    data_cfg = cfg.get("data", {})
    ann_path_cfg = data_cfg.get("annotation") or data_cfg.get("ann_path") or data_cfg.get("annotation_path")
    ann_path = override_annotation or (Path(ann_path_cfg).expanduser() if ann_path_cfg else None)
    if ann_path is None:
        raise ValueError("No annotation file provided. Please set data.annotation in the config or pass --annotation.")
    if not ann_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {ann_path}")

    data_root_cfg = data_cfg.get("root") or data_cfg.get("data_root")
    data_root = override_data_root or (Path(data_root_cfg).expanduser() if data_root_cfg else None)

    records = json.loads(ann_path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("Annotation JSON must contain a list of samples.")

    samples: List[Dict[str, Any]] = []
    for idx, record in enumerate(records):
        if not isinstance(record, dict):
            continue
        img_rel = record.get("image_path") or record.get("img_path") or record.get("image")
        if not img_rel:
            continue
        image_path = resolve_image_path(img_rel, data_root)
        if not image_path.exists():
            print(f"[WARN] Missing image: {image_path}")
            continue

        label = record.get("label")
        try:
            label_int = int(label) if label is not None else None
        except (TypeError, ValueError):
            label_int = None

        sample_id = record.get("id") or record.get("image_id") or Path(img_rel).stem or f"sample_{idx:05d}"

        samples.append(
            {
                "id": str(sample_id),
                "image_path": image_path,
                "label": label_int,
                "raw": record,
            }
        )

    if not samples:
        raise ValueError(f"No valid samples were loaded from annotation {ann_path}")

    return samples


def iter_samples(samples: List[Dict[str, Any]], limit: Optional[int] = None) -> Iterator[Tuple[int, Dict[str, Any]]]:
    max_count = len(samples) if limit is None else min(len(samples), limit)
    for index in range(max_count):
        yield index, samples[index]


def run_forensic_head(
    image: Image.Image,
    processor: AutoProcessor,
    visual_tap: QwenVisualTap,
    head: ForensicHead,
    device: torch.device,
) -> Tuple[float, torch.Tensor]:
    inputs = processor(
        images=[image],
        return_tensors="pt",
        do_rescale=True,
        size={"height": 448, "width": 448},
    )
    pixel_values = inputs["pixel_values"].to(device)
    image_grid_thw = inputs.get("image_grid_thw")
    if image_grid_thw is not None:
        image_grid_thw = image_grid_thw.to(device)
    else:
        B, _, H, W = pixel_values.shape
        patch = 14
        image_grid_thw = torch.tensor(
            [[B, H // patch, W // patch]], device=device, dtype=torch.int32
        )

    with torch.no_grad():
        grid_dict = visual_tap(pixel_values, image_grid_thw)
        grid_dict = {k: v.float().to(device) for k, v in grid_dict.items()}
        cls_logits, heatmap_logits = head(grid_dict)
        prob_tensor = torch.sigmoid(cls_logits).detach().to(device="cpu", dtype=torch.float32)
        prob = prob_tensor.item()
        if heatmap_logits.dim() == 3:
            heatmap_logits = heatmap_logits.unsqueeze(1)
        heatmap_prob = torch.sigmoid(heatmap_logits).detach().cpu()

    heatmap_prob = heatmap_prob[0, 0]
    return prob, heatmap_prob


def upscale_heatmap(heatmap: torch.Tensor, target_size: Tuple[int, int]) -> np.ndarray:
    target_h, target_w = int(target_size[0]), int(target_size[1])
    hm = heatmap.unsqueeze(0).unsqueeze(0)
    hm = F.interpolate(hm, size=(target_h, target_w), mode="bilinear", align_corners=False)
    hm = hm[0, 0]
    hm = hm.clamp(0.0, 1.0)
    return hm.numpy()


def create_evidence_image(image: Image.Image, heatmap: np.ndarray, cfg: Dict) -> Image.Image:
    alpha = float(cfg.get("alpha", 0.4))
    min_clip = float(cfg.get("min_clip", 0.0))
    max_clip = float(cfg.get("max_clip", 1.0))

    hm = np.clip(heatmap, min_clip, max_clip)
    if hm.max() > hm.min():
        hm = (hm - hm.min()) / (hm.max() - hm.min())
    else:
        hm = np.zeros_like(hm)
    hm = hm[..., None]

    base = np.array(image.convert("RGB"), dtype=np.float32)
    red = np.zeros_like(base)
    red[..., 0] = 255.0
    overlay = base * (1.0 - alpha * hm) + red * (alpha * hm)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return Image.fromarray(overlay)


def build_prompt(system_prompt: str, prob: float, user_prompt: str) -> str:
    label = "Fake" if prob >= 0.5 else "Real"
    return (
        f"{system_prompt.strip()}\n"
        f"Head predict result：{label}（p={prob:.2f}）\n"
        f"{user_prompt.strip()}"
    )


def select_user_prompt(user_prompt_cfg: Any, index: int, sample: Dict[str, Any]) -> str:
    if user_prompt_cfg is None:
        return "Is this image tampered or authentic? Decide and give an explanation."

    if isinstance(user_prompt_cfg, str):
        return user_prompt_cfg

    if isinstance(user_prompt_cfg, list):
        if not user_prompt_cfg:
            raise ValueError("user_prompt list in configuration is empty.")
        return user_prompt_cfg[index % len(user_prompt_cfg)] if len(user_prompt_cfg) == 1 else GLOBAL_RNG.choice(user_prompt_cfg)

    if isinstance(user_prompt_cfg, dict):
        options = user_prompt_cfg.get("options") or user_prompt_cfg.get("choices")
        if not options:
            raise ValueError("user_prompt configuration must include a non-empty 'options' list.")

        strategy = user_prompt_cfg.get("strategy", "random")
        if strategy == "sequential":
            return options[index % len(options)]
        if strategy == "by_label" and sample.get("label") is not None:
            by_label = user_prompt_cfg.get("by_label", {})
            label_key = str(sample["label"])
            if label_key in by_label:
                label_prompts = by_label[label_key]
                if isinstance(label_prompts, list) and label_prompts:
                    return GLOBAL_RNG.choice(label_prompts)
                if isinstance(label_prompts, str):
                    return label_prompts
        return GLOBAL_RNG.choice(options)

    raise TypeError("Unsupported user_prompt configuration type; expected str, list, or dict with options.")


def generate_explanation(
    model: Qwen2_5_VLForConditionalGeneration,
    processor: AutoProcessor,
    original_image: Image.Image,
    evidence_image: Image.Image,
    prompt: str,
    chat_system_prompt: str,
    device: torch.device,
    gen_cfg: Dict,
) -> str:
    system_content = chat_system_prompt.strip() or DEFAULT_CHAT_SYSTEM_PROMPT
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_content}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": original_image},
                {"type": "image", "image": evidence_image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[[original_image, evidence_image]], return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}

    generation_kwargs = {
        "max_new_tokens": int(gen_cfg.get("max_new_tokens", 256)),
        "temperature": float(gen_cfg.get("temperature", 0.0)),
        "top_p": float(gen_cfg.get("top_p", 0.9)),
    }

    with torch.inference_mode():
        output_ids = model.generate(**inputs, **generation_kwargs)
    responses = processor.batch_decode(output_ids, skip_special_tokens=True)
    return responses[0]


def extract_assistant_content(full_response: str) -> str:
    if not full_response:
        return ""

    search_space = [
        "assistant\n",
        "assistant: ",
        "assistant：",
        "Assistant\n",
        "Assistant: ",
    ]
    lowered = full_response.lower()
    for marker in search_space:
        marker_lower = marker.lower()
        idx = lowered.rfind(marker_lower)
        if idx != -1:
            return full_response[idx + len(marker) :].lstrip()

    special_markers = ["<|im_start|>assistant", "<|assistant|>"]
    for marker in special_markers:
        idx = full_response.rfind(marker)
        if idx != -1:
            return full_response[idx + len(marker) :].lstrip("\n\r ")

    return full_response.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Frozen multimodal inference pipeline")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config_inference.json"))
    parser.add_argument("--annotation", type=Path, default=None, help="Override annotation JSON for batch inference")
    parser.add_argument("--data_root", type=Path, default=None, help="Root directory used to resolve relative image paths")
    parser.add_argument("--image", type=Path, default=None, help="Run inference for a single image instead of dataset")
    parser.add_argument("--prompt", type=str, default=None, help="Override user prompt string")
    parser.add_argument("--output_csv", type=Path, default=None, help="Path to save CSV outputs")
    parser.add_argument("--evidence_dir", type=Path, default=None, help="Directory to store evidence overlays")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of samples to process")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    set_seed(cfg.get("seed"))

    model, processor, visual_tap, head, device = load_model_components(cfg)

    samples: List[Dict[str, Any]] = []
    if args.image is not None:
        img_path = Path(args.image).expanduser().resolve()
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")
        samples.append({"id": img_path.stem, "image_path": img_path, "label": None, "raw": {}})
    else:
        samples = load_annotation_samples(cfg, override_annotation=args.annotation, override_data_root=args.data_root)

    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive when provided")

    output_cfg = cfg.get("output", {})
    csv_path_cfg = output_cfg.get("csv") or output_cfg.get("output_csv")
    csv_path = args.output_csv or (Path(csv_path_cfg).expanduser() if csv_path_cfg else Path("task8_inference_results.csv"))
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    evidence_dir_cfg = output_cfg.get("evidence_dir") or output_cfg.get("evidence_path")
    evidence_dir = args.evidence_dir or (Path(evidence_dir_cfg).expanduser() if evidence_dir_cfg else None)
    if evidence_dir is not None:
        evidence_dir.mkdir(parents=True, exist_ok=True)

    system_prompt = cfg.get("system_prompt", "")
    chat_system_prompt = cfg.get("chat_system_prompt", DEFAULT_CHAT_SYSTEM_PROMPT)
    user_prompt_cfg = cfg.get("user_prompt")

    heatmap_cfg = cfg.get("heatmap", {})
    generation_cfg = cfg.get("generation", {})

    processed = 0
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        fieldnames = [
            "sample_id",
            "image_path",
            "label",
            "prediction",
            "probability_fake",
            "probability_real",
            "user_prompt",
            "full_prompt",
            "explanation",
            "evidence_path",
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for index, sample in iter_samples(samples, limit=args.limit):
            image = Image.open(sample["image_path"]).convert("RGB")
            prob, heatmap = run_forensic_head(image, processor, visual_tap, head, device)
            heatmap_np = upscale_heatmap(heatmap, image.size[::-1])
            evidence_image = create_evidence_image(image, heatmap_np, heatmap_cfg)

            evidence_path: Optional[Path] = None
            if evidence_dir is not None:
                evidence_path = evidence_dir / f"{sample['id']}_evidence.png"
                evidence_image.save(evidence_path)

            selected_user_prompt = args.prompt if args.prompt is not None else select_user_prompt(user_prompt_cfg, index, sample)
            full_prompt = build_prompt(system_prompt, prob, selected_user_prompt)
            raw_explanation = generate_explanation(
                model,
                processor,
                image,
                evidence_image,
                full_prompt,
                chat_system_prompt,
                device,
                generation_cfg,
            )
            explanation = extract_assistant_content(raw_explanation)

            prediction = "Fake" if prob >= 0.5 else "Real"
            writer.writerow(
                {
                    "sample_id": sample["id"],
                    "image_path": str(sample["image_path"]),
                    "label": sample.get("label"),
                    "prediction": prediction,
                    "probability_fake": f"{prob:.6f}",
                    "probability_real": f"{1.0 - prob:.6f}",
                    "user_prompt": selected_user_prompt,
                    "full_prompt": full_prompt,
                    "explanation": explanation,
                    "evidence_path": str(evidence_path) if evidence_path is not None else "",
                }
            )
            csv_file.flush()
            processed += 1
            print(f"[INFO] Processed {processed}/{len(samples) if args.limit is None else min(len(samples), args.limit)} samples.")

    visual_tap.close()


if __name__ == "__main__":
    main()
