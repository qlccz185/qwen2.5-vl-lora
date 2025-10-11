"""Dataset utilities for Task2 LM-supervised LoRA training without mapped datasets."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

from PIL import Image

import torch
from torch.utils.data import Dataset


def resize_square_pad_rgb(img: Image.Image, size: int, pad_color: tuple[int, int, int] = (128, 128, 128)) -> Image.Image:
    """Resize the input image to a square canvas while preserving the aspect ratio."""
    w, h = img.size
    if max(w, h) == size:
        return img.resize((size, size), Image.BICUBIC)
    scale = size / max(w, h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = img.resize((new_w, new_h), Image.BICUBIC)
    canvas = Image.new("RGB", (size, size), pad_color)
    canvas.paste(resized, ((size - new_w) // 2, (size - new_h) // 2))
    return canvas


@dataclass
class ForgerySample:
    image_path: str
    label: int


class ForgeryBinaryDataset(Dataset):
    """Load binary forgery annotations directly from raw JSON files."""

    def __init__(self, annotation_file: str | Path, data_root: str | Path) -> None:
        self.annotation_path = Path(annotation_file)
        if not self.annotation_path.exists():
            raise FileNotFoundError(f"Annotation file not found: {self.annotation_path}")
        self.data_root = Path(data_root)

        raw_items = json.loads(self.annotation_path.read_text(encoding="utf-8"))
        self.samples: List[ForgerySample] = []
        skipped = 0
        for item in raw_items:
            rel_path = item.get("image_path")
            if not isinstance(rel_path, str):
                skipped += 1
                continue
            abs_path = (self.data_root / rel_path).resolve()
            if not abs_path.exists():
                skipped += 1
                continue
            label = int(item.get("label", 0))
            self.samples.append(ForgerySample(image_path=str(abs_path), label=label))
        if not self.samples:
            raise RuntimeError(f"No valid samples found in {self.annotation_path}")
        if skipped:
            print(f"[Task2] Skipped {skipped} invalid items when reading {self.annotation_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        image = Image.open(sample.image_path).convert("RGB")
        return {"image": image, "label": sample.label, "path": sample.image_path}


class Task2Collator:
    """Turn raw PIL images into Qwen2.5-VL processor inputs."""

    def __init__(self, processor, prompt: str, image_size: int = 448) -> None:
        self.processor = processor
        self.prompt = prompt.strip()
        self.image_size = int(image_size)

    def __call__(self, batch: Sequence[dict]) -> dict:
        messages: List[dict] = []
        labels: List[int] = []
        paths: List[str] = []
        resized_images: List[Image.Image] = []
        for record in batch:
            image: Image.Image = record["image"]
            resized = resize_square_pad_rgb(image, self.image_size)
            resized_images.append(resized)
            labels.append(int(record["label"]))
            paths.append(record["path"])
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": resized},
                        {
                            "type": "text",
                            "text": self.prompt,
                        },
                    ],
                }
            )
        text_inputs = [
            self.processor.apply_chat_template(
                [msg], tokenize=False, add_generation_prompt=True
            )
            for msg in messages
        ]
        model_inputs = self.processor(
            text=text_inputs,
            images=resized_images,
            padding=True,
            return_tensors="pt",
        )
        model_inputs["labels"] = torch.tensor(labels, dtype=torch.float32)
        model_inputs["paths"] = paths
        return model_inputs
