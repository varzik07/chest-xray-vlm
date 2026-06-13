"""
M2 — Dataset & DataLoader for the Phase 1 baseline.

Loads the jsonl produced by prepare_data.py. For each sample it:
  - opens the X-ray PNG, converts to RGB (X-rays are grayscale -> 3 channels),
    resizes to 224x224, normalizes (ImageNet stats, since the encoder is
    ImageNet-pretrained).
  - tokenizes the report text with the GPT-2 tokenizer, appends <eos> so the
    model learns where a report ends.

collate_fn pads a batch of variable-length reports to the same length.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

ROOT = Path(__file__).resolve().parent.parent

# ImageNet normalization (the DenseNet encoder was pretrained on ImageNet).
IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


class CXRDataset(Dataset):
    def __init__(self, jsonl_path: str | Path, tokenizer, max_len: int = 100,
                 max_samples: int | None = None, transform=None):
        self.rows = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                self.rows.append(json.loads(line))
        if max_samples:
            self.rows = self.rows[:max_samples]
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.transform = transform or IMAGE_TRANSFORM

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]

        # --- image ---
        img = Image.open(ROOT / row["image"]).convert("RGB")
        pixel_values = self.transform(img)

        # --- text --- tokenize report, append <eos> so the model can stop.
        ids = self.tokenizer(
            row["report"],
            truncation=True,
            max_length=self.max_len - 1,
            return_attention_mask=False,
        )["input_ids"]
        ids = ids + [self.tokenizer.eos_token_id]
        input_ids = torch.tensor(ids, dtype=torch.long)

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "report": row["report"],
        }


def make_collate_fn(pad_token_id: int):
    """Pad input_ids/labels/attention_mask to the longest report in the batch."""
    def collate(batch: list[dict]) -> dict:
        pixel_values = torch.stack([b["pixel_values"] for b in batch])
        max_len = max(b["input_ids"].size(0) for b in batch)

        input_ids, attention_mask, labels = [], [], []
        for b in batch:
            ids = b["input_ids"]
            pad = max_len - ids.size(0)
            input_ids.append(torch.cat([ids, torch.full((pad,), pad_token_id)]))
            attention_mask.append(torch.cat([torch.ones(ids.size(0)), torch.zeros(pad)]))
            # labels: ignore padded positions with -100 (no loss there).
            labels.append(torch.cat([ids, torch.full((pad,), -100)]))

        return {
            "pixel_values": pixel_values,
            "input_ids": torch.stack(input_ids).long(),
            "attention_mask": torch.stack(attention_mask).long(),
            "labels": torch.stack(labels).long(),
        }
    return collate
