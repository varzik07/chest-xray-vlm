"""
M4 (v2) — QLoRA fine-tuning of SmolVLM, fixing the mode-collapse from v1.

v1 collapsed to a few "normal" templates and ignored the image. Four fixes:
  1. LoRA also on the VISION CONNECTOR (modality_projection.proj), not just
     the LLM -> the X-ray can actually influence the output.
  2. Oversample ABNORMAL reports (varied/hard) so the model stops defaulting
     to the repetitive normal template.
  3. A VALIDATION split + best-checkpoint saving -> we know when to stop.
  4. r=16 (more adapter capacity).

Loads SmolVLM 4-bit from a LOCAL folder, OFFLINE (see download_model.py).

  python src/train_qlora.py --max-samples 32 --epochs 1            # smoke test
  python src/train_qlora.py --epochs 3 --batch-size 16             # real run
"""

from __future__ import annotations

import argparse
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "SmolVLM-500M-Instruct"
REPORTS_DIR = ROOT / "data" / "ecgen-radiology"
OUT_DIR = ROOT / "outputs" / "smolvlm_lora"
PROMPT = "You are a radiologist. Describe the findings in this chest X-ray."


def build_abnormal_map() -> dict[str, bool]:
    """uid -> True if the report has any non-'normal' MeSH major label."""
    labels = {}
    for xml in REPORTS_DIR.glob("*.xml"):
        try:
            root = ET.parse(xml).getroot()
        except ET.ParseError:
            continue
        majors = [e.text.strip().lower() for e in root.iter("major") if e.text]
        labels[xml.stem] = not (majors and all(m == "normal" for m in majors))
    return labels


class CXRChatDataset(Dataset):
    def __init__(self, jsonl, abnormal_map=None, oversample_abnormal=1.0, max_samples=None):
        rows = [json.loads(l) for l in open(ROOT / "data" / jsonl, encoding="utf-8")]
        if max_samples:
            rows = rows[:max_samples]
        # Oversample abnormal cases by duplicating them.
        if abnormal_map and oversample_abnormal > 1.0:
            extra = int(oversample_abnormal - 1.0)  # whole extra copies
            frac = oversample_abnormal - 1.0 - extra
            abn = [r for r in rows if abnormal_map.get(r["uid"], False)]
            rows = rows + abn * extra + abn[: int(len(abn) * frac)]
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        return {"image": Image.open(ROOT / r["image"]).convert("RGB"),
                "report": r["report"]}


def build_collate(processor, image_token_id):
    pad_id = processor.tokenizer.pad_token_id

    def collate(batch):
        texts, images = [], []
        for ex in batch:
            messages = [
                {"role": "user", "content": [{"type": "image"},
                                             {"type": "text", "text": PROMPT}]},
                {"role": "assistant", "content": [{"type": "text", "text": ex["report"]}]},
            ]
            texts.append(processor.apply_chat_template(messages, add_generation_prompt=False).strip())
            images.append([ex["image"]])
        out = processor(text=texts, images=images, return_tensors="pt", padding=True)
        labels = out["input_ids"].clone()
        labels[labels == pad_id] = -100
        labels[labels == image_token_id] = -100
        out["labels"] = labels
        return out
    return collate


@torch.no_grad()
def evaluate_loss(model, loader, device):
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        total += model(**batch).loss.item()
        n += 1
    model.train()
    return total / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--oversample-abnormal", type=float, default=2.0)
    ap.add_argument("--max-samples", type=int, default=None)
    args = ap.parse_args()

    device = "cuda"
    processor = AutoProcessor.from_pretrained(MODEL_DIR)
    processor.image_processor.do_image_splitting = False
    image_token_id = processor.tokenizer.convert_tokens_to_ids("<image>")

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_DIR, quantization_config=bnb, device_map={"": 0})
    model = prepare_model_for_kbit_training(model)

    lora = LoraConfig(
        r=args.rank, lora_alpha=args.rank * 2, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",
                        "modality_projection.proj"],   # <-- vision connector
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    model.config.use_cache = False

    abn_map = build_abnormal_map()
    n_abn = sum(abn_map.values())
    print(f"abnormal reports: {n_abn}/{len(abn_map)}  (oversample x{args.oversample_abnormal})")

    collate = build_collate(processor, image_token_id)
    train_ds = CXRChatDataset("train.jsonl", abn_map, args.oversample_abnormal, args.max_samples)
    val_ds = CXRChatDataset("val.jsonl", max_samples=args.max_samples)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    print(f"train rows (after oversample): {len(train_ds)} | val rows: {len(val_ds)}")

    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    model.train()
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}")
        running = 0.0
        optim.zero_grad()
        for step, batch in enumerate(pbar, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            (out.loss / args.grad_accum).backward()
            if step % args.grad_accum == 0:
                optim.step()
                optim.zero_grad()
            running += out.loss.item()
            pbar.set_postfix(loss=f"{running/step:.3f}")

        val_loss = evaluate_loss(model, val_loader, device)
        print(f"epoch {epoch}: train_loss={running/len(train_loader):.3f}  val_loss={val_loss:.3f}")
        if val_loss < best_val:
            best_val = val_loss
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(OUT_DIR)
            processor.save_pretrained(OUT_DIR)
            print(f"  saved best -> {OUT_DIR} (val {val_loss:.3f})")


if __name__ == "__main__":
    main()
