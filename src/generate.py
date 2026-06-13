"""
M2 — Generate reports from a trained baseline checkpoint.

Examples:
  python src/generate.py --n 5                 # 5 random test X-rays
  python src/generate.py --image data/images/CXR1_1_IM-0001-3001.png
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from transformers import GPT2TokenizerFast

from dataset import IMAGE_TRANSFORM
from model import CXRReportModel
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
CKPT = ROOT / "outputs" / "baseline_best.pt"


def load_model(device):
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = CXRReportModel().to(device)
    state = torch.load(CKPT, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()
    return model, tokenizer


def write_report(model, tokenizer, image_path, device):
    img = Image.open(ROOT / image_path).convert("RGB")
    pixel_values = IMAGE_TRANSFORM(img).unsqueeze(0).to(device)
    out_ids = model.generate(pixel_values, max_new_tokens=100, num_beams=4,
                             no_repeat_ngram_size=3)
    return tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--image", type=str, default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_model(device)

    if args.image:
        print("PREDICTED:", write_report(model, tokenizer, args.image, device))
        return

    rows = [json.loads(l) for l in open(ROOT / "data" / "test.jsonl", encoding="utf-8")]
    for row in random.sample(rows, args.n):
        pred = write_report(model, tokenizer, row["image"], device)
        print("=" * 80)
        print("IMAGE     :", row["image"])
        print("GROUND TRUTH:", row["report"])
        print("PREDICTED   :", pred)


if __name__ == "__main__":
    main()
