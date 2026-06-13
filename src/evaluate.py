"""
M3 — Evaluate the Phase 1 baseline on the test set.

Generates a report for every test X-ray, saves predictions, and computes
BLEU/ROUGE-L/METEOR.

Run (after training finishes):
  python src/evaluate.py
  python src/evaluate.py --max-samples 100   # quick subset
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GPT2TokenizerFast

from dataset import CXRDataset, IMAGE_TRANSFORM
from metrics import compute_text_metrics, format_metrics
from model import CXRReportModel
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
CKPT = ROOT / "outputs" / "baseline_best.pt"


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out", type=str, default="outputs/predictions.jsonl")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    model = CXRReportModel().to(device)
    model.load_state_dict(torch.load(CKPT, map_location=device)["model"])
    model.eval()

    rows = [json.loads(l) for l in open(ROOT / "data" / "test.jsonl", encoding="utf-8")]
    if args.max_samples:
        rows = rows[:args.max_samples]

    preds, refs = [], []
    out_path = ROOT / args.out
    with out_path.open("w", encoding="utf-8") as fout:
        for i in tqdm(range(0, len(rows), args.batch_size), desc="generating"):
            chunk = rows[i:i + args.batch_size]
            imgs = torch.stack([
                IMAGE_TRANSFORM(Image.open(ROOT / r["image"]).convert("RGB"))
                for r in chunk
            ]).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out_ids = model.generate(imgs, max_new_tokens=100, num_beams=4,
                                         no_repeat_ngram_size=3)
            texts = tokenizer.batch_decode(out_ids, skip_special_tokens=True)
            for r, t in zip(chunk, texts):
                t = t.strip()
                preds.append(t)
                refs.append(r["report"])
                fout.write(json.dumps({"image": r["image"], "reference": r["report"],
                                       "prediction": t}, ensure_ascii=False) + "\n")

    print(f"\nWrote predictions -> {out_path}")
    metrics = compute_text_metrics(preds, refs)
    print("\n=== Baseline metrics (test set) ===")
    print(format_metrics(metrics))

    metrics_path = ROOT / "outputs" / "metrics_baseline.json"
    json.dump(metrics, open(metrics_path, "w"), indent=2)
    print(f"Saved -> {metrics_path}")


if __name__ == "__main__":
    main()
