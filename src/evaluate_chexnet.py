"""
v3 eval — CheXNet + GPT-2 on the test set. Mirrors evaluate.py.

  python src/evaluate_chexnet.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import GPT2TokenizerFast

from metrics import compute_text_metrics, format_metrics
from model_chexnet import CXRReportModelMedical, MEDICAL_TRANSFORM

ROOT = Path(__file__).resolve().parent.parent
CKPT = ROOT / "outputs" / "chexnet_best.pt"


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-beams", type=int, default=1)   # 1 = greedy
    ap.add_argument("--out", type=str, default="outputs/predictions_chexnet.jsonl")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    model = CXRReportModelMedical().to(device)
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
                MEDICAL_TRANSFORM(Image.open(ROOT / r["image"]).convert("RGB")) for r in chunk
            ]).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out_ids = model.generate(imgs, max_new_tokens=100, num_beams=args.num_beams,
                                         no_repeat_ngram_size=3)
            texts = tokenizer.batch_decode(out_ids, skip_special_tokens=True)
            for r, t in zip(chunk, texts):
                t = t.strip()
                preds.append(t); refs.append(r["report"])
                fout.write(json.dumps({"image": r["image"], "reference": r["report"],
                                       "prediction": t}, ensure_ascii=False) + "\n")

    print(f"\nWrote predictions -> {out_path}")
    metrics = compute_text_metrics(preds, refs)
    print("\n=== CheXNet+GPT-2 metrics (test set) ===")
    print(format_metrics(metrics))
    uniq = len(set(preds))
    print(f"unique predictions: {uniq}/{len(preds)} ({100*uniq/len(preds):.1f}%)")
    json.dump(metrics, open(ROOT / "outputs" / "metrics_chexnet.json", "w"), indent=2)


if __name__ == "__main__":
    main()
