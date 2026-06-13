"""
M4 eval — Evaluate the QLoRA SmolVLM on the test set and compare to baseline.

Loads SmolVLM 4-bit (offline) + the trained LoRA adapter, generates a report
for each test X-ray, and scores with the same BLEU/ROUGE-L/METEOR harness.

Run (after train_qlora finishes):
  python src/evaluate_qlora.py
  python src/evaluate_qlora.py --max-samples 100
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel

from metrics import compute_text_metrics, format_metrics

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "SmolVLM-500M-Instruct"
ADAPTER = ROOT / "outputs" / "smolvlm_lora"
PROMPT = "You are a radiologist. Describe the findings in this chest X-ray."


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--num-beams", type=int, default=1)   # 1 = greedy (fast, shows true diversity)
    ap.add_argument("--out", type=str, default="outputs/predictions_qlora.jsonl")
    args = ap.parse_args()

    device = "cuda"
    processor = AutoProcessor.from_pretrained(MODEL_DIR)
    processor.image_processor.do_image_splitting = False
    processor.tokenizer.padding_side = "left"     # required for batched generation

    # Load base in bf16 and MERGE the LoRA adapter -> fast inference (no 4-bit overhead).
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_DIR, torch_dtype=torch.bfloat16, device_map={"": 0})
    model = PeftModel.from_pretrained(model, ADAPTER)
    model = model.merge_and_unload()
    model.eval()

    rows = [json.loads(l) for l in open(ROOT / "data" / "test.jsonl", encoding="utf-8")]
    if args.max_samples:
        rows = rows[:args.max_samples]

    msg = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
    prompt_text = processor.apply_chat_template(msg, add_generation_prompt=True)

    preds, refs = [], []
    out_path = ROOT / args.out
    with out_path.open("w", encoding="utf-8") as fout:
        for i in tqdm(range(0, len(rows), args.batch_size), desc="generating"):
            chunk = rows[i:i + args.batch_size]
            imgs = [[Image.open(ROOT / r["image"]).convert("RGB")] for r in chunk]
            inputs = processor(text=[prompt_text] * len(chunk), images=imgs,
                               return_tensors="pt", padding=True).to(device)
            gen = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                 num_beams=args.num_beams, do_sample=False)
            new = gen[:, inputs["input_ids"].shape[1]:]
            texts = processor.batch_decode(new, skip_special_tokens=True)
            for r, pred in zip(chunk, texts):
                pred = pred.strip()
                preds.append(pred)
                refs.append(r["report"])
                fout.write(json.dumps({"image": r["image"], "reference": r["report"],
                                       "prediction": pred}, ensure_ascii=False) + "\n")

    print(f"\nWrote predictions -> {out_path}")
    metrics = compute_text_metrics(preds, refs)
    print("\n=== QLoRA SmolVLM metrics (test set) ===")
    print(format_metrics(metrics))
    json.dump(metrics, open(ROOT / "outputs" / "metrics_qlora.json", "w"), indent=2)


if __name__ == "__main__":
    main()
