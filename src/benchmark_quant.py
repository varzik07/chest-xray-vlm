"""
M5 — Quantization benchmark: fp16 vs 4-bit (NF4).

Measures VRAM footprint and inference latency of SmolVLM loaded in full
precision (fp16) vs 4-bit, to quantify the quantization win for the README.

  python src/benchmark_quant.py
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "SmolVLM-500M-Instruct"
PROMPT = "You are a radiologist. Describe the findings in this chest X-ray."
N = 20


def load(mode):
    if mode == "4bit":
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.bfloat16,
                                 bnb_4bit_use_double_quant=True)
        return AutoModelForImageTextToText.from_pretrained(
            MODEL_DIR, quantization_config=bnb, device_map={"": 0})
    return AutoModelForImageTextToText.from_pretrained(
        MODEL_DIR, torch_dtype=torch.float16, device_map={"": 0})


@torch.no_grad()
def bench(mode, processor, imgs):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    model = load(mode); model.eval()
    vram = torch.cuda.memory_allocated() / 1024**3
    text = processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}],
        add_generation_prompt=True)
    # warmup
    w = processor(text=[text], images=[[imgs[0]]], return_tensors="pt").to(model.device)
    model.generate(**w, max_new_tokens=40, do_sample=False)
    torch.cuda.synchronize(); t0 = time.time()
    for im in imgs:
        inp = processor(text=[text], images=[[im]], return_tensors="pt").to(model.device)
        model.generate(**inp, max_new_tokens=40, do_sample=False)
    torch.cuda.synchronize(); dt = (time.time() - t0) / len(imgs)
    del model; torch.cuda.empty_cache()
    return vram, dt


def main():
    processor = AutoProcessor.from_pretrained(MODEL_DIR)
    processor.image_processor.do_image_splitting = False
    rows = [json.loads(l) for l in open(ROOT / "data" / "test.jsonl", encoding="utf-8")][:N]
    imgs = [Image.open(ROOT / r["image"]).convert("RGB") for r in rows]

    print(f"Benchmarking on {N} images...\n")
    v16, t16 = bench("fp16", processor, imgs)
    v4, t4 = bench("4bit", processor, imgs)

    print(f"{'mode':<8}{'VRAM (GB)':<12}{'latency/img (s)':<16}")
    print(f"{'fp16':<8}{v16:<12.2f}{t16:<16.3f}")
    print(f"{'4-bit':<8}{v4:<12.2f}{t4:<16.3f}")
    print(f"\nVRAM reduction: {v16/v4:.1f}x smaller with 4-bit")


if __name__ == "__main__":
    main()
