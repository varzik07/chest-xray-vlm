"""
Chest X-ray Report Generator — Hugging Face Spaces app.

Loads SmolVLM-500M (from the HF hub) + our QLoRA adapter (bundled in this repo),
merges them, and generates a radiology-style report for an uploaded chest X-ray.

Runs on free CPU (float32) or GPU (float16) automatically. No bitsandbytes
(4-bit is GPU-only); on Spaces we serve the merged fp16/fp32 model.

⚠️ Research / educational demo only — NOT a medical device.
"""

from __future__ import annotations

from pathlib import Path

import gradio as gr
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel

ROOT = Path(__file__).resolve().parent
BASE_MODEL = "HuggingFaceTB/SmolVLM-500M-Instruct"   # downloaded at runtime
ADAPTER = ROOT / "smolvlm_lora"                       # bundled in this repo
PROMPT = "You are a radiologist. Describe the findings in this chest X-ray."

DISCLAIMER = (
    "⚠️ **Research / educational demo only — NOT a medical device.** "
    "Generated text may be inaccurate and must never be used for diagnosis."
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

print(f"Loading model on {DEVICE} ({DTYPE})...")
processor = AutoProcessor.from_pretrained(BASE_MODEL)
processor.image_processor.do_image_splitting = False
base = AutoModelForImageTextToText.from_pretrained(BASE_MODEL, torch_dtype=DTYPE)
model = PeftModel.from_pretrained(base, str(ADAPTER)).merge_and_unload().to(DEVICE)
model.eval()
print("Model ready.")


@torch.no_grad()
def generate_report(image, max_new_tokens: int = 80):
    if image is None:
        return "Please upload a chest X-ray image."
    image = image.convert("RGB")
    messages = [{"role": "user", "content": [{"type": "image"},
                                             {"type": "text", "text": PROMPT}]}]
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=[text], images=[[image]], return_tensors="pt").to(DEVICE)
    gen = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    new = gen[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(new, skip_special_tokens=True)[0].strip()


example_dir = ROOT / "examples"
examples = [str(p) for p in sorted(example_dir.glob("*.png"))] if example_dir.exists() else None

with gr.Blocks(title="Chest X-ray Report Generator") as demo:
    gr.Markdown("# 🩻 Chest X-ray Report Generator")
    gr.Markdown(
        "Fine-tuned **SmolVLM** (QLoRA, 4-bit training) on the Indiana University "
        "Chest X-ray dataset. Upload a chest X-ray to generate a findings report."
    )
    gr.Markdown(DISCLAIMER)
    with gr.Row():
        with gr.Column():
            img = gr.Image(type="pil", label="Chest X-ray", height=360)
            btn = gr.Button("Generate Report", variant="primary")
        with gr.Column():
            out = gr.Textbox(label="Generated findings", lines=8)
    if examples:
        gr.Examples(examples=examples, inputs=img)
    btn.click(generate_report, inputs=img, outputs=out)

if __name__ == "__main__":
    demo.launch()
