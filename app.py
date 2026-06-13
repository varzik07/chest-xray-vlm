"""
Chest X-ray Report Generator — Gradio demo (M6).

Loads SmolVLM (4-bit) + our QLoRA adapter and writes a radiology-style
report for an uploaded chest X-ray.

Run locally:   python app.py
Then open the printed local URL in your browser.

NOTE: research/educational demo only — NOT for clinical use.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import gradio as gr
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from peft import PeftModel

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models" / "SmolVLM-500M-Instruct"
ADAPTER = ROOT / "outputs" / "smolvlm_lora"
PROMPT = "You are a radiologist. Describe the findings in this chest X-ray."

DISCLAIMER = (
    "⚠️ **Research / educational demo only — NOT a medical device.** "
    "Generated text may be inaccurate and must never be used for diagnosis."
)

print("Loading model (4-bit + QLoRA adapter)...")
processor = AutoProcessor.from_pretrained(MODEL_DIR)
processor.image_processor.do_image_splitting = False
_bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                          bnb_4bit_compute_dtype=torch.bfloat16,
                          bnb_4bit_use_double_quant=True)
_model = AutoModelForImageTextToText.from_pretrained(
    MODEL_DIR, quantization_config=_bnb, device_map={"": 0})
model = PeftModel.from_pretrained(_model, ADAPTER)
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
    inputs = processor(text=[text], images=[[image]], return_tensors="pt").to(model.device)
    gen = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    new = gen[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(new, skip_special_tokens=True)[0].strip()


def build_demo():
    example_dir = ROOT / "data" / "images"
    examples = [str(p) for p in list(example_dir.glob("*.png"))[:4]] if example_dir.exists() else None

    with gr.Blocks(title="Chest X-ray Report Generator") as demo:
        gr.Markdown("# 🩻 Chest X-ray Report Generator")
        gr.Markdown(
            "Fine-tuned **SmolVLM** (QLoRA, 4-bit) on the Indiana University "
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
    return demo


if __name__ == "__main__":
    build_demo().launch()
