---
title: Chest X-ray Report Generator
emoji: 🩻
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.35.0
app_file: app.py
pinned: false
license: mit
---

# 🩻 Chest X-ray Report Generator

Fine-tuned **SmolVLM-500M** (QLoRA, 4-bit training) on the Indiana University
Chest X-ray dataset to generate radiology-style **findings** reports from a
chest X-ray image.

Upload a chest X-ray (or pick an example) and click **Generate Report**.

⚠️ **Research / educational demo only — NOT a medical device.** Generated text
may be inaccurate and must never be used for diagnosis.

Full project, training code, and write-up: see the GitHub repository.
