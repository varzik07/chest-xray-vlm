"""Generate a clean model-architecture diagram for the project (PPT/README)."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parent.parent
out = ROOT / "assets" / "architecture.png"
out.parent.mkdir(exist_ok=True)

fig, ax = plt.subplots(figsize=(12, 6.5))
ax.set_xlim(0, 12); ax.set_ylim(0, 7); ax.axis("off")
ax.set_title("Model Architecture — Vision-Language Model for Chest X-ray Reports",
             fontsize=15, fontweight="bold", pad=16)

boxes = [
    (0.4, "Chest X-ray\n(input image)", "#e8eef7", "#1a5fb4"),
    (3.0, "Vision Encoder\n(FROZEN)\n\nDenseNet / SigLIP /\nCheXNet (medical)", "#fde9d9", "#c45e00"),
    (6.0, "Connector\n(TRAINED)\n\nprojects image →\nLLM token space", "#e9f6e9", "#2a9d4a"),
    (9.0, "Language Model\n(decoder)\n\nGPT-2 / SmolLM2\nwrites the report", "#f3e9f7", "#7a3fa0"),
]
w, h, y = 2.3, 2.6, 3.4
centers = []
for x, label, fc, ec in boxes:
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                                linewidth=2, facecolor=fc, edgecolor=ec))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=10.5, color="#222")
    centers.append((x, x + w))

# arrows between boxes
for i in range(len(centers) - 1):
    ax.add_patch(FancyArrowPatch((centers[i][1], y + h / 2),
                                 (centers[i + 1][0], y + h / 2),
                                 arrowstyle="-|>", mutation_scale=22,
                                 linewidth=2, color="#555"))

# output box
ax.add_patch(FancyBboxPatch((9.0, 1.55), 2.6, 1.15, boxstyle="round,pad=0.06",
                            linewidth=1.5, facecolor="#fffbe6", edgecolor="#b59f00"))
ax.text(10.3, 2.12, '"The lungs are clear.\nNo pleural effusion..."', ha="center",
        va="center", fontsize=9.5, style="italic", color="#444")
ax.add_patch(FancyArrowPatch((10.15, y), (10.3, 2.75), arrowstyle="-|>",
                             mutation_scale=18, linewidth=2, color="#555"))

# variant legend (bottom)
ax.text(0.4, 1.15, "Three variants built & compared:", fontsize=11, fontweight="bold")
ax.text(0.4, 0.75,
        "1) Baseline:  DenseNet (ImageNet) + GPT-2        — full fine-tune\n"
        "2) SmolVLM:   SigLIP + SmolLM2                    — QLoRA (4-bit + LoRA)\n"
        "3) v3 (best): DenseNet (CheXNet, medical) + GPT-2 — BLEU-1 0.343 (+23%)",
        fontsize=9.5, family="monospace", color="#333", va="top")

plt.tight_layout()
plt.savefig(out, dpi=140, bbox_inches="tight")
print("Saved", out)
