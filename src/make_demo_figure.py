"""Build a 'demo result' figure (X-ray + generated report) for the README."""
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
IMG = ROOT / "data" / "images" / "CXR474_IM-2101-2001.png"
PRED = ("The lungs are clear. There is no pleural effusion or pneumothorax. "
        "The heart is normal. The skeletal structures are normal.")
TRUE = ("The lungs are clear. There is no pleural effusion or pneumothorax. "
        "The heart and mediastinum are normal. The skeletal structures are normal.")

fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 5),
                               gridspec_kw={"width_ratios": [1, 1.25]})
fig.suptitle("Chest X-ray Report Generator  —  SmolVLM (QLoRA, 4-bit)",
             fontsize=14, fontweight="bold")

axL.imshow(Image.open(IMG).convert("L"), cmap="gray")
axL.set_title("Input chest X-ray", fontsize=11)
axL.axis("off")

axR.axis("off")
axR.text(0.0, 0.95, "Generated report", fontsize=12, fontweight="bold",
         color="#1a5fb4", transform=axR.transAxes, va="top")
axR.text(0.0, 0.86, "\n".join(textwrap.wrap(PRED, 46)), fontsize=11,
         transform=axR.transAxes, va="top")
axR.text(0.0, 0.45, "Ground truth (radiologist)", fontsize=12, fontweight="bold",
         color="#2a9d4a", transform=axR.transAxes, va="top")
axR.text(0.0, 0.36, "\n".join(textwrap.wrap(TRUE, 46)), fontsize=11,
         transform=axR.transAxes, va="top")
axR.text(0.0, 0.05, "ROUGE-L 0.90  ·  research/educational demo, not for clinical use",
         fontsize=9, style="italic", color="#666", transform=axR.transAxes, va="top")

plt.tight_layout(rect=[0, 0, 1, 0.95])
out = ROOT / "assets" / "demo_result.png"
out.parent.mkdir(exist_ok=True)
plt.savefig(out, dpi=130, bbox_inches="tight")
print("Saved", out)
