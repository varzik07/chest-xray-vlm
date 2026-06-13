"""
Pre-download a HF model into the local cache in a MINIMAL process
(no torch/transformers imports) to avoid a native-DLL crash seen when
downloading from inside the heavy training process on this machine.

Symlinks are disabled (Windows blocks them without admin/Developer Mode),
so files are copied into the cache instead.

Usage:
  python src/download_model.py HuggingFaceTB/SmolVLM-500M-Instruct
"""
import os
import sys

# MUST be set before importing huggingface_hub.
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from pathlib import Path  # noqa: E402

from huggingface_hub import snapshot_download  # noqa: E402

model_id = sys.argv[1] if len(sys.argv) > 1 else "HuggingFaceTB/SmolVLM-500M-Instruct"
# Copy real files into models/<name> (no symlinks/blobs -> no Windows privilege issue).
local_dir = Path(__file__).resolve().parent.parent / "models" / model_id.split("/")[-1]
path = snapshot_download(
    model_id,
    local_dir=str(local_dir),
    allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "tokenizer*", "*.jinja"],
)
print("DOWNLOADED:", model_id)
print("PATH:", path)
