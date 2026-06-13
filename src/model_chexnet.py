"""
v3 — CheXNet (medical) encoder + GPT-2 decoder.

Same architecture as the Phase-1 baseline (src/model.py), but the vision
encoder is a DenseNet121 **pretrained on chest X-rays** (via torchxrayvision)
instead of ImageNet. This directly targets the diagnosed weakness: a
general-purpose encoder can't "see" chest pathology.

The medical encoder expects single-channel images normalized to [-1024, 1024]
(the torchxrayvision convention), so it uses its own transform.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchxrayvision as xrv
from torchvision import transforms
from transformers import GPT2LMHeadModel

# torchxrayvision preprocessing: grayscale, 224x224, pixels in [-1024, 1024].
MEDICAL_TRANSFORM = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),                                  # [1,224,224] in [0,1]
    transforms.Lambda(lambda x: (x * 2.0 - 1.0) * 1024.0),  # -> [-1024,1024]
])


class CXRReportModelMedical(nn.Module):
    def __init__(self, gpt2_name: str = "gpt2",
                 xrv_weights: str = "densenet121-res224-all"):
        super().__init__()

        # 1. Medical vision encoder (frozen) -------------------------------
        self.encoder = xrv.models.DenseNet(weights=xrv_weights)
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()
        enc_dim = 1024                                    # densenet121 feature channels

        # 3. Text decoder (GPT-2) ------------------------------------------
        self.gpt2 = GPT2LMHeadModel.from_pretrained(gpt2_name)
        hidden = self.gpt2.config.n_embd

        # 2. Projector (trained) -------------------------------------------
        self.projector = nn.Sequential(
            nn.Linear(enc_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden),
        )

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feats = self.encoder.features(pixel_values)   # [B,1024,7,7]
        feats = feats.flatten(2).transpose(1, 2)          # [B,49,1024]
        return self.projector(feats)

    def forward(self, pixel_values, input_ids, attention_mask, labels):
        img_embeds = self.encode_image(pixel_values)
        n_img = img_embeds.size(1)
        word_embeds = self.gpt2.transformer.wte(input_ids)
        inputs_embeds = torch.cat([img_embeds, word_embeds], dim=1)

        b = input_ids.size(0)
        img_mask = torch.ones(b, n_img, device=input_ids.device, dtype=attention_mask.dtype)
        full_mask = torch.cat([img_mask, attention_mask], dim=1)
        img_labels = torch.full((b, n_img), -100, device=input_ids.device, dtype=labels.dtype)
        full_labels = torch.cat([img_labels, labels], dim=1)

        out = self.gpt2(inputs_embeds=inputs_embeds, attention_mask=full_mask, labels=full_labels)
        return out.loss, out.logits

    @torch.no_grad()
    def generate(self, pixel_values, max_new_tokens: int = 100, **kwargs):
        self.eval()
        img_embeds = self.encode_image(pixel_values)
        b, n_img, _ = img_embeds.shape
        attn = torch.ones(b, n_img, device=pixel_values.device, dtype=torch.long)
        return self.gpt2.generate(inputs_embeds=img_embeds, attention_mask=attn,
                                  max_new_tokens=max_new_tokens,
                                  pad_token_id=self.gpt2.config.eos_token_id, **kwargs)

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]
