"""
M2 — The Phase 1 baseline VLM (LLaVA-style, but small).

Three parts:
  1. Vision encoder : DenseNet121 (ImageNet-pretrained), FROZEN.
                      A 224x224 image -> 49 feature tokens of dim 1024.
  2. Projector      : small MLP, TRAINED. Maps 1024-dim image tokens into
                      GPT-2's 768-dim embedding space ("image tokens").
  3. Text decoder   : GPT-2 small, TRAINED. We prepend the image tokens in
                      front of the word embeddings, so GPT-2 "reads" the
                      image as a prefix and writes the report.

Loss is only computed on the text positions (image positions are masked with
-100), so the model is rewarded for predicting the report words.
"""

from __future__ import annotations

import timm
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel


class CXRReportModel(nn.Module):
    def __init__(self, gpt2_name: str = "gpt2", vision_name: str = "densenet121"):
        super().__init__()

        # 1. Vision encoder (frozen) ---------------------------------------
        self.encoder = timm.create_model(vision_name, pretrained=True,
                                         num_classes=0, global_pool="")
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()
        enc_dim = self.encoder.num_features          # 1024 for densenet121

        # 3. Text decoder (GPT-2) ------------------------------------------
        self.gpt2 = GPT2LMHeadModel.from_pretrained(gpt2_name)
        hidden = self.gpt2.config.n_embd            # 768

        # 2. Projector (trained) -------------------------------------------
        self.projector = nn.Sequential(
            nn.Linear(enc_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )

    # ---- shared: turn images into GPT-2-space "image tokens" -------------
    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feats = self.encoder.forward_features(pixel_values)   # [B,1024,7,7]
        b, c, h, w = feats.shape
        feats = feats.flatten(2).transpose(1, 2)                  # [B,49,1024]
        return self.projector(feats)                              # [B,49,768]

    def forward(self, pixel_values, input_ids, attention_mask, labels):
        img_embeds = self.encode_image(pixel_values)              # [B,49,768]
        n_img = img_embeds.size(1)

        word_embeds = self.gpt2.transformer.wte(input_ids)        # [B,T,768]
        inputs_embeds = torch.cat([img_embeds, word_embeds], dim=1)

        # extend attention mask + labels to cover the image prefix
        b = input_ids.size(0)
        img_mask = torch.ones(b, n_img, device=input_ids.device, dtype=attention_mask.dtype)
        full_mask = torch.cat([img_mask, attention_mask], dim=1)
        img_labels = torch.full((b, n_img), -100, device=input_ids.device, dtype=labels.dtype)
        full_labels = torch.cat([img_labels, labels], dim=1)

        out = self.gpt2(inputs_embeds=inputs_embeds,
                       attention_mask=full_mask,
                       labels=full_labels)
        return out.loss, out.logits

    @torch.no_grad()
    def generate(self, pixel_values, max_new_tokens: int = 100, **kwargs):
        """Autoregressively write a report from the image alone."""
        self.eval()
        img_embeds = self.encode_image(pixel_values)              # [B,49,768]
        b, n_img, _ = img_embeds.shape
        attn = torch.ones(b, n_img, device=pixel_values.device, dtype=torch.long)
        # With inputs_embeds, generate() returns only the newly produced tokens.
        return self.gpt2.generate(
            inputs_embeds=img_embeds,
            attention_mask=attn,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.gpt2.config.eos_token_id,
            **kwargs,
        )

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]
