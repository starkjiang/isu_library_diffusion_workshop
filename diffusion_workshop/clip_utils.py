"""Thin wrapper around CLIP (Hugging Face `transformers`, pre-installed on Colab).

    enc = ClipEncoder(device)
    img_emb = enc.encode_images([pil_image, ...])   # (N, 512), unit length
    txt_emb = enc.encode_text(["a red rose", ...])  # (N, 512), unit length

With WORKSHOP_SMOKE=1 a deterministic fake encoder is used so notebooks can be
tested offline without downloading the 600 MB CLIP checkpoint.
"""
import hashlib

import torch
import torch.nn.functional as F

from . import SMOKE

CLIP_DIM = 512


class ClipEncoder:
    def __init__(self, device="cpu", model_name="openai/clip-vit-base-patch32"):
        self.device, self.dim = device, CLIP_DIM
        self.fake = SMOKE
        if not self.fake:
            from transformers import CLIPModel, CLIPProcessor

            self.model = CLIPModel.from_pretrained(model_name).to(device).eval()
            self.processor = CLIPProcessor.from_pretrained(model_name)

    # -- offline stand-in --------------------------------------------------- #
    def _fake(self, keys):
        rows = []
        for k in keys:
            seed = int(hashlib.sha256(k.encode()).hexdigest()[:8], 16)
            rows.append(torch.randn(self.dim, generator=torch.Generator().manual_seed(seed)))
        return F.normalize(torch.stack(rows), dim=-1).to(self.device)

    # -- real encoders ------------------------------------------------------ #
    @torch.no_grad()
    def encode_images(self, pil_images, batch_size=128):
        if self.fake:
            return self._fake([str(im.resize((4, 4)).tobytes()) for im in pil_images])
        out = []
        for i in range(0, len(pil_images), batch_size):
            px = self.processor(images=pil_images[i:i + batch_size], return_tensors="pt")["pixel_values"].to(self.device)
            feats = self.model.visual_projection(self.model.vision_model(pixel_values=px).pooler_output)
            out.append(F.normalize(feats.float(), dim=-1))
        return torch.cat(out)

    @torch.no_grad()
    def encode_text(self, prompts):
        if self.fake:
            return self._fake(list(prompts))
        tok = self.processor(text=list(prompts), return_tensors="pt", padding=True, truncation=True).to(self.device)
        pooled = self.model.text_model(input_ids=tok["input_ids"], attention_mask=tok["attention_mask"]).pooler_output
        return F.normalize(self.model.text_projection(pooled).float(), dim=-1)
