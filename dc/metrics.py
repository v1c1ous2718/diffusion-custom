"""Subject fidelity vs prompt fidelity, following the DreamBooth / Custom Diffusion evaluation protocol.

  CLIP-T : cosine(CLIP image, CLIP text of the prompt without <new1>)    -> does it follow the prompt?
  CLIP-I : mean cosine(CLIP image, each of the 4 training images)          -> does it look like the object?
  DINO   : mean cosine(DINO ViT-S/16 CLS, each training image)             -> same, less tolerant of class-level match
"""
from __future__ import annotations

import json

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor

from .gen import CONTEXTS, OUT, SEEDS, plain_prompt
from .train import DATA

CLIP = "openai/clip-vit-base-patch32"
DINO = "facebook/dino-vits16"


class Scorer:
    def __init__(self):
        self.clip = CLIPModel.from_pretrained(CLIP).cuda().eval()
        self.cproc = CLIPProcessor.from_pretrained(CLIP)
        self.dino = AutoModel.from_pretrained(DINO).cuda().eval()
        self.dproc = AutoImageProcessor.from_pretrained(DINO)
        refs = [Image.open(p).convert("RGB") for p in sorted(DATA.glob("*.jpg"))]
        self.ref_clip = self.clip_img(refs)
        self.ref_dino = self.dino_img(refs)
        self.text = {ctx: self.clip_txt(plain_prompt(ctx)) for ctx, _ in CONTEXTS}

    @torch.no_grad()
    def clip_img(self, ims):
        x = self.cproc(images=ims, return_tensors="pt")["pixel_values"].cuda()
        return F.normalize(self.clip.get_image_features(pixel_values=x), dim=-1)

    @torch.no_grad()
    def clip_txt(self, t):
        x = self.cproc(text=[t], return_tensors="pt", padding=True)
        return F.normalize(self.clip.get_text_features(**{k: v.cuda() for k, v in x.items()}), dim=-1)[0]

    @torch.no_grad()
    def dino_img(self, ims):
        x = self.dproc(images=ims, return_tensors="pt")["pixel_values"].cuda()
        return F.normalize(self.dino(pixel_values=x).last_hidden_state[:, 0], dim=-1)

    def score_dir(self, tag: str) -> dict:
        d = OUT / tag
        if (d / "scores.json").exists():
            return json.loads((d / "scores.json").read_text())
        rows = []
        for ctx, _ in CONTEXTS:
            ims = [Image.open(d / f"{ctx}_{s}.jpg").convert("RGB") for s in SEEDS]
            ci, di = self.clip_img(ims), self.dino_img(ims)
            ct = (ci @ self.text[ctx]).tolist()
            cI = (ci @ self.ref_clip.T).mean(1).tolist()
            dI = (di @ self.ref_dino.T).mean(1).tolist()
            for s, a, b, c in zip(SEEDS, ct, cI, dI):
                rows.append({"ctx": ctx, "seed": s, "clip_t": a, "clip_i": b, "dino": c})
        mean = {k: sum(r[k] for r in rows) / len(rows) for k in ["clip_t", "clip_i", "dino"]}
        res = {"tag": tag, "mean": mean, "rows": rows}
        (d / "scores.json").write_text(json.dumps(res, indent=1))
        return res
