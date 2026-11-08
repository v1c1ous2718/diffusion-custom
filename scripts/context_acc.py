"""Context accuracy: does the background follow the prompt? CLIP zero-shot 6-way classification of each image
into the six prompt contexts (texts describe only the scene, not the object)."""
import json
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from dc.gen import CONTEXTS, OUT, SEEDS

TEXTS = {"snow": "a photo of something in the snow", "beach": "a photo of something on a sandy beach",
         "kitchen": "a photo of something on a kitchen table", "jungle": "a photo of something in the jungle",
         "space": "a photo of something floating in outer space", "watercolor": "a watercolor painting"}
m = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").cuda().eval()
p = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
ctxs = [c for c, _ in CONTEXTS]
with torch.no_grad():
    t = p(text=[TEXTS[c] for c in ctxs], return_tensors="pt", padding=True)
    T = F.normalize(m.get_text_features(**{k: v.cuda() for k, v in t.items()}), dim=-1)
res = {}
for d in sorted(OUT.iterdir()):
    if not (d / "meta.json").exists():
        continue
    correct, per = 0, {}
    for i, c in enumerate(ctxs):
        ims = [Image.open(d / f"{c}_{s}.jpg").convert("RGB") for s in SEEDS]
        with torch.no_grad():
            x = p(images=ims, return_tensors="pt")["pixel_values"].cuda()
            pred = (F.normalize(m.get_image_features(pixel_values=x), dim=-1) @ T.T).argmax(1)
        k = (pred == i).sum().item(); correct += k; per[c] = k / len(SEEDS)
    res[d.name] = {"context_acc": correct / (len(ctxs) * len(SEEDS)), "per_context": per}
    print(f"{d.name:26s} {res[d.name]['context_acc']:.3f}")
(OUT.parent / "runs" / "context_acc.json").write_text(json.dumps(res, indent=1))
