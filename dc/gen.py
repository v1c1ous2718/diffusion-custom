"""Image generation for every configuration with fixed prompts and seeds (paired across configs)."""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline

from .train import DATA, ROOT, SD, TOKEN

SUBJ = "cat-shaped wood carving with rainbow stripes"
CONTEXTS = [
    ("snow", "photo of a {} in the snow"),
    ("beach", "photo of a {} on a sandy beach"),
    ("kitchen", "photo of a {} on a kitchen table"),
    ("jungle", "photo of a {} in the jungle"),
    ("space", "a {} floating in outer space"),
    ("watercolor", "a watercolor painting of a {}"),
]
BLIP_CONTEXTS = {"snow": "in the snow", "beach": "on a sandy beach", "kitchen": "on a kitchen table",
                 "jungle": "in the jungle", "space": "floating in outer space", "watercolor": "watercolor painting"}
SEEDS = list(range(8))
OUT = ROOT / "out"


def plain_prompt(ctx: str) -> str:
    return dict(CONTEXTS)[ctx].format(SUBJ)


def sd_pipe():
    pipe = StableDiffusionPipeline.from_pretrained(SD, variant="fp16", torch_dtype=torch.float16, safety_checker=None,
                                                   requires_safety_checker=False).to("cuda")
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def load_custom(pipe, ckpt: Path):
    """Install Custom Diffusion processors (as in the HW9 notebook; diffusers 0.33's load_attn_procs
    silently skips these weights) and the learned <new1> embedding."""
    from safetensors.torch import load_file
    sd = load_file(str(Path(ckpt) / "pytorch_custom_diffusion_weights.safetensors"), device="cpu")
    procs = pipe.unet.attn_processors
    procs.update(pipe.unet._process_custom_diffusion(state_dict=sd))
    pipe.unet.set_attn_processor(procs)
    pipe.unet.to(dtype=pipe.unet.dtype, device=pipe.unet.device)
    pipe.load_textual_inversion(str(ckpt), weight_name=f"{TOKEN}.safetensors")
    cross = [p for k, p in pipe.unet.attn_processors.items() if k.endswith("attn2.processor")]
    assert len(cross) == 16 and all(type(p).__name__.startswith("CustomDiffusion") for p in cross), \
        "Custom Diffusion weights were not installed on all 16 cross-attention layers"


def generate_sd(tag: str, ckpt: Path | None, guidance=9.5, steps=50):
    """ckpt=None -> plain SD 1.5 with the descriptive prompt (no learned token)."""
    out = OUT / tag
    if (out / "meta.json").exists():
        return
    out.mkdir(parents=True, exist_ok=True)
    pipe = sd_pipe()
    subj = SUBJ
    if ckpt is not None:
        load_custom(pipe, ckpt)
        subj = f"{TOKEN} {SUBJ}"
    t0 = time.time()
    for ctx, tmpl in CONTEXTS:
        prompt = tmpl.format(subj)
        gens = [torch.Generator("cuda").manual_seed(s) for s in SEEDS]
        ims = pipe([prompt] * len(SEEDS), num_inference_steps=steps, guidance_scale=guidance, generator=gens).images
        for s, im in zip(SEEDS, ims):
            im.save(out / f"{ctx}_{s}.jpg", quality=95)
    meta = {"tag": tag, "ckpt": str(ckpt) if ckpt else None, "guidance": guidance, "steps": steps,
            "sampler": "DDIM eta=0", "sec_per_image": (time.time() - t0) / (len(CONTEXTS) * len(SEEDS))}
    (out / "meta.json").write_text(json.dumps(meta))
    del pipe; torch.cuda.empty_cache()


def generate_blip(tag="blip_diffusion", guidance=7.5, steps=50, ref="1.jpg"):
    out = OUT / tag
    if (out / "meta.json").exists():
        return
    out.mkdir(parents=True, exist_ok=True)
    from diffusers.pipelines import BlipDiffusionPipeline
    from diffusers.utils import load_image
    pipe = BlipDiffusionPipeline.from_pretrained("Salesforce/blipdiffusion", mean=None, std=None  # as in HW9;
                                                 ).to("cuda")                                    # fp32: Q-Former fails in fp16
    pipe.set_progress_bar_config(disable=True)
    cond = load_image(str(DATA / ref))
    neg = ("over-exposure, under-exposure, saturated, duplicate, out of frame, lowres, cropped, worst quality, "
           "low quality, jpeg artifacts, morbid, mutilated, ugly, bad anatomy, bad proportions, deformed, blurry")
    t0 = time.time()
    for ctx, _ in CONTEXTS:
        for s in SEEDS:
            im = pipe(BLIP_CONTEXTS[ctx], cond, "wood carving", "wood carving", guidance_scale=guidance,
                      num_inference_steps=steps, neg_prompt=neg, height=512, width=512,
                      generator=torch.Generator("cuda").manual_seed(s)).images[0]
            im.save(out / f"{ctx}_{s}.jpg", quality=95)
    meta = {"tag": tag, "ref_image": ref, "guidance": guidance, "steps": steps,
            "sec_per_image": (time.time() - t0) / (len(CONTEXTS) * len(SEEDS))}
    (out / "meta.json").write_text(json.dumps(meta))
    del pipe; torch.cuda.empty_cache()
