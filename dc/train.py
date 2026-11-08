"""Custom Diffusion fine-tuning of SD 1.5, condensed from the GenAI-ML 2025 HW9 notebook.

Trainable: the new token <new1> embedding, plus cross-attention K/V ("kv") or Q/K/V/out ("full")
of all 16 cross-attention layers. Frozen UNet/VAE run in fp16 under autocast; trainable weights stay fp32.
Checkpoints (attention processors + token embedding) are written at every step in `save_at`.
"""
from __future__ import annotations

import itertools
import json
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import safetensors.torch
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import CustomDiffusionAttnProcessor2_0
from PIL import Image
from transformers import AutoTokenizer, CLIPTextModel

ROOT = Path(__file__).resolve().parents[1]
SD = "stable-diffusion-v1-5/stable-diffusion-v1-5"
DATA = ROOT / "data" / "hw9" / "data" / "custom_diffusion"
TOKEN, INIT_TOKEN = "<new1>", "ktn"
INSTANCE_PROMPT = "photo of a <new1> cat-shaped wood carving with rainbow stripes on its body and tail."


@dataclass
class Cfg:
    name: str
    params: str = "kv"            # kv | full
    lr: float = 5e-5              # per-sample LR, scaled by batch size as in HW9 (effective 1e-4 at bs 2)
    steps: int = 800
    bs: int = 2
    save_at: list = field(default_factory=lambda: [50, 100, 200, 400, 800])
    seed: int = 0
    aug: bool = True              # HW9 used aug=False; the Custom Diffusion paper's default is True


class InstanceDS(torch.utils.data.Dataset):
    """HW9 dataset (instance images only): random flip, optional random rescale + validity mask."""

    def __init__(self, tok, size=512, mask_size=64, aug=True, n_virtual=10_000):
        self.paths = sorted(DATA.glob("*.jpg"))
        self.tok, self.size, self.factor, self.aug, self.n = tok, size, size // mask_size, aug, n_virtual

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        im = Image.open(self.paths[i % len(self.paths)]).convert("RGB")
        if random.random() < 0.5:
            im = im.transpose(Image.FLIP_LEFT_RIGHT)
        s = self.size
        scale = s
        if self.aug:
            scale = np.random.randint(s // 3, s + 1) if np.random.uniform() < 0.66 else np.random.randint(int(1.2 * s), int(1.4 * s))
        im = np.asarray(im.resize((scale, scale), Image.BILINEAR), dtype=np.float32) / 127.5 - 1
        out = np.zeros((s, s, 3), np.float32)
        mask = np.zeros((s // self.factor, s // self.factor), np.float32)
        prompt = INSTANCE_PROMPT
        if scale > s:
            t, l = np.random.randint(0, scale - s + 1, 2)
            out = im[t:t + s, l:l + s]; mask[:] = 1
            prompt = random.choice(["zoomed in ", "close up "]) + prompt
        else:
            t, l = np.random.randint(0, s - scale + 1, 2)
            out[t:t + scale, l:l + scale] = im
            f = self.factor
            mask[t // f + 1:(t + scale) // f - 1, l // f + 1:(l + scale) // f - 1] = 1
            if scale < 0.6 * s:
                prompt = random.choice(["a far away ", "very small "]) + prompt
        ids = self.tok(prompt, truncation=True, padding="max_length", max_length=self.tok.model_max_length,
                       return_tensors="pt").input_ids[0]
        return torch.from_numpy(out).permute(2, 0, 1), torch.from_numpy(mask)[None], ids


def setup_custom_attn(unet, train_q_out: bool):
    st = unet.state_dict()
    procs = {}
    for name in unet.attn_processors:
        cross = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            hidden = list(reversed(unet.config.block_out_channels))[int(name[len("up_blocks.")])]
        else:
            hidden = unet.config.block_out_channels[int(name[len("down_blocks.")])]
        layer = name.split(".processor")[0]
        if cross is not None:
            w = {"to_k_custom_diffusion.weight": st[layer + ".to_k.weight"],
                 "to_v_custom_diffusion.weight": st[layer + ".to_v.weight"]}
            if train_q_out:
                w["to_q_custom_diffusion.weight"] = st[layer + ".to_q.weight"]
                w["to_out_custom_diffusion.0.weight"] = st[layer + ".to_out.0.weight"]
                w["to_out_custom_diffusion.0.bias"] = st[layer + ".to_out.0.bias"]
            p = CustomDiffusionAttnProcessor2_0(train_kv=True, train_q_out=train_q_out, hidden_size=hidden,
                                                cross_attention_dim=cross)
            p.load_state_dict({k: v.float() for k, v in w.items()})
            procs[name] = p.to(unet.device, torch.float32)
        else:
            procs[name] = CustomDiffusionAttnProcessor2_0(train_kv=False, train_q_out=False, hidden_size=hidden,
                                                          cross_attention_dim=None)
    unet.set_attn_processor(procs)
    return AttnProcsLayers(unet.attn_processors)


def train(cfg: Cfg) -> dict:
    out = ROOT / "ckpt" / cfg.name
    if (out / "log.json").exists():
        return json.loads((out / "log.json").read_text())
    out.mkdir(parents=True, exist_ok=True)
    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(SD, subfolder="tokenizer", use_fast=False)
    sched = DDPMScheduler.from_pretrained(SD, subfolder="scheduler")
    te = CLIPTextModel.from_pretrained(SD, subfolder="text_encoder", variant="fp16", torch_dtype=torch.float32).to(dev)
    vae = AutoencoderKL.from_pretrained(SD, subfolder="vae", variant="fp16", torch_dtype=torch.float16).to(dev)
    unet = UNet2DConditionModel.from_pretrained(SD, subfolder="unet", variant="fp16", torch_dtype=torch.float16).to(dev)
    unet.enable_gradient_checkpointing()

    tok.add_tokens(TOKEN)
    tid, iid = tok.convert_tokens_to_ids(TOKEN), tok.encode([INIT_TOKEN], add_special_tokens=False)[0]
    te.resize_token_embeddings(len(tok))
    emb = te.get_input_embeddings().weight
    emb.data[tid] = emb.data[iid]
    te.requires_grad_(False); emb.requires_grad_(True)
    emb_orig = emb.data.clone()
    keep = torch.arange(len(tok), device=dev) != tid
    vae.requires_grad_(False); unet.requires_grad_(False)
    layers = setup_custom_attn(unet, train_q_out=(cfg.params == "full"))
    n_train = sum(p.numel() for p in layers.parameters()) + emb.shape[1]

    lr = cfg.lr * cfg.bs
    opt = torch.optim.AdamW(itertools.chain([emb], layers.parameters()), lr=lr, betas=(0.9, 0.999),
                            weight_decay=1e-2, eps=1e-8)
    scaler = torch.amp.GradScaler()
    dl = torch.utils.data.DataLoader(InstanceDS(tok, aug=cfg.aug), batch_size=cfg.bs, shuffle=True, num_workers=2,
                                     drop_last=True)
    log, t0 = [], time.time()
    it = iter(dl)
    for step in range(1, cfg.steps + 1):
        x, mask, ids = (t.to(dev) for t in next(it))
        with torch.no_grad():
            lat = vae.encode(x.half()).latent_dist.sample() * vae.config.scaling_factor
        noise = torch.randn_like(lat)
        t = torch.randint(0, sched.config.num_train_timesteps, (lat.shape[0],), device=dev)
        noisy = sched.add_noise(lat, noise, t)
        with torch.autocast("cuda", dtype=torch.float16):
            cond = te(ids)[0]
            pred = unet(noisy, t, cond.half()).sample
        loss = F.mse_loss(pred.float(), noise.float(), reduction="none")
        loss = ((loss * mask).sum([1, 2, 3]) / mask.sum([1, 2, 3])).mean()
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        emb.grad[keep] = 0                                           # only the new token's row learns
        torch.nn.utils.clip_grad_norm_(itertools.chain([emb], layers.parameters()), 1.0)
        scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
        with torch.no_grad():                                        # AdamW weight decay would shrink every row;
            emb.data[keep] = emb_orig[keep]                          # restore all rows except <new1>
        log.append({"step": step, "loss": loss.item(), "time": time.time() - t0})
        if step % 50 == 0:
            print(f"[{cfg.name}] step {step} loss {np.mean([l['loss'] for l in log[-50:]]):.4f} ({time.time()-t0:.0f}s)",
                  flush=True)
        if step in cfg.save_at:
            d = out / f"step_{step:04d}"
            d.mkdir(exist_ok=True)
            unet.save_attn_procs(d, weight_name="pytorch_custom_diffusion_weights.safetensors", safe_serialization=True)
            safetensors.torch.save_file({TOKEN: emb[tid].detach().float().cpu()}, str(d / f"{TOKEN}.safetensors"))
    res = {"cfg": asdict(cfg), "trainable_params": n_train, "log": log,
           "train_time_s": time.time() - t0, "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9}
    (out / "log.json").write_text(json.dumps(res))
    return res
