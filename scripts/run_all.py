"""Full diffusion experiment: train 4 configs, generate for every saved step + baselines, guidance sweep, score."""
import json
import sys

from dc.gen import generate_blip, generate_sd
from dc.train import ROOT, Cfg, train

RUNS = [
    Cfg("kv_lr5e-5_noaug", params="kv", lr=5e-5, aug=False),   # HW9 notebook setting
    Cfg("kv_lr5e-5", params="kv", lr=5e-5),                    # + paper's resize augmentation
    Cfg("full_lr5e-5", params="full", lr=5e-5),                # + train Q and out projection too
    Cfg("kv_lr1e-5", params="kv", lr=1e-5),                    # lower LR
]
GUIDANCE = [3.0, 5.0, 7.5, 9.5, 12.0, 15.0]
GUIDANCE_CKPT = ("kv_lr5e-5_noaug", 200)                       # HW9 default: kv, 200 steps


def main(stages):
    if "train" in stages:
        for c in RUNS:
            r = train(c)
            print(f"trained {c.name}: {r['trainable_params']/1e6:.2f}M params, {r['train_time_s']:.0f}s, "
                  f"peak {r['peak_mem_gb']:.1f} GB", flush=True)
    if "gen" in stages:
        generate_sd("sd_baseline", None)
        for c in RUNS:
            for s in c.save_at:
                generate_sd(f"{c.name}_s{s:04d}", ROOT / "ckpt" / c.name / f"step_{s:04d}")
                print("generated", c.name, s, flush=True)
        generate_blip()
        name, s = GUIDANCE_CKPT
        for g in GUIDANCE:
            generate_sd(f"guidance_{g}", ROOT / "ckpt" / name / f"step_{s:04d}", guidance=g)
        print("generation done", flush=True)
    if "score" in stages:
        from dc.metrics import Scorer
        from dc.gen import OUT
        sc = Scorer()
        summary = {d.name: sc.score_dir(d.name)["mean"] for d in sorted(OUT.iterdir()) if (d / "meta.json").exists()}
        (ROOT / "runs").mkdir(exist_ok=True)
        (ROOT / "runs" / "summary.json").write_text(json.dumps(summary, indent=1))
        for k, v in summary.items():
            print(f"{k:28s} CLIP-T {v['clip_t']:.3f}  CLIP-I {v['clip_i']:.3f}  DINO {v['dino']:.3f}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["train", "gen", "score"])
