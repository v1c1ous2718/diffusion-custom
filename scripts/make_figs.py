"""Figures for the Custom Diffusion report from out/*/scores.json and ckpt/*/log.json."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT, CK, F = ROOT / "out", ROOT / "ckpt", ROOT / "report" / "figs"
F.mkdir(parents=True, exist_ok=True)
C = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "axes.axisbelow": True, "lines.linewidth": 2, "savefig.dpi": 200,
                     "savefig.bbox": "tight", "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
RUNS = [("kv_lr5e-5_noaug", "K/V, lr 5e-5, no resize aug (HW9)"), ("kv_lr5e-5", "K/V, lr 5e-5"),
        ("full_lr5e-5", "Q/K/V/out, lr 5e-5"), ("kv_lr1e-5", "K/V, lr 1e-5")]
STEPS = [50, 100, 200, 400, 800]
CTX = ["snow", "beach", "kitchen", "jungle", "space", "watercolor"]


def sc(tag):
    p = OUT / tag / "scores.json"
    return json.loads(p.read_text()) if p.exists() else None


def save(fig, n):
    fig.savefig(F / n); plt.close(fig); print("wrote", n)


def fig_tradeoff():
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    for i, (r, lab) in enumerate(RUNS):
        pts = [(s, sc(f"{r}_s{s:04d}")) for s in STEPS]
        pts = [(s, p["mean"]) for s, p in pts if p]
        if not pts: continue
        x = [p["clip_t"] for _, p in pts]; y = [p["dino"] for _, p in pts]
        ax.plot(x, y, "-o", color=C[i], label=lab, ms=5)
        for s, p in pts:
            if s in (50, 800):
                ax.annotate(str(s), (p["clip_t"], p["dino"]), textcoords="offset points", xytext=(5, 4), fontsize=7.5,
                            color=INK2)
    for tag, lab, mk in [("sd_baseline", "SD 1.5, text only", "s"), ("blip_diffusion", "BLIP-Diffusion (zero-shot)", "D")]:
        p = sc(tag)
        if p:
            ax.scatter(p["mean"]["clip_t"], p["mean"]["dino"], marker=mk, s=70, color=INK, zorder=4)
            ax.annotate(lab, (p["mean"]["clip_t"], p["mean"]["dino"]), textcoords="offset points", xytext=(7, -3),
                        fontsize=8.5, color=INK)
    ax.set_xlabel("CLIP-T: follows the prompt  →"); ax.set_ylabel("DINO: looks like the real object  →")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.set_title("Subject fidelity vs prompt fidelity (numbers = training steps)", loc="left", fontsize=11)
    save(fig, "tradeoff.png")


def fig_tradeoff_context():
    ca = json.loads((ROOT / "runs" / "context_acc.json").read_text())
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.axvline(1 / 6, color=INK2, ls=":", lw=1.2)
    ax.text(1 / 6 + 0.01, 0.27, "chance (1/6):\nbackground ignores\nthe prompt", fontsize=8, color=INK2)
    for i, (r, lab) in enumerate(RUNS):
        pts = [(s_, ca.get(f"{r}_s{s_:04d}"), sc(f"{r}_s{s_:04d}")) for s_ in STEPS]
        pts = [(s_, a["context_acc"], b["mean"]["dino"]) for s_, a, b in pts if a and b]
        jit = (i - 1.5) * 0.006                          # separate overlapping chance-level points
        ax.plot([x + jit for _, x, _ in pts], [y for _, _, y in pts], "-o", color=C[i], label=lab, ms=5)
        for s_, x, y in pts:
            if s_ in (50, 200, 800) and x > 0.2:
                ax.annotate(str(s_), (x + jit, y), textcoords="offset points", xytext=(5, 3), fontsize=7.5, color=INK2)
    ax.annotate("lr 5e-5 checkpoints from\n100-200 steps on: all at chance", (0.175, 0.645), textcoords="offset points",
                xytext=(30, 25), fontsize=8, color=INK2, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    for tag, lab, mk, off in [("sd_baseline", "SD 1.5, text only", "s", (-10, 6)),
                              ("blip_diffusion", "BLIP-Diffusion (zero-shot)", "D", (8, -12))]:
        if tag in ca and sc(tag):
            x, y = ca[tag]["context_acc"], sc(tag)["mean"]["dino"]
            ax.scatter(x, y, marker=mk, s=70, color=INK, zorder=4)
            ax.annotate(lab, (x, y), textcoords="offset points", xytext=off, fontsize=8.5, color=INK,
                        ha="right" if off[0] < 0 else "left")
    ax.set_xlabel("context accuracy: background matches the prompt (CLIP 6-way)  →")
    ax.set_ylabel("DINO: looks like the real object  →")
    ax.set_xlim(0.1, 0.78); ax.set_ylim(0.22, 0.72); ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.set_title("Fidelity vs editability (numbers = training steps)", loc="left", fontsize=11)
    save(fig, "tradeoff_context.png")


def fig_steps():
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    for i, (r, lab) in enumerate(RUNS):
        pts = [(s, sc(f"{r}_s{s:04d}")) for s in STEPS]
        pts = [(s, p["mean"]) for s, p in pts if p]
        for ax, k in zip(axes, ["dino", "clip_i", "clip_t"]):
            ax.plot([s for s, _ in pts], [p[k] for _, p in pts], "-o", color=C[i], label=lab, ms=4)
    base = sc("sd_baseline")
    for ax, k, t in zip(axes, ["dino", "clip_i", "clip_t"], ["DINO (subject)", "CLIP-I (subject)", "CLIP-T (prompt)"]):
        if base:
            ax.axhline(base["mean"][k], color=INK2, ls=":", lw=1.2)
        ax.set_xscale("log"); ax.set_xticks(STEPS, [str(s) for s in STEPS]); ax.minorticks_off()
        ax.set_title(t, loc="left", fontsize=10); ax.set_xlabel("training steps")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=8.5, loc="upper center", ncol=4, bbox_to_anchor=(0.5, -0.02))
    axes[0].text(55, base["mean"]["dino"] + 0.01, "plain SD", fontsize=8, color=INK2) if base else None
    save(fig, "steps.png")


def fig_guidance():
    gs = [3.0, 5.0, 7.5, 9.5, 12.0, 15.0]
    pts = [(g, sc(f"guidance_{g}")) for g in gs]
    pts = [(g, p) for g, p in pts if p]
    if not pts: return
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.0))
    for ax, k, t, c in [(axes[0], "clip_t", "CLIP-T (prompt)", C[0]), (axes[1], "dino", "DINO (subject)", C[1])]:
        m = [np.mean([r[k] for r in p["rows"]]) for _, p in pts]
        se = [np.std([r[k] for r in p["rows"]], ddof=1) / np.sqrt(len(p["rows"])) for _, p in pts]
        ax.errorbar([g for g, _ in pts], m, yerr=se, fmt="-o", color=c, capsize=3)
        ax.set_xlabel("classifier-free guidance scale"); ax.set_title(t + ", mean ± s.e. (48 images)", loc="left", fontsize=10)
    save(fig, "guidance.png")


def fig_context():
    tags = [("sd_baseline", "SD 1.5 text only"), ("blip_diffusion", "BLIP-Diffusion"),
            ("kv_lr5e-5_noaug_s0200", "K/V 200 steps (HW9)"), ("full_lr5e-5_s0800", "Q/K/V/out 800 steps")]
    tags = [(t, l, sc(t)) for t, l in tags if sc(t)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.2), sharey=False)
    w = 0.8 / len(tags)
    for i, (t, l, s) in enumerate(tags):
        for ax, k in zip(axes, ["clip_t", "dino"]):
            v = [np.mean([r[k] for r in s["rows"] if r["ctx"] == c]) for c in CTX]
            ax.bar(np.arange(len(CTX)) + (i - (len(tags) - 1) / 2) * w, v, w * 0.92, color=C[i], label=l)
    for ax, t in zip(axes, ["CLIP-T by prompt context", "DINO by prompt context"]):
        ax.set_xticks(range(len(CTX)), CTX); ax.set_title(t, loc="left", fontsize=10); ax.grid(axis="x", visible=False)
    axes[0].set_ylim(0.2, 0.4); axes[1].set_ylim(0, 0.8)
    axes[1].legend(frameon=False, fontsize=8, ncol=2)
    save(fig, "by_context.png")


def fig_grid():
    rows = [("sd_baseline", "SD 1.5\ntext only"), ("blip_diffusion", "BLIP-\nDiffusion"),
            ("kv_lr1e-5_s0200", "K/V lr 1e-5\n200 steps"),
            ("kv_lr5e-5_noaug_s0200", "K/V lr 5e-5\n200 steps (HW9)"), ("kv_lr5e-5_s0800", "K/V lr 5e-5\n800 steps"),
            ("full_lr5e-5_s0800", "Q/K/V/out\n800 steps")]
    rows = [r for r in rows if (OUT / r[0] / "meta.json").exists()]  # noqa
    if not rows: return
    S = 160
    fig, axes = plt.subplots(len(rows) + 1, len(CTX), figsize=(len(CTX) * 1.6, (len(rows) + 1) * 1.6))
    refs = sorted((ROOT / "data" / "hw9" / "data" / "custom_diffusion").glob("*.jpg"))
    for k, ax in enumerate(axes[0]):
        ax.axis("off")
        if k < len(refs):
            ax.imshow(Image.open(refs[k]).resize((S, S)))
    axes[0][0].set_title("training images (4)", loc="left", fontsize=8, color=INK2)
    for r, (tag, lab) in enumerate(rows, start=1):
        for c, ctx in enumerate(CTX):
            ax = axes[r][c]; ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values(): sp.set_visible(False)
            ax.imshow(Image.open(OUT / tag / f"{ctx}_0.jpg").resize((S, S)))
            if r == 1: ax.set_title(ctx, fontsize=8)
            if c == 0: ax.set_ylabel(lab, fontsize=8, rotation=0, ha="right", va="center")
    fig.subplots_adjust(wspace=0.03, hspace=0.22)
    save(fig, "grid.png")


def fig_loss():
    fig, ax = plt.subplots(figsize=(5.4, 2.8))
    for i, (r, lab) in enumerate(RUNS):
        p = CK / r / "log.json"
        if not p.exists(): continue
        L = [x["loss"] for x in json.loads(p.read_text())["log"]]
        y = np.convolve(L, np.ones(50) / 50, mode="valid")
        ax.plot(np.arange(len(y)) + 50, y, color=C[i], label=lab, lw=1.5)
    ax.set_xlabel("step"); ax.set_ylabel("masked MSE (50-step mean)")
    ax.legend(frameon=False, fontsize=7.5); ax.set_title("Training loss", loc="left", fontsize=11)
    save(fig, "loss.png")


if __name__ == "__main__":
    fig_tradeoff(); fig_tradeoff_context(); fig_steps(); fig_guidance(); fig_context(); fig_grid(); fig_loss()
