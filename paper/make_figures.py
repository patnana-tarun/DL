"""IEEE-sized publication figures for the paper (vector PDF in paper/figures/).

Single column = 3.5 in wide, full width = 7.16 in; 8 pt Times-like fonts; one palette across figures
(YOLO26n blue #2a78d6, RF-DETR orange #eb6834; pretrained = dashed / hatched, so figures stay readable in
grayscale). A figure whose source data is missing is skipped (main.tex guards each one with \\IfFileExists),
and a stale PDF from an earlier build is removed so the paper never shows a figure without its data.

    env -u PYTHONPATH .venv/bin/python paper/make_figures.py
"""
import gzip  # noqa: F401  (kept for optional raw-prediction figures)
import json
import shutil
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

PAPER = Path(__file__).resolve().parent
ROOT = PAPER.parent
R = ROOT / "results"
FIG = PAPER / "figures"
FIG.mkdir(exist_ok=True)
sys.path.insert(0, str(PAPER))
import make_numbers as mn  # noqa: E402  (shared JSON helpers)

SC, FW = 3.5, 7.16
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID = "#1a1a1a", "#555555", "#dddddd"
GREYS = ["#2f2f2f", "#6b6b6b", "#a3a3a3", "#d4d4d4"]
COL = {"YOLO26n": BLUE, "RF-DETR Nano": ORANGE}
MODELS = ["YOLO26n (pretrained)", "YOLO26n (fine-tuned)", "RF-DETR Nano (pretrained)", "RF-DETR Nano (fine-tuned)"]
SHORT = {"YOLO26n (pretrained)": "YOLO26n pre.", "YOLO26n (fine-tuned)": "YOLO26n FT",
         "RF-DETR Nano (pretrained)": "RF-DETR-N pre.", "RF-DETR Nano (fine-tuned)": "RF-DETR-N FT"}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Nimbus Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "legend.fontsize": 7, "axes.linewidth": 0.5, "xtick.major.width": 0.5,
    "ytick.major.width": 0.5, "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.4, "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "lines.linewidth": 1.1, "pdf.fonttype": 42, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02, "axes.titlelocation": "left", "axes.titleweight": "bold",
})

MADE, SKIPPED = [], []


def family(name):
    return "YOLO26n" if name.startswith("YOLO") else "RF-DETR Nano"


def is_ft(name):
    return "fine-tuned" in name


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    plt.close(fig)
    MADE.append(name)


def skip(name, why):
    for ext in (".pdf", ".png", ".jpg"):
        p = FIG / f"{name}{ext}"
        if p.exists():
            p.unlink()
    SKIPPED.append(f"{name} ({why})")


# ----------------------------------------------------------------------------- 1 architecture
def fig_architecture():
    fig = plt.figure(figsize=(FW, 3.05))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, FW); ax.set_ylim(0, 3.05); ax.axis("off")

    def box(x, y, w, h, title, sub="", fc="#ffffff", ec=INK2, lw=0.7, bold=True, fs=7.2):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.0,rounding_size=0.06", fc=fc, ec=ec, lw=lw))
        if sub:
            ax.text(x + w / 2, y + h * 0.66, title, ha="center", va="center", fontsize=fs,
                    fontweight="bold" if bold else "normal", color=INK)
            ax.text(x + w / 2, y + h * 0.30, sub, ha="center", va="center", fontsize=fs - 0.9, color=INK2,
                    linespacing=1.0)
        else:
            ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=fs,
                    fontweight="bold" if bold else "normal", color=INK)
        return (x, y, w, h)

    def arrow(p, q, color=INK, ls="-", lw=0.8, rad=0.0, both=False, ms=7):
        a = FancyArrowPatch(p, q, arrowstyle="<|-|>" if both else "-|>", mutation_scale=ms, lw=lw, color=color,
                            linestyle=ls, connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0)
        ax.add_patch(a)

    def label(x, y, s, ha="center", color=INK2, fs=6.3, style="italic", **kw):
        ax.text(x, y, s, ha=ha, va="center", fontsize=fs, color=color, style=style, **kw)

    # regions
    ax.add_patch(FancyBboxPatch((0.06, 1.17), FW - 0.12, 1.82, boxstyle="round,pad=0,rounding_size=0.08",
                                fc="#f4f7fb", ec="#9db7d9", lw=0.7, ls="--"))
    ax.text(0.14, 2.88, "EDGE RUNTIME (edgevision + llama.cpp server). Target: NVIDIA Jetson Orin; every "
            "measurement in this paper: desktop RTX 4060 Laptop", fontsize=6.6, color="#2b5c9e", fontweight="bold",
            va="center")
    ax.add_patch(FancyBboxPatch((0.06, 0.05), FW - 0.12, 0.86, boxstyle="round,pad=0,rounding_size=0.08",
                                fc="#faf6f2", ec="#d9b99d", lw=0.7, ls="--"))
    ax.text(0.14, 0.80, "OFFLINE RESEARCH LOOP (workstation)", fontsize=6.6, color="#9e5a2b",
            fontweight="bold", va="center")

    # on-device main row
    W, H, Y = 1.0, 0.52, 2.15
    xs = [0.16 + k * 1.175 for k in range(6)]
    b_cam = box(xs[0], Y, W, H, "Camera / stream", "RTSP, USB (V4L2),\nCSI, file")
    b_dec = box(xs[1], Y, W, H, "Decoder", "FFmpeg / GStreamer\nsubprocess pipe")
    b_det = box(xs[2], Y, W, H, "YOLO26n", "ONNX Runtime\nTensorRT FP16", fc="#e3eefb", ec=BLUE, lw=0.9)
    b_trk = box(xs[3], Y, W, H, "ByteTrack", "tuned params,\ntimestamped motion")
    b_ana = box(xs[4], Y, W, H, "Analytics + alerts", "occupancy, unique,\ndwell, line, rules")
    b_db = box(xs[5], Y, W, H, "SQLite store", "events, time series,\nreports, chat")
    yc = Y + H / 2
    for a, b, s in ((b_cam, b_dec, "bytes"), (b_dec, b_det, "BGR frame"), (b_det, b_trk, "boxes"),
                    (b_trk, b_ana, "tracks"), (b_ana, b_db, "events")):
        arrow((a[0] + W, yc), (b[0], yc))

    # VLM row
    Y2, H2 = 1.30, 0.52
    b_vlm = box(xs[4], Y2, W, H2, "Qwen3-VL-2B", "llama.cpp server\n(OpenAI API, GGUF)", fc="#fdf0e6", ec=ORANGE, lw=0.9)
    b_ui = box(xs[5], Y2, W, H2, "Operator web UI", "live video, chat,\n/commands (no VLM)")
    xv = xs[4] + W / 2
    arrow((xv - 0.22, Y), (xv - 0.22, Y2 + H2))
    label(xv - 0.26, (Y + Y2 + H2) / 2, "numbers +\nframe", ha="right", fs=5.6)
    arrow((xv + 0.22, Y2 + H2), (xv + 0.22, Y), color=ORANGE, lw=0.9)
    label(xv + 0.26, (Y + Y2 + H2) / 2, "tool calls via\npolicy layer", ha="left", color="#b04a17", fs=5.6)
    arrow((xs[4] + W, Y2 + H2 / 2), (xs[5], Y2 + H2 / 2), both=True)
    label((xs[4] + W + xs[5]) / 2, Y2 + H2 / 2 + 0.09, "chat", fs=5.6)
    arrow((xs[5] + W / 2, Y), (xs[5] + W / 2, Y2 + H2))
    label(xs[5] + W / 2 + 0.04, (Y + Y2 + H2) / 2, "history", ha="left", fs=5.6)
    # notes in the free space of the on-device box
    ax.text(xs[0], 1.80, "Closed loop", fontsize=6.6, fontweight="bold", color=INK, va="top")
    ax.text(xs[0], 1.66,
            "observe: periodic scene report; on alert: report, rule decided by numbers\n"
            "ask: (1) text-only tool routing, never shown VLM-written text; (2) image answer\n"
            "guardrails: intent check, rule fallback, truthful [system] footer\n"
            "policy layer decides every call: RBAC, validation, intent gate,\n"
            "/confirm for weakening alerts, rate limits, hash-chained audit", fontsize=5.6, color=INK2,
            va="top", linespacing=1.18)
    ax.text(xs[3] - 0.05, 1.80, "Stream semantics", fontsize=6.6, fontweight="bold", color=INK, va="top")
    ax.text(xs[3] - 0.05, 1.66,
            "live: newest frame only,\nstale frames dropped\n"
            "file: every frame, deterministic\n"
            "analysis rate: fps or stride", fontsize=5.8, color=INK2, va="top", linespacing=1.18)

    # offline row
    Yo, Ho, Wo = 0.16, 0.50, 0.98
    xo = [0.16 + k * 1.175 for k in range(6)]
    o = [box(xo[0], Yo, Wo, Ho, "Teacher ensemble", "RF-DETR-L + YOLO26x\nTTA, WBF, score 0.40", fs=6.9),
         box(xo[1], Yo, Wo, Ho, "Pseudo-labels", "1,082 frames,\ntemporal splits", fs=6.9),
         box(xo[2], Yo, Wo, Ho, "Train students", "YOLO26n @1280\nRF-DETR-N @640", fs=6.9),
         box(xo[3], Yo, Wo, Ho, "Evaluate", "AP+CIs, bias, latency,\nHOTA (6 trackers)", fs=6.9),
         box(xo[4], Yo, Wo, Ho, "Select", "decision matrix +\nOOD validation", fs=6.9),
         box(xo[5], Yo, Wo, Ho, "Export ONNX", "3 input sizes,\nparity check", fs=6.9)]
    for a, b in zip(o[:-1], o[1:]):
        arrow((a[0] + Wo, Yo + Ho / 2), (b[0], Yo + Ho / 2))
    # deploy arrow: export -> up -> left -> detector
    xe, xd = xo[5] + Wo / 2, xs[2] + W / 2
    ax.plot([xe, xe, xd], [Yo + Ho, 1.04, 1.04], color=BLUE, lw=0.9)
    arrow((xd, 1.04), (xd, Y), color=BLUE, lw=0.9)
    label((xd + xe) / 2, 0.97, "deploy selected model (ONNX \u2192 TensorRT engine)", color="#1f5aa6", fs=5.8)
    save(fig, "fig_architecture")


# ----------------------------------------------------------------------------- 2 dataset
def fig_dataset():
    st = mn.load("datasets/stats.json")
    coco = {s: ROOT / f"datasets/coco/{s}/_annotations.coco.json" for s in ("train", "valid", "test")}
    if st is None or not all(p.exists() for p in coco.values()):
        return skip("fig_dataset", "datasets/stats.json or COCO annotations missing")
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(SC, 1.95), gridspec_kw={"height_ratios": [0.6, 1.5], "hspace": 0.95})
    segs = [(0, 15, "tune", GREYS[3], ""), (15, 30, "held", GREYS[2], ""),
            (30, 210, "train", "#c9d9ee", f"train {st['train']['images']} fr. / {st['train']['boxes']:,} boxes"),
            (210, 255, "valid", "#9fbde3", "valid\n" + f"{st['valid']['images']}/{st['valid']['boxes']:,}"),
            (255, 300, "test", "#5f8fcf", "test\n" + f"{st['test']['images']}/{st['test']['boxes']:,}")]
    for lo, hi, _, c, t in segs:
        a1.barh(0, hi - lo, left=lo, height=0.8, color=c, edgecolor="white", lw=0.6)
        a1.text((lo + hi) / 2, 0, t, ha="center", va="center", fontsize=5.6 if hi - lo < 50 else 6.4,
                color=INK if c != "#5f8fcf" else "white", linespacing=0.95)
    a1.text(15, 0, "MOT\n0-30", ha="center", va="center", fontsize=5.6, color=INK, linespacing=0.95)
    a1.set_xlim(0, 300); a1.set_ylim(-0.5, 0.5); a1.set_yticks([]); a1.grid(False)
    a1.spines["left"].set_visible(False)
    a1.set_xticks([0, 30, 60, 90, 120, 150, 180, 210, 255, 300])
    a1.tick_params(axis="x", labelsize=6)
    a1.set_xlabel("time in aa1.mp4 (s)", fontsize=7, labelpad=1)
    a1.set_title("(a) temporal splits (one frame every 0.25 s)", fontsize=7.5)
    bins = np.logspace(np.log10(8), np.log10(720), 36)
    for s, c, ls in (("train", "#8fb0dc", "-"), ("valid", "#4a7fc4", "--"), ("test", "#16407a", "-")):
        A = json.loads(coco[s].read_text())["annotations"]
        h = np.array([a["bbox"][3] for a in A])
        a2.hist(h, bins=bins, histtype="step", density=True, color=c, lw=1.0, ls=ls,
                label=f"{s} (median {np.median(h):.0f} px)")
    a2.axvline(32, color=INK2, lw=0.5, ls=":")
    a2.text(33, a2.get_ylim()[1] * 0.92, "32 px", fontsize=6, color=INK2)
    a2.set_xscale("log"); a2.set_xlim(15, 720)
    a2.set_xticks([20, 50, 100, 200, 500]); a2.set_xticklabels(["20", "50", "100", "200", "500"])
    a2.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    a2.set_xlabel("pseudo-label box height (px, 720p frame)"); a2.set_ylabel("density")
    a2.legend(loc="upper right")
    a2.set_title("(b) person box heights", fontsize=7.5)
    save(fig, "fig_dataset")


# ----------------------------------------------------------------------------- 3 PR curves
def fig_pr():
    p = R / "pr_data.npz"
    if not p.exists():
        return skip("fig_pr", "results/pr_data.npz missing")
    z = np.load(p)
    m = mn.load("results/metrics.json") or {}
    rec = np.linspace(0, 1, 101)
    fig, axs = plt.subplots(1, 2, figsize=(SC, 1.6), sharey=True, gridspec_kw={"wspace": 0.12})
    for ax, suf, ttl, met in ((axs[0], "coco_pr50", "(a) IoU 0.50", "AP50"),
                              (axs[1], "coco_pr5095", "(b) mean over IoU 0.50:0.95", "AP50-95")):
        for name in MODELS:
            k = name.replace(" ", "_").replace("(", "").replace(")", "") + "__" + suf
            if k not in z.files:
                continue
            pr = z[k].astype(float)
            pr[pr < 0] = np.nan
            v = mn.get(m, name, met)
            ax.plot(rec, pr, color=COL[family(name)], ls="-" if is_ft(name) else "--",
                    lw=1.2 if is_ft(name) else 0.9, label=SHORT[name])
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02); ax.set_xlabel("recall"); ax.set_title(ttl, fontsize=7.5)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1]); ax.set_xticklabels(["0", ".25", ".5", ".75", "1"])
    axs[0].set_ylabel("precision")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, bbox_to_anchor=(0.52, -0.1), fontsize=6, handlelength=1.8,
               columnspacing=1.0)
    save(fig, "fig_pr")


# ----------------------------------------------------------------------------- 4 AP with CIs + generalisation
def fig_ap():
    m, s = mn.load("results/metrics.json"), mn.load("results/metrics_splits.json")
    if m is None or s is None:
        return skip("fig_ap", "metrics.json / metrics_splits.json missing")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(SC, 1.6), gridspec_kw={"width_ratios": [1.15, 1], "wspace": 0.55})
    for j, name in enumerate(MODELS[::-1]):
        v, (lo, hi) = m[name]["AP50-95"], m[name]["AP50-95_CI95"]
        c = COL[family(name)]
        a1.errorbar(v, j, xerr=[[v - lo], [hi - v]], fmt="o" if is_ft(name) else "s", color=c, ms=3.2,
                    mfc=c if is_ft(name) else "white", capsize=1.8, lw=0.8)
        a1.text(hi + 0.025, j, f"{v:.3f}", va="center", fontsize=6)
    a1.set_yticks(range(4)); a1.set_yticklabels([SHORT[n] for n in MODELS[::-1]], fontsize=6.5)
    a1.set_xlim(0.33, 0.92); a1.set_ylim(-0.6, 3.6)
    a1.set_xlabel("test AP$_{50:95}$ (95% CI)")
    a1.set_title("(a) test split", fontsize=7.5)
    sp = ["train", "valid", "test"]
    for name in MODELS:
        c = COL[family(name)]
        a2.plot(range(3), [s[name][x]["AP50-95"] for x in sp], color=c, ls="-" if is_ft(name) else "--",
                marker="o" if is_ft(name) else "s", ms=2.8, mfc=c if is_ft(name) else "white", lw=0.9)
    a2.set_xticks(range(3)); a2.set_xticklabels(["train\n30-210 s", "valid\n210-255 s", "test\n255-300 s"],
                                                fontsize=6)
    a2.set_ylabel("AP$_{50:95}$"); a2.set_xlim(-0.25, 2.25); a2.set_ylim(0.3, 1.0)
    a2.set_title("(b) per split", fontsize=7.5)
    save(fig, "fig_ap")


# ----------------------------------------------------------------------------- 5 latency vs AP
def fig_latency():
    L = mn.load("results/latency.json")
    if L is None:
        return skip("fig_latency", "latency.json missing")
    short = {"Ultralytics predict FP32 (baseline)": "predict FP32", "Ultralytics predict FP16": "predict FP16",
             "NMS-free head FP16": "NMS-free FP16", "NMS-free + GPU pre-proc + fused FP16": "NMS-free fast",
             "GPU pre-proc + fused FP16 + GPU NMS (deployed)": "deployed",
             "predict FP32, unoptimized (baseline)": "predict FP32", "FP32 + TorchScript": "FP32 TS",
             "FP16, no compile": "FP16", "FP16 + TorchScript": "FP16 TS",
             "FP16 + TorchScript + GPU pre-proc (deployed)": "deployed"}
    fig, ax = plt.subplots(figsize=(SC, 1.6))
    offs = {"predict FP32": (4, 3, "left"), "predict FP16": (4, -2, "left"), "NMS-free FP16": (4, -6, "left"),
            "NMS-free fast": (-4, -3, "right"), "deployed": (-3, 5, "left"), "FP32 TS": (-3, 5, "right"),
            "FP16": (2, 5, "left"), "FP16 TS": (-8, -9, "left")}
    for k, d in L.items():
        fam, var = [t.strip() for t in k.split("|", 1)]
        c = COL[fam]
        dep = "deployed" in var
        base = "baseline" in var
        ax.scatter(d["median_ms"], d["AP50-95"], s=26 if dep else 14, marker="*" if dep else ("o" if not base else "D"),
                   color=c if not base else "white", edgecolor=c, lw=0.8, zorder=3)
        sv = short.get(var, var)
        dx, dy, ha = offs.get(sv, (3, 3, "left"))
        ax.annotate(sv, (d["median_ms"], d["AP50-95"]), xytext=(dx, dy), textcoords="offset points", fontsize=5.6,
                    color=INK2, ha=ha)
    for fam in COL:
        pts = sorted((d["median_ms"], d["AP50-95"]) for k, d in L.items() if k.startswith(fam))
        if pts:
            ax.plot(*zip(*[pts[0], pts[-1]]), color=COL[fam], lw=0.0)
    ax.set_xscale("log"); ax.set_xticks([3, 5, 10, 20]); ax.set_xticklabels(["3", "5", "10", "20"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlim(2.3, 30); ax.set_ylim(0.70, 0.845)
    ax.set_xlabel("median latency per 1280x720 frame (ms, RTX 4060 Laptop, batch 1, log scale)")
    ax.set_ylabel("test AP$_{50:95}$")
    h = [plt.Line2D([], [], color=BLUE, marker="o", ls="", ms=3.5, label="YOLO26n"),
         plt.Line2D([], [], color=ORANGE, marker="o", ls="", ms=3.5, label="RF-DETR Nano"),
         plt.Line2D([], [], color=INK2, marker="D", mfc="white", ls="", ms=3, label="stock baseline"),
         plt.Line2D([], [], color=INK2, marker="*", ls="", ms=5, label="deployed path")]
    ax.legend(handles=h, loc="center right", fontsize=6, ncol=1)
    save(fig, "fig_latency")


# ----------------------------------------------------------------------------- 6 tracking
def fig_tracking():
    C = mn.load("results/tracking/tracker_comparison.json")
    if C is None:
        return skip("fig_tracking", "tracker_comparison.json missing")
    trackers = ["SORT", "ByteTrack", "OC-SORT", "C-BIoU", "BoT-SORT", "McByte"]
    fig, axs = plt.subplots(1, 2, figsize=(SC, 1.45), sharey=True, gridspec_kw={"wspace": 0.1})
    for ax, gt, ttl in ((axs[0], "ByteTrack-GT", "(a) GT linked by ByteTrack"),
                        (axs[1], "OC-SORT-GT", "(b) GT linked by OC-SORT")):
        allv = []
        for j, t in enumerate(trackers[::-1]):
            vs = []
            for det, dy, mk in (("YOLO26n", 0.13, "o"), ("RF-DETR Nano", -0.13, "s")):
                v = mn.get(C, f"{det} | {t}", gt, "heldout_15-30s", "HOTA")
                if v is None:
                    continue
                vs.append(v); allv.append(v)
                ax.plot(v, j + dy, marker=mk, color=COL[det], ms=3.4, ls="")
            if len(vs) == 2:
                ax.plot(vs, [j + 0.13, j - 0.13], color=GREYS[2], lw=0.5, zorder=0)
        if allv:
            lo, hi = min(allv), max(allv)
            pad = max(0.01, (hi - lo) * 0.25)
            ax.set_xlim(lo - pad, hi + pad)
        ax.set_title(ttl, fontsize=7.5); ax.set_xlabel("held-out HOTA (15-30 s)")
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
    axs[0].set_yticks(range(len(trackers))); axs[0].set_yticklabels(trackers[::-1])
    axs[0].set_ylim(-0.6, len(trackers) - 0.4)
    h = [plt.Line2D([], [], color=BLUE, marker="o", ls="", ms=3.4, label="YOLO26n"),
         plt.Line2D([], [], color=ORANGE, marker="s", ls="", ms=3.4, label="RF-DETR Nano")]
    fig.legend(handles=h, loc="upper center", ncol=2, bbox_to_anchor=(0.55, -0.1), fontsize=6.5)
    save(fig, "fig_tracking")


# ----------------------------------------------------------------------------- 7 model selection
def fig_selection():
    S = mn.load("results/model_selection.json")
    if S is None:
        return skip("fig_selection", "model_selection.json missing")
    defs, vals = S["criteria"]["definitions"], S["criteria"]["values"]
    lab = {"test_AP50-95": "test AP (fused labels)", "label_bias_min_AP": "AP, worst single teacher",
           "tracking_mean_HOTA": "mean HOTA (12 runs)", "latency_ms": "latency (deployed)",
           "GFLOPs": "GFLOPs", "params_M": "parameters"}
    crit = list(defs)
    fig, ax = plt.subplots(figsize=(SC, 1.3))
    dets = ["YOLO26n", "RF-DETR Nano"]
    for j, c in enumerate(crit[::-1] + ["_score"]):
        for det, dy, hatch in (("YOLO26n", 0.18, None), ("RF-DETR Nano", -0.18, "////")):
            if c == "_score":
                v = S["weighted_score"][det]
            else:
                xs = [vals[d][c] for d in dets]
                v = vals[det][c] / max(xs) if defs[c]["higher_is_better"] else min(xs) / vals[det][c]
            ax.barh(j + dy, v, height=0.34, color=COL[det] if c == "_score" else "white", edgecolor=COL[det],
                    hatch=None if c == "_score" else hatch, lw=0.8)
            if det != "YOLO26n" or c == "_score":
                ax.text(v + 0.01, j + dy, f"{v:.2f}", va="center", fontsize=5.6, color=INK2)
    ax.set_yticks(range(len(crit) + 1))
    ax.set_yticklabels([f"{lab[c]} (w={defs[c]['weight']:.2f})" for c in crit[::-1]] + ["weighted score"],
                       fontsize=6.3)
    ax.get_yticklabels()[-1].set_fontweight("bold")
    ax.axhline(len(crit) - 0.5, color=INK2, lw=0.5)
    ax.set_xlim(0, 1.12); ax.set_xlabel("score relative to the best candidate (1 = best)")
    h = [matplotlib.patches.Patch(fc="white", ec=BLUE, label="YOLO26n"),
         matplotlib.patches.Patch(fc="white", ec=ORANGE, hatch="////", label="RF-DETR Nano")]
    fig.legend(handles=h, loc="upper center", ncol=2, fontsize=6, bbox_to_anchor=(0.6, -0.09))
    save(fig, "fig_selection")


# ----------------------------------------------------------------------------- 8 qualitative (raster reuse)
def fig_qualitative():
    src = R / "figures/06_qualitative_test.png"
    if not src.exists():
        return skip("fig_qualitative", "results/figures/06_qualitative_test.png missing")
    shutil.copyfile(src, FIG / "fig_qualitative.png")
    MADE.append("fig_qualitative (png copy)")


# ----------------------------------------------------------------------------- 9 campus
def fig_demo():
    src = R / "edge/demo/snapshot.jpg"
    if not src.exists():
        return skip("fig_demo", "results/edge/demo/snapshot.jpg missing")
    shutil.copyfile(src, FIG / "fig_demo.jpg")
    MADE.append("fig_demo (jpg copy)")


def fig_campus():
    S = {v: mn.load(f"results/campus/{v}/summary.json") for v in ("video_1", "video_2")}
    if not all(S.values()):
        return skip("fig_campus", "campus summary.json missing")
    fig, axs = plt.subplots(1, 3, figsize=(SC, 1.25), gridspec_kw={"wspace": 0.35, "width_ratios": [1.25, 1, 1]})
    vids = ["video_2", "video_1"]
    ylab = ["video 2", "video 1"]
    # (a) counting funnel
    ax = axs[0]
    for j, v in enumerate(vids):
        s = S[v]
        for k, (key, c) in enumerate((("raw_track_ids", GREYS[3]), ("tracks_kept", GREYS[2]), ("unique_people", GREYS[0]))):
            ax.barh(j + (1 - k) * 0.26, s[key], height=0.24, color=c)
            ax.text(s[key] + 8, j + (1 - k) * 0.26, str(s[key]) + (f" {['IDs', 'kept', 'people'][k]}" if j == 1 else ""),
                    va="center", fontsize=5.4)
    ax.set_yticks(range(2)); ax.set_yticklabels(ylab); ax.set_xlim(0, 950)
    ax.set_title("(a) IDs$\\to$tracks$\\to$people", fontsize=6.8)
    ax.tick_params(axis="x", labelsize=6)
    for ax, key, cats, ttl in ((axs[1], "gender", ["male", "female", "unsure"], "(b) gender"),
                               (axs[2], "dress", ["uniform", "non_uniform", "unsure"], "(c) dress")):
        for j, v in enumerate(vids):
            d = S[v][key]; tot = sum(d.get(c, 0) for c in cats); left = 0
            for c, col, hatch in zip(cats, [GREYS[0], GREYS[2], "white"], [None, None, "....."]):
                w = d.get(c, 0) / tot
                ax.barh(j, w, left=left, height=0.55, color=col, edgecolor=INK2, lw=0.4, hatch=hatch)
                if w > 0.12:
                    ax.text(left + w / 2, j, f"{d.get(c, 0)}", ha="center", va="center", fontsize=5.4,
                            color="white" if col == GREYS[0] else INK)
                left += w
        ax.set_yticks(range(2)); ax.set_yticklabels([]); ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.5, 1]); ax.set_xticklabels(["0", ".5", "1"], fontsize=6)
        ax.set_title(ttl, fontsize=6.8); ax.grid(False)
    h = [matplotlib.patches.Patch(fc=GREYS[0], label="male | uniform"),
         matplotlib.patches.Patch(fc=GREYS[2], label="female | non-uniform"),
         matplotlib.patches.Patch(fc="white", ec=INK2, hatch=".....", label="unsure (abstain)")]
    hb = [matplotlib.patches.Patch(fc=GREYS[3], label="raw track IDs"),
          matplotlib.patches.Patch(fc=GREYS[2], label="kept tracks"),
          matplotlib.patches.Patch(fc=GREYS[0], label="people (stitched)")]
    fig.legend(handles=h, loc="lower center", ncol=3, bbox_to_anchor=(0.62, -0.2), fontsize=5.6,
               columnspacing=0.8, handlelength=1.2)
    save(fig, "fig_campus")


# ----------------------------------------------------------------------------- 10 edge + VLM (optional)
def fig_edge():
    E = mn.load("results/edge/edge_eval.json")
    files = mn.get(E, "files") or {}
    runs = [r for r in mn.EDGE_RUNS if mn.get(files, r, "heldout_15-30s", "HOTA") is not None]
    if not runs:
        return skip("fig_edge", "results/edge/edge_eval.json missing or has no file runs")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(SC, 1.7), gridspec_kw={"wspace": 0.62, "width_ratios": [1, 1.1]})
    hi = [r for r in runs if r.startswith("736")]
    if hi:
        x = [files[r]["analysis_fps"] for r in hi]; y = [files[r]["heldout_15-30s"]["HOTA"] for r in hi]
        a1.plot(x, y, "-o", color=BLUE, ms=3, label="736x1280")
    for r, mk, lab in (("544x960_s4", "s", "544x960"), ("384x640_s4", "^", "384x640")):
        if r in runs:
            a1.plot(files[r]["analysis_fps"], files[r]["heldout_15-30s"]["HOTA"], mk, color=GREYS[1], ms=3.5,
                    mfc="white", label=lab)
    ref = mn.get(files, "_reference_research_offline", "heldout_15-30s", "HOTA")
    if ref:
        a1.axhline(ref, color=INK2, ls=":", lw=0.7)
        lo_, hi_ = a1.get_ylim()
        a1.set_ylim(lo_, max(hi_, ref + 0.006))
        a1.text(60, ref + 0.0015, "offline study (PyTorch)", fontsize=5.6, color=INK2, va="bottom", ha="right")
    a1.set_xlabel("analysis rate (fps)"); a1.set_ylabel("held-out HOTA")
    a1.set_title("(a) accuracy vs frame stride", fontsize=7.5); a1.legend(fontsize=5.6, loc="lower right")
    stages = [("detect_ms", "detect", BLUE), ("track_ms", "track", GREYS[1]), ("analytics_ms", "analytics", GREYS[2]),
              ("annotate_ms", "annotate", GREYS[3])]
    for j, r in enumerate(runs[::-1]):
        left = 0
        for st, lab, c in stages:
            v = mn.get(files, r, "latency_ms", st, "p50")
            if v is None:
                continue
            a2.barh(j, v, left=left, color=c, height=0.6, label=lab if j == 0 else None)
            left += v
        a2.text(left + 1, j, f"{left:.1f}", va="center", fontsize=5.6)
    a2.set_yticks(range(len(runs)))
    a2.set_yticklabels([f"{r.split('_')[0].split('x')[0]} px, {files[r]['analysis_fps']:.0f} fps" for r in runs[::-1]], fontsize=6)
    a2.set_xlim(0, max(a2.get_xlim()[1], 10) * 1.2)
    a2.set_xlabel("median ms per frame"); a2.set_title("(b) per-stage latency", fontsize=7.5)
    a2.legend(fontsize=5.6, loc="lower right")
    save(fig, "fig_edge")


def fig_vlm_counting():
    E1 = mn.load("results/vlm/e1_counting.json")
    M = mn.get(E1, "metrics") or {}
    lab = {"a": "(a) detector alone", "b": "(b) VLM, image only", "b_sys": "(b') VLM + system prompt",
           "c": "(c) VLM + detector count", "d": "(d) VLM, count only (no image)"}
    rows = [(k, M[k]["MAE"], M[k].get("MAE_CI95")) for k in lab if mn.get(M, k, "MAE") is not None]
    if not rows:
        return skip("fig_vlm_counting", "results/vlm/e1_counting.json missing")
    fig, ax = plt.subplots(figsize=(SC, 0.45 + 0.2 * len(rows)))
    for j, (k, v, ci) in enumerate(rows[::-1]):
        col = BLUE if k == "a" else ORANGE
        ax.barh(j, v, color=col if k in ("a", "c") else "white", edgecolor=col, lw=0.8, height=0.6)
        if ci:
            ax.errorbar(v, j, xerr=[[v - ci[0]], [ci[1] - v]], color=INK, capsize=1.5, lw=0.6)
        ax.text((ci[1] if ci else v) + 0.08, j, f"{v:.2f}", va="center", fontsize=6)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([lab[k] for k, _, _ in rows[::-1]], fontsize=6.3)
    ax.set_xlabel("people-count MAE per test frame (95% bootstrap CI)")
    ax.set_xlim(0, max((ci[1] if ci else v) for _, v, ci in rows) * 1.18)
    save(fig, "fig_vlm_counting")


def fig_heldout():
    """Held-out scene: (a) AP on aa1-test vs held-out, (b) recall by person height, (c) HOTA per 30 s segment."""
    D = mn.load("results/test1/detection.json")
    T = mn.load("results/test1/tracking.json")
    if mn.get(D, "models") is None or T is None:
        return skip("fig_heldout", "results/test1/detection.json or tracking.json missing")
    PURPLE = "#7048e8"
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(FW, 1.95), gridspec_kw={"wspace": 0.32, "width_ratios": [1.25, 1, 1.1]})
    x = np.arange(len(MODELS)); w = 0.38
    for k, m in enumerate(MODELS):
        d = D["models"][m]; c = COL[family(m)]
        a1.bar(k - w / 2, d["aa1_test"]["AP50-95"], w, color=c, alpha=0.3, hatch=None if is_ft(m) else "///",
               edgecolor=c, lw=0.4, label="aa1 test" if k == 0 else None)
        lo, hi = d["AP50-95_CI95_block"]
        a1.bar(k + w / 2, d["AP50-95"], w, color=c, hatch=None if is_ft(m) else "///", edgecolor="white", lw=0.4,
               yerr=[[max(0, d["AP50-95"] - lo)], [max(0, hi - d["AP50-95"])]], capsize=1.5,
               error_kw={"elinewidth": 0.5, "ecolor": INK}, label="held-out" if k == 0 else None)
        a1.text(k + w / 2, hi + 0.02, f"{d['AP50-95']:.2f}", ha="center", fontsize=5.6)
    a1.set_xticks(x); a1.set_xticklabels(["YOLO26n\npre.", "YOLO26n\nFT", "RF-DETR-N\npre.", "RF-DETR-N\nFT"], fontsize=5.6)
    a1.set_ylim(0, 1); a1.set_ylabel("AP$_{50:95}$")
    a1.set_title("(a) aa1 test vs held-out scene", fontsize=7.5); a1.legend(fontsize=5.6, loc="upper right", ncol=2)
    for m in MODELS:
        rb = D["models"][m]["recall_by_height_px"]; keys = [k for k in rb if rb[k]["n_gt"] >= 200]
        a2.plot(range(len(keys)), [rb[k]["recall"] for k in keys], marker="o", ms=2.5, color=COL[family(m)],
                ls="-" if is_ft(m) else "--", label=SHORT[m])
    a2.set_xticks(range(len(keys))); a2.set_xticklabels([k.replace("px", "") for k in keys], fontsize=6)
    a2.set_xlabel("label height (px)"); a2.set_ylabel("recall (IoU 0.5)"); a2.set_ylim(0, 1.02)
    a2.set_title("(b) recall at the frozen threshold", fontsize=7.5); a2.legend(fontsize=5.4, loc="lower right")
    seg = T["info"]["segment_s"]
    for name, lab, c, ls in (("YOLO26n", "YOLO26n FT", BLUE, "-"), ("RF-DETR Nano", "RF-DETR-N FT", ORANGE, "-"),
                             ("YOLO26n (pretrained)", "YOLO26n pre.", PURPLE, "--")):
        if name in T:
            h = [m["HOTA"] for m in T[name]["aa1_tuned"]["segments"]]
            a3.plot(np.arange(len(h)) * seg + seg / 2, h, marker="o", ms=2.5, color=c, ls=ls,
                    label=f"{lab} ({T[name]['aa1_tuned']['full']['HOTA']:.2f})")
    a3.set_ylim(0, 1); a3.set_xlabel("time (s)"); a3.set_ylabel("HOTA")
    a3.set_title(f"(c) ByteTrack HOTA per {seg:.0f} s segment", fontsize=7.5)
    a3.legend(fontsize=5.4, loc="upper left", handlelength=1.6, borderaxespad=0.2)
    save(fig, "fig_heldout")


# ----------------------------------------------------------------------------- revision figures
ABL = {"model_only": ("model only", "#7fb0e6"), "guardrails": ("+ guardrails (v1)", "#2a78d6"),
       "policy": ("+ policy layer", "#163f73")}          # ordinal: one hue, light -> dark (validated)
GREEN, PURPLE2 = "#2e9e6b", "#8a5cc2"                     # categorical slots 3-4 after blue/orange (validated)
SAFETY_LABEL = {"A_authorized": "authorized (success)", "B_ambiguous": "ambiguous", "C_conflicting": "conflicting",
                "D_unauthorized": "unauthorized role", "E_malformed": "malformed / out of range",
                "F_alert_disabling": "alert-disabling", "G_direct_injection": "direct injection",
                "H_frame_injection": "frame-text injection", "I_tool_result_injection": "tool-result injection",
                "J_multi_turn": "multi-turn escalation"}


def fig_safety():
    """E6: share of cases handled safely per category and configuration (Wilson 95% CI)."""
    S = mn.load("results/revision/vlm_safety.json")
    if S is None:
        return skip("fig_safety", "results/revision/vlm_safety.json missing")
    cats = list(SAFETY_LABEL)
    fig, ax = plt.subplots(figsize=(SC, 2.75))
    h = 0.26
    for j, (cfg, (lab, c)) in enumerate(ABL.items()):
        for k, cat in enumerate(cats):
            r = S["summary"][cfg]["by_category"][cat]
            lo, hi = r.get("ci95") or _wilson(r["k"], r["n"])
            y = k + (j - 1) * h
            ax.barh(y, r["rate"], h * 0.9, color=c, label=lab if k == 0 else None)
            ax.errorbar(r["rate"], y, xerr=[[r["rate"] - lo], [hi - r["rate"]]], fmt="none", ecolor="#8c8c8c",
                        elinewidth=0.6, capsize=1.6, capthick=0.6)
    ax.set_yticks(range(len(cats)))
    ax.set_yticklabels([f"{SAFETY_LABEL[c]} ({S['categories'][c]})" for c in cats], fontsize=6.2)
    ax.invert_yaxis(); ax.set_xlim(0, 1.0); ax.set_xlabel("share handled safely (no unrequested change)")
    ax.legend(fontsize=6, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=3, handlelength=1.2, columnspacing=0.8)
    ax.grid(axis="y", visible=False)
    save(fig, "fig_safety")


def _wilson(k, n, z=1.959964):
    if not n:
        return [0, 1]
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    hw = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0, c - hw), min(1, c + hw)]


def _ms(name):
    return mn.load(f"results/revision/multiscene/{name.replace(' ', '_').replace('/', '_')}.json")


def fig_tradeoff():
    """In-domain (aa1 test) vs held-out (test1) AP for every adaptation variant; WiSE paths as curves."""
    pre, ft = _ms("YOLO26n pretrained"), _ms("YOLO26n ft80")
    if pre is None or ft is None:
        return skip("fig_tradeoff", "multiscene results for pretrained / ft80 missing")
    ap = lambda r, s: mn.get(r, "scenes", s, "AP50-95")  # noqa: E731
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(FW, 2.45), gridspec_kw={"wspace": 0.28})
    for ax, (sx, sy, xl, yl) in zip((a1, a2), (("aa1_test", "test1", "aa1 test AP$_{50:95}$ (in-domain)",
                                                 "held-out test1 AP$_{50:95}$"),
                                                ("aa1_test", "coco_val", "aa1 test AP$_{50:95}$ (in-domain)",
                                                 "COCO person val AP$_{50:95}$"))):
        ax.plot([ap(pre, sx), ap(ft, sx)], [ap(pre, sy), ap(ft, sy)], ls=":", color=INK2, lw=0.8,
                label="linear pretrained$\\rightarrow$FT")
        for tag, lab, c, bn, lw, ls in (("ft80", "WiSE (FT-aa1)", BLUE, "", 1.0, "-"),
                                        ("replay", "WiSE (FT+replay)", GREEN, "", 1.0, "-"),
                                        ("multiscene", "WiSE (multi-scene)", PURPLE2, "", 1.0, "-"),
                                        ("ft80", "WiSE (FT-aa1), BN re-estimated", "#9cb9dd", "_bnre", 0.7, ":")):
            pts = [(a / 10, _ms(f"WiSE {tag}_a{a / 10:.1f}{bn}")) for a in range(11)]
            pts = [(a, r) for a, r in pts if r]
            if len(pts) < 2:
                continue
            ax.plot([ap(r, sx) for _, r in pts], [ap(r, sy) for _, r in pts], marker="o", ms=1.8 if bn else 2.4,
                    color=c, ls=ls, lw=lw, label=lab, zorder=3 if not bn else 2)
            if tag == "ft80" and not bn:
                for a, r in pts:
                    if a in (0.3, 0.5, 0.8):
                        ax.annotate(f"$\\alpha$={a:.1f}", (ap(r, sx), ap(r, sy)), fontsize=5.4, color=c,
                                    xytext=(4, 2), textcoords="offset points")
        for name, lab, mk, c in (("YOLO26n pretrained", "pretrained", "s", INK), ("YOLO26n ft80", "FT-aa1", "o", BLUE),
                                 ("YOLO26n replay", "FT+replay", "D", GREEN), ("YOLO26n freeze", "frozen backbone", "v", INK2),
                                 ("YOLO26n ignore", "ignore uncertain", "^", ORANGE),
                                 ("YOLO26n multiscene", "multi-scene", "P", PURPLE2),
                                 ("RF-DETR-N FT-aa1 @640", "RF-DETR-N FT", "X", ORANGE)):
            r = _ms(name)
            if r and ap(r, sx) is not None and ap(r, sy) is not None:
                ax.scatter(ap(r, sx), ap(r, sy), marker=mk, s=18, color=c, edgecolor="white", lw=0.4, zorder=4,
                           label=lab)
        ax.set_xlabel(xl); ax.set_ylabel(yl)
    a1.set_title("(a) in-domain vs held-out scene", fontsize=7.5)
    a2.set_title("(b) in-domain vs generic people (forgetting)", fontsize=7.5)
    h, l = a1.get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=6, fontsize=5.8, handlelength=1.6,
               columnspacing=0.9)
    save(fig, "fig_tradeoff")


TRANSFER_ROWS = [("YOLO26n pretrained", "YOLO26n pretrained"), ("YOLO26n ft80", "YOLO26n FT aa1"),
                 ("YOLO26n v1", "YOLO26n FT scene 1"), ("YOLO26n v2", "YOLO26n FT scene 2"),
                 ("YOLO26n multiscene", "YOLO26n FT aa1+1+2"), ("RF-DETR-N pretrained", "RF-DETR-N pretrained"),
                 ("RF-DETR-N FT-aa1 @640", "RF-DETR-N FT aa1")]
TRANSFER_COLS = [("aa1_test", "aa1"), ("v1_test", "scene 1"), ("v2_test", "scene 2"), ("test1", "held-out"),
                 ("coco_val", "COCO")]


def fig_transfer():
    rows = [(lab, _ms(n)) for n, lab in TRANSFER_ROWS]
    rows = [(lab, r) for lab, r in rows if r]
    if len(rows) < 3:
        return skip("fig_transfer", "multiscene results missing")
    M_ = np.array([[mn.get(r, "scenes", s, "AP50-95") or np.nan for s, _ in TRANSFER_COLS] for _, r in rows], float)
    fig, ax = plt.subplots(figsize=(SC, 0.3 * len(rows) + 0.75))
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("blues", ["#f2f7fd", "#7fb0e6", "#2a78d6", "#163f73"])
    ax.imshow(M_, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for (i_, j_), v in np.ndenumerate(M_):
        if not np.isnan(v):
            best = np.nanargmax(M_[:, j_]) == i_
            ax.text(j_, i_, f"{v:.2f}", ha="center", va="center", fontsize=6.2,
                    color="white" if v > 0.55 else INK, fontweight="bold" if best else "normal")
    ax.set_xticks(range(len(TRANSFER_COLS))); ax.set_xticklabels([c for _, c in TRANSFER_COLS], fontsize=6.4)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([lab for lab, _ in rows], fontsize=6.4)
    ax.xaxis.tick_top(); ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    save(fig, "fig_transfer")


def fig_resolution():
    sizes = (384, 512, 640, 768, 960, 1280)
    series = [("YOLO26n pretrained", "YOLO26n pre.", INK, "--"), ("YOLO26n ft80", "YOLO26n FT @1280", BLUE, "-"),
              ("YOLO26n res640", "YOLO26n FT @640", GREEN, "-"), ("RF-DETR-N pretrained", "RF-DETR-N pre.", ORANGE, "--"),
              ("RF-DETR-N FT-aa1", "RF-DETR-N FT @640", ORANGE, "-")]
    have = [s for s in series if _ms(f"{s[0]} @640")]
    if len(have) < 2:
        return skip("fig_resolution", "resolution sweep missing")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(FW, 2.0), gridspec_kw={"wspace": 0.25})
    for base, lab, c, ls in have:
        for ax, sc in ((a1, "aa1_test"), (a2, "test1")):
            xs = [z for z in sizes if _ms(f"{base} @{z}")]
            ys = [mn.get(_ms(f"{base} @{z}"), "scenes", sc, "AP50-95") for z in xs]
            ax.plot(xs, ys, marker="o", ms=2.5, color=c, ls=ls, label=lab)
    for ax, t in ((a1, "(a) aa1 test (in-domain)"), (a2, "(b) held-out test1")):
        ax.set_xscale("log", base=2); ax.set_xticks(sizes); ax.set_xticklabels([str(z) for z in sizes], fontsize=6)
        ax.minorticks_off(); ax.set_xlabel("inference input (long side, px)"); ax.set_ylabel("AP$_{50:95}$")
        ax.set_title(t, fontsize=7.5); ax.set_ylim(0, 1)
    a1.legend(fontsize=5.6, loc="lower right")
    save(fig, "fig_resolution")


def main():
    for fn in (fig_architecture, fig_dataset, fig_pr, fig_ap, fig_latency, fig_tracking, fig_selection,
               fig_qualitative, fig_campus, fig_edge, fig_vlm_counting, fig_demo, fig_heldout,
               fig_safety, fig_tradeoff, fig_transfer, fig_resolution):
        try:
            fn()
        except Exception as e:  # noqa: BLE001  (one broken figure must not stop the build)
            skip(fn.__name__, f"error: {type(e).__name__}: {e}")
    print("figures written:", ", ".join(MADE))
    if SKIPPED:
        print("figures skipped:", "; ".join(SKIPPED), file=sys.stderr)


if __name__ == "__main__":
    main()
