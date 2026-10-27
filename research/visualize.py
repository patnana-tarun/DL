"""Render all evaluation figures into results/figures/."""
import json
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd

import paths  # noqa: F401  (sets the working directory)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

R = Path("results"); FIG = R / "figures"; FIG.mkdir(parents=True, exist_ok=True)
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
COLOR = {"YOLO26n": "#2a78d6", "RF-DETR Nano": "#eb6834", "Teacher": "#8a8984"}
RF_FT = "RF-DETR Nano (fine-tuned)"
MODELS = ["YOLO26n (pretrained)", "YOLO26n (fine-tuned)", "RF-DETR Nano (pretrained)", RF_FT]

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "axes.edgecolor": GRID,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "legend.frameon": False, "lines.linewidth": 2, "figure.dpi": 110, "savefig.dpi": 150,
})


def family(name):
    return "YOLO26n" if name.startswith("YOLO") else "RF-DETR Nano"


def style(name):
    ft = "fine-tuned" in name
    return dict(color=COLOR[family(name)], linestyle="-" if ft else "--", alpha=1 if ft else 0.75)


def key(name):
    return name.replace(" ", "_").replace("(", "").replace(")", "")


def load_preds(name, split):
    """Raw detections written by evaluate.py (gzipped COCO results list)."""
    import gzip
    with gzip.open(R / "preds" / f"{name.replace(' ', '_')}__{split}.json.gz", "rt") as f:
        return json.load(f)


def save(fig, name):
    fig.tight_layout(); fig.savefig(FIG / name, bbox_inches="tight"); plt.close(fig)
    print("wrote", FIG / name)


# ------------------------------------------------------------ dataset
def dataset_stats():
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.8))
    splits, counts, heights = ["train", "valid", "test"], [], []
    for s in splits:
        j = json.load(open(f"datasets/coco/{s}/_annotations.coco.json"))
        counts.append((len(j["images"]), len(j["annotations"])))
        heights.append([x["bbox"][3] for x in j["annotations"]])
    x = np.arange(3)
    bars = a.bar(x, [c[1] for c in counts], width=0.55, color=COLOR["Teacher"], edgecolor=SURF, linewidth=2)
    for r, (ni, nb) in zip(bars, counts):
        a.text(r.get_x() + r.get_width() / 2, r.get_height(), f"{nb:,} boxes\n{ni} frames", ha="center",
               va="bottom", color=INK2, fontsize=9)
    a.set_xticks(x, ["train (30–210 s)", "valid (210–255 s)", "test (255–300 s)"])
    a.set_ylabel("person boxes"); a.set_title("Pseudo-labelled dataset by split"); a.margins(y=0.2)
    a.grid(axis="x", visible=False)
    allh = np.concatenate(heights)
    b.hist(allh, bins=np.arange(0, allh.max() + 20, 15), color=COLOR["Teacher"], edgecolor=SURF, linewidth=1)
    for v, lab in ((32, "COCO small < 32²"), (96, "large > 96²")):
        b.axvline(v, color=INK2, lw=1, ls=":"); b.text(v + 3, b.get_ylim()[1] * 0.92, lab, color=INK2, fontsize=8)
    b.set_xlabel("box height (px, 1280×720 frame)"); b.set_ylabel("boxes"); b.set_title("Person size distribution")
    save(fig, "01_dataset.png")


# ------------------------------------------------------------ detection
def detection():
    m = json.load(open(R / "metrics.json")); z = np.load(R / "pr_data.npz")
    rec = np.linspace(0, 1, 101)

    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2))
    for n in MODELS:
        a.plot(rec, np.clip(z[f"{key(n)}__coco_pr50"], 0, None), label=f"{n}  AP50={m[n]['AP50']:.3f}", **style(n))
        b.plot(z[f"{key(n)}__conf"], z[f"{key(n)}__f1"], label=n, **style(n))
        bc = m[n]["conf"]; f1 = np.interp(bc, z[f"{key(n)}__conf"], z[f"{key(n)}__f1"])
        b.plot([bc], [f1], "o", ms=8, color=style(n)["color"], mec=SURF, mew=2)
    a.set(xlabel="recall", ylabel="precision", xlim=(0, 1), ylim=(0, 1.02), title="Precision–recall (IoU 0.5, test)")
    a.legend(loc="lower left", fontsize=8.5)
    b.set(xlabel="confidence threshold", ylabel="F1", xlim=(0, 1), ylim=(0, 1), title="F1 vs confidence (● = threshold chosen on valid)")
    save(fig, "02_pr_f1_curves.png")

    cols = ["AP50-95", "AP50", "AP75", "AR@300", "F1", "det_accuracy"]
    fig, ax = plt.subplots(figsize=(12, 4.2)); w = 0.2; x = np.arange(len(cols))
    for i, n in enumerate(MODELS):
        ft = "fine-tuned" in n
        bars = ax.bar(x + (i - 1.5) * w, [m[n][c] for c in cols], w - 0.02, label=n, color=COLOR[family(n)],
                      alpha=1 if ft else 0.45, hatch=None if ft else "///", edgecolor=SURF, linewidth=1.5)
        if ft:
            for r in bars:
                ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.01, f"{r.get_height():.2f}",
                        ha="center", va="bottom", fontsize=8, color=INK2)
    ax.set_xticks(x, ["mAP50-95", "mAP50", "mAP75", "AR@300", "F1 (valid thr.)", "TP/(TP+FP+FN)"]); ax.set_ylim(0, 1.08)
    ax.set_title("Detection metrics on held-out test split (255–300 s)"); ax.grid(axis="x", visible=False)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize=8.5)
    save(fig, "03_detection_metrics.png")

    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4))
    sizes = ["AP_small", "AP_medium", "AP_large"]; x = np.arange(3)
    for i, n in enumerate(MODELS):
        ft = "fine-tuned" in n
        a.bar(x + (i - 1.5) * w, [max(m[n][s], 0) for s in sizes], w - 0.02, color=COLOR[family(n)],
              alpha=1 if ft else 0.45, hatch=None if ft else "///", edgecolor=SURF, linewidth=1.5, label=n)
        b.plot(m[n]["latency_ms"], m[n]["AP50-95"], "o", ms=11, color=COLOR[family(n)], alpha=1 if ft else 0.5,
               mec=SURF, mew=2)
        b.annotate(n, (m[n]["latency_ms"], m[n]["AP50-95"]), xytext=(8, -3), textcoords="offset points",
                   fontsize=8.5, color=INK2)
    a.set_xticks(x, ["small (<32²)", "medium", "large (>96²)"]); a.set_ylim(0, 1); a.grid(axis="x", visible=False)
    a.set(ylabel="mAP50-95", title="mAP by person size"); a.legend(fontsize=7.5, loc="upper left")
    b.set(xlabel="median latency per 1280×720 frame (ms, RTX 4060, batch 1)", ylabel="mAP50-95",
          title="Accuracy vs speed"); b.set_xlim(0, max(v["latency_ms"] for v in m.values()) * 1.6); b.margins(y=0.15)
    save(fig, "04_size_and_speed.png")


def training_curves():
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    y = pd.read_csv("runs/yolo26n/results.csv"); y.columns = [c.strip() for c in y.columns]
    c = COLOR["YOLO26n"]
    axes[0].plot(y["epoch"], y["train/box_loss"], color=c, label="train box"); axes[0].plot(
        y["epoch"], y["val/box_loss"], color=c, ls="--", label="val box")
    axes[0].set(title="YOLO26n loss", xlabel="epoch"); axes[0].legend()
    cm = len(y) - 20
    axes[0].axvline(cm, color=INK2, lw=1, ls=":"); axes[0].text(cm - 2, axes[0].get_ylim()[1] * 0.97, "mosaic off",
                                                                ha="right", va="top", fontsize=8, color=INK2)
    axes[2].plot(y["epoch"], y["metrics/mAP50-95(B)"], color=c, label="YOLO26n")
    c = COLOR["RF-DETR Nano"]
    for run, tag, ls in (("rfdetr_nano_v2", "cosine LR", "-"),):
        rf = Path(f"runs/{run}/metrics.csv")
        if not rf.exists():
            continue
        ep = pd.read_csv(rf).groupby("epoch").mean(numeric_only=True).reset_index()
        tl = ep["train/loss"].dropna(); axes[1].plot(ep.loc[tl.index, "epoch"], tl, color=c, ls=ls, label=tag)
        mp = ep["val/ema_mAP_50_95"].dropna()
        axes[2].plot(ep.loc[mp.index, "epoch"], mp, color=c, ls=ls, label=f"RF-DETR Nano {tag}")
    axes[1].set(title="RF-DETR Nano train loss (total)", xlabel="epoch"); axes[1].legend()
    axes[2].set(title="Validation mAP50-95", xlabel="epoch", ylim=(0.6, 0.9)); axes[2].legend(fontsize=8)
    save(fig, "05_training_curves.png")


def qualitative(n_frames=3):
    gt = json.load(open("datasets/coco/test/_annotations.coco.json"))
    m = json.load(open(R / "metrics.json"))
    preds = {c: load_preds(c, "test") for c in MODELS}
    imgs = gt["images"][:: max(1, len(gt["images"]) // n_frames)][:n_frames]
    cols = ["Teacher (pseudo-GT)", "YOLO26n (fine-tuned)", RF_FT]
    fig, axes = plt.subplots(n_frames, 3, figsize=(16, 2.35 * n_frames))
    for r, im in enumerate(imgs):
        img = cv2.imread(f"datasets/coco/test/{im['file_name']}")
        for c, col in enumerate(cols):
            f = img.copy()
            if c == 0:
                boxes = [a["bbox"] for a in gt["annotations"] if a["image_id"] == im["id"]]; bgr = (255, 255, 255)
            else:
                th = m[col]["conf"]
                boxes = [d["bbox"] for d in preds[col] if d["image_id"] == im["id"] and d["score"] >= th]
                bgr = tuple(int(COLOR[family(col)][i:i + 2], 16) for i in (5, 3, 1))
            for x, y, w, h in boxes:
                cv2.rectangle(f, (int(x), int(y)), (int(x + w), int(y + h)), bgr, 3)
            ax = axes[r, c]; ax.imshow(f[330:, :, ::-1]); ax.axis("off")  # crop: people are in the lower half
            ax.set_title(f"{col} — {len(boxes)} persons" if r == 0 else f"{len(boxes)} persons", fontsize=10)
    save(fig, "06_qualitative_test.png")


# ------------------------------------------------------------ tracking
def tracking():
    p = R / "tracking/tracking_metrics.json"
    if not p.exists():
        return
    t = json.load(open(p)); models = list(t)
    cols = ["HOTA", "MOTA", "IDF1", "DetA", "AssA"]
    fig, ax = plt.subplots(figsize=(11, 4)); w = 0.2; x = np.arange(len(cols))
    for i, (mname, tag) in enumerate([(mm, tg) for mm in models for tg in ("default", "tuned")]):
        v = t[mname][tag]["heldout_15-30s"]; tuned = tag == "tuned"
        bars = ax.bar(x + (i - 1.5) * w, [v[c] for c in cols], w - 0.02, color=COLOR[mname],
                      alpha=1 if tuned else 0.45, hatch=None if tuned else "///", edgecolor=SURF, linewidth=1.5,
                      label=f"{mname} — {'tuned' if tuned else 'default'} ByteTrack")
        if tuned:
            for r in bars:
                ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.01, f"{r.get_height():.2f}",
                        ha="center", va="bottom", fontsize=8, color=INK2)
    ax.set_xticks(x, cols); ax.set_ylim(0, 1.08); ax.grid(axis="x", visible=False)
    ax.set_title("ByteTrack on held-out 15–30 s (params tuned on 0–15 s)")
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize=8.5)
    save(fig, "07_tracking_metrics.png")

    g = pd.DataFrame(json.load(open(R / "tracking/grid_search.json")))
    params = ["min_conf", "nms_iou", "track_activation_threshold", "high_conf_det_threshold", "lost_track_buffer",
              "minimum_iou_threshold", "minimum_consecutive_frames"]
    fig, axes = plt.subplots(1, 7, figsize=(20, 3.4), sharey=True)
    for ax, prm in zip(axes, params):
        for mname in models:
            s = g[g.model == mname].groupby(prm)["HOTA"].max()
            ax.plot(range(len(s)), s.values, "o-", color=COLOR[mname], ms=7, mec=SURF, mew=2, label=mname)
            ax.set_xticks(range(len(s)), [str(v) for v in s.index])
        ax.set_title({"min_conf": "min det confidence", "nms_iou": "NMS IoU (0 = off)"}.get(prm, prm.replace("_", " ")),
                     fontsize=9.5); ax.set_xlabel("value")
    axes[0].set_ylabel("best HOTA (0–15 s)"); axes[0].legend(fontsize=8)
    fig.suptitle("ByteTrack grid search: best HOTA reachable at each parameter value", x=0.01, ha="left",
                 fontweight="bold")
    save(fig, "08_bytetrack_grid.png")


# ------------------------------------------------------------ generalisation, latency, uncertainty
def splits_fig():
    p = R / "metrics_splits.json"
    if not p.exists():
        return
    m = json.load(open(p)); names = [n for n in m if n in MODELS]
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.3), gridspec_kw={"width_ratios": [3, 2]})
    x = np.arange(len(names)); w = 0.26
    for i, (sp, alpha) in enumerate((("train", 0.4), ("valid", 0.7), ("test", 1.0))):
        bars = a.bar(x + (i - 1) * w, [m[n][sp]["AP50-95"] for n in names], w - 0.03,
                     color=[COLOR[family(n)] for n in names], alpha=alpha, edgecolor=SURF, linewidth=1.5)
        for r in bars:
            a.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.008, f"{r.get_height():.3f}", ha="center",
                   va="bottom", fontsize=7, color=INK2, rotation=90)
    a.set_xticks(x, [n.replace(" (", "\n(") for n in names], fontsize=8.5); a.set_ylim(0, 1.1)
    a.set(ylabel="mAP50-95", title="Train / valid / test mAP50-95 (light → dark)"); a.grid(axis="x", visible=False)
    ft = [n for n in names if "diagnosis" in m[n]]
    y = np.arange(len(ft))
    b.barh(y - 0.2, [m[n]["diagnosis"]["train_minus_test"] for n in ft], 0.36, color=[COLOR[family(n)] for n in ft],
           edgecolor=SURF, label="fine-tuned: train − test")
    b.barh(y + 0.2, [m[n]["diagnosis"]["pretrained_train_minus_test"] for n in ft], 0.36,
           color=[COLOR[family(n)] for n in ft], alpha=0.4, hatch="///", edgecolor=SURF,
           label="pretrained: train − test (split difficulty)")
    for yi, n in zip(y, ft):
        e = m[n]["diagnosis"]["excess_gap_overfit"]
        b.text(max(m[n]["diagnosis"]["train_minus_test"], m[n]["diagnosis"]["pretrained_train_minus_test"]) + 0.004,
               yi, f"excess {e:+.3f}", va="center", fontsize=8.5, color=INK2)
    b.set_yticks(y, [n.replace(" (", "\n(") for n in ft], fontsize=8.5); b.axvline(0, color=INK2, lw=1)
    b.set(xlabel="mAP50-95 gap", title="Overfitting check: excess train−test gap"); b.legend(fontsize=7.5, loc="lower right")
    b.margins(x=0.35); b.grid(axis="y", visible=False)
    save(fig, "09_split_generalisation.png")


def latency_fig():
    p = R / "latency.json"
    if not p.exists():
        return
    L = {k: v for k, v in json.load(open(p)).items() if "median_ms" in v}
    names = list(L)[::-1]; y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(11, 0.42 * len(names) + 1.2))
    bars = ax.barh(y, [L[n]["median_ms"] for n in names], 0.62, color=[COLOR[family(n)] for n in names],
                   edgecolor=SURF, linewidth=1.5)
    for bar, n in zip(bars, names):   # per-bar alpha: matplotlib 3.11 rejects a list here
        bar.set_alpha(1 if "deployed" in n else 0.5)
    for yi, n in zip(y, names):
        ax.text(L[n]["median_ms"] + 0.2, yi, f"{L[n]['median_ms']:.2f} ms · AP {L[n]['AP50-95']:.3f}", va="center",
                fontsize=8.5, color=INK2)
    ax.set_yticks(y, [n.replace(" | ", " · ") for n in names], fontsize=8.5)
    ax.set(xlabel="median end-to-end latency per 1280×720 frame (ms, RTX 4060 Laptop, batch 1)",
           title="Inference variants: latency and test mAP50-95 (solid = deployed path)")
    ax.margins(x=0.25); ax.grid(axis="y", visible=False)
    save(fig, "10_latency_variants.png")


def ci_fig():
    m = json.load(open(R / "metrics.json"))
    names = [n for n in m if "AP50-95_CI95" in m[n]][::-1]
    if not names:
        return
    fig, axes = plt.subplots(1, 3, figsize=(13, 0.45 * len(names) + 1.4), sharey=True)
    for ax, k in zip(axes, ("AP50-95", "AP50", "F1")):
        for yi, n in enumerate(names):
            lo, hi = m[n][f"{k}_CI95"]; c = COLOR[family(n)]; ft = "fine-tuned" in n
            ax.plot([lo, hi], [yi, yi], color=c, lw=2, alpha=1 if ft else 0.5)
            ax.plot([m[n][k]], [yi], "o", ms=8, color=c, mec=SURF, mew=2, alpha=1 if ft else 0.5)
            ax.text(hi, yi + 0.18, f"{m[n][k]:.3f} [{lo:.3f}, {hi:.3f}]", fontsize=7.5, color=INK2, ha="right")
        ax.set_title(f"{k} (test, 95% bootstrap CI)", fontsize=10.5); ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(names)), names, fontsize=8.5)
    save(fig, "11_confidence_intervals.png")


if __name__ == "__main__":
    dataset_stats()
    if (R / "metrics.json").exists():
        detection(); qualitative(); ci_fig()
    splits_fig(); latency_fig()
    training_curves()
    tracking()
