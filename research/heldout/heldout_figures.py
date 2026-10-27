"""Held-out test video, figures (results/<name>/figures/*.png), in the style of research/visualize.py.

  01_qualitative   teacher labels vs the three models of interest on three frames (busy / medium / sparse)
  02_transfer      AP50-95 on aa1-test vs this video (block-bootstrap CIs), recall by person height
  03_pr            precision-recall at IoU 0.5 on this video, frozen operating points marked
  04_tracking      ByteTrack (aa1 defaults / aa1-tuned) and six trackers; HOTA per 30 s segment
  05_analytics     occupancy over time (GT vs models), foot-traffic heatmap with the counting line, crossings
  06_edge          deployed runtime: HOTA and unique people per input size / analysis rate

    python research/heldout/heldout_figures.py --name test1 --video videos/test1.mp4
"""
import argparse
import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths as P  # noqa: E402,F401

import cv2  # noqa: E402
import matplotlib  # noqa: E402
import numpy as np  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from visualize import COLOR, INK2, SURF  # noqa: E402  (shared style; importing applies the rcParams)

MODELS = ["YOLO26n (pretrained)", "YOLO26n (fine-tuned)", "RF-DETR Nano (pretrained)", "RF-DETR Nano (fine-tuned)"]
SHORT = {"YOLO26n (pretrained)": "YOLO26n pre.", "YOLO26n (fine-tuned)": "YOLO26n FT",
         "RF-DETR Nano (pretrained)": "RF-DETR-N pre.", "RF-DETR Nano (fine-tuned)": "RF-DETR-N FT"}
GT_COL = "#2f9e44"
PRE_COL = "#7048e8"          # COCO-pretrained YOLO26n (not in the aa1 palette: it was not a tracking candidate there)
TRK = [("YOLO26n", "YOLO26n FT", COLOR["YOLO26n"]), ("RF-DETR Nano", "RF-DETR-N FT", COLOR["RF-DETR Nano"]),
       ("YOLO26n (pretrained)", "YOLO26n pre.", PRE_COL)]


def family(n):
    return "YOLO26n" if n.startswith("YOLO") else "RF-DETR Nano"


def col(n):
    return COLOR[family(n)]


def save(fig, out, name):
    fig.tight_layout(); fig.savefig(out / name, bbox_inches="tight"); plt.close(fig)
    print("wrote", out / name)


def qualitative(out, data, det, figs):
    from pycocotools.coco import COCO
    gt = COCO(str(data / "coco/test/_annotations.coco.json"))
    frozen = det["dataset"]["frozen_thresholds_from_aa1_valid"]
    counts = {i: len(gt.getAnnIds(imgIds=i)) for i in gt.getImgIds()}
    order = sorted(counts, key=counts.get)
    picks = [order[int(len(order) * q)] for q in (0.98, 0.75, 0.4)]
    show = ["YOLO26n (fine-tuned)", "YOLO26n (pretrained)", "RF-DETR Nano (fine-tuned)"]
    preds = {m: json.load(gzip.open(out / "preds" / f"{m.replace(' ', '_')}__test.json.gz", "rt")) for m in show}
    fig, axes = plt.subplots(len(picks), 1 + len(show), figsize=(18, 2.9 * len(picks)))
    for r, img_id in enumerate(picks):
        im = gt.loadImgs(img_id)[0]
        img = cv2.imread(str(data / "coco/test" / im["file_name"]))[:, :, ::-1]
        H, W = img.shape[:2]
        crop = (0, int(0.25 * H), W, H)                     # the sky / canopy carries no people
        for c, (title, boxes, color) in enumerate(
                [("teacher labels", [a["bbox"] for a in gt.loadAnns(gt.getAnnIds(imgIds=img_id))], GT_COL)] +
                [(SHORT[m], [d["bbox"] for d in preds[m] if d["image_id"] == img_id and d["score"] >= frozen[m]], col(m))
                 for m in show]):
            ax = axes[r, c]; ax.imshow(img[crop[1]:crop[3], crop[0]:crop[2]]); ax.axis("off")
            for x, y, w, h in boxes:
                ax.add_patch(plt.Rectangle((x - crop[0], y - crop[1]), w, h, fill=False, ec=color, lw=1.1))
            ax.set_title(f"{title}: {len(boxes)}  (t = {im['t']:.0f} s)", fontsize=10)
    fig.suptitle("Held-out video, boxes at the operating thresholds frozen from aa1 (top 25% of the frame cropped)",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    save(fig, figs, "01_qualitative.png")


def transfer(det, sig, figs):
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={"width_ratios": [1.1, 1]})
    x = np.arange(len(MODELS)); w = 0.38
    aa1 = [det["models"][m]["aa1_test"]["AP50-95"] for m in MODELS]
    ho = [det["models"][m]["AP50-95"] for m in MODELS]
    ci = np.array([det["models"][m]["AP50-95_CI95_block"] for m in MODELS])
    a.bar(x - w / 2, aa1, w, color=[col(m) for m in MODELS], alpha=0.35, label="aa1 test (in-domain)")
    a.bar(x + w / 2, ho, w, color=[col(m) for m in MODELS], label="held-out video",
          yerr=np.clip([np.array(ho) - ci[:, 0], ci[:, 1] - np.array(ho)], 0, None), capsize=3, ecolor=INK2)
    for k, (u, v) in enumerate(zip(aa1, ho)):
        a.text(k - w / 2, u + 0.01, f"{u:.3f}", ha="center", fontsize=8, color=INK2)
        a.text(k + w / 2, v + 0.03, f"{v:.3f}", ha="center", fontsize=8, fontweight="bold")
    a.set_xticks(x, [SHORT[m] for m in MODELS]); a.set_ylim(0, 1); a.set_ylabel("AP50-95")
    a.set_title("Transfer: aa1 test vs held-out video (95% block-bootstrap CI)"); a.legend(loc="upper right")
    for m in MODELS:
        rb = det["models"][m]["recall_by_height_px"]
        keys = list(rb); xs = np.arange(len(keys))
        b.plot(xs, [rb[k]["recall"] for k in keys], marker="o", color=col(m),
               ls="-" if "fine" in m else "--", label=SHORT[m])
    n = [rb[k]["n_gt"] for k in keys]
    b.set_xticks(xs, [f"{k}\n(n={c})" for k, c in zip(keys, n)], fontsize=8)
    b.set(ylim=(0, 1.02), ylabel="recall @ IoU 0.5, frozen threshold", xlabel="label box height",
          title="Recall by person height")
    b.legend(loc="lower right")
    save(fig, figs, "02_transfer.png")


def pr(out, det, figs):
    z = np.load(out / "pr_data.npz")
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    for m in MODELS:
        k = m.replace(" ", "_").replace("(", "").replace(")", "")
        ax.plot(np.linspace(0, 1, 101), z[f"{k}__coco_pr50"], color=col(m), ls="-" if "fine" in m else "--",
                label=f"{SHORT[m]} (AP50 {det['models'][m]['AP50']:.3f})")
        d = det["models"][m]
        ax.plot(d["recall"], d["precision"], "o", color=col(m), ms=6, mec="white")
    ax.set(xlabel="recall", ylabel="precision", xlim=(0, 1), ylim=(0, 1.02),
           title="Held-out PR at IoU 0.5 (dots: frozen aa1 thresholds)")
    ax.legend(loc="lower left")
    save(fig, figs, "03_pr.png")


def tracking(trk, tc, figs):
    S = ["YOLO26n", "RF-DETR Nano"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.3), gridspec_kw={"width_ratios": [1, 1.1, 1.3]})
    ax = axes[0]; mets = ["HOTA", "MOTA", "IDF1"]; x = np.arange(len(mets))
    combos = [(s, lab, c, t) for s, lab, c in TRK if s in trk for t in ("default", "aa1_tuned")]
    w = 0.8 / len(combos)
    for k, (s, lab, c, tag) in enumerate(combos):
        v = [trk[s][tag]["full"][m] for m in mets]
        ax.bar(x + (k - (len(combos) - 1) / 2) * w, v, w, color=c, alpha=1 if tag == "aa1_tuned" else 0.4,
               label=f"{lab}, {'aa1 params' if tag == 'aa1_tuned' else 'default'}")
    ax.set_xticks(x, mets); ax.set_ylim(0, 1); ax.set_title("ByteTrack, whole video"); ax.legend(fontsize=8)
    ax = axes[1]
    trackers = sorted({k.split(" | ")[1] for k in tc if " | " in k})
    for g, mk in (("ByteTrack-GT", "o"), ("OC-SORT-GT", "s")):
        for s in S:
            ys = [tc[f"{s} | {t}"][g]["full"]["HOTA"] for t in trackers]
            ax.scatter(ys, np.arange(len(trackers)) + (0.12 if s == S[0] else -0.12), marker=mk, color=COLOR[s],
                       facecolors=COLOR[s] if g == "ByteTrack-GT" else "none", s=36,
                       label=f"{s}, {g}")
    ax.set_yticks(np.arange(len(trackers)), trackers); ax.set_xlabel("HOTA (whole video)")
    ax.set_title("Six trackers, parameters frozen from aa1"); ax.legend(fontsize=7, loc="lower left")
    ax = axes[2]
    for s, lab, c in TRK:
        if s not in trk:
            continue
        h = [m["HOTA"] for m in trk[s]["aa1_tuned"]["segments"]]
        ax.plot(np.arange(len(h)) * trk["info"]["segment_s"] + trk["info"]["segment_s"] / 2, h, marker="o",
                color=c, label=lab)
    p = trk["paired_segments"]["aa1_tuned HOTA"]
    ax.set(xlabel="segment centre (s)", ylabel="HOTA", ylim=(0, 1),
           title=f"HOTA per {trk['info']['segment_s']:.0f} s segment, aa1 params\n"
                 f"(YOLO26n FT vs RF-DETR-N FT: Wilcoxon p = {p['wilcoxon_p']:.3f})")
    ax.legend()
    save(fig, figs, "04_tracking.png")


def analytics(out, an, video, figs):
    z = np.load(out / "tracks.npz")
    pts = z["pts"]
    cap = cv2.VideoCapture(video); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); stack = []
    for k in range(15):                  # median of 15 frames: a people-free backdrop (fixed camera)
        for _ in range(n // 15 - 1 if k else 0):
            cap.grab()
        ok, f = cap.read()
        if ok:
            stack.append(f)
    frame = np.median(np.stack(stack), axis=0).astype(np.uint8); H, W = frame.shape[:2]
    fig = plt.figure(figsize=(16, 8.2))
    g = fig.add_gridspec(2, 3, height_ratios=[1, 1.35])
    ax = fig.add_subplot(g[0, :2])
    s = an["occupancy_series"]
    series = [("GT", "pseudo-GT", GT_COL)] + [t for t in TRK if t[0] in an]
    for k, lab, c in series:
        y = np.convolve(s[k], np.ones(4) / 4, mode="same")
        ax.plot(s["t"], y, color=c, lw=1.2, label=f"{lab} (mean {an[k]['occupancy_mean']:.1f})")
    ax.set(xlabel="time (s)", ylabel="people in frame", title="Occupancy (tracked people, 2 s moving average)")
    ax.legend(ncol=4, loc="upper right", fontsize=8)
    ax = fig.add_subplot(g[0, 2])
    keys = [k for k, _, _ in series]; cols = [c for _, _, c in series]
    dirs = list(an["GT"]["crossings"])
    labels = an["GT"]["direction_labels"]
    x = np.arange(len(keys)); w = 0.38
    for j, d in enumerate(dirs):
        ax.bar(x + (j - 0.5) * w, [an[k]["crossings"][d] for k in keys], w, color=cols,
               alpha=1 if j == 1 else 0.45, label=labels[d], hatch=None if j == 1 else "//")
    ax.set_xticks(x, [lab for _, lab, _ in series], fontsize=8); ax.set_ylabel("line crossings")
    ax.set_title("Counting-line crossings"); ax.legend(fontsize=8)
    ax = fig.add_subplot(g[1, :2])
    gx, gy = 48, 27; heat = np.zeros((gy, gx))
    i = 0
    while f"GT/{i}" in z:
        for x1, y1, x2, y2, _ in z[f"GT/{i}"]:
            heat[min(gy - 1, int(y2 / H * gy)), min(gx - 1, int((x1 + x2) / 2 / W * gx))] += 1
        i += 1
    hm = cv2.GaussianBlur(cv2.resize(heat, (W, H), interpolation=cv2.INTER_LINEAR), (0, 0), 15)
    ax.imshow(frame[:, :, ::-1]); ax.imshow(np.ma.masked_less(hm, 0.03 * hm.max()), cmap="magma", alpha=0.6, extent=(0, W, H, 0))
    (lx1, ly1), (lx2, ly2) = an["line_norm"]
    ax.plot([lx1 * W, lx2 * W], [ly1 * H, ly2 * H], color="white", lw=2, ls="--")
    ax.axis("off"); ax.set_title("Foot-point heatmap of the pseudo-GT tracks (median background) and the counting line,\n"
                                 "fixed before evaluation")
    ax = fig.add_subplot(g[1, 2]); ax.axis("off")
    rows = [("", *(lab.replace("pseudo-GT", "GT").replace("YOLO26n ", "Y26n ").replace("RF-DETR-N ", "RF-N ")
                   for _, lab, _ in series))]
    rows.append(("unique people", *(str(an[k]["unique_people"]) for k in keys)))
    rows.append(("mean / max in frame", *(f"{an[k]['occupancy_mean']:.1f} / {an[k]['occupancy_max']}" for k in keys)))
    rows.append(("occupancy MAE", "-", *(f"{an[k]['occupancy_MAE']:.2f}" for k in keys[1:])))
    rows.append(("crossings", *(str(an[k]["crossing_events_n"]) for k in keys)))
    rows.append(("crossing F1 (±1 s)", "-", *(f"{an[k]['crossing_match']['F1']:.2f}" for k in keys[1:])))
    rows.append(("median dwell (s)", *(f"{an[k]['dwell_s_median']:.1f}" for k in keys)))
    for r, row in enumerate(rows):
        for c, v in enumerate(row):
            ax.text([0, 0.42, 0.58, 0.76, 0.94][c], 0.92 - r * 0.12, v, fontsize=9.5, transform=ax.transAxes,
                    fontweight="bold" if r == 0 or c == 0 else None, color=INK2 if c == 0 else None)
    ax.set_title("Analytics vs pseudo-GT (whole video)")
    save(fig, figs, "05_analytics.png")


def edge(ed, figs):
    runs = [k for k in ed if not k.startswith("_") and k != "protocol"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4))
    x = np.arange(len(runs))
    ref = ed["_reference_research_offline"]["full_video"]["HOTA"]
    a.bar(x, [ed[r]["full_video"]["HOTA"] for r in runs], color=COLOR["YOLO26n"], alpha=0.85)
    a.axhline(ref, color=INK2, ls="--", lw=1); a.text(len(runs) - 0.5, ref + 0.01, f"offline PyTorch {ref:.3f}", ha="right", fontsize=8)
    for k, r in enumerate(runs):
        a.text(k, ed[r]["full_video"]["HOTA"] + 0.01, f"{ed[r]['full_video']['HOTA']:.3f}", ha="center", fontsize=8)
    lab = [f"{r.split('_')[0]}\n{ed[r]['analysis_fps']:.0f} fps" for r in runs]
    a.set_xticks(x, lab, fontsize=8); a.set(ylim=(0, 1), ylabel="HOTA (whole video)", title="Deployed runtime: tracking")
    b.bar(x - 0.2, [ed[r]["unique_people"] for r in runs], 0.4, color=COLOR["YOLO26n"], label="runtime")
    b.bar(x + 0.2, [ed[r]["gt_unique_people"] for r in runs], 0.4, color=GT_COL, alpha=0.6, label="pseudo-GT")
    b.set_xticks(x, lab, fontsize=8); b.set(ylabel="unique people", title="Unique people (tracks ≥ 0.25 s)"); b.legend()
    save(fig, figs, "06_edge.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True); ap.add_argument("--video", required=True)
    ap.add_argument("--out", default=None); ap.add_argument("--data", default=None)
    a = ap.parse_args()
    out = Path(a.out or f"results/{a.name}"); data = Path(a.data or f"datasets/{a.name}")
    figs = out / "figures"; figs.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.facecolor": SURF})
    det = json.load(open(out / "detection.json")); sig = json.load(open(out / "significance.json"))
    qualitative(out, data, det, figs); transfer(det, sig, figs); pr(out, det, figs)
    tracking(json.load(open(out / "tracking.json")), json.load(open(out / "tracker_comparison.json")), figs)
    analytics(out, json.load(open(out / "analytics.json")), a.video, figs)
    if (out / "edge.json").exists():
        edge(json.load(open(out / "edge.json")), figs)


if __name__ == "__main__":
    main()
