"""Held-out test video, stage 2: person detection, test only (no threshold, parameter or weight is fitted here).

All four models of the aa1 study are run on the sampled JPEG frames (deployed fast paths, conf >= 0.001) and
scored against the fused teacher labels with the evaluate.py protocol:
  COCO AP / AR (pycocotools), LAMR, P / R / F1 / counting at the operating threshold FROZEN from the aa1
  valid split (results/metrics_splits.json). The best-F1 threshold on this video is reported as a diagnostic
  only ("oracle"), to show how far the calibrated threshold drifts on a new domain.
  CIs: moving-block bootstrap over frames (B = 1000, blocks of BLOCK_S seconds), because frames 0.25 s
  apart are strongly correlated; the i.i.d. frame bootstrap of the aa1 study is kept for comparison.
  Paired tests on the same resamples: fine-tuned vs fine-tuned, and fine-tuned vs its own pretrained model
  (does aa1 fine-tuning transfer, or did it overfit to the aa1 scene?).
  Recall by person height (px) at the frozen threshold, IoU 0.5.
  Label bias: re-scored against RF-DETR-Large-only and YOLO26x-only labels; teacher-vs-teacher agreement as
  a ceiling reference.
  Transfer gap: aa1 test (results/metrics.json) vs this video.
  Latency: median of the deployed path on these 1920x1080 frames.

    python research/heldout/heldout_eval.py --name test1
Writes results/<name>/detection.json, significance.json, pr_data.npz, preds/*.json.gz.
"""
import argparse
import contextlib
import copy
import gzip
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths as P  # noqa: E402,F401  (project root as working directory)

import numpy as np  # noqa: E402
from pycocotools.coco import COCO  # noqa: E402

import evaluate as E  # noqa: E402

B = 1000
BLOCK_S = 10.0
HEIGHT_BINS = [(8, 32), (32, 64), (64, 128), (128, 256), (256, 2000)]
FUSED_THR, MIN_BOX = 0.40, 8


def load_gt(path):
    with contextlib.redirect_stdout(io.StringIO()):
        return COCO(str(path))


def gt_from(gt, per_frame, key):
    """COCO object with the same images and `key` boxes from teachers.json (score >= 0.40, min side >= 8 px)."""
    d = copy.deepcopy(gt.dataset); d["annotations"] = []
    by_name = {im["file_name"]: im["id"] for im in d["images"]}
    for f in per_frame:
        for x1, y1, x2, y2, s in f[key]:
            if s >= FUSED_THR and min(x2 - x1, y2 - y1) >= MIN_BOX:
                w, h = x2 - x1, y2 - y1
                d["annotations"].append({"id": len(d["annotations"]) + 1, "image_id": by_name[f["file_name"]],
                                         "category_id": 1, "bbox": [x1, y1, w, h], "area": w * h, "iscrowd": 0})
    c = COCO(); c.dataset = d
    with contextlib.redirect_stdout(io.StringIO()):
        c.createIndex()
    return c


def as_dets(gt, per_frame, key):
    """A teacher's own boxes (all scores >= 0.15) as a detection list, for teacher-vs-teacher agreement."""
    by_name = {im["file_name"]: im["id"] for im in gt.dataset["images"]}
    return [{"image_id": by_name[f["file_name"]], "category_id": 1, "score": s,
             "bbox": [x1, y1, x2 - x1, y2 - y1]} for f in per_frame for x1, y1, x2, y2, s in f[key]]


def block_indices(n, block, rng):
    """Moving-block bootstrap: ceil(n / block) random contiguous blocks, truncated to n frames."""
    starts = rng.integers(0, n - block + 1, int(np.ceil(n / block)))
    return np.concatenate([np.arange(s, s + block) for s in starts])[:n]


def recall_by_height(gt, dets, conf):
    """Recall at IoU 0.5 per GT box-height bin (greedy, score-ordered, dets >= conf)."""
    by_img = {}
    for d in dets:
        if d["score"] >= conf:
            by_img.setdefault(d["image_id"], []).append(d)
    hit = {b: [0, 0] for b in HEIGHT_BINS}
    for img_id in gt.getImgIds():
        g = np.array([a["bbox"] for a in gt.loadAnns(gt.getAnnIds(imgIds=img_id))], float).reshape(-1, 4)
        ds = sorted(by_img.get(img_id, []), key=lambda d: -d["score"])
        used = np.zeros(len(g), bool)
        if len(ds) and len(g):
            ious = E.iou_matrix(np.array([d["bbox"] for d in ds], float), g)
            for i in range(len(ds)):
                row = np.where(used, -1, ious[i]); j = row.argmax()
                if row[j] >= 0.5:
                    used[j] = True
        for k, (x, y, w, h) in enumerate(g):
            for b in HEIGHT_BINS:
                if b[0] <= h < b[1]:
                    hit[b][0] += int(used[k]); hit[b][1] += 1
    return {f"{a}-{b}px": {"recall": h / max(n, 1), "n_gt": n} for (a, b), (h, n) in hit.items()}


def main():
    import torch
    import detectors as D
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--data", default=None, help="default datasets/<name>")
    ap.add_argument("--out", default=None, help="default results/<name>")
    a = ap.parse_args()
    data = Path(a.data or f"datasets/{a.name}"); out = Path(a.out or f"results/{a.name}")
    (out / "preds").mkdir(parents=True, exist_ok=True)
    split = data / "coco" / "test"
    E.DATA = split.parent
    gt = load_gt(split / "_annotations.coco.json")
    teachers = json.load(open(data / "teachers.json"))
    frozen = {n: v["valid"]["conf"] for n, v in json.load(open("results/metrics_splits.json")).items()}
    aa1 = json.load(open("results/metrics.json"))
    n_img = len(gt.getImgIds())
    period = np.median(np.diff([im["t"] for im in gt.loadImgs(gt.getImgIds())]))
    block = min(n_img, max(1, int(round(BLOCK_S / period))))

    res, per_img, preds = {}, {}, {}
    npz = {}
    for name, factory in D.model_zoo(fast=True).items():
        print(f"== {name}", flush=True)
        run = factory()
        _, dets, t = E.run_split(run, "test", timed=True)
        del run; torch.cuda.empty_cache()
        with gzip.open(out / "preds" / f"{name.replace(' ', '_')}__test.json.gz", "wt") as f:
            json.dump(dets, f)
        Ev, m = E.coco_eval(gt, dets)
        per = E.match_images(gt, dets)
        m.update(E.operating_point(per, frozen[name])); m["LAMR"] = E.lamr(per)
        curve = E.pr_curve(per); oc = float(curve["conf"][curve["f1"].argmax()])
        m["oracle_threshold_diagnostic"] = {k: E.operating_point(per, oc)[k] for k in
                                            ("conf", "precision", "recall", "F1", "count_MAE")}
        m["latency_ms_1080p"] = float(np.median(t))
        m["recall_by_height_px"] = recall_by_height(gt, dets, frozen[name])
        m["aa1_test"] = {k: aa1[name][k] for k in ("AP50-95", "AP50", "AP75", "AP_small", "F1", "LAMR", "count_MAE")}
        res[name] = m; per_img[name] = per; preds[name] = dets
        key = name.replace(" ", "_").replace("(", "").replace(")", "")
        npz[f"{key}__coco_pr50"] = Ev.eval["precision"][0, :, 0, 0, 2]
        for k, v in curve.items():
            npz[f"{key}__{k}"] = v
        print(f"  AP {m['AP50-95']:.4f} AP50 {m['AP50']:.4f} F1@frozen {m['F1']:.4f} (thr {frozen[name]:.2f}) "
              f"oracle F1 {m['oracle_threshold_diagnostic']['F1']:.4f} (thr {oc:.2f})  LAMR {m['LAMR']:.4f}  "
              f"lat {m['latency_ms_1080p']:.1f} ms", flush=True)

    # bootstrap CIs and paired tests (same resamples for every model)
    rng = np.random.default_rng(0)
    schemes = {"block": [block_indices(n_img, block, rng) for _ in range(B)],
               "iid": [rng.integers(0, n_img, n_img) for _ in range(B)]}
    boots = {s: {n: E.bootstrap(per_img[n], frozen[n], idx) for n in per_img} for s, idx in schemes.items()}
    for s, bs in boots.items():
        for n, b in bs.items():
            for j, k in enumerate(("AP50-95", "AP50", "F1")):
                res[n][f"{k}_CI95_{s}"] = [float(x) for x in np.percentile(b[:, j], [2.5, 97.5])]
    pairs = [("YOLO26n (fine-tuned)", "RF-DETR Nano (fine-tuned)"),
             ("YOLO26n (fine-tuned)", "YOLO26n (pretrained)"),
             ("RF-DETR Nano (fine-tuned)", "RF-DETR Nano (pretrained)"),
             ("YOLO26n (pretrained)", "RF-DETR Nano (pretrained)")]
    sig = {}
    for x, y in pairs:
        sig[f"{x} vs {y}"] = {}
        for s, bs in boots.items():
            for j, k in ((0, "AP50-95"), (2, "F1")):   # bootstrap() columns: AP50-95, AP50, F1
                d = bs[x][:, j] - bs[y][:, j]
                obs = (res[x][k] - res[y][k])
                sig[f"{x} vs {y}"][f"{k}_{s}"] = {
                    "delta": float(obs), "CI95": [float(v) for v in np.percentile(d, [2.5, 97.5])],
                    "p_two_sided": float(min(1, 2 * min((d <= 0).mean(), (d >= 0).mean())))}

    # label bias: each teacher alone as GT; teacher-vs-teacher agreement as a reference ceiling
    gts = {"fused": gt, "rfdetr_large": gt_from(gt, teachers, "rfdetr_large"), "yolo26x": gt_from(gt, teachers, "yolo26x")}
    bias = {"label_counts": {k: len(g.getAnnIds()) for k, g in gts.items()}}
    for n, dets in preds.items():
        bias[n] = {k: {m: v for m, v in E.coco_eval(g, dets)[1].items() if m in ("AP50-95", "AP50")} for k, g in gts.items()}
    for t_det, t_gt in (("rfdetr_large", "yolo26x"), ("yolo26x", "rfdetr_large"), ("rfdetr_large", "fused"), ("yolo26x", "fused")):
        bias[f"teacher {t_det} vs labels {t_gt}"] = {m: v for m, v in E.coco_eval(gts[t_gt], as_dets(gt, teachers, t_det))[1].items()
                                                    if m in ("AP50-95", "AP50")}

    heights = np.array([a["bbox"][3] for a in gt.dataset["annotations"]])
    counts = np.bincount([a["image_id"] for a in gt.dataset["annotations"]], minlength=n_img + 1)[1:]
    info = {"images": n_img, "boxes": int(len(heights)), "period_s": float(period), "block_frames": block,
            "people_per_frame_mean": float(counts.mean()), "people_per_frame_max": int(counts.max()),
            "frames_without_people": int((counts == 0).sum()),
            "box_height_px_median": float(np.median(heights)), "box_height_px_p10_p90": np.percentile(heights, [10, 90]).tolist(),
            "share_small_area_lt_32sq": float(np.mean([a["area"] < 32 ** 2 for a in gt.dataset["annotations"]])),
            "frozen_thresholds_from_aa1_valid": frozen}
    json.dump({"dataset": info, "models": res, "label_bias": bias}, open(out / "detection.json", "w"), indent=2)
    json.dump(sig, open(out / "significance.json", "w"), indent=2)
    np.savez(out / "pr_data.npz", **npz)
    print(json.dumps(info, indent=1)); print(json.dumps(sig, indent=1))
    print(json.dumps({k: v for k, v in bias.items()}, indent=1))


if __name__ == "__main__":
    main()
