"""Evaluate every model on the train / valid / test splits with one shared protocol.

Metrics (person class, pseudo-GT from the teacher ensemble):
  COCO        AP50-95, AP50, AP75, AP S/M/L, AR@1/100/300, AR S/M/L          (pycocotools)
  Operating   P, R, F1, TP/FP/FN, detection accuracy TP/(TP+FP+FN), mean IoU of TPs, FPPI
              at IoU 0.5. The confidence threshold is the best-F1 threshold on the VALID split,
              then frozen for train/test (choosing it on test would be optimistic).
  Pedestrian  log-average miss rate LAMR (MR^-2, Caltech/CityPersons protocol, FPPI 1e-2..1e0)
  Counting    per-frame people count MAE / RMSE / mean count accuracy at the operating threshold
  Stats       95% bootstrap CIs over test images (B=1000) for AP50-95 / AP50 / F1, plus paired
              bootstrap tests between models on the same resampled images
  Diagnosis   train-test gap of each fine-tuned model minus the same gap for its pretrained
              counterpart (the pretrained model never saw any split, so its gap is pure split
              difficulty; what is left over is overfitting)
  Cost        params, GFLOPs at the inference resolution, test-split latency (deployed fast path)

Writes results/metrics.json (test split, used by visualize.py), results/metrics_splits.json,
results/significance.json, results/pr_data.npz, results/preds/*.json.gz (raw detections).
"""
import argparse
import contextlib
import gzip
import io
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import paths as P

OUT = Path("results")
DATA = Path("datasets/coco")
SPLITS = ("train", "valid", "test")
IOUS = np.linspace(0.5, 0.95, 10)
REC_GRID = np.linspace(0, 1, 101)
B = 1000
RNG = np.random.default_rng(0)


# ---------------------------------------------------------------- COCO
def coco_eval(gt, dets):
    with contextlib.redirect_stdout(io.StringIO()):
        dt = gt.loadRes(dets) if dets else None
        E = COCOeval(gt, dt, "bbox")
        E.params.maxDets = [1, 100, 300]
        E.evaluate(); E.accumulate(); E.summarize()
    s = E.stats
    return E, {"AP50-95": s[0], "AP50": s[1], "AP75": s[2], "AP_small": s[3], "AP_medium": s[4],
               "AP_large": s[5], "AR@1": s[6], "AR@100": s[7], "AR@300": s[8], "AR_small": s[9],
               "AR_medium": s[10], "AR_large": s[11]}


# ---------------------------------------------------------------- matching
def iou_matrix(d, g):
    """d, g: [N,4] / [M,4] xywh -> IoU [N,M]."""
    x1 = np.maximum(d[:, None, 0], g[None, :, 0]); y1 = np.maximum(d[:, None, 1], g[None, :, 1])
    x2 = np.minimum(d[:, None, 0] + d[:, None, 2], g[None, :, 0] + g[None, :, 2])
    y2 = np.minimum(d[:, None, 1] + d[:, None, 3], g[None, :, 1] + g[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    return inter / (d[:, None, 2] * d[:, None, 3] + g[None, :, 2] * g[None, :, 3] - inter)


def match_images(gt, dets, max_det=300):
    """COCO-style greedy matching per image at all 10 IoU thresholds.

    Returns a list (one per image) of dicts: scores [N], tp [10,N] bool, iou [N] (IoU of the
    IoU-0.5 match, 0 if unmatched), n_gt.
    """
    by_img = {}
    for d in dets:
        by_img.setdefault(d["image_id"], []).append(d)
    out = []
    for img_id in gt.getImgIds():
        g = np.array([a["bbox"] for a in gt.loadAnns(gt.getAnnIds(imgIds=img_id))], float).reshape(-1, 4)
        ds = sorted(by_img.get(img_id, []), key=lambda d: -d["score"])[:max_det]
        sc = np.array([d["score"] for d in ds], float)
        db = np.array([d["bbox"] for d in ds], float).reshape(-1, 4)
        tp = np.zeros((len(IOUS), len(ds)), bool); miou = np.zeros(len(ds))
        if len(ds) and len(g):
            ious = iou_matrix(db, g)
            for t, thr in enumerate(IOUS):
                used = np.zeros(len(g), bool)
                for i in range(len(ds)):
                    row = np.where(used, -1, ious[i])
                    j = row.argmax()
                    if row[j] >= thr:
                        used[j] = True; tp[t, i] = True
                        if t == 0:
                            miou[i] = row[j]
        out.append({"scores": sc, "tp": tp, "iou": miou, "n_gt": len(g)})
    return out


def ap_from(scores, tp, n_gt, w=None):
    """COCO 101-point interpolated AP for each IoU row of tp. w = per-detection weights (bootstrap)."""
    if n_gt == 0 or len(scores) == 0:          # no GT, or no detections at all -> AP 0
        return np.zeros(len(tp))
    o = np.argsort(-scores, kind="mergesort")
    w = np.ones(len(scores)) if w is None else w
    w = w[o]; tp = tp[:, o]
    ctp = np.cumsum(tp * w, 1); cfp = np.cumsum(~tp * w, 1)
    rec = ctp / n_gt; prec = ctp / np.maximum(ctp + cfp, 1e-12)
    prec = np.maximum.accumulate(prec[:, ::-1], 1)[:, ::-1]      # precision envelope
    aps = []
    for r, p in zip(rec, prec):
        idx = np.searchsorted(r, REC_GRID, side="left")
        aps.append(np.where(idx < len(p), p[np.minimum(idx, len(p) - 1)], 0).mean())
    return np.array(aps)


def flatten(per):
    sc = np.concatenate([p["scores"] for p in per]) if per else np.zeros(0)
    tp = np.concatenate([p["tp"] for p in per], 1)
    iou = np.concatenate([p["iou"] for p in per])
    img = np.concatenate([np.full(len(p["scores"]), i) for i, p in enumerate(per)])
    return sc, tp, iou, img


def pr_curve(per):
    """P/R/F1 at IoU 0.5 on a 0.01..0.99 confidence grid."""
    sc, tp, _, _ = flatten(per); tp = tp[0]; n_gt = sum(p["n_gt"] for p in per)
    grid = np.linspace(0.01, 0.99, 99)
    P, R = [], []
    for c in grid:
        k = sc >= c; t = tp[k].sum(); n = k.sum()
        P.append(t / n if n else 1.0); R.append(t / max(n_gt, 1))
    P, R = np.array(P), np.array(R)
    return {"conf": grid, "precision": P, "recall": R, "f1": 2 * P * R / np.maximum(P + R, 1e-9)}


def operating_point(per, conf):
    sc, tp, iou, img = flatten(per); tp = tp[0]
    k = sc >= conf
    TP = int(tp[k].sum()); FP = int(k.sum() - TP); n_gt = sum(p["n_gt"] for p in per); FN = n_gt - TP
    P = TP / max(TP + FP, 1); R = TP / max(n_gt, 1)
    pred_n = np.bincount(img[k], minlength=len(per)); gt_n = np.array([p["n_gt"] for p in per])
    err = pred_n - gt_n
    return {"conf": conf, "precision": P, "recall": R, "F1": 2 * P * R / max(P + R, 1e-9),
            "TP": TP, "FP": FP, "FN": FN, "det_accuracy": TP / max(TP + FP + FN, 1),
            "mean_IoU_TP": float(iou[k & tp].mean()) if TP else 0.0, "FPPI": FP / len(per),
            "count_MAE": float(np.abs(err).mean()), "count_RMSE": float(np.sqrt((err ** 2).mean())),
            "count_accuracy": float(np.mean(1 - np.abs(err) / np.maximum(gt_n, 1)))}


def lamr(per):
    """Log-average miss rate over 9 FPPI points log-spaced in [1e-2, 1e0] (Dollar et al.)."""
    sc, tp, _, _ = flatten(per); tp = tp[0]; n_gt = sum(p["n_gt"] for p in per)
    o = np.argsort(-sc); tp = tp[o]
    fppi = np.cumsum(~tp) / len(per); mr = 1 - np.cumsum(tp) / n_gt
    refs = np.logspace(-2, 0, 9); vals = []
    for r in refs:
        i = np.where(fppi <= r)[0]
        vals.append(mr[i[-1]] if len(i) else 1.0)
    return float(np.exp(np.mean(np.log(np.maximum(vals, 1e-10)))))


def bootstrap(per, conf, idx_sets):
    """AP50-95, AP50, F1 on each resampled image set."""
    sc, tp, _, img = flatten(per); n_gt = np.array([p["n_gt"] for p in per])
    out = []
    for idx in idx_sets:
        cnt = np.bincount(idx, minlength=len(per)).astype(float)
        w = cnt[img]; ng = (cnt * n_gt).sum()
        ap = ap_from(sc, tp, ng, w)
        k = sc >= conf
        t = (w * tp[0] * k).sum(); f = (w * ~tp[0] * k).sum()
        P, R = t / max(t + f, 1e-9), t / max(ng, 1)
        out.append((ap.mean(), ap[0], 2 * P * R / max(P + R, 1e-9)))
    return np.array(out)


# ---------------------------------------------------------------- cost
def model_cost(name):
    """Params and forward GFLOPs at the inference resolution (batch 1)."""
    from torch.utils.flop_counter import FlopCounterMode
    if name.startswith("YOLO"):
        from ultralytics import YOLO
        net = YOLO(P.YOLO_PRETRAINED if "pretrained" in name else P.YOLO_FT).model.cuda().eval()
        x = torch.zeros(1, 3, 736, 1280, device="cuda")
    else:
        import detectors as D
        from rfdetr import RFDETRNano
        m = RFDETRNano(pretrain_weights=P.RFDETR_PRETRAINED) if "pretrained" in name else \
            RFDETRNano(pretrain_weights=D.rfdetr_best(P.RFDETR_FT_RUN), resolution=640)
        net = m.model.model.cuda().eval(); r = m.model.resolution
        x = torch.zeros(1, 3, r, r, device="cuda")
    params = sum(p.numel() for p in net.parameters())
    net.requires_grad_(False)   # FlopCounterMode's module tracker breaks on grad-tracking params
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        net(x)
    return {"params_M": params / 1e6, "GFLOPs": fc.get_total_flops() / 1e9}


# ---------------------------------------------------------------- main
def run_split(run, split, timed=False):
    d = DATA / split
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(d / "_annotations.coco.json"))
    dets, times = [], []
    imgs = gt.loadImgs(gt.getImgIds())
    if timed:
        for im in imgs[:20]:
            run(cv2.imread(str(d / im["file_name"])))
    for im in imgs:
        img = cv2.imread(str(d / im["file_name"]))
        torch.cuda.synchronize(); t0 = time.perf_counter()
        boxes, scores = run(img)
        torch.cuda.synchronize(); times.append(1000 * (time.perf_counter() - t0))
        for (x1, y1, x2, y2), s in zip(boxes, scores):
            dets.append({"image_id": im["id"], "category_id": 1, "score": float(s),
                         "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]})
    return gt, dets, np.array(times)


def main():
    """python research/evaluate.py                                       # aa1 test protocol
    python research/evaluate.py --data datasets/street/coco --out results/street  # zero-shot transfer to a new video
        [--yolo "YOLO26n (street-ft)=runs/street_yolo/weights/best.pt"]
        [--rfdetr "RF-DETR Nano (street-ft)=runs/street_rfdetr/checkpoint_best_total.pth"]
    """
    global OUT, DATA
    import detectors as D
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DATA)); ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--yolo", action="append", default=[], help="extra 'name=weights' YOLO models")
    ap.add_argument("--rfdetr", action="append", default=[], help="extra 'name=weights' RF-DETR models")
    a = ap.parse_args()
    DATA, OUT = Path(a.data), Path(a.out)
    (OUT / "preds").mkdir(parents=True, exist_ok=True)
    zoo = D.model_zoo(fast=True)
    for spec in a.yolo:
        n, w = spec.split("="); zoo[n] = lambda w=w: D.yolo(w)
    for spec in a.rfdetr:
        n, w = spec.split("="); zoo[n] = lambda w=w: D.rfdetr(w, resolution=640)
    splits, per_img, npz = {}, {}, {}
    n_test = None
    for name, factory in zoo.items():
        print(f"== {name}", flush=True)
        run = factory()
        splits[name] = {}
        for split in ("valid", "train", "test"):
            gt, dets, t = run_split(run, split, timed=split == "test")
            with gzip.open(OUT / "preds" / f"{name.replace(' ', '_')}__{split}.json.gz", "wt") as f:
                json.dump(dets, f)
            E, m = coco_eval(gt, dets)
            per = match_images(gt, dets)
            if split == "valid":
                curve = pr_curve(per); conf = float(curve["conf"][curve["f1"].argmax()])
            m.update(operating_point(per, conf)); m["LAMR"] = lamr(per)
            if split == "test":
                m["latency_ms"] = float(np.median(t)); m["FPS"] = 1000 / m["latency_ms"]
                per_img[name] = per; n_test = len(per)
                curve_t = pr_curve(per)
                key = name.replace(" ", "_").replace("(", "").replace(")", "")
                npz[f"{key}__coco_pr50"] = E.eval["precision"][0, :, 0, 0, 2]
                npz[f"{key}__coco_pr5095"] = E.eval["precision"][:, :, 0, 0, 2].mean(0)
                for k, v in curve_t.items():
                    npz[f"{key}__{k}"] = v
            splits[name][split] = m
            print(f"  {split:5s} AP {m['AP50-95']:.4f}  AP50 {m['AP50']:.4f}  F1 {m['F1']:.4f}  "
                  f"acc {m['det_accuracy']:.4f}  LAMR {m['LAMR']:.4f}", flush=True)
        del run; torch.cuda.empty_cache()
        splits[name]["cost"] = model_cost(name)

    # bootstrap CIs + paired tests on the test split (same resamples for every model)
    idx_sets = [RNG.integers(0, n_test, n_test) for _ in range(B)]
    boot = {n: bootstrap(per_img[n], splits[n]["valid"]["conf"], idx_sets) for n in per_img}
    for n, b in boot.items():
        for j, k in enumerate(("AP50-95", "AP50", "F1")):
            lo, hi = np.percentile(b[:, j], [2.5, 97.5])
            splits[n]["test"][f"{k}_CI95"] = [float(lo), float(hi)]
    sig = {}
    names = list(boot)
    for i, a in enumerate(names):
        for bname in names[i + 1:]:
            d = boot[a][:, 0] - boot[bname][:, 0]
            sig[f"{a} vs {bname}"] = {"delta_AP50-95": float(np.mean(d)),
                                      "CI95": [float(x) for x in np.percentile(d, [2.5, 97.5])],
                                      "p_two_sided": float(min(1, 2 * min((d <= 0).mean(), (d >= 0).mean())))}

    # overfitting diagnosis
    for n in splits:
        if "fine-tuned" not in n:
            continue
        pre = next(p for p in splits if p.startswith(n.split(" (")[0]) and "pretrained" in p)
        ft_gap = splits[n]["train"]["AP50-95"] - splits[n]["test"]["AP50-95"]
        pt_gap = splits[pre]["train"]["AP50-95"] - splits[pre]["test"]["AP50-95"]
        splits[n]["diagnosis"] = {"train_minus_test": ft_gap, "pretrained_train_minus_test": pt_gap,
                                  "excess_gap_overfit": ft_gap - pt_gap,
                                  "train_minus_valid": splits[n]["train"]["AP50-95"] - splits[n]["valid"]["AP50-95"]}

    json.dump(splits, open(OUT / "metrics_splits.json", "w"), indent=2)
    json.dump(sig, open(OUT / "significance.json", "w"), indent=2)
    json.dump({n: {**splits[n]["test"], **splits[n]["cost"]} for n in splits}, open(OUT / "metrics.json", "w"), indent=2)
    np.savez(OUT / "pr_data.npz", **npz)
    print(json.dumps({n: splits[n].get("diagnosis") for n in splits}, indent=1))
    print(json.dumps(sig, indent=1))


if __name__ == "__main__":
    main()
