"""Independent re-computation of the published numbers.

Every headline metric is recomputed with code that shares nothing with the script that produced it:

detection  (needs results/metrics_splits.json + results/preds/*.json.gz from evaluate.py)
  D1  COCO AP50-95 / AP50 / AP75 with a from-scratch 101-point AP (vectorised greedy matching),
      against pycocotools                                                   tol 0.002
  D2  P / R / F1 at the frozen operating threshold with optimal (Hungarian) matching at IoU 0.5,
      against evaluate.py's greedy matching                                 tol 0.01
  D3  the operating threshold is the best-F1 threshold of the VALID split (re-derived)
  D4  Ultralytics' own validator on the test split for the fine-tuned YOLO26n (a different
      matcher, interpolation and NMS path)                                  tol 0.02
  D5  label counts in the COCO files match datasets/stats.json, and the YOLO txt labels match COCO
  D6  every bootstrap CI contains its point estimate; significance agrees with the CIs
tracking   (needs results/tracking/{detections.npz,tracking_metrics.json})
  T1  re-running the tuned tracker reproduces the stored held-out HOTA / MOTA / IDF1 exactly
  T2  MOTA / IDF1 / ID switches with py-motmetrics (independent CLEAR-MOT + IDF1 implementation)
                                                                            tol 0.01 (IDSW: 10 %)

    python research/crosscheck.py detection|tracking|all
Writes results/crosscheck_<mode>.json; exits non-zero if any check fails.
"""
import contextlib
import gzip
import io
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

import paths as P  # noqa: F401  (project root as working directory)

R = Path("results")
DATA = Path("datasets/coco")


def load_gt(split):
    j = json.load(open(DATA / split / "_annotations.coco.json"))
    g = {im["id"]: [] for im in j["images"]}
    for a in j["annotations"]:
        g[a["image_id"]].append(a["bbox"])
    return {k: np.array(v, float).reshape(-1, 4) for k, v in g.items()}


def load_dets(name, split):
    with gzip.open(R / "preds" / f"{name.replace(' ', '_')}__{split}.json.gz", "rt") as f:
        d = json.load(f)
    out = {}
    for x in d:
        out.setdefault(x["image_id"], []).append([*x["bbox"], x["score"]])
    return {k: np.array(v, float) for k, v in out.items()}


def iou_xywh(a, b):
    ax2, ay2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx2, by2 = b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(ax2[:, None], bx2[None]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    ih = np.clip(np.minimum(ay2[:, None], by2[None]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = iw * ih
    return inter / (a[:, 2:].prod(1)[:, None] + b[:, 2:].prod(1)[None] - inter)


# ---------------------------------------------------------------- D1: AP from scratch
def ap_scratch(gt, dets, thr, max_det=300):
    """COCO-style AP at one IoU threshold: per image, detections in descending score order take the
    best still-unmatched GT with IoU >= thr; precision envelope sampled at 101 recall points."""
    scores, tps, n_gt = [], [], 0
    for img, g in gt.items():
        n_gt += len(g)
        d = dets.get(img, np.zeros((0, 5)))
        d = d[np.argsort(-d[:, 4], kind="stable")][:max_det]
        tp = np.zeros(len(d), bool)
        if len(d) and len(g):
            ious = iou_xywh(d[:, :4], g)
            free = np.ones(len(g), bool)
            for i in range(len(d)):
                cand = np.where(free & (ious[i] >= thr), ious[i], -1)
                j = int(cand.argmax())
                if cand[j] >= thr:
                    free[j] = False; tp[i] = True
        scores.append(d[:, 4]); tps.append(tp)
    s = np.concatenate(scores); tp = np.concatenate(tps)
    o = np.argsort(-s, kind="mergesort"); tp = tp[o]
    ctp = np.cumsum(tp); cfp = np.cumsum(~tp)
    rec = ctp / n_gt; prec = ctp / (ctp + cfp)
    for i in range(len(prec) - 2, -1, -1):                 # monotone envelope, written as a plain loop
        prec[i] = max(prec[i], prec[i + 1])
    pts = []
    for r in np.linspace(0, 1, 101):
        k = np.searchsorted(rec, r, side="left")
        pts.append(prec[k] if k < len(prec) else 0.0)
    return float(np.mean(pts))


# ---------------------------------------------------------------- D2: Hungarian operating point
def op_hungarian(gt, dets, conf):
    TP = FP = FN = 0
    for img, g in gt.items():
        d = dets.get(img, np.zeros((0, 5)))
        d = d[d[:, 4] >= conf]
        if len(d) and len(g):
            ious = iou_xywh(d[:, :4], g)
            r, c = linear_sum_assignment(-((ious >= 0.5) + 1e-3 * ious))   # max #matches, then IoU
            m = int((ious[r, c] >= 0.5).sum())
        else:
            m = 0
        TP += m; FP += len(d) - m; FN += len(g) - m
    P_, R_ = TP / max(TP + FP, 1), TP / max(TP + FN, 1)
    return {"precision": P_, "recall": R_, "F1": 2 * P_ * R_ / max(P_ + R_, 1e-9), "TP": TP, "FP": FP, "FN": FN}


def best_f1_conf(gt, dets):
    grid = np.linspace(0.01, 0.99, 99)
    f1 = []
    for c in grid:   # greedy matching, as the protocol defines the threshold
        TP = FP = 0; n_gt = 0
        for img, g in gt.items():
            n_gt += len(g)
            d = dets.get(img, np.zeros((0, 5))); d = d[d[:, 4] >= c]
            d = d[np.argsort(-d[:, 4], kind="stable")]
            if len(d) and len(g):
                ious = iou_xywh(d[:, :4], g); free = np.ones(len(g), bool)
                for i in range(len(d)):
                    cand = np.where(free, ious[i], -1); j = int(cand.argmax())
                    if cand[j] >= 0.5:
                        free[j] = False; TP += 1
                    else:
                        FP += 1
            else:
                FP += len(d)
        p, r = TP / max(TP + FP, 1), TP / max(n_gt, 1)
        f1.append(2 * p * r / max(p + r, 1e-9))
    return float(grid[int(np.argmax(f1))])


def check(rows, name, ok, **info):
    rows.append({"check": name, "pass": bool(ok), **info})
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {json.dumps(info, default=float)}", flush=True)


def detection():
    rows = []
    M = json.load(open(R / "metrics_splits.json"))
    gts = {s: load_gt(s) for s in ("train", "valid", "test")}
    for name in M:
        for split in ("train", "valid", "test"):
            dets = load_dets(name, split)
            ref = M[name][split]
            mine = {"AP50-95": np.mean([ap_scratch(gts[split], dets, t) for t in np.linspace(0.5, 0.95, 10)]),
                    "AP50": ap_scratch(gts[split], dets, 0.5), "AP75": ap_scratch(gts[split], dets, 0.75)}
            diff = max(abs(mine[k] - ref[k]) for k in mine)
            check(rows, f"D1 AP from scratch vs pycocotools | {name} | {split}", diff <= 0.002,
                  scratch={k: round(v, 4) for k, v in mine.items()},
                  pycocotools={k: round(ref[k], 4) for k in mine}, max_abs_diff=round(diff, 5))
            hung = op_hungarian(gts[split], dets, ref["conf"])
            d = abs(hung["F1"] - ref["F1"])
            check(rows, f"D2 F1 Hungarian vs greedy | {name} | {split}", d <= 0.01,
                  hungarian_F1=round(hung["F1"], 4), greedy_F1=round(ref["F1"], 4), conf=ref["conf"])
        c = best_f1_conf(gts["valid"], load_dets(name, "valid"))
        check(rows, f"D3 operating threshold = best valid F1 | {name}", abs(c - M[name]["test"]["conf"]) < 1e-6,
              rederived=c, stored=M[name]["test"]["conf"])

    # D4: Ultralytics' validator (independent matcher + its own NMS path) on the fine-tuned YOLO26n
    from ultralytics import YOLO
    with contextlib.redirect_stdout(io.StringIO()):
        v = YOLO(P.YOLO_FT).val(data="datasets/yolo/data.yaml", split="test", imgsz=1280, batch=8, conf=0.001,
                                iou=0.7, max_det=300, plots=False, verbose=False, project="/tmp", name="xcheck",
                                exist_ok=True)
    ul = {"AP50-95": float(v.box.map), "AP50": float(v.box.map50), "AP75": float(v.box.map75)}
    ref = M["YOLO26n (fine-tuned)"]["test"]
    diff = max(abs(ul[k] - ref[k]) for k in ul)
    check(rows, "D4 Ultralytics val vs our protocol | YOLO26n (fine-tuned) | test", diff <= 0.02,
          ultralytics={k: round(x, 4) for k, x in ul.items()}, ours={k: round(ref[k], 4) for k in ul},
          max_abs_diff=round(diff, 4))

    # D5: label bookkeeping
    st = json.load(open("datasets/stats.json"))
    for s, g in gts.items():
        n_box = sum(len(b) for b in g.values())
        n_txt = sum(len(Path(p).read_text().split("\n")) - 1 for p in Path(f"datasets/yolo/{s}/labels").glob("*.txt"))
        check(rows, f"D5 label counts | {s}", n_box == st[s]["boxes"] == n_txt and len(g) == st[s]["images"],
              coco_boxes=n_box, yolo_txt_boxes=n_txt, stats_boxes=st[s]["boxes"], images=len(g))

    # D6: CIs and significance are internally consistent
    for name in M:
        t = M[name]["test"]
        for k in ("AP50-95", "AP50", "F1"):
            lo, hi = t[f"{k}_CI95"]
            check(rows, f"D6 CI contains estimate | {name} | {k}", lo <= t[k] <= hi, est=round(t[k], 4), ci=[lo, hi])
    sig = json.load(open(R / "significance.json"))
    for pair, s in sig.items():
        lo, hi = s["CI95"]
        consistent = (lo > 0 or hi < 0) == (s["p_two_sided"] < 0.05)
        check(rows, f"D6 significance consistent with CI | {pair}", consistent, delta=round(s["delta_AP50-95"], 4),
              ci=[round(lo, 4), round(hi, 4)], p=s["p_two_sided"])
    return rows


def tracking():
    import track
    rows = []
    fps, cache = track.load_cache()
    n = len(cache["teacher"]); half = n // 2
    gt = track.build_gt(fps, cache["teacher"])
    rep = json.load(open(track.OUT / "tracking_metrics.json"))
    import motmetrics as mm
    mh = mm.metrics.create()
    for s in track.STUDENTS:
        pred = track.run_tracker(cache[s], rep[s]["best_params"], fps)
        now = track.score(gt, pred, half, n); ref = rep[s]["tuned"]["heldout_15-30s"]
        d = max(abs(now[k] - ref[k]) for k in ("HOTA", "MOTA", "IDF1"))
        check(rows, f"T1 tracker re-run reproduces stored metrics | {s}", d < 1e-9,
              rerun={k: round(now[k], 4) for k in ("HOTA", "MOTA", "IDF1")},
              stored={k: round(ref[k], 4) for k in ("HOTA", "MOTA", "IDF1")})
        acc = mm.MOTAccumulator(auto_id=True)
        for (gb, gi), (pb, pi) in zip(gt[half:n], pred[half:n]):
            if len(gb) and len(pb):
                g4 = np.c_[gb[:, :2], gb[:, 2:] - gb[:, :2]]; p4 = np.c_[pb[:, :2], pb[:, 2:] - pb[:, :2]]
                dist = 1 - iou_xywh(g4, p4); dist[dist > 0.5] = np.nan
            else:
                dist = np.full((len(gb), len(pb)), np.nan)
            acc.update(list(map(int, gi)), list(map(int, pi)), dist)
        m = mh.compute(acc, metrics=["mota", "idf1", "num_switches"], name="x").iloc[0]
        ok = abs(m["mota"] - ref["MOTA"]) <= 0.01 and abs(m["idf1"] - ref["IDF1"]) <= 0.01 \
            and abs(m["num_switches"] - ref["IDSW"]) <= max(3, 0.1 * ref["IDSW"])
        check(rows, f"T2 py-motmetrics vs trackers.eval | {s}", ok,
              motmetrics={"MOTA": round(float(m["mota"]), 4), "IDF1": round(float(m["idf1"]), 4),
                          "IDSW": int(m["num_switches"])},
              trackers_eval={"MOTA": round(ref["MOTA"], 4), "IDF1": round(ref["IDF1"], 4), "IDSW": ref["IDSW"]})
    return rows


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    res = {}
    if mode in ("detection", "all"):
        res["detection"] = detection()
    if mode in ("tracking", "all"):
        res["tracking"] = tracking()
    for k, rows in res.items():
        json.dump(rows, open(R / f"crosscheck_{k}.json", "w"), indent=1, default=float)
    fails = [r["check"] for rows in res.values() for r in rows if not r["pass"]]
    total = sum(len(r) for r in res.values())
    print(f"\n{total - len(fails)}/{total} checks passed" + ("" if not fails else "\nFAILED:\n  " + "\n  ".join(fails)))
    sys.exit(1 if fails else 0)
