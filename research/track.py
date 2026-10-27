"""ByteTrack on the first 30 s of aa1.mp4 with the fine-tuned YOLO26n and RF-DETR Nano.

Steps (each cached, so re-runs are cheap):
  detect  - run teacher ensemble (cached after the first run) + both students on every frame of 0-30 s
  gt      - build pseudo ground-truth tracks from the teacher detections
  tune    - grid-search ByteTrack params per student on 0-15 s (by HOTA)
  report  - score the best params on the unseen 15-30 s and on the full 30 s
  render  - write annotated videos (per model + side-by-side)
"""
import itertools
import json
import subprocess
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from trackers import ByteTrackTracker
from trackers.eval import (box_iou, compute_clear_metrics, compute_hota_metrics,
                           compute_identity_metrics)

import paths as P

VIDEO = P.VIDEO_AA1
OUT = Path("results/tracking")
CACHE = OUT / "detections.npz"
SECONDS = 30
STUDENTS = ["YOLO26n", "RF-DETR Nano"]
YOLO_W = P.YOLO_FT
RF_CKPT = f"{P.RFDETR_FT_RUN}/checkpoint_best_total.pth"
# min_conf / nms_iou pre-filter detections before ByteTrack; the rest are ByteTrackTracker args.
# A first 243-combo pass put every optimum on the grid edge, so the ranges extend past it.
GRID = {
    "min_conf": [0.05, 0.1, 0.2, 0.3],
    "nms_iou": [0.0, 0.6],             # 0 = no NMS (RF-DETR emits near-duplicate boxes without it)
    "track_activation_threshold": [0.5, 0.6, 0.7, 0.8],
    "high_conf_det_threshold": [0.4, 0.5, 0.6, 0.7],
    "lost_track_buffer": [30, 60],
    "minimum_iou_threshold": [0.2, 0.3, 0.4, 0.5],
    "minimum_consecutive_frames": [3, 5, 8],
}
# Second pass: the first grid's optima again sat on its edges, so refine around them.
GRID_REFINE = {
    "min_conf": [0.3, 0.4, 0.5, 0.6],   # 0.6: YOLO's optimum sat on 0.5, the old edge
    "nms_iou": [0.0, 0.6],
    "track_activation_threshold": [0.7, 0.8, 0.9],
    "high_conf_det_threshold": [0.4, 0.5, 0.6],
    "lost_track_buffer": [15, 30],
    "minimum_iou_threshold": [0.4, 0.5, 0.6],
    "minimum_consecutive_frames": [1, 2, 3],
}
# Pseudo-GT uses the same 0.40 fused-score cut as the detection labels (prepare_data.FUSED_THR);
# a stricter 0.5 activation dropped real people and showed up as student "false positives".
GT_PARAMS = dict(track_activation_threshold=0.4, high_conf_det_threshold=0.4, lost_track_buffer=120,
                 minimum_iou_threshold=0.2, minimum_consecutive_frames=3)
GT_MIN_LEN = 15          # drop pseudo-GT tracks shorter than 0.25 s


def video_info():
    cap = cv2.VideoCapture(VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS)
    return cap, fps, int(round(SECONDS * fps))


# ---------------------------------------------------------------- detect
def detect():
    """Teacher + student detections for every frame of 0-30 s.

    The teacher pass (RF-DETR Large + YOLO26x TTA, ~1 min per 100 frames) is reused from an
    existing cache; only the students are re-run, through the deployed fast paths in detectors.py.
    """
    import detectors as D

    cap, fps, n = video_info()
    teacher = load_cache()[1]["teacher"] if CACHE.exists() else None
    if teacher is None:
        from ensemble_boxes import weighted_boxes_fusion
        from rfdetr import RFDETRLarge
        from ultralytics import YOLO
        teacher_rf, teacher_yolo = RFDETRLarge(), YOLO(P.YOLO_TEACHER)
    students = {"YOLO26n": D.yolo(YOLO_W, conf=0.05),
                "RF-DETR Nano": D.rfdetr(RF_CKPT, resolution=640, conf=0.05)}
    out = {k: [] for k in ("teacher", *STUDENTS)}
    for i in range(n):
        ok, img = cap.read()
        if not ok:
            break
        H, W = img.shape[:2]
        if teacher is None:   # same recipe as the training labels
            rgb = img[:, :, ::-1].copy()
            d = teacher_rf.predict(rgb, threshold=0.15); m = d.class_id == 1
            r = teacher_yolo.predict(img, imgsz=1280, conf=0.15, classes=[0], augment=True, verbose=False)[0]
            bl = [np.clip(d.xyxy[m] / [W, H, W, H], 0, 1), np.clip(r.boxes.xyxy.cpu().numpy() / [W, H, W, H], 0, 1)]
            sl = [d.confidence[m], r.boxes.conf.cpu().numpy()]
            fb, fs, _ = weighted_boxes_fusion(bl, sl, [np.zeros(len(s)) for s in sl], iou_thr=0.55,
                                              skip_box_thr=0.15, conf_type="avg")
            out["teacher"].append(np.c_[fb * [W, H, W, H], fs])
        else:
            out["teacher"].append(teacher[i])
        # students keep low-score boxes too: ByteTrack's second association stage uses them
        for s, run in students.items():
            b, sc = run(img)
            out[s].append(np.c_[b, sc])
        if i % 300 == 0:
            print(f"detect {i}/{n}", {k: len(v[-1]) for k, v in out.items()}, flush=True)
    np.savez_compressed(CACHE, fps=fps, **{f"{k}/{i}": a for k, v in out.items() for i, a in enumerate(v)})


def load_cache():
    z = np.load(CACHE)
    fps = float(z["fps"])
    per = {}
    for key in z.files:
        if key == "fps":
            continue
        k, i = key.rsplit("/", 1)
        per.setdefault(k, {})[int(i)] = z[key]
    return fps, {k: [v[i] for i in range(len(v))] for k, v in per.items()}


# ---------------------------------------------------------------- tracking + metrics
def run_tracker(dets, params, fps):
    params = dict(params)
    min_conf, nms_iou = params.pop("min_conf", 0.0), params.pop("nms_iou", 0.0)
    t = ByteTrackTracker(frame_rate=fps, **params)
    res = []
    for a in dets:
        a = a[a[:, 4] >= min_conf]
        d = sv.Detections(xyxy=a[:, :4].astype(np.float32), confidence=a[:, 4].astype(np.float32),
                          class_id=np.zeros(len(a), int))
        if nms_iou and len(d):
            d = d.with_nms(threshold=nms_iou, class_agnostic=True)
        o = t.update(d)
        keep = o.tracker_id >= 0 if len(o) else np.zeros(0, bool)
        res.append((o.xyxy[keep], o.tracker_id[keep]))
    return res


def build_gt(fps, teacher):
    dets = [a[a[:, 4] >= 0.4] for a in teacher]
    tr = run_tracker(dets, GT_PARAMS, fps)
    lengths = {}
    for _, ids in tr:
        for j in ids:
            lengths[j] = lengths.get(j, 0) + 1
    return [(b[[lengths[j] >= GT_MIN_LEN for j in ids]], ids[[lengths[j] >= GT_MIN_LEN for j in ids]])
            for b, ids in tr]


def score(gt, pred, lo, hi):
    g_ids, p_ids, sims = [], [], []
    for (gb, gi), (pb, pi) in zip(gt[lo:hi], pred[lo:hi]):
        g_ids.append(gi); p_ids.append(pi)
        sims.append(box_iou(gb, pb) if len(gb) and len(pb) else np.zeros((len(gb), len(pb))))
    clear = compute_clear_metrics(g_ids, p_ids, sims)
    hota = compute_hota_metrics(g_ids, p_ids, sims)
    ident = compute_identity_metrics(g_ids, p_ids, sims)
    n_ids = len({int(j) for _, ids in pred[lo:hi] for j in ids})
    return {"HOTA": float(np.mean(hota["HOTA"])), "DetA": float(np.mean(hota["DetA"])),
            "AssA": float(np.mean(hota["AssA"])), "MOTA": clear["MOTA"], "MOTP": clear["MOTP"],
            "IDF1": ident["IDF1"], "IDSW": int(clear["IDSW"]), "Frag": int(clear["Frag"]),
            "MT": int(clear["MT"]), "ML": int(clear["ML"]), "FP": int(clear["CLR_FP"]),
            "FN": int(clear["CLR_FN"]), "pred_IDs": n_ids,
            "gt_IDs": len({int(j) for _, ids in gt[lo:hi] for j in ids})}


_SHARED = {}   # set before the Pool forks so workers inherit it instead of unpickling per job


def _grid_job(args):
    name, params = args
    fps, cache, gt, hi = (_SHARED[k] for k in ("fps", "cache", "gt", "hi"))
    return name, params, score(gt, run_tracker(cache[name][:hi], params, fps), 0, hi)


def tune_and_report():
    fps, cache = load_cache()
    n = len(cache["teacher"]); half = n // 2
    gt = build_gt(fps, cache["teacher"])
    json.dump({"frames": n, "gt_tracks": len({int(j) for _, ids in gt for j in ids}), "gt_params": GT_PARAMS},
              open(OUT / "pseudo_gt_info.json", "w"), indent=2)
    combos = {tuple(v): dict(zip(g, v)) for g in (GRID, GRID_REFINE) for v in itertools.product(*g.values())}
    combos = list(combos.values())
    _SHARED.update(fps=fps, cache=cache, gt=gt, hi=half)
    jobs = [(s, p) for s in STUDENTS for p in combos]
    with Pool(18) as pool:
        rows = []
        for i, r in enumerate(pool.imap_unordered(_grid_job, jobs, chunksize=8)):
            rows.append(r)
            if i % 500 == 0:
                print(f"grid {i}/{len(jobs)}", flush=True)
    json.dump([{"model": s, **p, **m} for s, p, m in rows], open(OUT / "grid_search.json", "w"))
    report = {}
    default = dict(track_activation_threshold=0.7, high_conf_det_threshold=0.6, lost_track_buffer=30,
                   minimum_iou_threshold=0.1, minimum_consecutive_frames=2)
    for s in STUDENTS:
        best = max((r for r in rows if r[0] == s), key=lambda r: r[2]["HOTA"])[1]
        report[s] = {"best_params": best}
        for tag, params in (("default", default), ("tuned", best)):
            pred = run_tracker(cache[s], params, fps)
            report[s][tag] = {"tune_0-15s": score(gt, pred, 0, half), "heldout_15-30s": score(gt, pred, half, n),
                              "full_0-30s": score(gt, pred, 0, n)}
        print(s, json.dumps(report[s]["tuned"]["heldout_15-30s"]), flush=True)
    json.dump(report, open(OUT / "tracking_metrics.json", "w"), indent=2)
    return report


# ---------------------------------------------------------------- render
def render(report):
    fps, cache = load_cache()
    tracks = {s: run_tracker(cache[s], report[s]["best_params"], fps) for s in STUDENTS}
    cap, _, n = video_info()
    ok, img = cap.read(); H, W = img.shape[:2]; cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    palette = sv.ColorPalette.from_matplotlib("tab20", 20)
    ann = {s: (sv.BoxAnnotator(color=palette, thickness=2, color_lookup=sv.ColorLookup.TRACK),
               sv.LabelAnnotator(color=palette, text_scale=0.5, text_padding=3, color_lookup=sv.ColorLookup.TRACK),
               sv.TraceAnnotator(color=palette, trace_length=90, thickness=2, color_lookup=sv.ColorLookup.TRACK))
           for s in STUDENTS}
    raw = {s: cv2.VideoWriter(str(OUT / f"_{s}.avi"), cv2.VideoWriter_fourcc(*"MJPG"), fps, (W, H)) for s in STUDENTS}
    side = cv2.VideoWriter(str(OUT / "_side.avi"), cv2.VideoWriter_fourcc(*"MJPG"), fps, (W, H // 2))
    for i in range(n):
        ok, img = cap.read()
        if not ok:
            break
        frames = []
        for s in STUDENTS:
            b, ids = tracks[s][i]
            d = sv.Detections(xyxy=b, tracker_id=ids, class_id=np.zeros(len(b), int))
            f = img.copy()
            box, lab, trace = ann[s]
            f = trace.annotate(f, d); f = box.annotate(f, d)
            f = lab.annotate(f, d, labels=[f"#{j}" for j in ids])
            seen = len({int(j) for _, x in tracks[s][:i + 1] for j in x})
            text = f"{s} + ByteTrack  t={i / fps:5.2f}s  live={len(ids)}  IDs so far={seen}"
            (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(f, (0, 0), (tw + 20, 44), (0, 0, 0), -1)
            cv2.putText(f, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            raw[s].write(f); frames.append(cv2.resize(f, (W // 2, H // 2)))
        side.write(np.hstack(frames))
    for w in (*raw.values(), side):
        w.release()
    ff = "ffmpeg"
    for src, dst in [*((OUT / f"_{s}.avi", OUT / f"track_{s.replace(' ', '_')}.mp4") for s in STUDENTS),
                     (OUT / "_side.avi", OUT / "track_side_by_side.mp4")]:
        subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(src), "-c:v", "libx264", "-crf", "20",
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst)], check=True)
        src.unlink()


if __name__ == "__main__":
    # python research/track.py [detect tune render]                                      # aa1, first 30 s
    # python research/track.py --video street.mp4 --out results/street/tracking --yolo runs/street_yolo26n/weights/best.pt
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("steps", nargs="*", default=["detect", "tune", "render"])
    ap.add_argument("--video", default=VIDEO); ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--seconds", type=float, default=SECONDS)
    ap.add_argument("--yolo", default=None, help="YOLO weights (default: aa1 fine-tuned)")
    ap.add_argument("--rfdetr", default=None, help="RF-DETR checkpoint (default: best aa1 run)")
    a = ap.parse_args()
    VIDEO, OUT, SECONDS = a.video, Path(a.out), a.seconds
    CACHE = OUT / "detections.npz"
    YOLO_W = a.yolo or YOLO_W
    RF_CKPT = a.rfdetr or RF_CKPT
    OUT.mkdir(parents=True, exist_ok=True)
    steps = a.steps
    if "detect" in steps:
        detect()
    rep = tune_and_report() if "tune" in steps else json.load(open(OUT / "tracking_metrics.json"))
    if "render" in steps:
        render(rep)
