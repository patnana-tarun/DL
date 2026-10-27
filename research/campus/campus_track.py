"""Stage 1 of the campus analytics: detect + track every person, keep the best crops per track.

Detector: YOLO26x (the teacher from the aa1 study) at 1280 px. The fine-tuned YOLO26n won that study,
but it was fitted to Times Square pseudo-labels at 720p; on a new domain the large pretrained model
is the more accurate choice, and nothing here needs real time.
Tracker: BoT-SORT with ReID (native YOLO features) + global motion compensation, 3 s track buffer.

Per track we keep the top-K crops by a quality score (box height, confidence, not cut by the frame
edge, not overlapped by another person), at least MIN_GAP frames apart so the votes in stage 2
see different poses.

    python research/campus/campus_track.py --video videos/video_1.mp4
Writes results/campus/<name>/tracks.npz and results/campus/<name>/crops/<id>_<frame>.jpg
"""
import argparse
import heapq
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths as P  # noqa: E402,F401  (project root as working directory)

import cv2
import numpy as np
from ultralytics import YOLO

TRACKER_CFG = """tracker_type: botsort
track_high_thresh: 0.4
track_low_thresh: 0.1
new_track_thresh: 0.5
track_buffer: {buffer}
match_thresh: 0.8
fuse_score: True
gmc_method: sparseOptFlow
proximity_thresh: 0.5
appearance_thresh: 0.8
with_reid: True
model: auto
"""
K, MIN_GAP, EVERY = 6, 12, 3      # crops kept per track, min frames between them, score every Nth frame


def quality(box, others, W, H, conf):
    x1, y1, x2, y2 = box
    h = y2 - y1
    edge = 0.35 if (x1 < 4 or y1 < 4 or x2 > W - 4 or y2 > H - 4) else 1.0
    ov = 0.0
    for o in others:                              # fraction of this box covered by any other person
        ix = max(0, min(x2, o[2]) - max(x1, o[0])); iy = max(0, min(y2, o[3]) - max(y1, o[1]))
        ov = max(ov, ix * iy / ((x2 - x1) * h + 1e-6))
    return h * conf * edge * (1 - ov) ** 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--model", default=P.YOLO_TEACHER)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.1)
    a = ap.parse_args()
    name = Path(a.video).stem
    out = Path("results/campus") / name; (out / "crops").mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(a.video); fps = cap.get(cv2.CAP_PROP_FPS); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W, H = int(cap.get(3)), int(cap.get(4)); cap.release()
    cfg = out / "botsort_reid.yaml"; cfg.write_text(TRACKER_CFG.format(buffer=int(3 * fps)))

    model = YOLO(a.model)
    rows, best = [], {}                           # rows: frame,id,x1,y1,x2,y2,conf ; best[id] = heap of (score, frame, crop)
    for f, r in enumerate(model.track(a.video, stream=True, persist=True, tracker=str(cfg), classes=[0],
                                      imgsz=a.imgsz, conf=a.conf, half=True, verbose=False)):
        if f % 500 == 0:
            print(f"{name}: frame {f}/{n}", flush=True)
        b = r.boxes
        if b.id is None:
            continue
        xyxy, ids, cf = b.xyxy.cpu().numpy(), b.id.int().cpu().numpy(), b.conf.cpu().numpy()
        for bx, i, c in zip(xyxy, ids, cf):
            rows.append((f, i, *bx, c))
        if f % EVERY:
            continue
        for k, (bx, i, c) in enumerate(zip(xyxy, ids, cf)):
            if bx[3] - bx[1] < 80:                # too small to say anything about clothes / ID card
                continue
            s = quality(bx, np.delete(xyxy, k, 0), W, H, c)
            h = best.setdefault(int(i), [])
            if any(abs(f - g) < MIN_GAP for _, g, _ in h):
                # replace a close-in-time crop only if this one is better
                j = next(j for j, (_, g, _) in enumerate(h) if abs(f - g) < MIN_GAP)
                if s <= h[j][0]:
                    continue
                h.pop(j); heapq.heapify(h)
            elif len(h) >= K and s <= h[0][0]:
                continue
            x1, y1, x2, y2 = bx; pw, ph = 0.08 * (x2 - x1), 0.04 * (y2 - y1)   # small context margin
            crop = r.orig_img[max(0, int(y1 - ph)):int(y2 + ph), max(0, int(x1 - pw)):int(x2 + pw)].copy()
            heapq.heappush(h, (s, f, crop))
            if len(h) > K:
                heapq.heappop(h)

    for f in (out / "crops").glob("*.jpg"):
        f.unlink()
    for i, h in best.items():
        for s, f, crop in h:
            cv2.imwrite(str(out / "crops" / f"{i}_{f}_{s:.0f}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
    rows = np.array(rows, dtype=np.float64)
    np.savez_compressed(out / "tracks.npz", rows=rows, fps=fps, W=W, H=H, n=n)
    print(f"{name}: {n} frames, {len(np.unique(rows[:, 1]))} raw track IDs, crops for {len(best)} tracks")


if __name__ == "__main__":
    main()
