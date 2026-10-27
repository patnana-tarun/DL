"""Build a person-detection dataset from a video using teacher pseudo-labels.

Frames after the tracking hold-out (default: first 30 s) are sampled, labelled by an
RF-DETR Large + YOLO26x ensemble fused with WBF, split by time into train/valid/test
(contiguous blocks, so neighbouring near-duplicate frames never straddle two splits
except at the boundaries), and written in both YOLO and COCO formats.

    python research/prepare_data.py                                   # aa1.mp4 -> datasets/ (as published)
    python research/prepare_data.py --video street.mp4 --name street  # -> datasets/street/, 2/3-1/6-1/6 time split
"""
import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
from ensemble_boxes import weighted_boxes_fusion
from rfdetr import RFDETRLarge
from ultralytics import YOLO

import paths as P

VIDEO = P.VIDEO_AA1
ROOT = Path("datasets")
STEP_S = 0.25                      # sample a frame every 0.25 s
SPLITS = {"train": (29.9, 210), "valid": (210, 255), "test": (255, 300.6)}   # aa1.mp4 split, in seconds
FUSED_THR = 0.40                   # keep fused boxes above this score
MIN_BOX = 8                        # drop boxes smaller than this (px, min side)


def split_of(t):
    for name, (a, b) in SPLITS.items():
        if a <= t < b:
            return name


def main():
    global VIDEO, ROOT, STEP_S, SPLITS, YOLO_DIR, COCO_DIR, FRAMES_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=VIDEO)
    ap.add_argument("--name", default="", help="dataset sub-folder; empty = datasets/ (aa1)")
    ap.add_argument("--holdout", type=float, default=30, help="leading seconds kept out for tracking")
    ap.add_argument("--end", type=float, default=None, help="last second to use (default: whole video)")
    ap.add_argument("--step", type=float, default=STEP_S)
    ap.add_argument("--fractions", default="0.667,0.167,0.166", help="train,valid,test time fractions")
    a = ap.parse_args()
    VIDEO, STEP_S = a.video, a.step
    ROOT = Path("datasets") / a.name if a.name else ROOT
    YOLO_DIR, COCO_DIR, FRAMES_DIR = ROOT / "yolo", ROOT / "coco", ROOT / "frames"
    cap = cv2.VideoCapture(VIDEO)
    dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    end = min(a.end or dur, dur)
    if a.name:   # new videos: split the post-hold-out time range by fractions
        f = [float(x) for x in a.fractions.split(",")]
        edges = np.cumsum([a.holdout - 0.1] + [(end - a.holdout) * x for x in f])
        edges[-1] = end + 0.1
        SPLITS = {s: (edges[i], edges[i + 1]) for i, s in enumerate(("train", "valid", "test"))}
    print("splits (s):", {k: (round(x, 1), round(y, 1)) for k, (x, y) in SPLITS.items()})

    for d in (FRAMES_DIR, *(YOLO_DIR / s / k for s in SPLITS for k in ("images", "labels")),
              *(COCO_DIR / s for s in SPLITS)):
        d.mkdir(parents=True, exist_ok=True)

    fps = cap.get(cv2.CAP_PROP_FPS)
    n = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), int(end * fps) + 1)
    step = int(round(STEP_S * fps))
    frames = []
    for idx in range(int(a.holdout * fps), n, step):
        name = f"f{idx:06d}.jpg"
        if (FRAMES_DIR / name).exists():
            frames.append((idx, idx / fps, name))
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, img = cap.read()
        if not ok:
            continue
        cv2.imwrite(str(FRAMES_DIR / name), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        frames.append((idx, idx / fps, name))
    H, W = cv2.imread(str(FRAMES_DIR / frames[0][2])).shape[:2]
    print(f"extracted {len(frames)} frames ({W}x{H})")

    rf = RFDETRLarge()
    yolo = YOLO(P.YOLO_TEACHER)

    coco = {s: {"images": [], "annotations": [], "categories": [{"id": 1, "name": "person", "supercategory": "none"}]}
            for s in SPLITS}
    stats = {s: [0, 0] for s in SPLITS}
    for i, (idx, t, name) in enumerate(frames):
        path = str(FRAMES_DIR / name)
        img = cv2.imread(path)
        boxes, scores = [], []

        d = rf.predict(img[:, :, ::-1].copy(), threshold=0.15)
        m = d.class_id == 1
        boxes.append(d.xyxy[m] / [W, H, W, H]); scores.append(d.confidence[m])

        r = yolo.predict(path, imgsz=1280, conf=0.15, classes=[0], augment=True, verbose=False)[0]
        boxes.append(r.boxes.xyxy.cpu().numpy() / [W, H, W, H]); scores.append(r.boxes.conf.cpu().numpy())

        boxes = [np.clip(b, 0, 1) for b in boxes]
        fb, fs, _ = weighted_boxes_fusion(boxes, scores, [np.zeros(len(s)) for s in scores],
                                          weights=[1, 1], iou_thr=0.55, skip_box_thr=0.15, conf_type="avg")
        keep = fs >= FUSED_THR
        fb = fb[keep] * [W, H, W, H]
        fb = fb[np.minimum(fb[:, 2] - fb[:, 0], fb[:, 3] - fb[:, 1]) >= MIN_BOX]

        s = split_of(t)
        if s is None:
            continue
        stats[s][0] += 1; stats[s][1] += len(fb)
        os.symlink(os.path.abspath(path), YOLO_DIR / s / "images" / name) if not (YOLO_DIR / s / "images" / name).exists() else None
        os.symlink(os.path.abspath(path), COCO_DIR / s / name) if not (COCO_DIR / s / name).exists() else None
        with open(YOLO_DIR / s / "labels" / name.replace(".jpg", ".txt"), "w") as f:
            for x1, y1, x2, y2 in fb:
                f.write(f"0 {(x1+x2)/2/W:.6f} {(y1+y2)/2/H:.6f} {(x2-x1)/W:.6f} {(y2-y1)/H:.6f}\n")
        img_id = len(coco[s]["images"]) + 1
        coco[s]["images"].append({"id": img_id, "file_name": name, "width": W, "height": H, "frame": idx})
        for x1, y1, x2, y2 in fb:
            w, h = float(x2 - x1), float(y2 - y1)
            coco[s]["annotations"].append({"id": len(coco[s]["annotations"]) + 1, "image_id": img_id, "category_id": 1,
                                           "bbox": [float(x1), float(y1), w, h], "area": w * h, "iscrowd": 0})
        if i % 100 == 0:
            print(f"{i}/{len(frames)} {name} -> {len(fb)} persons", flush=True)

    for s in SPLITS:
        with open(COCO_DIR / s / "_annotations.coco.json", "w") as f:
            json.dump(coco[s], f)
    (YOLO_DIR / "data.yaml").write_text(
        f"path: {YOLO_DIR.resolve()}\ntrain: train/images\nval: valid/images\ntest: test/images\nnames:\n  0: person\n")
    json.dump({s: {"images": a, "boxes": b} for s, (a, b) in stats.items()}, open(ROOT / "stats.json", "w"), indent=1)
    print(stats)


if __name__ == "__main__":
    main()
