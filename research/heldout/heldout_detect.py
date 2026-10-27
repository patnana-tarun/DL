"""Held-out test video, stage 1: one sequential pass over the whole video (test only, nothing is trained or tuned).

For EVERY frame (native rate, read sequentially because the phone video has a variable frame rate):
  teacher   RF-DETR Large + YOLO26x (TTA) fused with WBF, the same recipe as the aa1 labels (prepare_data.py)
  students  fine-tuned YOLO26n and RF-DETR Nano through the deployed fast paths, conf >= 0.05 (as track.py)
  pts       presentation timestamp of the frame (s)
-> <out>/detections_full.npz, the tracking input (pseudo-GT tracks are built from the teacher boxes).

Every STEP_S seconds of PTS (first frame at or after k * STEP_S) the frame is also written as JPEG q95 to
datasets/<name>/frames/ and the teachers are re-run on the decoded JPEG, exactly as for the aa1 detection
splits (labels and students see the same compressed image). Each teacher's own boxes are kept for the
label-bias check.
-> datasets/<name>/coco/test/_annotations.coco.json   (fused labels, fused score >= 0.40, min side >= 8 px)
   datasets/<name>/teachers.json                      (per-frame fused / RF-DETR Large / YOLO26x boxes + scores)

The pass is chunked (CHUNK frames per part) and resumes from the last finished part.

    python research/heldout/heldout_detect.py --video videos/test1.mp4 --name test1
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths as P  # noqa: E402  (project root as working directory)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from ensemble_boxes import weighted_boxes_fusion  # noqa: E402

STEP_S = 0.25          # detection-set sampling period (s of PTS), as prepare_data.py
FUSED_THR = 0.40       # label cut on the fused score (prepare_data.FUSED_THR)
MIN_BOX = 8            # px, min side (prepare_data.MIN_BOX)
CHUNK = 2000
STUDENT_CONF = 0.05    # track.py keeps low boxes for ByteTrack's second association


class Teacher:
    def __init__(self):
        from rfdetr import RFDETRLarge
        from ultralytics import YOLO
        self.rf, self.yolo = RFDETRLarge(), YOLO(P.YOLO_TEACHER)

    def __call__(self, img):
        """BGR uint8 -> fused [N,5] (x1 y1 x2 y2 score, pixels), plus each teacher's raw [N,5] at score >= 0.15."""
        H, W = img.shape[:2]
        d = self.rf.predict(img[:, :, ::-1].copy(), threshold=0.15); m = d.class_id == 1
        r = self.yolo.predict(img, imgsz=1280, conf=0.15, classes=[0], augment=True, verbose=False)[0]
        raw = {"rfdetr_large": np.c_[d.xyxy[m], d.confidence[m]],
               "yolo26x": np.c_[r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()]}
        bl = [np.clip(v[:, :4] / [W, H, W, H], 0, 1) for v in raw.values()]
        sl = [v[:, 4] for v in raw.values()]
        fb, fs, _ = weighted_boxes_fusion(bl, sl, [np.zeros(len(s)) for s in sl], weights=[1, 1], iou_thr=0.55,
                                          skip_box_thr=0.15, conf_type="avg")
        return np.c_[fb * [W, H, W, H], fs].reshape(-1, 5), raw


def pack(arrs):
    """list of [n_i,5] -> (concatenated [sum n_i,5], offsets [len+1])."""
    off = np.cumsum([0] + [len(a) for a in arrs])
    return (np.concatenate(arrs) if len(arrs) else np.zeros((0, 5))).astype(np.float32), off


def unpack(data, off):
    return [data[off[i]:off[i + 1]] for i in range(len(off) - 1)]


def load_full(path):
    """detections_full.npz -> (pts [N], {"teacher"|"YOLO26n"|"RF-DETR Nano": [N arrays [n,5]]})."""
    z = np.load(path)
    keys = sorted({k.split("__")[0] for k in z.files if "__" in k})
    return z["pts"], {k: unpack(z[f"{k}__data"], z[f"{k}__off"]) for k in keys}


def main():
    import detectors as D
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True); ap.add_argument("--name", required=True)
    ap.add_argument("--out", default=None, help="default results/<name>")
    ap.add_argument("--data", default=None, help="default datasets/<name>")
    a = ap.parse_args()
    out = Path(a.out or f"results/{a.name}"); parts = out / "parts"; parts.mkdir(parents=True, exist_ok=True)
    data = Path(a.data or f"datasets/{a.name}"); frames_dir = data / "frames"; frames_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(a.video)
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    teacher = Teacher()
    students = {"YOLO26n": D.yolo(P.YOLO_FT, conf=STUDENT_CONF),
                "RF-DETR Nano": D.rfdetr(D.rfdetr_best(P.RFDETR_FT_RUN), resolution=640, conf=STUDENT_CONF)}
    i, next_t, k_sample = 0, 0.0, 0
    sampled = json.load(open(data / "teachers.json")) if (data / "teachers.json").exists() else []
    done = sorted(parts.glob("part_*.npz"))
    if done:   # resume: skip finished parts (grab() decodes without converting, ~1 ms per frame)
        last = np.load(done[-1]); i = int(last["end"]); next_t = float(last["next_t"])
        sampled = [s for s in sampled if s["frame"] < i]; k_sample = len(sampled)
        for _ in range(i):
            cap.grab()
        print(f"resume at frame {i}, {k_sample} sampled frames kept", flush=True)
    buf = {k: [] for k in ("teacher", *students)}; pts = []; start = i
    while True:
        ok, img = cap.read()
        if not ok:
            break
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        fused, _ = teacher(img)
        buf["teacher"].append(fused)
        for s, run in students.items():
            b, sc = run(img); buf[s].append(np.c_[b, sc].reshape(-1, 5))
        pts.append(t)
        if t >= next_t - 1e-6:   # detection-set frame: JPEG round trip, teachers on the decoded JPEG
            name = f"f{i:06d}.jpg"
            cv2.imwrite(str(frames_dir / name), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            fj, raw = teacher(cv2.imread(str(frames_dir / name)))
            sampled.append({"frame": i, "t": round(t, 4), "file_name": name,
                            "fused": fj.round(2).tolist(), **{k: v.round(2).tolist() for k, v in raw.items()}})
            k_sample += 1
            while next_t <= t + 1e-6:
                next_t += STEP_S
        i += 1
        if i % CHUNK == 0 or i == n_total:
            np.savez_compressed(parts / f"part_{start:06d}.npz", start=start, end=i, next_t=next_t, pts=np.array(pts),
                                **{f"{k}__{x}": y for k, v in buf.items() for x, y in zip(("data", "off"), pack(v))})
            json.dump(sampled, open(data / "teachers.json", "w"))
            print(f"frame {i}/{n_total}  t={t:.1f}s  sampled {k_sample}  teacher boxes {len(fused)}", flush=True)
            buf = {k: [] for k in buf}; pts = []; start = i
    if pts:
        np.savez_compressed(parts / f"part_{start:06d}.npz", start=start, end=i, next_t=next_t, pts=np.array(pts),
                            **{f"{k}__{x}": y for k, v in buf.items() for x, y in zip(("data", "off"), pack(v))})
        json.dump(sampled, open(data / "teachers.json", "w"))

    # merge parts -> detections_full.npz
    allp = [np.load(p) for p in sorted(parts.glob("part_*.npz"))]
    merged = {"pts": np.concatenate([p["pts"] for p in allp])}
    for k in ("teacher", *students):
        merged[f"{k}__data"], merged[f"{k}__off"] = pack([x for p in allp for x in unpack(p[f"{k}__data"], p[f"{k}__off"])])
    np.savez_compressed(out / "detections_full.npz", **merged)
    print("frames", len(merged["pts"]), "of", n_total)

    # COCO test split from the fused JPEG labels
    H, W = cv2.imread(str(frames_dir / sampled[0]["file_name"])).shape[:2]
    coco = {"images": [], "annotations": [], "categories": [{"id": 1, "name": "person", "supercategory": "none"}]}
    for s in sampled:
        img_id = len(coco["images"]) + 1
        coco["images"].append({"id": img_id, "file_name": s["file_name"], "width": W, "height": H,
                               "frame": s["frame"], "t": s["t"]})
        for x1, y1, x2, y2, sc in s["fused"]:
            if sc < FUSED_THR or min(x2 - x1, y2 - y1) < MIN_BOX:
                continue
            w, h = x2 - x1, y2 - y1
            coco["annotations"].append({"id": len(coco["annotations"]) + 1, "image_id": img_id, "category_id": 1,
                                        "bbox": [x1, y1, w, h], "area": w * h, "iscrowd": 0, "score": sc})
    split = data / "coco" / "test"; split.mkdir(parents=True, exist_ok=True)
    for s in sampled:   # images next to the annotations, as in datasets/coco/<split>/
        if not (split / s["file_name"]).exists():
            (split / s["file_name"]).symlink_to((frames_dir / s["file_name"]).resolve())
    json.dump(coco, open(split / "_annotations.coco.json", "w"))
    stats = {"video": a.video, "frames_total": len(merged["pts"]), "duration_s": float(merged["pts"][-1]),
             "test_images": len(coco["images"]), "test_boxes": len(coco["annotations"]), "width": W, "height": H}
    json.dump(stats, open(data / "stats.json", "w"), indent=1)
    print(stats)


if __name__ == "__main__":
    main()
