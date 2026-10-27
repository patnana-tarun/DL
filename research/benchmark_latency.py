"""Latency benchmark: every inference variant of fine-tuned YOLO26n / RF-DETR Nano.

Each variant is timed end-to-end on the 182 test frames (BGR uint8 in -> boxes out,
including pre/post-processing, batch 1) and re-scored with the COCO evaluator, so a
speed-up that costs accuracy is visible. Writes results/latency.json.
"""
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from pycocotools.coco import COCO

import detectors as D
import paths as P
from evaluate import coco_eval

SPLIT_DIR = Path("datasets/coco/test")
OUT = Path("results")
YOLO_W = P.YOLO_FT


def yolo_stock(half, nms=True):
    from ultralytics import YOLO
    m = YOLO(YOLO_W)

    def run(img):
        r = m.predict(img, imgsz=1280, conf=0.001, max_det=300, half=half, nms=nms, verbose=False)[0]
        return r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()
    return run


def rf_opt(ckpt, **opt):
    from rfdetr import RFDETRNano
    m = RFDETRNano(pretrain_weights=ckpt, resolution=640)
    m.inference(**opt)

    def run(img):
        d = m.predict(img[:, :, ::-1].copy(), threshold=0.001, include_source_image=False)
        return d.xyxy, d.confidence
    return run


def variants():
    rf = D.rfdetr_best(P.RFDETR_FT_RUN)
    return {
        "YOLO26n | Ultralytics predict FP32 (baseline)": lambda: yolo_stock(False),
        "YOLO26n | Ultralytics predict FP16": lambda: yolo_stock(True),
        "YOLO26n | NMS-free head FP16": lambda: yolo_stock(True, nms=False),
        "YOLO26n | NMS-free + GPU pre-proc + fused FP16": lambda: D.yolo(YOLO_W, nms=False),
        "YOLO26n | GPU pre-proc + fused FP16 + GPU NMS (deployed)": lambda: D.yolo(YOLO_W),
        "RF-DETR Nano | predict FP32, unoptimized (baseline)": lambda: D.rfdetr(rf, False, 640),
        "RF-DETR Nano | FP32 + TorchScript": lambda: rf_opt(rf, compile=True, dtype=torch.float32),
        "RF-DETR Nano | FP16, no compile": lambda: rf_opt(rf, compile=False, dtype=torch.float16),
        "RF-DETR Nano | FP16 + TorchScript": lambda: rf_opt(rf, compile=True, dtype=torch.float16),
        "RF-DETR Nano | FP16 + TorchScript + GPU pre-proc (deployed)": lambda: D.rfdetr(rf, True, 640),
    }


def time_model(run, frames, passes=2):
    """Returns (COCO-format detections, per-frame latency in ms) from the last pass."""
    for img in list(frames.values())[:20]:
        run(img)
    for _ in range(passes):
        dets, times = [], []
        for img_id, img in frames.items():
            torch.cuda.synchronize(); t0 = time.perf_counter()
            boxes, scores = run(img)
            torch.cuda.synchronize(); times.append(1000 * (time.perf_counter() - t0))
            for (x1, y1, x2, y2), s in zip(boxes, scores):
                dets.append({"image_id": img_id, "category_id": 1, "score": float(s),
                             "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]})
    return dets, np.array(times)


def main():
    gt = COCO(str(SPLIT_DIR / "_annotations.coco.json"))
    frames = {im["id"]: cv2.imread(str(SPLIT_DIR / im["file_name"])) for im in gt.loadImgs(gt.getImgIds())}
    res = {}
    for name, factory in variants().items():
        try:
            dets, t = time_model(factory(), frames)
        except Exception as e:  # one variant failing (e.g. compile unsupported) should not kill the sweep
            print(f"!! {name}: {type(e).__name__}: {e}", flush=True)
            res[name] = {"error": f"{type(e).__name__}: {e}"}
            continue
        _, m = coco_eval(gt, dets)
        res[name] = {"median_ms": float(np.median(t)), "p90_ms": float(np.percentile(t, 90)),
                     "mean_ms": float(t.mean()), "FPS": float(1000 / np.median(t)),
                     "AP50-95": m["AP50-95"], "AP50": m["AP50"]}
        print(f"{name:62s} {res[name]['median_ms']:6.2f} ms  p90 {res[name]['p90_ms']:6.2f}  "
              f"AP {m['AP50-95']:.4f}  AP50 {m['AP50']:.4f}", flush=True)
        torch.cuda.empty_cache()
    json.dump(res, open(OUT / "latency.json", "w"), indent=2)


if __name__ == "__main__":
    main()
