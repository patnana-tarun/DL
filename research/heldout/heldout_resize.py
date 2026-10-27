"""Held-out test video, diagnostic: sensitivity to the downscaling filter (1920x1080 -> 1280x720 network input).

The deployed runtime has FFmpeg scale the stream to 1280x720 (swscale default: bicubic) before the ONNX detector,
while the research path decodes at full size and downsamples on the GPU (bilinear, no antialiasing). On aa1
(native 720p) the two paths agree; on this 1080p video they do not, so every model is re-scored on the detection
frames after downscaling with each filter, and the boxes are mapped back to the full frame.
  research path   YOLO26n fine-tuned / pretrained (detectors.yolo, PyTorch FP16) on frames pre-scaled by
                  bilinear (= what the GPU path does itself), bicubic (~ FFmpeg) and area (antialiased)
  deployed path   the edge runtime's ONNX detector (edgevision.detector, 736x1280, CPU) on the same three inputs,
                  plus on frames scaled by FFmpeg itself (the exact deployed resampler)

    python research/heldout/heldout_resize.py --name test1
Writes results/<name>/resize_sensitivity.json.
"""
import argparse
import contextlib
import io
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths as P  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from pycocotools.coco import COCO  # noqa: E402

import evaluate as E  # noqa: E402

sys.path.insert(0, str(P.DEPLOY))
from edgevision import config as ecfg  # noqa: E402
from edgevision.detector import build  # noqa: E402

FILTERS = {"bilinear": cv2.INTER_LINEAR, "bicubic": cv2.INTER_CUBIC, "area": cv2.INTER_AREA}
W, H = 1280, 720


def ffmpeg_scale(path):
    """Scale one JPEG with FFmpeg's default swscale filter, exactly as edgevision.stream does."""
    out = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vf", f"scale={W}:{H}", "-f", "rawvideo",
                          "-pix_fmt", "bgr24", "pipe:1"], capture_output=True, check=True).stdout
    return np.frombuffer(out, np.uint8).reshape(H, W, 3)


def score(gt, run, imgs, scaled, conf):
    dets = []
    for im in imgs:
        x = scaled(im)
        b, sc = run(x)
        s = im["width"] / x.shape[1]
        for (x1, y1, x2, y2), v in zip(np.asarray(b) * s, sc):
            dets.append({"image_id": im["id"], "category_id": 1, "score": float(v),
                         "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]})
    m = E.coco_eval(gt, dets)[1]
    op = E.operating_point(E.match_images(gt, dets), conf)
    return {"AP50-95": m["AP50-95"], "AP50": m["AP50"], "AP_small": m["AP_small"], "F1": op["F1"],
            "recall": op["recall"], "precision": op["precision"]}


def main():
    import detectors as D
    import torch
    ap = argparse.ArgumentParser(); ap.add_argument("--name", required=True)
    a = ap.parse_args()
    split = Path(f"datasets/{a.name}/coco/test")
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(split / "_annotations.coco.json"))
    imgs = gt.loadImgs(gt.getImgIds())
    frozen = {n: v["valid"]["conf"] for n, v in json.load(open("results/metrics_splits.json")).items()}

    def scaled_by(f):
        def load(im):
            img = cv2.imread(str(split / im["file_name"]))
            if f == "full":
                return img
            if f == "ffmpeg":
                return ffmpeg_scale(split / im["file_name"])
            return cv2.resize(img, (W, H), interpolation=FILTERS[f])
        return load

    res = {}
    for name, weights in (("YOLO26n (fine-tuned)", P.YOLO_FT), ("YOLO26n (pretrained)", P.YOLO_PRETRAINED)):
        run = D.yolo(weights)
        res[f"research | {name}"] = {f: score(gt, run, imgs, scaled_by(f), frozen[name])
                                     for f in ("full", *FILTERS)}
        del run; torch.cuda.empty_cache()
        print(name, {f: round(v["AP50-95"], 3) for f, v in res[f"research | {name}"].items()}, flush=True)
    cfg = ecfg.load(None, ["detector.providers=[cpu]", "detector.weights=models/yolo26n_person_736x1280.onnx"])["detector"]
    det = build(cfg)
    res["deployed ONNX 736x1280 | YOLO26n (fine-tuned)"] = {
        f: score(gt, det, imgs, scaled_by(f), frozen["YOLO26n (fine-tuned)"]) for f in ("full", *FILTERS, "ffmpeg")}
    print("ONNX", {f: round(v["AP50-95"], 3) for f, v in res["deployed ONNX 736x1280 | YOLO26n (fine-tuned)"].items()})
    res["protocol"] = __doc__.split("\n\n")[1].strip()
    res["note"] = "full = the 1920x1080 frame given to the detector, which downsamples it itself (research: GPU bilinear; ONNX: cv2 INTER_LINEAR)"
    json.dump(res, open(f"results/{a.name}/resize_sensitivity.json", "w"), indent=2)


if __name__ == "__main__":
    main()
