"""Small-person-aware labels for aa1 ("ignore" variant of train_variants.py) and the label-threshold diagnostic.

Diagnostic: the 0.40 cut on the fused teacher score drops many small people. A dropped person is not "unknown"
to the detector; it is background, so fine-tuning learns to suppress exactly the small, far people that the
held-out scene is full of. We measure, per box-height bin, the share of fused teacher boxes (score >= 0.15)
that fall below the cut, on aa1 train and on every labelled scene.

Variant: re-run the teachers on the aa1 training frames and grey out (value 114, the letterbox colour) every
fused box with 0.15 <= score < 0.40 that does not overlap a kept label (IoU < 0.3), so those regions carry no
gradient either way, a cheap stand-in for ignore regions, which Ultralytics does not support.
  -> datasets/revision/aa1_ignore/train/{images,labels}, results/revision/label_threshold.json

    python research/revision/aa1_ignore.py
"""
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "heldout"))
import paths as P  # noqa: E402,F401

import cv2  # noqa: E402
import numpy as np  # noqa: E402

LO, CUT, MIN_BOX = 0.15, 0.40, 8
BINS = [(8, 32), (32, 64), (64, 128), (128, 256), (256, 4000)]


def iou(a, b):
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    ar = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])  # noqa: E731
    return inter / (ar(a)[:, None] + ar(b)[None] - inter + 1e-9)


def below_cut_share(fused_lists):
    """fused_lists: per-frame [[x1,y1,x2,y2,s], ...] (s >= 0.15) -> per height bin: n, share with s < CUT."""
    out = {}
    allb = np.array([b for f in fused_lists for b in f], float).reshape(-1, 5)
    h = allb[:, 3] - allb[:, 1]; w = allb[:, 2] - allb[:, 0]
    ok = np.minimum(h, w) >= MIN_BOX
    for a, b in BINS:
        k = ok & (h >= a) & (h < b)
        out[f"{a}-{b}px"] = {"n": int(k.sum()), "below_cut": float((allb[k, 4] < CUT).mean()) if k.any() else None,
                             "median_score": float(np.median(allb[k, 4])) if k.any() else None}
    return out


def main():
    from heldout_detect import Teacher
    teacher = Teacher()
    src = Path("datasets/yolo/train")
    dst = Path("datasets/revision/aa1_ignore/train")
    (dst / "images").mkdir(parents=True, exist_ok=True); (dst / "labels").mkdir(parents=True, exist_ok=True)
    fused_all, greyed, n_img = [], 0, 0
    for img_path in sorted((src / "images").glob("*.jpg")):
        img = cv2.imread(str(img_path)); H, W = img.shape[:2]
        fused, _ = teacher(img)
        fused_all.append(fused.tolist())
        lab = np.loadtxt(src / "labels" / (img_path.stem + ".txt"), ndmin=2)
        kept = (np.c_[lab[:, 1] - lab[:, 3] / 2, lab[:, 2] - lab[:, 4] / 2, lab[:, 1] + lab[:, 3] / 2,
                      lab[:, 2] + lab[:, 4] / 2] * [W, H, W, H]) if len(lab) else np.zeros((0, 4))
        unc = fused[(fused[:, 4] >= LO) & (fused[:, 4] < CUT)][:, :4]
        if len(unc) and len(kept):
            unc = unc[iou(unc, kept).max(1) < 0.3]
        for x1, y1, x2, y2 in unc.astype(int):
            img[max(0, y1):y2, max(0, x1):x2] = 114
        greyed += len(unc); n_img += 1
        cv2.imwrite(str(dst / "images" / img_path.name), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        shutil.copy(src / "labels" / (img_path.stem + ".txt"), dst / "labels" / (img_path.stem + ".txt"))
    res = {"aa1_train": below_cut_share(fused_all), "aa1_ignore_variant": {"images": n_img, "greyed_regions": greyed}}
    for name in ("test1", "scene_v1", "scene_v2"):
        p = Path(f"datasets/{name}/teachers.json")
        if p.exists():
            res[name] = below_cut_share([r["fused"] for r in json.load(open(p))])
    Path("results/revision").mkdir(parents=True, exist_ok=True)
    json.dump(res, open("results/revision/label_threshold.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
