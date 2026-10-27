"""Temporal train / valid / test splits for the two campus scenes labelled by heldout_detect.py
(datasets/scene_v{1,2}/teachers.json, frames every 0.25 s, fused teacher boxes with scores).

Same label recipe as aa1 (fused score >= 0.40, min side >= 8 px). Contiguous time blocks: train = first 60 %,
valid = next 15 %, test = last 25 % of the video, so the test frames are the latest and never interleave with
training frames.
  -> datasets/scene_vN/yolo/{train,valid,test}/{images,labels}   (YOLO, symlinked JPEGs)
     datasets/scene_vN/splits/{train,valid,test}/_annotations.coco.json  (COCO, evaluate.py layout)

    python research/revision/make_scene_splits.py scene_v1 scene_v2
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths as P  # noqa: E402,F401

import cv2  # noqa: E402

FRACTIONS = {"train": (0.0, 0.60), "valid": (0.60, 0.75), "test": (0.75, 1.0001)}
FUSED_THR, MIN_BOX = 0.40, 8


def main(name):
    root = Path("datasets") / name
    rows = json.load(open(root / "teachers.json"))
    dur = max(r["t"] for r in rows)
    H, W = cv2.imread(str(root / "frames" / rows[0]["file_name"])).shape[:2]
    out = {}
    for split, (a, b) in FRACTIONS.items():
        sel = [r for r in rows if a * dur <= r["t"] < b * dur]
        yi, yl, cd = (root / "yolo" / split / "images", root / "yolo" / split / "labels", root / "splits" / split)
        for d in (yi, yl, cd):
            d.mkdir(parents=True, exist_ok=True)
        coco = {"images": [], "annotations": [], "categories": [{"id": 1, "name": "person", "supercategory": "none"}]}
        for r in sel:
            src = (root / "frames" / r["file_name"]).resolve()
            for d in (yi, cd):
                if not (d / r["file_name"]).exists():
                    os.symlink(src, d / r["file_name"])
            boxes = [b for b in r["fused"] if b[4] >= FUSED_THR and min(b[2] - b[0], b[3] - b[1]) >= MIN_BOX]
            (yl / r["file_name"].replace(".jpg", ".txt")).write_text("".join(
                f"0 {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}\n"
                for x1, y1, x2, y2, _ in boxes))
            img_id = len(coco["images"]) + 1
            coco["images"].append({"id": img_id, "file_name": r["file_name"], "width": W, "height": H,
                                   "frame": r["frame"], "t": r["t"]})
            for x1, y1, x2, y2, s in boxes:
                coco["annotations"].append({"id": len(coco["annotations"]) + 1, "image_id": img_id, "category_id": 1,
                                            "bbox": [x1, y1, x2 - x1, y2 - y1], "area": (x2 - x1) * (y2 - y1),
                                            "iscrowd": 0, "score": s})
        json.dump(coco, open(cd / "_annotations.coco.json", "w"))
        out[split] = {"images": len(sel), "boxes": len(coco["annotations"]),
                      "t_range_s": [round(a * dur, 1), round(min(b, 1) * dur, 1)]}
    json.dump(out, open(root / "split_stats.json", "w"), indent=1)
    print(name, out)


if __name__ == "__main__":
    for n in sys.argv[1:]:
        main(n)
