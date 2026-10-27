"""Label-bias check: re-score the test predictions against each teacher on its own.

The pseudo-GT fuses RF-DETR Large and YOLO26x. A student from the same family as one
teacher may inherit that teacher's box conventions and score higher for it. Scoring every
model against RF-DETR-Large-only and YOLO26x-only labels (same 0.40 score cut) shows how
much each model's result depends on which teacher drew the boxes.

Needs results/preds/*__test.json.gz from evaluate.py. Writes results/teacher_bias.json.
"""
import contextlib
import copy
import gzip
import io
import json
from pathlib import Path

import cv2
import numpy as np
from pycocotools.coco import COCO

import paths as P
from evaluate import coco_eval

SPLIT = Path("datasets/coco/test")
THR = 0.40


def teacher_gts(gt):
    from rfdetr import RFDETRLarge
    from ultralytics import YOLO
    rf, yo = RFDETRLarge(), YOLO(P.YOLO_TEACHER)
    out = {"RF-DETR Large only": [], "YOLO26x only": []}
    for im in gt.loadImgs(gt.getImgIds()):
        img = cv2.imread(str(SPLIT / im["file_name"]))
        d = rf.predict(img[:, :, ::-1].copy(), threshold=THR); m = d.class_id == 1
        r = yo.predict(img, imgsz=1280, conf=THR, classes=[0], augment=True, verbose=False)[0]
        for k, boxes in (("RF-DETR Large only", d.xyxy[m]), ("YOLO26x only", r.boxes.xyxy.cpu().numpy())):
            for x1, y1, x2, y2 in boxes:
                if min(x2 - x1, y2 - y1) >= 8:
                    out[k].append((im["id"], [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]))
    return out


def as_coco(gt, anns):
    d = copy.deepcopy(gt.dataset)
    d["annotations"] = [{"id": i + 1, "image_id": img, "category_id": 1, "bbox": b, "area": b[2] * b[3], "iscrowd": 0}
                        for i, (img, b) in enumerate(anns)]
    c = COCO()
    c.dataset = d
    with contextlib.redirect_stdout(io.StringIO()):
        c.createIndex()
    return c


def main():
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(SPLIT / "_annotations.coco.json"))
    gts = {"Fused (published labels)": gt, **{k: as_coco(gt, v) for k, v in teacher_gts(gt).items()}}
    res = {}
    for f in sorted(Path("results/preds").glob("*__test.json.gz")):
        name = f.name.split("__")[0].replace("_", " ")
        with gzip.open(f, "rt") as fh:
            dets = json.load(fh)
        res[name] = {g: {k: v for k, v in coco_eval(G, dets)[1].items() if k in ("AP50-95", "AP50", "AP75")}
                     for g, G in gts.items()}
        print(name, {g: round(v["AP50-95"], 4) for g, v in res[name].items()}, flush=True)
    res["_label_counts"] = {g: len(G.getAnnIds()) for g, G in gts.items()}
    json.dump(res, open("results/teacher_bias.json", "w"), indent=2)


if __name__ == "__main__":
    main()
