"""Fine-tune YOLO26n on the pseudo-labelled person dataset.

    python research/train_yolo.py                                                   # aa1 (published run)
    python research/train_yolo.py --data datasets/street/yolo/data.yaml --name street_yolo26n \
        --weights runs/yolo26n/weights/best.pt --epochs 80                 # continue from the aa1 model
"""
import argparse

from ultralytics import YOLO

import paths as P

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="datasets/yolo/data.yaml")
    ap.add_argument("--name", default="yolo26n")
    ap.add_argument("--weights", default=P.YOLO_PRETRAINED)
    ap.add_argument("--epochs", type=int, default=150)
    a = ap.parse_args()
    model = YOLO(a.weights)
    model.train(
        data=a.data,
        epochs=a.epochs,
        patience=40,
        imgsz=1280,            # native frame width; people are often small
        batch=12,
        single_cls=True,
        optimizer="AdamW",
        lr0=0.001,
        cos_lr=True,
        close_mosaic=20,       # val loss rises slightly once mosaic stops (mild overfit); best.pt is kept by val mAP
        fliplr=0.5,
        scale=0.5,
        mixup=0.1,
        seed=0,
        deterministic=False,
        project="runs",
        name=a.name,
        exist_ok=True,
        plots=True,
    )
    # Evaluate the best checkpoint on the held-out test split
    best = YOLO(f"runs/{a.name}/weights/best.pt")
    best.val(data=a.data, split="test", imgsz=1280, batch=12,
             project="runs", name=f"{a.name}_test", exist_ok=True, plots=True)
