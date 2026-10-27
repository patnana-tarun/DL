"""Fine-tuning variants for the scene-adaptation study (review items 3 and 4). Same recipe as train_yolo.py
(AdamW, lr 1e-3, cosine, batch 12, seed 0, 1280 px) unless the variant says otherwise. Every variant keeps the
80-class COCO head (person = class 0, the other classes simply get no positives), so its weights stay
architecturally identical to the COCO-pretrained model and can be weight-interpolated (wise.py).

  ft80        aa1 pseudo-labels                                  (reference for all variants)
  replay      aa1 + COCO val2017 half-0 person replay, ~1:1      (Scenes100-style rehearsal)
  freeze      aa1, backbone frozen (11 modules), lr 3e-4, 40 ep  (regularised fine-tuning)
  res640      aa1 at 640 px                                      (resolution control vs RF-DETR Nano @ 640)
  multiscene  aa1 + campus scene_v1 + scene_v2 pseudo-labels     (more scenes, same label recipe)
  v1, v2      one campus scene each                              (transfer matrix)
  ignore      aa1, teacher boxes with fused score 0.15-0.40 greyed out in the images instead of being
              background (small-person-aware labels)
Model selection inside each run uses only that run's own validation split (never test1).

    python research/revision/train_variants.py ft80 replay ...
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths as P  # noqa: E402

from ultralytics import YOLO  # noqa: E402

ROOT = Path("datasets/revision")


def names80():
    return YOLO(P.YOLO_PRETRAINED).names


def write_yaml(name, train, val):
    """train/val: lists of image directories (absolute)."""
    d = ROOT / name; d.mkdir(parents=True, exist_ok=True)
    nm = names80()
    y = "train:\n" + "".join(f"  - {Path(t).resolve()}\n" for t in train) + \
        "val:\n" + "".join(f"  - {Path(v).resolve()}\n" for v in val) + \
        "names:\n" + "".join(f"  {k}: {v}\n" for k, v in nm.items())
    (d / "data.yaml").write_text(y)
    return str(d / "data.yaml")


AA1 = "datasets/yolo"
V1, V2 = "datasets/scene_v1/yolo", "datasets/scene_v2/yolo"

VARIANTS = {
    "ft80": dict(train=[f"{AA1}/train/images"], val=[f"{AA1}/valid/images"]),
    "replay": dict(train=[f"{AA1}/train/images", "datasets/coco_replay/images"], val=[f"{AA1}/valid/images"]),
    "freeze": dict(train=[f"{AA1}/train/images"], val=[f"{AA1}/valid/images"], freeze=11, lr0=3e-4, epochs=40,
                   close_mosaic=10, patience=20),
    "res640": dict(train=[f"{AA1}/train/images"], val=[f"{AA1}/valid/images"], imgsz=640, batch=24),
    "multiscene": dict(train=[f"{AA1}/train/images", f"{V1}/train/images", f"{V2}/train/images"],
                       val=[f"{AA1}/valid/images", f"{V1}/valid/images", f"{V2}/valid/images"]),
    "v1": dict(train=[f"{V1}/train/images"], val=[f"{V1}/valid/images"]),
    "v2": dict(train=[f"{V2}/train/images"], val=[f"{V2}/valid/images"]),
    "ignore": dict(train=["datasets/revision/aa1_ignore/train/images"], val=[f"{AA1}/valid/images"]),
}


def train(name):
    v = dict(VARIANTS[name])
    data = write_yaml(name, v.pop("train"), v.pop("val"))
    cfg = dict(data=data, epochs=150, patience=40, imgsz=1280, batch=12, single_cls=False, optimizer="AdamW",
               lr0=0.001, cos_lr=True, close_mosaic=20, fliplr=0.5, scale=0.5, mixup=0.1, seed=0,
               deterministic=False, project=str(Path("runs/revision").resolve()), name=name, exist_ok=True,
               plots=True, workers=6)
    cfg.update(v)
    model = YOLO(P.YOLO_PRETRAINED)
    model.train(**cfg)
    # Ultralytics may nest the run under runs/<task>/...; downstream scripts expect runs/revision/<name>
    save_dir, want = Path(model.trainer.save_dir).resolve(), Path("runs/revision", name)
    if save_dir != want.resolve():
        if want.is_symlink() or not want.exists():
            want.unlink(missing_ok=True); want.symlink_to(save_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("variants", nargs="+", choices=list(VARIANTS))
    for n in ap.parse_args().variants:
        if Path(f"runs/revision/{n}/weights/best.pt").exists():
            print("skip (done):", n); continue
        print("== train", n, flush=True)
        train(n)
