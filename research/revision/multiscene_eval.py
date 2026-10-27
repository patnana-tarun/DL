"""Every detector of the revision on every scene, with one protocol (review items 3, 4 and 5).

Scenes (pseudo-labels from the same teacher recipe, fused score >= 0.40, except COCO):
  aa1_test   aa1 255-300 s (182 frames)             in-domain for aa1-trained models
  v1_test    campus scene 1, last 25 % (phone, building / mural, near-field)
  v2_test    campus scene 2, last 25 % (phone at ground level, backlit)
  test1      held-out campus road, all 1,191 frames (TEST ONLY: never used to train, tune or select)
  coco_val   600 COCO val2017 images (500 with people, 100 without; human labels), at 640 px
Validation sets (threshold and model / blend selection): aa1_valid, v1_valid, v2_valid, coco_val.

Per (model, scene): COCO AP / AP50 / AP_small (pycocotools), P / R / F1 / FPPI at the model's operating
threshold (best F1 on its own training-scene validation split, frozen everywhere else), recall by person
height; 95 % CIs by circular block bootstrap over frames (5 s blocks; i.i.d. for COCO images).
Predictions are cached in results/revision/preds/, so re-scoring is cheap.

    python research/revision/multiscene_eval.py --set wise_ft80      # see MODEL_SETS
    python research/revision/multiscene_eval.py --models "name=path[@imgsz]" ... --scenes aa1_test test1
-> results/revision/multiscene/<model>.json
"""
import argparse
import contextlib
import gzip
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "heldout"))
import paths as P  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from pycocotools.coco import COCO  # noqa: E402

import evaluate as E  # noqa: E402
import tempstats as T  # noqa: E402
from heldout_eval import recall_by_height  # noqa: E402

SCENES = {
    "aa1_valid": "datasets/coco/valid", "aa1_test": "datasets/coco/test",
    "v1_valid": "datasets/scene_v1/splits/valid", "v1_test": "datasets/scene_v1/splits/test",
    "v2_valid": "datasets/scene_v2/splits/valid", "v2_test": "datasets/scene_v2/splits/test",
    "test1": "datasets/test1/coco/test", "coco_val": "datasets/coco_person_val/coco/test",
}
EVAL_SCENES = ["aa1_test", "v1_test", "v2_test", "test1", "coco_val"]
VALID_SCENES = ["aa1_valid", "v1_valid", "v2_valid", "coco_val"]
BLOCK_S, B = 5.0, 1000
OUT = Path("results/revision")


def gt_of(scene):
    with contextlib.redirect_stdout(io.StringIO()):
        return COCO(f"{SCENES[scene]}/_annotations.coco.json")


def runner(spec, scene):
    """'path[@imgsz]' (YOLO .pt, default 1280; COCO images at 640) or 'rfdetr:path|pretrained[@res]' -> run(img)."""
    import detectors as D
    path, _, res = spec.partition("@")
    if path.startswith("rfdetr:"):
        w = path[len("rfdetr:"):]
        return D.rfdetr(P.RFDETR_PRETRAINED if w == "pretrained" else w, True, int(res) if res else None)
    return D.yolo(path, True, imgsz=640 if scene == "coco_val" and not res else int(res or 1280))


def predict(name, spec, scene, cache=True):
    f = OUT / "preds" / f"{name.replace(' ', '_').replace('/', '_')}__{scene}.json.gz"
    if cache and f.exists():
        return json.load(gzip.open(f))
    run = runner(spec, scene)
    gt = gt_of(scene); dets = []
    for im in gt.loadImgs(gt.getImgIds()):
        boxes, scores = run(cv2.imread(f"{SCENES[scene]}/{im['file_name']}"))
        dets += [{"image_id": im["id"], "category_id": 1, "score": float(s),
                  "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]}
                 for (x1, y1, x2, y2), s in zip(boxes, scores)]
    del run; torch.cuda.empty_cache()
    f.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(f, "wt") as fh:
        json.dump(dets, fh)
    return dets


def threshold(name, spec, valid):
    """Best-F1 confidence on the pooled validation scenes of the model's training data."""
    pers = []
    for s in valid:
        pers += E.match_images(gt_of(s), predict(name, spec, s))
    c = E.pr_curve(pers)
    return float(c["conf"][c["f1"].argmax()])


def score(name, spec, scene, conf, boot=True):
    gt = gt_of(scene); dets = predict(name, spec, scene)
    _, m = E.coco_eval(gt, dets)
    per = E.match_images(gt, dets)
    m.update(E.operating_point(per, conf))
    m["recall_by_height_px"] = recall_by_height(gt, dets, conf)
    if boot:
        imgs = gt.loadImgs(gt.getImgIds())
        period = float(np.median(np.diff([i["t"] for i in imgs]))) if "t" in imgs[0] else 0.25   # aa1: every 0.25 s
        b = 1 if scene == "coco_val" else max(1, int(round(BLOCK_S / period)))
        bs = E.bootstrap(per, conf, T.resamples(len(per), b, B, "cbb", seed=0))
        m["AP50-95_CI95"], m["F1_CI95"], m["block_frames"] = T.ci(bs[:, 0]), T.ci(bs[:, 2]), b
    return {k: v for k, v in m.items() if not isinstance(v, np.ndarray)}


def evaluate(name, spec, valid, scenes=EVAL_SCENES, extra_valid=VALID_SCENES, boot=True):
    out = OUT / "multiscene" / f"{name.replace(' ', '_').replace('/', '_')}.json"
    conf = threshold(name, spec, valid)
    res = {"name": name, "spec": spec, "threshold_from": valid, "conf": conf, "scenes": {}, "valid": {}}
    for s in scenes:
        if Path(f"{SCENES[s]}/_annotations.coco.json").exists():
            res["scenes"][s] = score(name, spec, s, conf, boot)
    for s in extra_valid:            # validation-set AP for blend / model selection (never test1)
        if Path(f"{SCENES[s]}/_annotations.coco.json").exists():
            gt = gt_of(s)
            res["valid"][s] = E.coco_eval(gt, predict(name, spec, s))[1]["AP50-95"]
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out, "w"), indent=1)
    print(name, {s: round(v["AP50-95"], 3) for s, v in res["scenes"].items()},
          {s: round(v, 3) for s, v in res["valid"].items()}, flush=True)
    return res


def model_sets():
    import detectors as D
    aa1v = ["aa1_valid"]
    base = {"YOLO26n pretrained": (P.YOLO_PRETRAINED, aa1v), "YOLO26n FT-aa1 (1-cls)": (P.YOLO_FT, aa1v),
            "RF-DETR-N pretrained": ("rfdetr:pretrained", aa1v),
            "RF-DETR-N FT-aa1 @640": (f"rfdetr:{D.rfdetr_best(P.RFDETR_FT_RUN)}@640", aa1v)}
    v = {n: (f"runs/revision/{n}/weights/best.pt", vs) for n, vs in (
        ("ft80", aa1v), ("replay", aa1v), ("freeze", aa1v), ("ignore", aa1v),
        ("multiscene", ["aa1_valid", "v1_valid", "v2_valid"]), ("v1", ["v1_valid"]), ("v2", ["v2_valid"]))}
    v["res640"] = ("runs/revision/res640/weights/best.pt@640", aa1v)
    sets = {"base": base, "variants": {f"YOLO26n {k}": x for k, x in v.items()}}
    wise = {}
    for f in sorted(Path("runs/revision/wise").glob("*.pt")):
        wise[f"WiSE {f.stem}"] = (str(f), aa1v)
    sets["wise"] = wise
    # inference-resolution sweep (no retraining): every size for the aa1 models
    sweep = {}
    for sz in (384, 512, 640, 768, 960, 1280):
        sweep[f"YOLO26n pretrained @{sz}"] = (f"{P.YOLO_PRETRAINED}@{sz}", aa1v)
        sweep[f"YOLO26n ft80 @{sz}"] = (f"runs/revision/ft80/weights/best.pt@{sz}", aa1v)
        sweep[f"YOLO26n res640 @{sz}"] = (f"runs/revision/res640/weights/best.pt@{sz}", aa1v)
        sweep[f"RF-DETR-N FT-aa1 @{sz}"] = (f"rfdetr:{D.rfdetr_best(P.RFDETR_FT_RUN)}@{sz}", aa1v)
        sweep[f"RF-DETR-N pretrained @{sz}"] = (f"rfdetr:pretrained@{sz}", aa1v)
    sets["ressweep"] = sweep
    return sets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--only", default="", help="substring filter on model names")
    ap.add_argument("--scenes", nargs="*", default=EVAL_SCENES)
    ap.add_argument("--force", action="store_true", help="re-score models whose result is newer than the weights")
    a = ap.parse_args()
    sets = model_sets()
    for s in a.set:
        for name, (spec, valid) in sets[s].items():
            if a.only and a.only not in name:
                continue
            w = spec.split("@")[0].removeprefix("rfdetr:")
            if w != "pretrained" and not Path(w).exists():
                print("missing", name, w); continue
            out = OUT / "multiscene" / f"{name.replace(' ', '_').replace('/', '_')}.json"
            if out.exists() and out.stat().st_mtime > Path(w if w != "pretrained" else __file__).stat().st_mtime \
                    and not a.force:
                print("done:", name); continue
            try:
                # block-bootstrap CIs for the reported models; the sweeps (WiSE alphas, input sizes) only need
                # point estimates for selection and plots
                evaluate(name, spec, valid, a.scenes, boot=s in ("base", "variants"))
            except Exception as e:  # noqa: BLE001  (one failing model must not stop the set)
                print(f"!! {name} failed: {type(e).__name__}: {e}", flush=True)
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
