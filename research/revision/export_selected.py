"""Export the model chosen by the shift-aware selection (select_robust.py) for the edge runtime and verify it.

The first decision matrix (research/select_model.py) exported the fine-tuned YOLO26n. The shift-aware matrix picks
a weight average of the pretrained and fine-tuned weights instead, so this script exports that checkpoint at the
three runtime input sizes, scores every export through the runtime's own ONNX detector on the aa1 test split and on
test1, and makes it the runtime default. The fine-tuned exports stay in edge_deploy/models/ because the streamed
experiments in the paper were run with them.

    python research/revision/export_selected.py

Writes results/revision/export_selected.json and edge_deploy/models/{selected_model.json, yolo26n_wise_person_<h>x<w>.onnx}.
"""
import contextlib
import io
import json
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from pycocotools.coco import COCO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths as P  # noqa: E402
from evaluate import coco_eval  # noqa: E402
from select_model import SIZES, DEPLOY_MODELS  # noqa: E402

sys.path.insert(0, str(P.DEPLOY))
from edgevision import detector as edge_detector  # noqa: E402

R = Path("results/revision")
SPLITS = {"aa1_test": "datasets/coco/test", "test1": "datasets/test1/coco/test"}
TOL = 0.005   # allowed |ONNX - PyTorch| AP at 736x1280


def winner():
    sel = json.load(open(R / "selection_robust.json"))
    name = sel["winner"]                                   # e.g. "WiSE ft80_a0.3"
    tag = name.split()[-1]
    if not name.startswith("WiSE"):
        raise SystemExit(f"export is written for WiSE blends, got {name}")
    pt = Path("runs/revision/wise") / f"{tag}.pt"
    ms = json.load(open(R / "multiscene" / f"{name.replace(' ', '_')}.json"))
    return name, pt, ms


def export(pt):
    from ultralytics import YOLO
    files = {}
    for h, w in SIZES:
        with contextlib.redirect_stdout(io.StringIO()):
            f = YOLO(str(pt)).export(format="onnx", imgsz=(h, w), dynamic=False, simplify=True, half=False,
                                     nms=None, device="cpu")
        dst = DEPLOY_MODELS / f"yolo26n_wise_person_{h}x{w}.onnx"
        shutil.move(f, dst)
        dst.with_suffix(".json").write_text(json.dumps({"imgsz": [h, w], "classes": "COCO-80, person = 0",
                                                        "head": "one-to-many + NMS", "source": str(pt)}, indent=1))
        files[(h, w)] = dst
        print("exported", dst, f"{dst.stat().st_size / 1e6:.1f} MB", flush=True)
    return files


def score(files):
    res = {}
    for split, d in SPLITS.items():
        with contextlib.redirect_stdout(io.StringIO()):
            gt = COCO(f"{d}/_annotations.coco.json")
        imgs = gt.loadImgs(gt.getImgIds())
        for (h, w), f in files.items():
            det = edge_detector.OnnxDetector({"weights": str(f), "providers": ["cpu"], "fp16": False, "conf": 0.001,
                                              "iou": 0.7, "max_det": 300, "trt_cache": "/tmp/trt"})
            dets, times = [], []
            for im in imgs:
                img = cv2.imread(f"{d}/{im['file_name']}")
                t = time.perf_counter(); b, s = det(img); times.append(1000 * (time.perf_counter() - t))
                dets += [{"image_id": im["id"], "category_id": 1, "score": float(sc),
                          "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]}
                         for (x1, y1, x2, y2), sc in zip(b, s)]
            _, m = coco_eval(gt, dets)
            res.setdefault(f"{h}x{w}", {"file": str(f.relative_to(P.ROOT))})[split] = {
                "AP50-95": m["AP50-95"], "AP50": m["AP50"], "AP_small": m["AP_small"],
                "cpu_onnxruntime_ms_p50": float(np.median(times))}
            print(f"{split} {h}x{w}: AP {m['AP50-95']:.4f}  ORT-CPU {np.median(times):.1f} ms", flush=True)
    return res


def pt_ap(ms, split):
    """PyTorch AP of the same checkpoint from the multi-scene evaluation."""
    return ms["scenes"][split]["AP50-95"]


def main():
    name, pt, ms = winner()
    files = export(pt)
    res = score(files)
    big = f"{SIZES[0][0]}x{SIZES[0][1]}"
    parity = {s: {"pytorch_AP50-95": pt_ap(ms, s), "onnx_AP50-95": res[big][s]["AP50-95"]} for s in SPLITS}
    for v in parity.values():
        v["abs_diff"] = float(abs(v["onnx_AP50-95"] - v["pytorch_AP50-95"]))
        v["pass"] = bool(v["abs_diff"] <= TOL)
    old = json.load(open(DEPLOY_MODELS / "selected_model.json"))
    sel = dict(old)
    sel.update({"model": f"YOLO26n {name} (shift-aware selection)", "weights_pt": str(pt),
                "onnx": {k: v["file"].replace("edge_deploy/", "") for k, v in res.items()},
                "recommended": {"desktop": f"models/yolo26n_wise_person_{big}.onnx",
                                "jetson_orin_nano": "models/yolo26n_wise_person_544x960.onnx",
                                "jetson_orin_nx_agx": f"models/yolo26n_wise_person_{big}.onnx"},
                "test_metrics_onnx": {k: {s: v[s]["AP50-95"] for s in SPLITS} for k, v in res.items()},
                "previous": {"model": old["model"], "weights_pt": old["weights_pt"], "onnx": old["onnx"],
                             "note": "fine-tuned YOLO26n; used for the streamed runtime experiments in the paper"},
                "note": "tracker parameters were tuned for the fine-tuned YOLO26n and are reused unchanged"})
    (DEPLOY_MODELS / "selected_model.json").write_text(json.dumps(sel, indent=1))
    out = {"selected": name, "weights_pt": str(pt), "exports": res, "parity_736x1280": parity}
    json.dump(out, open(R / "export_selected.json", "w"), indent=1)
    print(json.dumps(parity, indent=1))
    if not all(v["pass"] for v in parity.values()):
        raise SystemExit("ONNX parity check FAILED")


if __name__ == "__main__":
    main()
