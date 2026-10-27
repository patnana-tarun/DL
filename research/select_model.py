"""Pick the detector to deploy, export it for the edge, and verify the export.

1. Decision matrix over the two fine-tuned students, from the stored results (no re-running):
     detection   test AP50-95 under the fused labels                      (results/metrics.json)
     label bias  the lower of AP vs RF-DETR-Large-only / YOLO26x-only labels (results/teacher_bias.json)
     tracking    mean held-out HOTA over 6 trackers x 2 GT linkings        (tracking/tracker_comparison.json)
     latency     deployed fast path, median ms per 1280x720 frame          (results/latency.json)
     compute     GFLOPs at the inference resolution, parameters            (results/metrics.json)
   Each criterion is scored relative to the best candidate (1.0 = best), then weighted. The choice is
   re-checked under alternative weightings (accuracy-only, speed-only, equal, bias-aware) so it is not an
   artefact of one set of weights.
2. Export the winner to ONNX (one-to-many head, external NMS: the variant the latency study deployed) at
   three input sizes for Jetson: 736x1280 (study resolution), 544x960, 384x640.
3. Verify every export through the edge runtime's own detector (edge_deploy/edgevision/detector.py,
   ONNX Runtime) on the test split: AP must match the PyTorch reference at 1280 within 0.005.

    python research/select_model.py
Writes results/model_selection.json and edge_deploy/models/{selected_model.json, yolo26n_person_<h>x<w>.onnx}.
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

import paths as P
from evaluate import coco_eval

sys.path.insert(0, str(P.DEPLOY))
from edgevision import detector as edge_detector  # noqa: E402

R = Path("results")
CAND = {"YOLO26n": "YOLO26n (fine-tuned)", "RF-DETR Nano": "RF-DETR Nano (fine-tuned)"}
LAT = {"YOLO26n": "YOLO26n | GPU pre-proc + fused FP16 + GPU NMS (deployed)",
       "RF-DETR Nano": "RF-DETR Nano | FP16 + TorchScript + GPU pre-proc (deployed)"}
CRITERIA = {   # name: (higher_is_better, weight)
    "test_AP50-95": (True, 0.30),
    "label_bias_min_AP": (True, 0.10),
    "tracking_mean_HOTA": (True, 0.25),
    "latency_ms": (False, 0.20),
    "GFLOPs": (False, 0.10),
    "params_M": (False, 0.05),
}
ALT_WEIGHTS = {
    "accuracy only": {"test_AP50-95": 1},
    "speed only": {"latency_ms": 1},
    "tracking only": {"tracking_mean_HOTA": 1},
    "equal": {k: 1 for k in CRITERIA},
    "bias-aware (label-bias AP replaces fused AP)": {"label_bias_min_AP": 0.4, "tracking_mean_HOTA": 0.25,
                                                     "latency_ms": 0.2, "GFLOPs": 0.1, "params_M": 0.05},
}
SIZES = [(736, 1280), (544, 960), (384, 640)]
DEPLOY_MODELS = P.DEPLOY / "models"


def gather():
    m = json.load(open(R / "metrics.json")); lat = json.load(open(R / "latency.json"))
    tb = json.load(open(R / "teacher_bias.json")); tc = json.load(open(R / "tracking/tracker_comparison.json"))
    rows = {}
    for c, name in CAND.items():
        hota = [v[g]["heldout_15-30s"]["HOTA"] for k, v in tc.items() if k.startswith(c + " |")
                for g in ("ByteTrack-GT", "OC-SORT-GT")]
        rows[c] = {"test_AP50-95": m[name]["AP50-95"],
                   "label_bias_min_AP": min(tb[name]["RF-DETR Large only"]["AP50-95"], tb[name]["YOLO26x only"]["AP50-95"]),
                   "tracking_mean_HOTA": float(np.mean(hota)),
                   "latency_ms": lat[LAT[c]]["median_ms"], "GFLOPs": m[name]["GFLOPs"], "params_M": m[name]["params_M"],
                   "_tracking_wins": None}
    # head-to-head tracker wins (same tracker, same GT)
    wins = {c: 0 for c in CAND}; n = 0
    for trk in {k.split(" | ")[1] for k in tc}:
        for g in ("ByteTrack-GT", "OC-SORT-GT"):
            a = tc[f"YOLO26n | {trk}"][g]["heldout_15-30s"]["HOTA"]; b = tc[f"RF-DETR Nano | {trk}"][g]["heldout_15-30s"]["HOTA"]
            wins["YOLO26n" if a > b else "RF-DETR Nano"] += 1; n += 1
    for c in CAND:
        rows[c]["_tracking_wins"] = f"{wins[c]}/{n}"
    return rows


def score(rows, weights):
    out = {}
    for c, r in rows.items():
        s = 0.0
        for k, w in weights.items():
            hib = CRITERIA[k][0]
            vals = [rows[x][k] for x in rows]
            s += w * (r[k] / max(vals) if hib else min(vals) / r[k])
        out[c] = s / sum(weights.values())
    return out


def export(sizes):
    from ultralytics import YOLO
    DEPLOY_MODELS.mkdir(parents=True, exist_ok=True)
    files = {}
    for h, w in sizes:
        with contextlib.redirect_stdout(io.StringIO()):
            f = YOLO(P.YOLO_FT).export(format="onnx", imgsz=(h, w), dynamic=False, simplify=True, half=False,
                                       nms=None, device="cpu")
        dst = DEPLOY_MODELS / f"yolo26n_person_{h}x{w}.onnx"
        shutil.move(f, dst)
        (dst.with_suffix(".json")).write_text(json.dumps({"imgsz": [h, w], "classes": ["person"], "head": "one-to-many + NMS",
                                                          "source": P.YOLO_FT}, indent=1))
        files[(h, w)] = dst
        print("exported", dst, f"{dst.stat().st_size / 1e6:.1f} MB", flush=True)
    return files


def verify(files, ref_ap):
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO("datasets/coco/test/_annotations.coco.json")
    imgs = gt.loadImgs(gt.getImgIds())
    frames = [(im["id"], cv2.imread(f"datasets/coco/test/{im['file_name']}")) for im in imgs]
    res = {}
    for (h, w), f in files.items():
        det = edge_detector.OnnxDetector({"weights": str(f), "providers": ["cpu"], "fp16": False, "conf": 0.001,
                                          "iou": 0.7, "max_det": 300, "trt_cache": "/tmp/trt"})
        dets, times = [], []
        for img_id, img in frames:
            t = time.perf_counter(); b, s = det(img); times.append(1000 * (time.perf_counter() - t))
            dets += [{"image_id": img_id, "category_id": 1, "score": float(sc),
                      "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]} for (x1, y1, x2, y2), sc in zip(b, s)]
        _, m = coco_eval(gt, dets)
        res[f"{h}x{w}"] = {"file": str(f.relative_to(P.ROOT)), "AP50-95": m["AP50-95"], "AP50": m["AP50"],
                           "AP_small": m["AP_small"], "AR@300": m["AR@300"],
                           "cpu_onnxruntime_ms_p50": float(np.median(times)), "provider": det.provider}
        print(f"{h}x{w}: AP {m['AP50-95']:.4f} AP50 {m['AP50']:.4f}  ORT-CPU {np.median(times):.1f} ms", flush=True)
    k = f"{SIZES[0][0]}x{SIZES[0][1]}"
    res["parity_vs_pytorch_1280"] = {"pytorch_AP50-95": ref_ap, "onnx_AP50-95": res[k]["AP50-95"],
                                     "abs_diff": abs(res[k]["AP50-95"] - ref_ap),
                                     "pass": abs(res[k]["AP50-95"] - ref_ap) <= 0.005}
    return res


def main():
    rows = gather()
    main_score = score(rows, {k: w for k, (_, w) in CRITERIA.items()})
    winner = max(main_score, key=main_score.get)
    alt = {name: score(rows, w) for name, w in ALT_WEIGHTS.items()}
    alt_winner = {name: max(s, key=s.get) for name, s in alt.items()}
    print(json.dumps({"criteria": rows, "weighted_score": main_score, "winner": winner, "alternative_winners": alt_winner},
                     indent=1, default=float))
    if winner != "YOLO26n":
        raise SystemExit("the export below is written for YOLO; RF-DETR export is not implemented")
    files = export(SIZES)
    ver = verify(files, json.load(open(R / "metrics.json"))[CAND[winner]]["AP50-95"])
    trk = json.load(open(R / "tracking/tracking_metrics.json"))[winner]["best_params"]
    sel = {"model": winner, "weights_pt": P.YOLO_FT, "onnx": {k: v["file"].replace("edge_deploy/", "") for k, v in ver.items()
                                                             if isinstance(v, dict) and "file" in v},
           "recommended": {"desktop": "models/yolo26n_person_736x1280.onnx",
                           "jetson_orin_nano": "models/yolo26n_person_544x960.onnx",
                           "jetson_orin_nx_agx": "models/yolo26n_person_736x1280.onnx"},
           "detector": {"conf": 0.05, "iou": 0.7, "max_det": 300}, "tracker": trk,
           "test_metrics_onnx": {k: {m: v[m] for m in ("AP50-95", "AP50", "AP_small")} for k, v in ver.items() if "file" in v}}
    (DEPLOY_MODELS / "selected_model.json").write_text(json.dumps(sel, indent=1))
    out = {"criteria": {"definitions": {k: {"higher_is_better": h, "weight": w} for k, (h, w) in CRITERIA.items()},
                        "values": rows},
           "weighted_score": main_score, "winner": winner, "alternative_weightings": {n: {"scores": alt[n], "winner": alt_winner[n]}
                                                                                       for n in alt},
           "winner_robust": len(set(alt_winner.values()) | {winner}) == 1, "export_verification": ver}
    json.dump(out, open(R / "model_selection.json", "w"), indent=1, default=float)
    print(json.dumps(out["alternative_weightings"], indent=1, default=float))
    if not ver["parity_vs_pytorch_1280"]["pass"]:
        raise SystemExit("ONNX parity check FAILED")


if __name__ == "__main__":
    main()
