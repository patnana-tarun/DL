"""Review item 5: how much do the detection CIs and paired tests depend on the resampling scheme?

For the aa1 test split (182 frames, 0.25 s apart) and the held-out test1 video (1,191 frames), the saved
predictions are re-scored with the evaluate.py protocol and resampled with
  i.i.d. frames (the original aa1 analysis), and circular-block / moving-block bootstraps with blocks of
  2.5, 5 and 10 s, plus the Politis-White automatic block length of the per-frame TP and FP series.
Each scheme uses B resamples shared by all models (paired). p-values use the (k + 1) / (B + 1) correction.
Nothing is re-run on the GPU and no threshold changes (frozen aa1-valid thresholds).

    python research/revision/temporal_ci.py            # -> results/revision/temporal_ci.json
"""
import contextlib
import gzip
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths as P  # noqa: E402,F401

import numpy as np  # noqa: E402
from pycocotools.coco import COCO  # noqa: E402

import evaluate as E  # noqa: E402
import tempstats as T  # noqa: E402

B = 2000
BLOCKS_S = (2.5, 5.0, 10.0)
MODELS = ("YOLO26n (pretrained)", "YOLO26n (fine-tuned)", "RF-DETR Nano (pretrained)", "RF-DETR Nano (fine-tuned)")
PAIRS = [("YOLO26n (fine-tuned)", "RF-DETR Nano (fine-tuned)"), ("YOLO26n (fine-tuned)", "YOLO26n (pretrained)"),
         ("RF-DETR Nano (fine-tuned)", "RF-DETR Nano (pretrained)")]
SETS = {"aa1_test": ("datasets/coco/test/_annotations.coco.json", "results/preds/{}__test.json.gz"),
        "test1": ("datasets/test1/coco/test/_annotations.coco.json", "results/test1/preds/{}__test.json.gz")}


def per_frame_series(per, conf):
    tp = np.array([int((p["tp"][0] & (p["scores"] >= conf)).sum()) for p in per])
    fp = np.array([int((~p["tp"][0] & (p["scores"] >= conf)).sum()) for p in per])
    return tp, fp


def main():
    frozen = {n: v["valid"]["conf"] for n, v in json.load(open("results/metrics_splits.json")).items()}
    out = {"B": B, "note": "CIs: 2.5/97.5 percentiles; p: two-sided, (k+1)/(B+1) corrected; blocks in seconds",
           "sets": {}}
    for set_name, (gt_path, pred_fmt) in SETS.items():
        with contextlib.redirect_stdout(io.StringIO()):
            gt = COCO(gt_path)
        imgs = gt.loadImgs(gt.getImgIds())
        period = float(np.median(np.diff([im["t"] for im in imgs]))) if "t" in imgs[0] else 0.25
        n = len(imgs)
        per = {}
        for m in MODELS:
            dets = json.load(gzip.open(pred_fmt.format(m.replace(" ", "_"))))
            per[m] = E.match_images(gt, dets)
        # automatic block length on the per-frame TP / FP series of every model (max over series)
        pw = {}
        for m in MODELS:
            tp, fp = per_frame_series(per[m], frozen[m])
            pw[m] = {"tp": T.optimal_block_length(tp), "fp": T.optimal_block_length(fp)}
        b_auto = int(np.ceil(max(max(v["tp"]["cb"], v["fp"]["cb"]) for v in pw.values())))
        schemes = {"iid": ("cbb", 1)}
        for s in BLOCKS_S:
            schemes[f"cbb_{s:g}s"] = ("cbb", max(1, int(round(s / period))))
            schemes[f"mbb_{s:g}s"] = ("mbb", max(1, int(round(s / period))))
        schemes["cbb_auto"] = ("cbb", b_auto)
        res = {"frames": n, "period_s": period, "politis_white": pw, "auto_block_frames": b_auto,
               "auto_block_s": b_auto * period, "schemes": {}}
        point = {m: (lambda mm: (mm["AP50-95"], mm["F1"]))(
            {**E.coco_eval(gt, json.load(gzip.open(pred_fmt.format(m.replace(' ', '_')))))[1],
             **E.operating_point(per[m], frozen[m])}) for m in MODELS}
        for sname, (kind, b) in schemes.items():
            idx = T.resamples(n, b, B, kind, seed=0)
            boot = {m: E.bootstrap(per[m], frozen[m], idx) for m in MODELS}
            r = {"block_frames": b, "block_s": b * period, "n_blocks": int(np.ceil(n / b)), "models": {}, "pairs": {}}
            for m in MODELS:
                r["models"][m] = {"AP50-95": point[m][0], "AP50-95_CI95": T.ci(boot[m][:, 0]),
                                  "AP50-95_se": float(boot[m][:, 0].std()),
                                  "F1": point[m][1], "F1_CI95": T.ci(boot[m][:, 2])}
            for x, y in PAIRS:
                d = boot[x][:, 0] - boot[y][:, 0]
                r["pairs"][f"{x} vs {y}"] = {"delta_AP50-95": point[x][0] - point[y][0], "CI95": T.ci(d),
                                             "se": float(d.std()), "p_two_sided": T.boot_p(d)}
            res["schemes"][sname] = r
            print(set_name, sname, b, {m: [round(v, 3) for v in r["models"][m]["AP50-95_CI95"]] for m in MODELS},
                  flush=True)
        # how much wider than i.i.d. (ratio of bootstrap standard errors)
        se0 = {m: res["schemes"]["iid"]["models"][m]["AP50-95_se"] for m in MODELS}
        res["se_ratio_vs_iid"] = {s: {m: r["models"][m]["AP50-95_se"] / se0[m] for m in MODELS}
                                  for s, r in res["schemes"].items()}
        out["sets"][set_name] = res
    Path("results/revision").mkdir(parents=True, exist_ok=True)
    json.dump(out, open("results/revision/temporal_ci.json", "w"), indent=1)


if __name__ == "__main__":
    main()
