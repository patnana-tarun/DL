"""Held-out test video, stage 5: visual audit of the pseudo-labels where the models disagree with them.

The detection metrics measure agreement with the teacher ensemble. To estimate how much of the disagreement is
label error, a random sample of N_FRAMES test frames is audited by eye:
  student "false positives"  detections at the frozen operating threshold with IoU < 0.1 to every label
                             (IoU 0.1-0.5 is localisation, counted automatically, not audited)
  student "misses"           labels with IoU < 0.1 to every detection at the threshold
of the three models of interest (YOLO26n fine-tuned / pretrained, RF-DETR Nano fine-tuned), merged across models
(IoU > 0.5 = one region), plus a random sample of N_LABELS labels for the label precision.
Each region is shown as a crop with context; the verdict per region is written to <out>/audit/verdicts.tsv:
  person      a real person (at least partly visible)
  not_person  not a person (sign, pole, shadow, reflection, statue, part of a bicycle, ...)
  duplicate   a second box on a person who already has a box
  unsure      too small / blurred to decide
Steps:  sheets  -> <out>/audit/sheet_*.jpg + regions.json (crops numbered by region id)
        score   -> <out>/audit/audit.json (rates with Wilson 95% CIs; per-model corrected precision / recall)

    python research/heldout/heldout_audit.py sheets|score --name test1
"""
import argparse
import contextlib
import gzip
import io
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths as P  # noqa: E402,F401

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from pycocotools.coco import COCO  # noqa: E402

import evaluate as E  # noqa: E402

N_FRAMES, N_LABELS, SEED = 40, 80, 0
MODELS = ["YOLO26n (fine-tuned)", "YOLO26n (pretrained)", "RF-DETR Nano (fine-tuned)"]
TILE = 220            # px, square tiles
COLS, ROWS = 6, 4
VERDICTS = ("person", "not_person", "duplicate", "unsure")


def wilson(k, n, z=1.96):
    if n == 0:
        return [0.0, 0.0]
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, c - h), min(1.0, c + h)]


def xyxy(b):
    return np.array([b[0], b[1], b[0] + b[2], b[1] + b[3]], float)


def iou(a, b):
    """xywh arrays [N,4], [M,4] -> [N,M]."""
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    return E.iou_matrix(np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4))


def regions(gt, preds, frozen, img_ids):
    out = []
    for img_id in img_ids:
        g = [a["bbox"] for a in gt.loadAnns(gt.getAnnIds(imgIds=img_id))]
        cand = []   # (kind, model, bbox)
        for m in MODELS:
            d = [x["bbox"] for x in preds[m] if x["image_id"] == img_id and x["score"] >= frozen[m]]
            I = iou(d, g)
            for k, b in enumerate(d):
                if not len(g) or I[k].max() < 0.1:
                    cand.append(("unlabelled_detection", m, b))
            for k, b in enumerate(g):
                if not len(d) or I[:, k].max() < 0.1:
                    cand.append(("missed_label", m, b))
        merged = []
        for kind, m, b in cand:
            for r in merged:
                if r["kind"] == kind and iou([b], [r["bbox"]])[0, 0] > 0.5:
                    r["models"].append(m); break
            else:
                merged.append({"kind": kind, "models": [m], "bbox": [float(v) for v in b], "image_id": int(img_id)})
        out += merged
    return out


def tile(img, b, rid, kind):
    x1, y1, x2, y2 = xyxy(b)
    w, h = x2 - x1, y2 - y1
    side = max(w, h) * 2.2 + 40
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    H, W = img.shape[:2]
    a, c = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
    bb, d = int(min(W, cx + side / 2)), int(min(H, cy + side / 2))
    crop = img[c:d, a:bb].copy()
    col = (0, 200, 255) if kind == "unlabelled_detection" else (255, 80, 255) if kind == "missed_label" else (80, 220, 80)
    cv2.rectangle(crop, (int(x1 - a), int(y1 - c)), (int(x2 - a), int(y2 - c)), col, 1)
    s = (TILE - 22) / max(crop.shape[:2])
    crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA)
    t = np.full((TILE, TILE, 3), 30, np.uint8)
    t[22:22 + crop.shape[0], :crop.shape[1]] = crop
    cv2.putText(t, f"#{rid} {kind[:6]} h{int(h)}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["sheets", "score"])
    ap.add_argument("--name", required=True)
    a = ap.parse_args()
    data, out = Path(f"datasets/{a.name}"), Path(f"results/{a.name}")
    au = out / "audit"; au.mkdir(parents=True, exist_ok=True)
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(data / "coco/test/_annotations.coco.json"))
    frozen = {n: v["valid"]["conf"] for n, v in json.load(open("results/metrics_splits.json")).items()}
    preds = {m: json.load(gzip.open(out / "preds" / f"{m.replace(' ', '_')}__test.json.gz", "rt")) for m in MODELS}
    rng = np.random.default_rng(SEED)
    img_ids = sorted(rng.choice(gt.getImgIds(), N_FRAMES, replace=False).tolist())

    if a.step == "sheets":
        regs = regions(gt, preds, frozen, img_ids)
        anns = [x for i in img_ids for x in gt.loadAnns(gt.getAnnIds(imgIds=i))]
        for k in rng.choice(len(anns), min(N_LABELS, len(anns)), replace=False):
            regs.append({"kind": "random_label", "models": [], "bbox": anns[k]["bbox"], "image_id": anns[k]["image_id"]})
        for r_id, r in enumerate(regs):
            r["id"] = r_id
        json.dump({"frames": img_ids, "regions": regs}, open(au / "regions.json", "w"), indent=1)
        files = {im["id"]: im["file_name"] for im in gt.loadImgs(img_ids)}
        cache, tiles = {}, []
        for r in regs:
            if r["image_id"] not in cache:
                cache = {r["image_id"]: cv2.imread(str(data / "coco/test" / files[r["image_id"]]))}
            tiles.append(tile(cache[r["image_id"]], r["bbox"], r["id"], r["kind"]))
        per = COLS * ROWS
        for s in range(0, len(tiles), per):
            chunk = tiles[s:s + per] + [np.zeros((TILE, TILE, 3), np.uint8)] * (per - len(tiles[s:s + per]))
            sheet = np.vstack([np.hstack(chunk[r * COLS:(r + 1) * COLS]) for r in range(ROWS)])
            cv2.imwrite(str(au / f"sheet_{s // per:02d}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
        kinds = {k: sum(r["kind"] == k for r in regs) for k in ("unlabelled_detection", "missed_label", "random_label")}
        print(f"{len(regs)} regions {kinds} in {len(img_ids)} frames -> {(len(tiles) + per - 1) // per} sheets")
        if not (au / "verdicts.tsv").exists():
            (au / "verdicts.tsv").write_text("id\tverdict\tnote\n" + "".join(f"{r['id']}\t\t\n" for r in regs))
        return

    regs = json.load(open(au / "regions.json"))["regions"]
    v = {}
    for line in (au / "verdicts.tsv").read_text().splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) >= 2 and parts[1]:
            assert parts[1] in VERDICTS, line
            v[int(parts[0])] = parts[1]
    missing = [r["id"] for r in regs if r["id"] not in v]
    assert not missing, f"no verdict for regions {missing[:20]}"
    res = {"frames": len(img_ids), "protocol": __doc__.split("\n\n")[0], "auditor": "AI model, from crops"}
    for kind in ("random_label", "unlabelled_detection", "missed_label"):
        rs = [r for r in regs if r["kind"] == kind]
        c = {x: sum(v[r["id"]] == x for r in rs) for x in VERDICTS}
        dec = c["person"] + c["not_person"] + c["duplicate"]
        res[kind] = {"n": len(rs), **c, "person_rate_decided": c["person"] / max(dec, 1),
                     "person_rate_CI95": wilson(c["person"], dec)}
    # per model on the audited frames: errors that the audit attributes to the labels
    gts = {i: [x["bbox"] for x in gt.loadAnns(gt.getAnnIds(imgIds=i))] for i in img_ids}
    for m in MODELS:
        d = [x for x in preds[m] if x["image_id"] in gts and x["score"] >= frozen[m]]
        per = E.match_images(_sub(gt, img_ids), [x for x in preds[m] if x["image_id"] in gts])
        op = E.operating_point(per, frozen[m])
        fp_reg = [r for r in regs if r["kind"] == "unlabelled_detection" and m in r["models"]]
        fn_reg = [r for r in regs if r["kind"] == "missed_label" and m in r["models"]]
        fp_real = sum(v[r["id"]] == "person" for r in fp_reg)          # "FP" that is a real unlabelled person
        fn_fake = sum(v[r["id"]] in ("not_person", "duplicate") for r in fn_reg)   # "miss" of a bogus label
        TP, FP, FN = op["TP"], op["FP"], op["FN"]
        res[m] = {"threshold": frozen[m], "detections": len(d), "TP": TP, "FP": FP, "FN": FN,
                  "precision": op["precision"], "recall": op["recall"], "F1": op["F1"],
                  "fp_far_from_labels": len(fp_reg), "fp_real_person": fp_real,
                  "fn_far_from_detections": len(fn_reg), "fn_label_not_person": fn_fake}
        TPc, FPc, FNc = TP + fp_real, FP - fp_real, FN - fn_fake
        Pc, Rc = TPc / max(TPc + FPc, 1), TPc / max(TPc + FNc, 1)
        res[m]["audit_corrected"] = {"precision": Pc, "recall": Rc, "F1": 2 * Pc * Rc / max(Pc + Rc, 1e-9),
                                     "note": "recall denominator ignores people that no audited model detected"}
    json.dump(res, open(au / "audit.json", "w"), indent=2)
    print(json.dumps(res, indent=1))


def _sub(gt, img_ids):
    d = {"images": gt.loadImgs(img_ids), "annotations": gt.loadAnns(gt.getAnnIds(imgIds=img_ids)),
         "categories": gt.dataset["categories"]}
    c = COCO(); c.dataset = d
    with contextlib.redirect_stdout(io.StringIO()):
        c.createIndex()
    return c


if __name__ == "__main__":
    main()
