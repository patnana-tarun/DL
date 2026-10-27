"""Human ground truth: inter-annotator agreement, adjudication, and evaluation of every model (and of the teacher
pseudo-labels) against the humans. See PROTOCOL.md.

    python research/annotation/human_gt.py agreement  --task det_main [--a A --b B]
    python research/annotation/human_gt.py adjudicate --task det_main [--a A --b B] [--adjudicated C]
    python research/annotation/human_gt.py eval       --task det_main
    python research/annotation/human_gt.py agreement  --task track_aa1        (tracking clip: HOTA / IDF1 A vs B)

agreement   per frame, A's and B's non-ignore boxes are matched one-to-one (Hungarian, IoU >= 0.5); a box that
            overlaps the other annotator's ignore box is left out. Matched F1 = 2M / (nA + nB), mean IoU of the
            matches, both by person height and by scene; per-frame count difference. Before adjudication.
adjudicate  matched pairs -> the mean box; unmatched boxes -> listed for a third person, who annotates only the
            frames with a disagreement, blind, as task <task>_adj (written here: tasks/<task>_adj/); with
            --adjudicated C an unmatched box is kept if C drew a box on it (IoU >= 0.5), otherwise it becomes an
            IGNORE region; without C every unmatched box is kept as an IGNORE region (COCO iscrowd = 1: detections on
            them are neither true nor false positives, the Caltech convention), so uncertain people never count
            against a model. -> datasets/human/<task>/<scene>/_annotations.coco.json (+ disagreements.json)
eval        every cached model prediction (results/revision/preds) and the teacher labels scored against the
            human GT of the same frames: AP, AP50, recall by height; the ranking of models under human GT vs
            under pseudo-labels on the same frames (Kendall tau). -> results/revision/human_eval_<task>.json
"""
import argparse
import contextlib
import copy
import gzip
import io
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths as P  # noqa: E402,F401

import numpy as np  # noqa: E402
from pycocotools.coco import COCO  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402
from scipy.stats import kendalltau  # noqa: E402

import evaluate as E  # noqa: E402

HERE = Path(__file__).resolve().parent
BINS = [(0, 32), (32, 64), (64, 128), (128, 10000)]
SCENE_GT = {"aa1": "datasets/coco/test", "scene_v1": "datasets/scene_v1/splits/test",
            "scene_v2": "datasets/scene_v2/splits/test", "test1": "datasets/test1/coco/test"}
PRED_SCENE = {"aa1": "aa1_test", "scene_v1": "v1_test", "scene_v2": "v2_test", "test1": "test1"}


def load(task, who):
    p = HERE / "labels" / who / f"{task}.json"
    if not p.exists():
        raise SystemExit(f"no labels for annotator {who!r} on {task}: {p}")
    return json.load(open(p))


def arr(boxes, ignore=False):
    sel = [b for b in boxes if bool(b.get("ignore")) == ignore]
    return np.array([[b["x1"], b["y1"], b["x2"], b["y2"]] for b in sel], float).reshape(-1, 4), sel


def iou(a, b):
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    ar = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])  # noqa: E731
    return inter / (ar(a)[:, None] + ar(b)[None] - inter + 1e-9)


def ioa(a, b):
    """Share of each box in a covered by its best box in b (for ignore overlap)."""
    if not len(a) or not len(b):
        return np.zeros(len(a))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    return (inter / ((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]))[:, None]).max(1)


def match(a, b, thr=0.5):
    m = iou(a, b)
    if not m.size:
        return []
    r, c = linear_sum_assignment(-m)
    return [(i, j, m[i, j]) for i, j in zip(r, c) if m[i, j] >= thr]


def frame_pair(la, lb):
    """-> (A boxes, B boxes, matches) after removing boxes covered by the other annotator's ignore regions."""
    a, _ = arr(la); b, _ = arr(lb); ai, _ = arr(la, True); bi, _ = arr(lb, True)
    a = a[ioa(a, bi) < 0.5] if len(a) else a
    b = b[ioa(b, ai) < 0.5] if len(b) else b
    return a, b, match(a, b)


def agreement(task, A, B):
    man = json.load(open(HERE / "tasks" / task / "manifest.json"))
    la, lb = load(task, A), load(task, B)
    common = [it for it in man["items"] if it["file"] in la and it["file"] in lb]
    if man["kind"] == "track":
        return track_agreement(man, common, la, lb)
    tot = {"nA": 0, "nB": 0, "M": 0, "ious": []}
    by_bin = {f"{lo}-{hi}px": {"nA": 0, "nB": 0, "M": 0} for lo, hi in BINS}
    by_scene, count_diff = {}, []
    for it in common:
        a, b, m = frame_pair(la[it["file"]], lb[it["file"]])
        sc = by_scene.setdefault(it["scene"], {"nA": 0, "nB": 0, "M": 0})
        for d in (tot, sc):
            d["nA"] += len(a); d["nB"] += len(b); d["M"] += len(m)
        tot["ious"] += [x[2] for x in m]
        count_diff.append(len(a) - len(b))
        ha = a[:, 3] - a[:, 1] if len(a) else np.zeros(0); hb = b[:, 3] - b[:, 1] if len(b) else np.zeros(0)
        hm = [(a[i, 3] - a[i, 1] + b[j, 3] - b[j, 1]) / 2 for i, j, _ in m]
        for lo, hi in BINS:
            k = f"{lo}-{hi}px"
            by_bin[k]["nA"] += int(((ha >= lo) & (ha < hi)).sum()); by_bin[k]["nB"] += int(((hb >= lo) & (hb < hi)).sum())
            by_bin[k]["M"] += int(sum(lo <= h < hi for h in hm))
    f1 = lambda d: round(2 * d["M"] / max(d["nA"] + d["nB"], 1), 4)  # noqa: E731
    return {"task": task, "annotators": [A, B], "frames": len(common), "boxes_A": tot["nA"], "boxes_B": tot["nB"],
            "matched": tot["M"], "matched_F1": f1(tot), "mean_IoU": round(float(np.mean(tot["ious"])), 4) if tot["ious"] else None,
            "IoU_ge_0.7": round(float(np.mean(np.array(tot["ious"]) >= 0.7)), 4) if tot["ious"] else None,
            "count_MAE_per_frame": round(float(np.mean(np.abs(count_diff))), 3) if count_diff else None,
            "by_height": {k: {**v, "F1": f1(v)} for k, v in by_bin.items()},
            "by_scene": {k: {**v, "F1": f1(v)} for k, v in by_scene.items()}}


def track_agreement(man, items, la, lb):
    from track import score   # HOTA / CLEAR / identity metrics (trackers.eval), as for the models
    def seq(lab):
        out = []
        for it in items:
            b, sel = arr(lab[it["file"]])
            out.append((b, np.array([x.get("tid", -1) for x in sel], int)))
        return out
    A, B = seq(la), seq(lb)
    return {"task": man["task"], "frames": len(items), "B_vs_A": score(A, B, 0, len(items)),
            "A_vs_B": score(B, A, 0, len(items))}


def adjudicate(task, A, B, C=None):
    man = json.load(open(HERE / "tasks" / task / "manifest.json"))
    la, lb = load(task, A), load(task, B)
    verdicts = load(f"{task}_adj", C) if C else {}
    per_scene, disagreements = {}, []
    for it in man["items"]:
        if it["file"] not in la or it["file"] not in lb:
            continue
        a, b, m = frame_pair(la[it["file"]], lb[it["file"]])
        keep = [((a[i] + b[j]) / 2).tolist() for i, j, _ in m]
        ua = [a[i].tolist() for i in range(len(a)) if i not in {x[0] for x in m}]
        ub = [b[j].tolist() for j in range(len(b)) if j not in {x[1] for x in m}]
        ign = [x.tolist() for x in np.vstack([arr(la[it["file"]], True)[0], arr(lb[it["file"]], True)[0]])]
        vc = verdicts.get(it["file"])                 # adjudicator: their own non-ignore boxes decide
        if vc is not None:
            vb, _ = arr(vc)
            for u in ua + ub:
                (keep if len(vb) and iou(np.array([u]), vb).max() >= 0.5 else ign).append(u)
        else:
            ign += ua + ub
        disagreements += [{"file": it["file"], "box": u, "from": w} for w, us in (("A", ua), ("B", ub)) for u in us]
        per_scene.setdefault(it["scene"], []).append((it, keep, ign))
    out_root = Path("datasets/human") / task
    stats = {}
    for scene, rows in per_scene.items():
        src = json.load(open(Path(SCENE_GT[scene]) / "_annotations.coco.json"))
        by_name = {im["file_name"]: im for im in src["images"]}
        coco = {"images": [], "annotations": [], "categories": src["categories"]}
        for it, keep, ign in rows:
            im = copy.deepcopy(by_name[it["file"].split("__", 1)[1]]); coco["images"].append(im)
            for (x1, y1, x2, y2), crowd in [(k, 0) for k in keep] + [(g, 1) for g in ign]:
                coco["annotations"].append({"id": len(coco["annotations"]) + 1, "image_id": im["id"], "category_id": 1,
                                            "bbox": [x1, y1, x2 - x1, y2 - y1], "area": (x2 - x1) * (y2 - y1),
                                            "iscrowd": crowd})
        d = out_root / scene; d.mkdir(parents=True, exist_ok=True)
        json.dump(coco, open(d / "_annotations.coco.json", "w"))
        stats[scene] = {"frames": len(rows), "people": sum(len(k) for _, k, _ in rows),
                        "ignore": sum(len(g) for _, _, g in rows)}
    json.dump(disagreements, open(out_root / "disagreements.json", "w"), indent=0)
    if not C:
        adj_task(man, {d["file"] for d in disagreements})
    json.dump(stats, open(out_root / "stats.json", "w"), indent=1)
    print(json.dumps(stats, indent=1), f"\n{len(disagreements)} unmatched boxes -> {out_root}/disagreements.json")


def adj_task(man, files):
    """The adjudicator's task: only the frames with a disagreement, same file names, images copied."""
    d = HERE / "tasks" / f"{man['task']}_adj"
    (d / "images").mkdir(parents=True, exist_ok=True)
    items = [it for it in man["items"] if it["file"] in files]
    for it in items:
        shutil.copy2(HERE / "tasks" / man["task"] / "images" / it["file"], d / "images" / it["file"])
    json.dump({**man, "task": f"{man['task']}_adj", "items": items}, open(d / "manifest.json", "w"), indent=1)
    print(f"adjudication task: {len(items)} frames -> {d}  (open ?task={man['task']}_adj&annotator=C)")


def human_coco(task, scene):
    with contextlib.redirect_stdout(io.StringIO()):
        return COCO(str(Path("datasets/human") / task / scene / "_annotations.coco.json"))


def evaluate(task):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "heldout"))
    from heldout_eval import recall_by_height
    res = {"task": task, "scenes": {}}
    for scene in sorted(p.name for p in (Path("datasets/human") / task).iterdir() if p.is_dir()):
        hg = human_coco(task, scene)
        ids = set(hg.getImgIds())
        with contextlib.redirect_stdout(io.StringIO()):
            pg_full = COCO(str(Path(SCENE_GT[scene]) / "_annotations.coco.json"))
        pg = copy.deepcopy(pg_full.dataset)                 # pseudo-labels on exactly the same frames
        pg["images"] = [im for im in pg["images"] if im["id"] in ids]
        pg["annotations"] = [a for a in pg["annotations"] if a["image_id"] in ids]
        pgc = COCO(); pgc.dataset = pg
        with contextlib.redirect_stdout(io.StringIO()):
            pgc.createIndex()
        teacher = [{"image_id": a["image_id"], "category_id": 1, "bbox": a["bbox"], "score": a.get("score", 1.0)}
                   for a in pg["annotations"]]
        rows = {"teacher pseudo-labels": {"vs_human": E.coco_eval(hg, teacher)[1]}}
        rows["teacher pseudo-labels"]["recall_by_height_px"] = recall_by_height(hg, teacher, 0.0)
        for f in sorted(Path("results/revision/preds").glob(f"*__{PRED_SCENE[scene]}.json.gz")):
            name = f.name.split("__")[0]
            dets = [d for d in json.load(gzip.open(f)) if d["image_id"] in ids]
            ms = json.load(open(Path("results/revision/multiscene") / f"{name}.json")) \
                if (Path("results/revision/multiscene") / f"{name}.json").exists() else {}
            conf = ms.get("conf", 0.3)
            rows[name] = {"vs_human": E.coco_eval(hg, dets)[1], "vs_pseudo": E.coco_eval(pgc, dets)[1],
                          "recall_by_height_px": recall_by_height(hg, dets, conf), "conf": conf}
        models = [k for k in rows if "vs_pseudo" in rows[k]]
        tau = kendalltau([rows[k]["vs_human"]["AP50-95"] for k in models], [rows[k]["vs_pseudo"]["AP50-95"] for k in models])
        res["scenes"][scene] = {"frames": len(ids), "human_people": sum(1 for a in hg.dataset["annotations"] if not a["iscrowd"]),
                                "models": rows, "kendall_tau_AP_human_vs_pseudo": {"tau": float(tau.statistic),
                                                                                   "p": float(tau.pvalue), "n": len(models)}}
        print(scene, "teacher AP vs human", round(rows["teacher pseudo-labels"]["vs_human"]["AP50-95"], 3), "tau", round(tau.statistic, 3))
    Path("results/revision").mkdir(parents=True, exist_ok=True)
    json.dump(res, open(f"results/revision/human_eval_{task}.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["agreement", "adjudicate", "eval"])
    ap.add_argument("--task", required=True); ap.add_argument("--a", default="A"); ap.add_argument("--b", default="B")
    ap.add_argument("--adjudicated", default=None)
    x = ap.parse_args()
    if x.cmd == "agreement":
        r = agreement(x.task, x.a, x.b)
        Path("results/revision").mkdir(parents=True, exist_ok=True)
        json.dump(r, open(f"results/revision/agreement_{x.task}.json", "w"), indent=1, default=float)
        print(json.dumps(r, indent=1, default=float))
    elif x.cmd == "adjudicate":
        adjudicate(x.task, x.a, x.b, x.adjudicated)
    else:
        evaluate(x.task)
