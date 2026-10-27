"""Shift-aware model selection (review section 13, "decision-matrix validity").

The original matrix (research/select_model.py) scored candidates on the aa1 scene only, so its robustness to the
weights said nothing about robustness to a new camera. This matrix adds an out-of-domain criterion measured on
VALIDATION data of cameras that no candidate was trained on (campus scene 1 / 2 validation splits), never on test1.
test1 is then used once, afterwards, to check whether the selection procedure picked well.

Criteria (1 = best candidate):  in-domain AP (aa1 test)  0.30 | OOD validation AP  0.40 | latency  0.20 | GFLOPs 0.10

    python research/revision/select_robust.py        -> results/revision/selection_robust.json
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths as P  # noqa: E402,F401

import numpy as np  # noqa: E402

MS = Path("results/revision/multiscene")
WEIGHTS = {"aa1_AP": 0.30, "ood_val_AP": 0.40, "latency_ms": 0.20, "GFLOPs": 0.10}
ALT = {"in-domain only": {"aa1_AP": 1}, "OOD only": {"ood_val_AP": 1}, "equal": {k: 1 for k in WEIGHTS},
       "original (no OOD)": {"aa1_AP": 0.55, "latency_ms": 0.30, "GFLOPs": 0.15}}
HIGHER = {"aa1_AP": True, "ood_val_AP": True, "latency_ms": False, "GFLOPs": False}
# The scenario: a detector is adapted to ONE camera (aa1) and must then be chosen for cameras nobody labelled.
# Candidates are therefore the models adapted with aa1 data only (plus COCO replay); the campus scenes 1 and 2 are
# unseen by all of them, so their validation splits are one common out-of-domain criterion. Models trained on the
# campus scenes (multi-scene, scene-only) answer a different question and are not candidates. WiSE alpha = 0 / 1
# duplicate the pretrained / fine-tuned endpoints and are dropped.
OOD_VAL = ("v1_valid", "v2_valid")
EXCLUDE = ("multiscene", "YOLO26n v1", "YOLO26n v2", "aa1-1cls", "_a0.0", "_a1.0")
COST = {"YOLO26n": {"latency_ms": 3.7, "GFLOPs": 13.6}, "RF-DETR": {"latency_ms": 7.9, "GFLOPs": 97.3}}


def candidates():
    out = {}
    for f in sorted(MS.glob("*.json")):
        r = json.load(open(f)); n = r["name"]
        if "@" in n or "aa1_test" not in r["scenes"] or any(x in n for x in EXCLUDE) \
                or not all(s in r["valid"] for s in OOD_VAL):
            continue
        fam = "RF-DETR" if n.startswith("RF-DETR") else "YOLO26n"
        out[n] = {"aa1_AP": r["scenes"]["aa1_test"]["AP50-95"],
                  "ood_val_AP": float(np.mean([r["valid"][s] for s in OOD_VAL])), "ood_val_scenes": list(OOD_VAL),
                  **COST[fam], "test1_AP": r["scenes"].get("test1", {}).get("AP50-95")}
    return out


def score(rows, w):
    s = {}
    for c, r in rows.items():
        t = 0.0
        for k, wk in w.items():
            vals = [x[k] for x in rows.values()]
            t += wk * (r[k] / max(vals) if HIGHER[k] else min(vals) / r[k])
        s[c] = t / sum(w.values())
    return s


def main():
    rows = candidates()
    main_s = score(rows, WEIGHTS)
    win = max(main_s, key=main_s.get)
    alt = {k: max((s := score(rows, w)), key=s.get) for k, w in ALT.items()}
    t1 = {c: r["test1_AP"] for c, r in rows.items() if r["test1_AP"] is not None}
    rank_t1 = sorted(t1, key=t1.get, reverse=True)
    res = {"weights": WEIGHTS, "candidates": rows, "scores": main_s, "winner": win, "alternative_winners": alt,
           "check_on_test1": {"winner_test1_AP": t1.get(win), "winner_rank_on_test1": rank_t1.index(win) + 1
                              if win in rank_t1 else None, "n": len(rank_t1), "best_on_test1": rank_t1[0] if rank_t1 else None,
                              "original_matrix_choice_test1_AP": t1.get("YOLO26n FT-aa1 (1-cls)")}}
    Path("results/revision").mkdir(parents=True, exist_ok=True)
    json.dump(res, open("results/revision/selection_robust.json", "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("winner", "alternative_winners", "check_on_test1")}, indent=1))


if __name__ == "__main__":
    main()
