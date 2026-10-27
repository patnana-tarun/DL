"""Inter-annotator agreement and adjudication on hand-made labels (research/annotation/human_gt.py)."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "annotation"))
import human_gt as H  # noqa: E402


def b(x1, y1, x2, y2, **k):
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2, **k}


def test_match_and_ignore_overlap():
    la = [b(0, 0, 10, 40), b(100, 0, 110, 40), b(200, 0, 205, 10)]
    lb = [b(1, 0, 11, 40), b(195, 0, 210, 15, ignore=True)]
    a, bb, m = H.frame_pair(la, lb)
    assert len(a) == 2 and len(bb) == 1 and len(m) == 1      # A's tiny box sits in B's ignore region
    assert m[0][2] > 0.8


def test_agreement_and_adjudication(tmp_path, monkeypatch):
    monkeypatch.setattr(H, "HERE", tmp_path)
    (tmp_path / "tasks" / "t" / "images").mkdir(parents=True)
    (tmp_path / "tasks" / "t" / "images" / "aa1__f1.jpg").write_bytes(b"x")
    json.dump({"task": "t", "kind": "det", "items": [{"file": "aa1__f1.jpg", "scene": "aa1", "t": 0}]},
              open(tmp_path / "tasks" / "t" / "manifest.json", "w"))
    for who, boxes in (("A", [b(0, 0, 10, 40), b(50, 0, 60, 40)]), ("B", [b(0, 0, 10, 42)])):
        (tmp_path / "labels" / who).mkdir(parents=True)
        json.dump({"aa1__f1.jpg": boxes}, open(tmp_path / "labels" / who / "t.json", "w"))
    r = H.agreement("t", "A", "B")
    assert r["matched"] == 1 and r["boxes_A"] == 2 and r["boxes_B"] == 1
    assert np.isclose(r["matched_F1"], 2 / 3, atol=1e-4)
    monkeypatch.setattr(H, "SCENE_GT", {"aa1": str(tmp_path / "src")})
    (tmp_path / "src").mkdir()
    json.dump({"images": [{"id": 7, "file_name": "f1.jpg", "width": 100, "height": 100}], "annotations": [],
               "categories": [{"id": 1, "name": "person"}]}, open(tmp_path / "src" / "_annotations.coco.json", "w"))
    monkeypatch.chdir(tmp_path)
    H.adjudicate("t", "A", "B")
    g = json.load(open(tmp_path / "datasets/human/t/aa1/_annotations.coco.json"))
    assert g["images"][0]["id"] == 7                          # same image ids as the pseudo-label file
    assert sorted(a["iscrowd"] for a in g["annotations"]) == [0, 1]   # matched -> person, unmatched -> ignore
    m = json.load(open(tmp_path / "tasks" / "t_adj" / "manifest.json"))
    assert m["task"] == "t_adj" and [it["file"] for it in m["items"]] == ["aa1__f1.jpg"]   # only disagreement frames
    (tmp_path / "labels" / "C").mkdir(parents=True)
    json.dump({"aa1__f1.jpg": [b(50, 0, 60, 41)]}, open(tmp_path / "labels" / "C" / "t_adj.json", "w"))
    H.adjudicate("t", "A", "B", "C")
    g = json.load(open(tmp_path / "datasets/human/t/aa1/_annotations.coco.json"))
    assert [a["iscrowd"] for a in g["annotations"]] == [0, 0]   # C confirmed A's extra person
