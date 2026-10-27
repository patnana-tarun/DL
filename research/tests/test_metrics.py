"""Hand-computable checks of the detection metrics in research/evaluate.py and research/crosscheck.py."""
import contextlib
import copy
import io

import numpy as np
import pytest
from pycocotools.coco import COCO

import crosscheck
import evaluate as E

IOUS = E.IOUS


def make_coco(gt_by_img, size=(1280, 720)):
    """In-memory COCO ground truth. gt_by_img: {image_id: [[x, y, w, h], ...]}"""
    ds = {"images": [{"id": i, "width": size[0], "height": size[1], "file_name": f"{i}.jpg"} for i in gt_by_img],
          "annotations": [], "categories": [{"id": 1, "name": "person"}]}
    k = 1
    for i, boxes in gt_by_img.items():
        for b in boxes:
            ds["annotations"].append({"id": k, "image_id": i, "category_id": 1, "bbox": list(map(float, b)),
                                      "area": float(b[2] * b[3]), "iscrowd": 0})
            k += 1
    c = COCO()
    c.dataset = ds
    with contextlib.redirect_stdout(io.StringIO()):
        c.createIndex()
    return c


def det(img, box, score):
    return {"image_id": img, "category_id": 1, "bbox": list(map(float, box)), "score": float(score)}


def ap_of(gt, dets):
    per = E.match_images(gt, dets)
    sc, tp, _, _ = E.flatten(per)
    return E.ap_from(sc, tp, sum(p["n_gt"] for p in per))


GT4 = {1: [[0, 0, 100, 100], [200, 0, 100, 100]], 2: [[0, 300, 50, 150], [400, 400, 80, 200]]}


# ---------------------------------------------------------------- IoU
def test_iou_matrix_hand_values():
    d = np.array([[0, 0, 10, 10], [5, 5, 10, 10]], float)
    g = np.array([[0, 0, 10, 10], [5, 0, 10, 10], [20, 20, 5, 5], [0, 0, 20, 20]], float)
    m = E.iou_matrix(d, g)
    assert m.shape == (2, 4)
    np.testing.assert_allclose(m[0], [1.0, 50 / 150, 0.0, 100 / 400])
    np.testing.assert_allclose(m[1], [25 / 175, 50 / 150, 0.0, 100 / 400])


def test_iou_matrix_symmetric_and_touching():
    rng = np.random.default_rng(0)
    a = np.c_[rng.uniform(0, 100, (6, 2)), rng.uniform(1, 50, (6, 2))]
    b = np.c_[rng.uniform(0, 100, (4, 2)), rng.uniform(1, 50, (4, 2))]
    np.testing.assert_allclose(E.iou_matrix(a, b), E.iou_matrix(b, a).T)
    assert ((E.iou_matrix(a, b) >= 0) & (E.iou_matrix(a, b) <= 1)).all()
    assert E.iou_matrix(np.array([[0, 0, 10, 10.]]), np.array([[10, 0, 10, 10.]]))[0, 0] == 0.0
    np.testing.assert_allclose(crosscheck.iou_xywh(a, b), E.iou_matrix(a, b))   # the independent copy agrees


# ---------------------------------------------------------------- matching + AP
def test_perfect_detections_ap_one():
    gt = make_coco(GT4)
    dets = [det(i, b, 0.9 - 0.1 * k) for i, bs in GT4.items() for k, b in enumerate(bs)]
    per = E.match_images(gt, dets)
    assert [p["n_gt"] for p in per] == [2, 2]
    assert all(p["tp"].all() for p in per) and all(np.allclose(p["iou"], 1.0) for p in per)
    np.testing.assert_allclose(ap_of(gt, dets), np.ones(10))


def test_shifted_detections_iou_06():
    """Every detection is its GT shifted by 25 px (IoU = 75/125 = 0.6): a hit up to IoU 0.55, a miss from 0.65."""
    gt_boxes = {1: [[0, 0, 100, 100], [300, 0, 100, 100]], 2: [[0, 200, 100, 100]]}
    gt = make_coco(gt_boxes)
    dets = [det(i, [b[0] + 25, b[1], 100, 100], 0.8) for i, bs in gt_boxes.items() for b in bs]
    per = E.match_images(gt, dets)
    np.testing.assert_allclose(np.concatenate([p["iou"] for p in per]), 0.6)
    ap = ap_of(gt, dets)
    assert ap[0] == 1.0 and ap[1] == 1.0                  # AP50 = 1
    assert (ap[3:] == 0).all()                            # 0.65 ... 0.95, incl. AP75 (index 5)
    assert ap[5] == 0.0
    assert IOUS[0] == 0.5 and IOUS[5] == pytest.approx(0.75)


def test_fp_above_all_tps():
    """4 GT, 4 TPs (0.8..0.5) under one FP (0.9): precision 0, 1/2, 2/3, 3/4, 4/5 -> envelope 0.8 everywhere."""
    gt = make_coco(GT4)
    tps = [det(i, b, s) for (i, b), s in zip([(i, b) for i, bs in GT4.items() for b in bs], [0.8, 0.7, 0.6, 0.5])]
    fp = det(1, [900, 500, 50, 50], 0.9)
    ap = ap_of(gt, tps + [fp])
    np.testing.assert_allclose(ap, 0.8)


def test_missing_gt_caps_recall():
    """2 of 4 people found, perfectly: precision 1 up to recall 0.5 -> 51 of 101 recall points."""
    gt = make_coco(GT4)
    dets = [det(1, GT4[1][0], 0.9), det(2, GT4[2][1], 0.8)]
    np.testing.assert_allclose(ap_of(gt, dets), 51 / 101)


def test_duplicate_detection_is_fp_and_greedy_order():
    gt = make_coco({1: [[0, 0, 100, 100]]})
    dets = [det(1, [0, 0, 100, 100], 0.6), det(1, [5, 0, 100, 100], 0.9)]    # higher score matches first
    per = E.match_images(gt, dets)
    assert per[0]["scores"].tolist() == [0.9, 0.6]
    assert per[0]["tp"][0].tolist() == [True, False]
    assert per[0]["iou"][0] == pytest.approx(95 / 105)
    np.testing.assert_allclose(ap_of(gt, dets)[0], 1.0)  # the FP ranks after the only TP


def test_best_iou_gt_taken_not_first():
    gt = make_coco({1: [[0, 0, 100, 100], [10, 0, 100, 100]]})
    per = E.match_images(gt, [det(1, [10, 0, 100, 100], 0.9), det(1, [0, 0, 100, 100], 0.8)])
    assert per[0]["tp"].all()                               # each detection takes its exact GT


def test_max_det_truncation():
    gt = make_coco({1: [[0, 0, 10, 10]]})
    dets = [det(1, [100 + i, 100, 10, 10], 0.5 - i * 0.001) for i in range(5)] + [det(1, [0, 0, 10, 10], 0.1)]
    per = E.match_images(gt, dets, max_det=5)
    assert len(per[0]["scores"]) == 5 and not per[0]["tp"].any()


def test_image_without_detections_or_gt():
    gt = make_coco({1: [[0, 0, 10, 10]], 2: []})
    per = E.match_images(gt, [det(2, [0, 0, 10, 10], 0.9)])
    assert per[0]["n_gt"] == 1 and len(per[0]["scores"]) == 0
    assert per[1]["n_gt"] == 0 and per[1]["tp"].shape == (10, 1) and not per[1]["tp"].any()
    np.testing.assert_allclose(ap_of(gt, [det(2, [0, 0, 10, 10], 0.9)]), 0.0)


def test_ap_from_edge_cases():
    assert (E.ap_from(np.array([0.5]), np.ones((10, 1), bool), 0) == 0).all()           # no GT
    sc = np.array([0.9, 0.8, 0.7]); tp = np.array([[True, False, True]] * 10)
    a = E.ap_from(sc, tp, 2)
    np.testing.assert_allclose(E.ap_from(sc, tp, 2, w=np.ones(3)), a)                  # unit weights = none
    # precision 1 at recall .5, 2/3 at recall 1 -> 51 points at 1, 50 at 2/3
    np.testing.assert_allclose(a, (51 + 50 * 2 / 3) / 101)
    # weight 2 on an image's detections == that image twice
    w = E.ap_from(sc, tp, 3, w=np.array([1.0, 1.0, 2.0]))
    dup = E.ap_from(np.array([0.9, 0.8, 0.7, 0.7]), np.array([[True, False, True, True]] * 10), 3)
    np.testing.assert_allclose(w, dup)


def test_ap_from_no_detections():
    assert (E.ap_from(np.zeros(0), np.zeros((10, 0), bool), 3) == 0).all()


# ---------------------------------------------------------------- operating point, LAMR
def test_operating_point_counts():
    gt = make_coco({1: [[0, 0, 100, 100], [300, 0, 100, 100]], 2: [[0, 0, 50, 50]]})
    dets = [det(1, [0, 0, 100, 100], 0.9), det(1, [700, 0, 50, 50], 0.8), det(1, [300, 0, 100, 100], 0.3),
            det(2, [600, 600, 50, 50], 0.95)]
    per = E.match_images(gt, dets)
    op = E.operating_point(per, 0.5)
    assert (op["TP"], op["FP"], op["FN"]) == (1, 2, 2)
    assert op["precision"] == pytest.approx(1 / 3) and op["recall"] == pytest.approx(1 / 3)
    assert op["F1"] == pytest.approx(1 / 3) and op["det_accuracy"] == pytest.approx(0.2)
    assert op["FPPI"] == 1.0 and op["mean_IoU_TP"] == pytest.approx(1.0)
    assert op["count_MAE"] == 0.0 and op["count_RMSE"] == 0.0 and op["count_accuracy"] == 1.0
    op = E.operating_point(per, 0.2)
    assert (op["TP"], op["FP"], op["FN"]) == (2, 2, 1)
    assert op["count_MAE"] == pytest.approx(0.5) and op["count_RMSE"] == pytest.approx(np.sqrt(0.5))
    assert op["count_accuracy"] == pytest.approx(0.75)
    op = E.operating_point(per, 0.99)
    assert (op["TP"], op["FP"], op["FN"]) == (0, 0, 3) and op["precision"] == 0 and op["mean_IoU_TP"] == 0.0


def test_pr_curve_consistent_with_operating_point():
    gt = make_coco(GT4)
    dets = [det(1, GT4[1][0], 0.9), det(1, [900, 0, 10, 10], 0.6), det(2, GT4[2][0], 0.4)]
    per = E.match_images(gt, dets)
    c = E.pr_curve(per)
    for conf in (0.3, 0.5, 0.7):
        i = int(np.argmin(np.abs(c["conf"] - conf)))
        op = E.operating_point(per, c["conf"][i])
        assert c["precision"][i] == pytest.approx(op["precision"]) and c["recall"][i] == pytest.approx(op["recall"])


def _lamr_data(seed=0, n_img=20, drop=0.0, fps_per_img=0, fp_score=0.95):
    rng = np.random.default_rng(seed)
    gt_boxes, dets = {}, []
    for i in range(1, n_img + 1):
        boxes = [[100 * k, 50, 40, 100] for k in range(3)]
        gt_boxes[i] = boxes
        for b in boxes:
            if rng.random() >= drop:
                dets.append(det(i, b, rng.uniform(0.5, 0.9)))
        for _ in range(fps_per_img):
            dets.append(det(i, [900, 500, 30, 30], fp_score))
    return E.match_images(make_coco(gt_boxes), dets)


def test_lamr_monotonic_sanity():
    perfect = E.lamr(_lamr_data())
    assert perfect <= 1e-9                                            # no misses at any FPPI
    missed = E.lamr(_lamr_data(drop=0.3))
    more_missed = E.lamr(_lamr_data(drop=0.6))
    assert 0 < missed < more_missed <= 1
    with_fps = E.lamr(_lamr_data(fps_per_img=1))                     # 1 confident FP per image (FPPI 1)
    assert with_fps > perfect and with_fps <= 1
    nothing = E.match_images(make_coco({1: [[0, 0, 10, 10]]}), [])
    assert E.lamr(nothing) == pytest.approx(1.0)
    # low-score FPs (ranked after every TP) cost nothing
    assert E.lamr(_lamr_data(fps_per_img=1, fp_score=0.01)) == pytest.approx(perfect)


# ---------------------------------------------------------------- independent implementations
def random_dataset(seed, n_img=25, p_fp=0.4):
    rng = np.random.default_rng(seed)
    gt_boxes, dets = {}, []
    for i in range(1, n_img + 1):
        n = int(rng.integers(0, 9))
        xy = rng.uniform(0, 1000, (n, 2)); wh = rng.uniform(10, 150, (n, 2))
        gt_boxes[i] = np.c_[xy, wh].round(2).tolist()
        for b in gt_boxes[i]:
            if rng.random() < 0.85:                                   # found, with localisation noise
                jit = rng.normal(0, 0.12, 4) * [b[2], b[3], b[2], b[3]]
                bb = [b[0] + jit[0], b[1] + jit[1], max(2, b[2] + jit[2]), max(2, b[3] + jit[3])]
                dets.append(det(i, bb, rng.uniform(0.05, 1)))
                if rng.random() < 0.15:                               # duplicate
                    dets.append(det(i, [v + 3 for v in bb], rng.uniform(0.05, 1)))
        while rng.random() < p_fp:                                    # background false positives
            dets.append(det(i, [*rng.uniform(0, 1000, 2), *rng.uniform(10, 100, 2)], rng.uniform(0.05, 1)))
    return gt_boxes, dets


@pytest.mark.parametrize("seed", range(5))
def test_crosscheck_ap_scratch_agrees(seed):
    gt_boxes, dets = random_dataset(seed)
    ours = ap_of(make_coco(gt_boxes), dets)
    g = {i: np.array(b, float).reshape(-1, 4) for i, b in gt_boxes.items()}
    d = {}
    for x in dets:
        d.setdefault(x["image_id"], []).append([*x["bbox"], x["score"]])
    d = {k: np.array(v, float) for k, v in d.items()}
    scratch = np.array([crosscheck.ap_scratch(g, d, t) for t in IOUS])
    np.testing.assert_allclose(scratch, ours, atol=1e-12)
    assert 0 < ours[0] < 1 and ours[0] >= ours[-1]


@pytest.mark.parametrize("seed", range(4))
def test_pycocotools_agreement(seed):
    gt_boxes, dets = random_dataset(seed + 100)
    gt = make_coco(gt_boxes)
    _, m = E.coco_eval(gt, copy.deepcopy(dets))                       # loadRes mutates its input
    ap = ap_of(gt, dets)
    assert m["AP50-95"] == pytest.approx(ap.mean(), abs=1e-4)
    assert m["AP50"] == pytest.approx(ap[0], abs=1e-4)
    assert m["AP75"] == pytest.approx(ap[5], abs=1e-4)


def test_pycocotools_agreement_hand_cases():
    gt = make_coco(GT4)
    tps = [det(i, b, s) for (i, b), s in zip([(i, b) for i, bs in GT4.items() for b in bs], [0.8, 0.7, 0.6, 0.5])]
    _, m = E.coco_eval(gt, copy.deepcopy(tps + [det(1, [900, 500, 50, 50], 0.9)]))
    assert m["AP50"] == pytest.approx(0.8, abs=1e-6) and m["AP50-95"] == pytest.approx(0.8, abs=1e-6)
    _, m = E.coco_eval(gt, copy.deepcopy([det(1, GT4[1][0], 0.9), det(2, GT4[2][1], 0.8)]))
    assert m["AP50"] == pytest.approx(51 / 101, abs=1e-6)
