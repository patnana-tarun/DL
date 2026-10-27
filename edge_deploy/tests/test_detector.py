
import cv2
import numpy as np
import pytest
from conftest import ONNX_384, ROOT

from edgevision import config, detector
from edgevision.detector import decode, letterbox, nms

TEST_IMG_DIR = ROOT / "datasets" / "coco" / "test"
YOLO_PT = ROOT / "runs" / "yolo26n" / "weights" / "best.pt"


def first_test_image():
    imgs = sorted(TEST_IMG_DIR.glob("*.jpg"))
    if not imgs:
        pytest.skip("datasets/coco/test has no images")
    return cv2.imread(str(imgs[0]))


def iou_xyxy(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    iw = (np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0])).clip(0)
    ih = (np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1])).clip(0)
    inter = iw * ih
    area = lambda x: (x[:, 2] - x[:, 0]).clip(0) * (x[:, 3] - x[:, 1]).clip(0)  # noqa: E731
    return inter / (area(a)[:, None] + area(b)[None] - inter + 1e-9)


# ---------------------------------------------------------------- letterbox
@pytest.mark.parametrize("hw, out_hw", [((720, 1280), (384, 640)), ((480, 640), (384, 640)),
                                        ((1080, 1080), (384, 640)), ((101, 333), (64, 64)),
                                        ((384, 640), (384, 640)), ((300, 200), (736, 1280))])
def test_letterbox_geometry(hw, out_hw):
    H, W = hw
    img = np.full((H, W, 3), 200, np.uint8)
    out, s, (px, py) = letterbox(img, out_hw)
    oh, ow = out_hw
    assert out.shape == (oh, ow, 3) and out.dtype == np.uint8
    assert s == pytest.approx(min(oh / H, ow / W))
    nh, nw = round(H * s), round(W * s)
    assert nh <= oh and nw <= ow and (nh == oh or nw == ow)        # fills one side exactly
    assert abs(px - (ow - nw - px)) <= 1 and abs(py - (oh - nh - py)) <= 1   # centred
    assert (out[py:py + nh, px:px + nw] == 200).all()             # picture region
    pad = np.ones((oh, ow), bool); pad[py:py + nh, px:px + nw] = False
    assert (out[pad] == 114).all()                                 # padding value


def test_letterbox_identity_when_same_size():
    img = np.random.default_rng(0).integers(0, 255, (384, 640, 3), dtype=np.uint8)
    out, s, pad = letterbox(img, (384, 640))
    assert s == 1.0 and pad == (0, 0) and np.array_equal(out, img)


@pytest.mark.parametrize("hw", [(720, 1280), (1080, 810), (500, 500)])
def test_letterbox_inverse_mapping_round_trip(hw):
    """A box drawn in the frame, found in the letterboxed image and mapped back lands where it started."""
    H, W = hw
    rng = np.random.default_rng(1)
    for _ in range(5):
        x1, y1 = int(rng.integers(0, W - 200)), int(rng.integers(0, H - 200))
        x2, y2 = x1 + int(rng.integers(60, 200)), y1 + int(rng.integers(60, 200))
        img = np.zeros((H, W, 3), np.uint8)
        img[y1:y2, x1:x2] = 255
        out, s, (px, py) = letterbox(img, (384, 640), pad_value=0)
        ys, xs = np.where(out[:, :, 0] > 127)
        found = np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], float)
        back = (found - [px, py, px, py]) / s
        assert np.abs(back - [x1, y1, x2, y2]).max() <= 1.5 / s + 1
        # the forward map of the original box agrees with what is in the image
        fwd = np.array([x1, y1, x2, y2]) * s + [px, py, px, py]
        assert np.abs(fwd - found).max() <= 1.5


# ---------------------------------------------------------------- NMS
def nms_reference(boxes, scores, thr):
    """Brute force: walk boxes by score, keep one if it overlaps no kept box by more than thr."""
    keep = []
    for i in sorted(range(len(boxes)), key=lambda i: -scores[i]):
        if all(iou_xyxy(boxes[i:i + 1], boxes[j:j + 1])[0, 0] <= thr for j in keep):
            keep.append(i)
    return keep


def random_boxes(rng, n, size=200):
    xy = rng.uniform(0, size, (n, 2))
    wh = rng.uniform(5, 60, (n, 2))
    return np.c_[xy, xy + wh].astype(np.float32), rng.uniform(0, 1, n).astype(np.float32)


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("thr", [0.3, 0.5, 0.7])
def test_nms_matches_brute_force(seed, thr):
    rng = np.random.default_rng(seed)
    b, s = random_boxes(rng, 150)
    got = nms(b, s, thr)
    assert got.dtype == np.int64
    assert got.tolist() == nms_reference(b, s, thr)
    assert np.all(np.diff(s[got]) <= 0)                              # highest score first


def test_nms_edge_cases():
    assert nms(np.zeros((1, 4), np.float32) + [0, 0, 10, 10], np.array([0.5], np.float32), 0.5).tolist() == [0]
    b = np.array([[0, 0, 10, 10], [0, 0, 10, 10], [20, 20, 30, 30]], np.float32)
    assert sorted(nms(b, np.array([0.9, 0.8, 0.7], np.float32), 0.5).tolist()) == [0, 2]
    assert nms(b, np.array([0.9, 0.8, 0.7], np.float32), 1.0).tolist() == [0, 1, 2]   # nothing above IoU 1


# ---------------------------------------------------------------- decode
def raw_output(boxes_xyxy, cls_scores, n_anchors=5040):
    """(1, 4+nc, N) like the exported one-to-many head: cx cy w h + class scores. The real head always has
    N = 5040 anchors at 384x640 (48x80 + 24x40 + 12x20); the given boxes come first, the rest score 0."""
    b = np.asarray(boxes_xyxy, np.float32)
    cxcywh = np.c_[(b[:, :2] + b[:, 2:]) / 2, b[:, 2:] - b[:, :2]]
    rows = np.c_[cxcywh, np.asarray(cls_scores, np.float32).reshape(len(b), -1)]
    pad = np.zeros((n_anchors - len(rows), rows.shape[1]), np.float32)
    pad[:, :4] = [320, 192, 10, 10]
    return np.r_[rows, pad].T[None]


def test_decode_raw_single_class():
    boxes = [[10, 10, 50, 110], [12, 12, 52, 112], [200, 20, 240, 120], [300, 30, 340, 130], [5, 5, 6, 6]]
    scores = [0.9, 0.8, 0.6, 0.04, 0.3]                    # #2 duplicates #1, #4 is below conf
    out = raw_output(boxes, scores)
    assert out.shape == (1, 5, 5040)
    b, s = decode(out, conf=0.05, iou=0.7, max_det=300)
    assert s.tolist() == pytest.approx([0.9, 0.6, 0.3])
    np.testing.assert_allclose(b, np.array(boxes, np.float32)[[0, 2, 4]], atol=1e-4)
    b, s = decode(out, conf=0.05, iou=0.7, max_det=2)     # max_det keeps the best
    assert s.tolist() == pytest.approx([0.9, 0.6])
    b, s = decode(out, conf=0.95, iou=0.7, max_det=300)
    assert b.shape == (0, 4) and s.shape == (0,)


def test_decode_raw_many_anchors_layout():
    """Real head shape (1, 5, 5040): N >> 4+nc, so it must be transposed, not read as (N, 5)."""
    rng = np.random.default_rng(0)
    n = 5040
    xy = rng.uniform(0, 600, (n, 2)); wh = rng.uniform(10, 50, (n, 2))
    sc = np.zeros(n); sc[[7, 4000]] = [0.8, 0.5]
    out = raw_output(np.c_[xy, xy + wh], sc)
    assert out.shape == (1, 5, 5040)
    b, s = decode(out, 0.05, 0.7, 300)
    assert s.tolist() == pytest.approx([0.8, 0.5])
    np.testing.assert_allclose(b, np.c_[xy, xy + wh][[7, 4000]], atol=1e-3)


def test_decode_raw_multiclass_uses_person_column():
    boxes = [[0, 0, 10, 20], [100, 0, 110, 20], [200, 0, 210, 20]]
    cls = np.zeros((3, 80)); cls[0, 0] = 0.7; cls[1, 2] = 0.95; cls[1, 0] = 0.01; cls[2, 0] = 0.4
    b, s = decode(raw_output(boxes, cls), conf=0.05, iou=0.7, max_det=300)
    assert s.tolist() == pytest.approx([0.7, 0.4])                   # the car (class 2) is ignored
    np.testing.assert_allclose(b, np.array(boxes, np.float32)[[0, 2]], atol=1e-4)


def test_decode_end2end_with_class_filter():
    K = 300                                                          # exported NMS-free head: (1, 300, 6)
    out = np.zeros((1, K, 6), np.float32)
    out[0, 0] = [10, 10, 50, 110, 0.9, 0]
    out[0, 1] = [60, 10, 90, 110, 0.85, 2]                           # not a person
    out[0, 2] = [100, 10, 150, 110, 0.5, 0]
    out[0, 3] = [160, 10, 190, 110, 0.03, 0]                         # below conf
    b, s = decode(out, conf=0.05, iou=0.7, max_det=300)
    assert s.tolist() == pytest.approx([0.9, 0.5])
    np.testing.assert_allclose(b, [[10, 10, 50, 110], [100, 10, 150, 110]])


def test_decode_end2end_single_class_keeps_all():
    out = np.zeros((1, 300, 6), np.float32)
    out[0, :3] = [[0, 0, 5, 5, 0.9, 0], [10, 0, 15, 5, 0.6, 0], [20, 0, 25, 5, 0.2, 0]]
    b, s = decode(out, conf=0.1, iou=0.7, max_det=300)
    assert s.tolist() == pytest.approx([0.9, 0.6, 0.2]) and b.shape == (3, 4)


def test_decode_accepts_2d_input():
    out = raw_output([[0, 0, 10, 10]], [0.9])[0]
    b, s = decode(out, 0.05, 0.7, 10)
    assert len(b) == 1


# ---------------------------------------------------------------- ONNX model
def onnx_cfg(**kw):
    c = dict(config.DEFAULTS["detector"], weights=str(ONNX_384), providers=["cpu"])
    c.update(kw)
    return c


@pytest.fixture(scope="module")
def onnx_det():
    if not ONNX_384.exists():
        pytest.skip("ONNX model missing")
    return detector.build(onnx_cfg())


def test_onnx_detector_on_real_frame(onnx_det):
    img = first_test_image()
    H, W = img.shape[:2]
    assert onnx_det.hw == (384, 640) and onnx_det.provider == "CPUExecutionProvider"
    b, s = onnx_det(img)
    assert b.dtype == np.float32 and s.dtype == np.float32 and b.shape == (len(s), 4)
    assert len(b) > 0, "aa1 test frames contain people"
    assert (b[:, 0] >= 0).all() and (b[:, 1] >= 0).all() and (b[:, 2] <= W).all() and (b[:, 3] <= H).all()
    assert (b[:, 2] >= b[:, 0]).all() and (b[:, 3] >= b[:, 1]).all()
    assert ((s > onnx_det.cfg["conf"]) & (s <= 1.0)).all()
    assert np.all(np.diff(s) <= 0), "scores sorted, highest first"
    assert len(b) <= onnx_det.cfg["max_det"]
    strong = b[s > 0.5]
    assert len(strong) >= 1 and ((strong[:, 3] - strong[:, 1]) > (strong[:, 2] - strong[:, 0]) * 0.8).mean() > 0.5
    b2, s2 = onnx_det(img)                                           # deterministic
    np.testing.assert_array_equal(b, b2)


def test_onnx_detector_relative_weights_and_resized_input(onnx_det):
    det = detector.build(onnx_cfg(weights="models/yolo26n_person_384x640.onnx", conf=0.25))
    img = first_test_image()
    small = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
    b, s = det(small)
    B, S = det(img)
    assert len(b) > 0 and (b[:, 2] <= small.shape[1]).all() and (b[:, 3] <= small.shape[0]).all()
    assert abs(len(b) - len(B)) <= max(3, 0.3 * len(B))              # same people at half resolution


def test_onnx_detector_empty_frame(onnx_det):
    b, s = onnx_det(np.zeros((720, 1280, 3), np.uint8))
    assert b.shape[1] == 4 and len(b) == len(s)


def test_unknown_provider_falls_back_to_cpu():
    det = detector.build(onnx_cfg(providers=["tensorrt", "cuda"]))   # not available here -> CPU
    assert det.provider == "CPUExecutionProvider" or det.provider in ("CUDAExecutionProvider",
                                                                      "TensorrtExecutionProvider")


@pytest.mark.slow
@pytest.mark.skipif(not YOLO_PT.exists(), reason="fine-tuned weights missing")
def test_onnx_matches_ultralytics(onnx_det):
    from ultralytics import YOLO
    img = first_test_image()
    det = detector.build(onnx_cfg(conf=0.25))
    b, s = det(img)
    r = YOLO(str(YOLO_PT)).predict(img, imgsz=(384, 640), conf=0.25, device="cpu", verbose=False)[0]
    ub, us = r.boxes.xyxy.numpy(), r.boxes.conf.numpy()
    assert len(ub) > 0
    assert abs(len(b) - len(ub)) <= max(2, round(0.1 * len(ub))), (len(b), len(ub))
    from scipy.optimize import linear_sum_assignment
    iou = iou_xyxy(b, ub)
    ri, ci = linear_sum_assignment(-iou)
    matched = iou[ri, ci]
    matched = matched[matched > 0.3]
    assert len(matched) >= 0.9 * min(len(b), len(ub))
    assert matched.mean() > 0.9, matched.mean()
