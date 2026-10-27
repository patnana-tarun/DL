"""Hand-computable checks of the held-out-video helpers in research/heldout/."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "heldout"))
import heldout_audit as HA  # noqa: E402
import heldout_detect as HD  # noqa: E402
import heldout_eval as HE  # noqa: E402
import heldout_track as HT  # noqa: E402
from test_metrics import make_coco  # noqa: E402


def test_pack_unpack_roundtrip_keeps_empty_frames():
    arrs = [np.ones((2, 5)), np.zeros((0, 5)), np.arange(15.).reshape(3, 5)]
    data, off = HD.pack(arrs)
    back = HD.unpack(data, off)
    assert list(off) == [0, 2, 2, 5]
    assert [len(a) for a in back] == [2, 0, 3]
    np.testing.assert_array_equal(back[2], arrs[2])


def test_block_bootstrap_draws_contiguous_blocks_of_full_length():
    rng = np.random.default_rng(0)
    idx = HE.block_indices(100, 10, rng)
    assert len(idx) == 100 and idx.min() >= 0 and idx.max() < 100
    for blk in idx.reshape(10, 10):
        assert np.all(np.diff(blk) == 1)
    assert len(HE.block_indices(95, 10, rng)) == 95       # truncated to n


def test_recall_by_height_bins_and_threshold():
    # two GT people: 40 px tall (32-64 bin) and 150 px tall (128-256 bin); only the tall one is found above 0.5
    gt = make_coco({1: [[0, 0, 20, 40], [100, 100, 60, 150]]})
    dets = [{"image_id": 1, "category_id": 1, "score": 0.9, "bbox": [100, 100, 60, 150]},
            {"image_id": 1, "category_id": 1, "score": 0.3, "bbox": [0, 0, 20, 40]}]
    r = HE.recall_by_height(gt, dets, 0.5)
    assert r["32-64px"] == {"recall": 0.0, "n_gt": 1}
    assert r["128-256px"] == {"recall": 1.0, "n_gt": 1}
    assert HE.recall_by_height(gt, dets, 0.2)["32-64px"]["recall"] == 1.0


def test_crossing_events_match_by_time_and_direction():
    gt = [(1.0, "a_to_b"), (5.0, "b_to_a"), (9.0, "a_to_b")]
    pred = [(1.4, "a_to_b"), (5.2, "a_to_b"), (9.9, "a_to_b"), (20.0, "b_to_a")]
    m = HT.match_events(gt, pred, dt=1.0)
    assert m["tp"] == 2                      # 1.0<->1.4 and 9.0<->9.9; 5.0 has the wrong direction
    assert m["precision"] == pytest.approx(2 / 4) and m["recall"] == pytest.approx(2 / 3)


def test_crossing_match_is_one_to_one():
    m = HT.match_events([(1.0, "a_to_b"), (1.1, "a_to_b")], [(1.05, "a_to_b")], dt=1.0)
    assert m["tp"] == 1 and m["recall"] == 0.5


def test_segments_cover_every_frame_once():
    pts = np.arange(0, 95, 0.5)          # 95 s at 2 fps
    old = HT.SEG_S
    try:
        HT.SEG_S = 30.0
        segs = HT.segments(pts)
    finally:
        HT.SEG_S = old
    assert segs[0][0] == 0 and segs[-1][1] == len(pts)
    assert all(a[1] == b[0] for a, b in zip(segs, segs[1:]))
    assert len(segs) == 3                # the 5 s tail is merged into the last segment


def test_wilson_interval():
    lo, hi = HA.wilson(8, 10)
    assert lo == pytest.approx(0.490, abs=1e-3) and hi == pytest.approx(0.943, abs=1e-3)
    assert HA.wilson(0, 0) == [0.0, 0.0]
