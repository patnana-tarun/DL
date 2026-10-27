import numpy as np
import pytest

from edgevision.config import DEFAULTS
from edgevision.tracker import Tracker

BASE = dict(DEFAULTS["tracker"])


def person(i, x0=100, y0=100, vx=5, vy=0, w=50, h=120):
    return [x0 + vx * i, y0 + vy * i, x0 + vx * i + w, y0 + vy * i + h]


def run(frames, cfg=None, fps=30, ts=None):
    """frames: list of (boxes, scores). ts: optional list of timestamps (live mode)."""
    t = Tracker(dict(BASE, **(cfg or {})), fps)
    out = []
    for i, (b, c) in enumerate(frames):
        tb, ids = t.update(np.array(b, np.float32).reshape(-1, 4), np.array(c, np.float32),
                           None if ts is None else ts[i])
        assert tb.dtype == np.float32 and tb.shape == (len(ids), 4)
        out.append((tb, ids))
    return out


def all_ids(out):
    return {int(j) for _, ids in out for j in ids}


def test_config_is_not_mutated():
    cfg = dict(BASE)
    Tracker(cfg, 30)
    assert cfg == BASE


def test_single_person_keeps_one_id():
    out = run([([person(i)], [0.9]) for i in range(30)])
    assert len(all_ids(out)) == 1
    present = [len(ids) for _, ids in out]
    assert sum(present) >= 30 - BASE["minimum_consecutive_frames"]   # confirmed after a few frames
    assert present[-10:] == [1] * 10
    tb, _ = out[-1]
    np.testing.assert_allclose(tb[0], person(29), atol=6)            # box follows the person


def test_diagonal_motion_and_unit_ids():
    out = run([([person(i, vx=4, vy=3)], [0.8]) for i in range(30)])
    assert len(all_ids(out)) == 1
    assert all(ids.dtype.kind == "i" for _, ids in out)


def test_two_separated_people_get_two_ids():
    frames = [([person(i), person(i, x0=600, vx=-5)], [0.9, 0.85]) for i in range(30)]
    out = run(frames)
    assert len(all_ids(out)) == 2
    for tb, ids in out[5:]:
        assert len(ids) == 2
        left = ids[np.argmin(tb[:, 0])]
        assert left == out[-1][1][np.argmin(out[-1][0][:, 0])]       # identities never swap


def test_min_conf_filter_drops_low_score_detections():
    frames = [([person(i)], [0.9]) for i in range(10)] + [([person(i)], [0.15]) for i in range(10, 30)]
    with_filter = run(frames)                                        # min_conf 0.2
    without = run(frames, {"min_conf": 0.0})
    assert [len(ids) for _, ids in with_filter[12:]] == [0] * 18
    assert [len(ids) for _, ids in without[12:]] == [1] * 18           # ByteTrack's 2nd stage keeps the track
    assert len(all_ids(without)) == 1


def test_low_score_only_never_starts_a_track():
    out = run([([person(i)], [0.3]) for i in range(30)])
    assert all_ids(out) == set()


def test_nms_prefilter_merges_duplicates():
    frames = [([person(i), [v + 4 for v in person(i)]], [0.9, 0.85]) for i in range(30)]
    with_nms = run(frames)                                           # nms_iou 0.6
    without = run(frames, {"nms_iou": 0.0})
    assert len(all_ids(with_nms)) == 1
    assert all(len(ids) <= 1 for _, ids in with_nms)
    assert len(all_ids(without)) == 2                                # duplicates become a second track


def test_timestamps_live_mode():
    ts = [i / 30 for i in range(30)]
    out = run([([person(i)], [0.9]) for i in range(30)], ts=ts)
    assert len(all_ids(out)) == 1 and len(out[-1][1]) == 1


def test_timestamps_with_dropped_frames():
    """Live stream that delivers every 3rd frame (analyser slower than camera): timestamps keep one ID."""
    keep = list(range(0, 90, 3))
    out = run([([person(i)], [0.9]) for i in keep], ts=[i / 30 for i in keep])
    assert len(all_ids(out)) == 1
    assert [len(ids) for _, ids in out[-10:]] == [1] * 10


def test_lower_frame_rate_tracker():
    """Analysis at 10 fps (source.fps=10): a 15 px/frame mover still keeps its ID."""
    out = run([([person(i, vx=15)], [0.9]) for i in range(20)], fps=10)
    assert len(all_ids(out)) == 1


def test_empty_frames():
    out = run([([], []) for _ in range(5)])
    for tb, ids in out:
        assert tb.shape == (0, 4) and ids.shape == (0,)
    out = run([([person(i)], [0.9]) for i in range(10)] + [([], [])] * 3 + [([person(i)], [0.9]) for i in range(13, 25)])
    assert len(out[11][1]) == 0
    assert len(all_ids(out)) == 1                                    # re-associated after a short gap


@pytest.mark.parametrize("n", [1, 20])
def test_many_people(n):
    frames = [([person(i, x0=10 + 60 * k, y0=50 + (k % 3) * 150, vx=1) for k in range(n)], [0.9] * n)
              for i in range(15)]
    out = run(frames)
    assert len(all_ids(out)) == n and len(out[-1][1]) == n
