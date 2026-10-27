"""Compare multi-object trackers on the first 30 s (reuses track.py's cached detections).

For each detector x tracker: random search over the tracker's parameters on 0-15 s
(by HOTA), then report on the unseen 15-30 s. Every result is scored against two
pseudo-GTs built from the teacher detections, one linked by ByteTrack (as in track.py) and
one by OC-SORT, because a GT linked by tracker X favours tracker X. A ranking that holds
under both is not an artefact of how the GT was built.

Writes results/tracking/tracker_comparison.json.
"""
import json
import random
from multiprocessing import Pool

import cv2
import numpy as np
import supervision as sv
import trackers as T

import paths  # noqa: F401  (sets the working directory)
import track

N_TRIALS = {"SORT": 60, "ByteTrack": 60, "OC-SORT": 60, "C-BIoU": 60, "BoT-SORT": 24, "McByte": 24}
NEEDS_FRAME = {"BoT-SORT", "McByte"}      # camera-motion compensation reads the image
PRE = {"min_conf": [0.1, 0.2, 0.3, 0.4, 0.5], "nms_iou": [0.0, 0.6]}
BUF = [15, 30, 60]
SPACES = {
    "SORT": (T.SORTTracker, dict(track_activation_threshold=[0.3, 0.4, 0.5, 0.6, 0.7], lost_track_buffer=BUF,
                                 minimum_iou_threshold=[0.2, 0.3, 0.4, 0.5], minimum_consecutive_frames=[1, 2, 3, 5])),
    "ByteTrack": (T.ByteTrackTracker, dict(track_activation_threshold=[0.5, 0.6, 0.7, 0.8, 0.9],
                                           high_conf_det_threshold=[0.4, 0.5, 0.6], lost_track_buffer=BUF,
                                           minimum_iou_threshold=[0.2, 0.3, 0.4, 0.5, 0.6],
                                           minimum_consecutive_frames=[1, 2, 3])),
    "OC-SORT": (T.OCSORTTracker, dict(high_conf_det_threshold=[0.4, 0.5, 0.6, 0.7], lost_track_buffer=BUF,
                                      minimum_iou_threshold=[0.2, 0.3, 0.4, 0.5], minimum_consecutive_frames=[1, 2, 3],
                                      direction_consistency_weight=[0.1, 0.2, 0.4], delta_t=[1, 3, 5])),
    "C-BIoU": (T.CBIoUTracker, dict(track_activation_threshold=[0.5, 0.6, 0.7, 0.8], high_conf_det_threshold=[0.4, 0.5, 0.6],
                                    lost_track_buffer=BUF, minimum_consecutive_frames=[1, 2, 3],
                                    buffer_ratio_first=[0.2, 0.3, 0.5], buffer_ratio_second=[0.3, 0.5, 0.7])),
    "BoT-SORT": (T.BoTSORTTracker, dict(track_activation_threshold=[0.5, 0.6, 0.7, 0.8], high_conf_det_threshold=[0.4, 0.5, 0.6],
                                        lost_track_buffer=BUF, minimum_iou_threshold_first_assoc=[0.1, 0.2, 0.3],
                                        minimum_consecutive_frames=[1, 2, 3])),
    "McByte": (T.McByteTracker, dict(track_activation_threshold=[0.5, 0.6, 0.7, 0.8], high_conf_det_threshold=[0.4, 0.5, 0.6],
                                     lost_track_buffer=BUF, minimum_iou_threshold_first_assoc=[0.1, 0.2, 0.3],
                                     minimum_consecutive_frames=[1, 2, 3])),
}


def frames(n):
    cap = cv2.VideoCapture(track.VIDEO)
    for _ in range(n):
        ok, img = cap.read()
        if not ok:
            return
        yield img


def run(tracker_name, dets, params, fps):
    params = dict(params)
    min_conf, nms_iou = params.pop("min_conf", 0.0), params.pop("nms_iou", 0.0)
    t = SPACES[tracker_name][0](frame_rate=fps, **params)
    imgs = frames(len(dets)) if tracker_name in NEEDS_FRAME else iter(lambda: None, 0)
    res = []
    for a, img in zip(dets, imgs):
        a = a[a[:, 4] >= min_conf]
        d = sv.Detections(xyxy=a[:, :4].astype(np.float32), confidence=a[:, 4].astype(np.float32),
                          class_id=np.zeros(len(a), int))
        if nms_iou and len(d):
            d = d.with_nms(threshold=nms_iou, class_agnostic=True)
        o = t.update(d, frame=img) if img is not None else t.update(d)
        keep = o.tracker_id >= 0 if len(o) else np.zeros(0, bool)
        res.append((o.xyxy[keep], o.tracker_id[keep]))
    return res


def gt_from(tracker_name, params, fps, teacher):
    tr = run(tracker_name, [a[a[:, 4] >= 0.4] for a in teacher], params, fps)
    lengths = {}
    for _, ids in tr:
        for j in ids:
            lengths[j] = lengths.get(j, 0) + 1
    return [(b[[lengths[j] >= track.GT_MIN_LEN for j in ids]], ids[[lengths[j] >= track.GT_MIN_LEN for j in ids]])
            for b, ids in tr]


_S = {}


def job(args):
    det, trk, params = args
    fps, cache, gts, half = _S["fps"], _S["cache"], _S["gts"], _S["half"]
    pred = run(trk, cache[det][:half], params, fps)
    return det, trk, params, track.score(gts["ByteTrack-GT"], pred, 0, half)["HOTA"]


def main():
    fps, cache = track.load_cache()
    n = len(cache["teacher"]); half = n // 2
    gts = {"ByteTrack-GT": track.build_gt(fps, cache["teacher"]),
           "OC-SORT-GT": gt_from("OC-SORT", dict(high_conf_det_threshold=0.4, lost_track_buffer=120,
                                                 minimum_iou_threshold=0.2, minimum_consecutive_frames=3), fps,
                                 cache["teacher"])}
    rnd = random.Random(0)
    jobs = []
    for trk, (_, space) in SPACES.items():
        full = {**PRE, **space}
        seen = set()
        while len(seen) < N_TRIALS[trk]:
            seen.add(tuple(rnd.choice(v) for v in full.values()))
        jobs += [(det, trk, dict(zip(full, c))) for det in track.STUDENTS for c in seen]
    _S.update(fps=fps, cache=cache, gts=gts, half=half)
    with Pool(16) as pool:
        rows = []
        for i, r in enumerate(pool.imap_unordered(job, jobs, chunksize=2)):
            rows.append(r)
            if i % 100 == 0:
                print(f"search {i}/{len(jobs)}", flush=True)
    out = {}
    for det in track.STUDENTS:
        for trk in SPACES:
            best = max((r for r in rows if r[0] == det and r[1] == trk), key=lambda r: r[3])[2]
            pred = run(trk, cache[det], best, fps)
            out[f"{det} | {trk}"] = {"params": best, **{g: {"heldout_15-30s": track.score(gt, pred, half, n)}
                                                         for g, gt in gts.items()}}
            m = out[f"{det} | {trk}"]
            print(f"{det:13s} {trk:10s} HOTA {m['ByteTrack-GT']['heldout_15-30s']['HOTA']:.3f} "
                  f"(OC-SORT-GT {m['OC-SORT-GT']['heldout_15-30s']['HOTA']:.3f})  "
                  f"IDF1 {m['ByteTrack-GT']['heldout_15-30s']['IDF1']:.3f}  "
                  f"IDSW {m['ByteTrack-GT']['heldout_15-30s']['IDSW']}", flush=True)
    json.dump(out, open(track.OUT / "tracker_comparison.json", "w"), indent=2)


if __name__ == "__main__":
    main()
