"""Held-out test video, stage 3: multi-object tracking and crowd analytics, test only.

Input: <out>/detections_full.npz from heldout_detect.py (teacher + student boxes for every frame).
Every tracker parameter is FROZEN from the aa1 study; nothing is searched on this video.
  pseudo-GT   teacher boxes (fused score >= 0.40) linked by ByteTrack with the aa1 GT_PARAMS (track.py), tracks
              shorter than 15 frames dropped; a second GT linked by OC-SORT (track_compare.py), because a GT
              linked by tracker X favours tracker X.
  bytetrack   both students with the aa1-tuned and the ByteTrack default parameters (track.py report); with the
              `pretrained` step also the COCO-pretrained YOLO26n (the best held-out detector), run on every frame and
              tracked with the YOLO26n aa1-tuned and the default parameters (it was never tuned for tracking)
  trackers    6 trackers x 2 students with the aa1-selected parameters (results/tracking/tracker_comparison.json)
  scoring     HOTA / MOTA / IDF1 / ... on the whole video (track.score), and per contiguous 30 s segment for a
              paired YOLO26n-vs-RF-DETR comparison (Wilcoxon signed-rank over segments)
  analytics   edgevision.analytics.Analytics (the deployed definitions) on GT and model tracks with frame PTS:
              occupancy, unique people, dwell, line crossings on a counting line fixed BEFORE looking at any
              result (LINE), crossing events matched to GT events (same direction, |dt| <= 1 s) -> P / R / F1

    python research/heldout/heldout_track.py --name test1 --video videos/test1.mp4 [steps: pretrained bytetrack trackers analytics]
Writes results/<name>/tracking.json, tracker_comparison.json, analytics.json, tracks.npz.
"""
import argparse
import copy
import json
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths as P  # noqa: E402  (project root as working directory)

import numpy as np  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

import track  # noqa: E402
import track_compare as TC  # noqa: E402
from heldout_detect import load_full, unpack  # noqa: E402

sys.path.insert(0, str(P.DEPLOY))
from edgevision import config as ecfg  # noqa: E402
from edgevision.analytics import Analytics  # noqa: E402

STUDENTS = track.STUDENTS
PRETRAINED = "YOLO26n (pretrained)"
SEG_S = 30.0
# Counting line, normalised (x, y). Chosen from one frame before any result was computed: a horizontal segment
# across the left footpath and the road at 75% of the height, where people walk towards / away from the camera;
# it stops short of the right footpath, where people walk parallel to it.
LINE = [[0.25, 0.75], [0.80, 0.75]]
MATCH_DT = 1.0
DEFAULT_BT = dict(track_activation_threshold=0.7, high_conf_det_threshold=0.6, lost_track_buffer=30,
                  minimum_iou_threshold=0.1, minimum_consecutive_frames=2)
OCSORT_GT = dict(high_conf_det_threshold=0.4, lost_track_buffer=120, minimum_iou_threshold=0.2,
                 minimum_consecutive_frames=3)

_S = {}


def segments(pts):
    """Frame index bounds of contiguous SEG_S-second segments (the last one absorbs a short tail)."""
    edges = list(np.searchsorted(pts, np.arange(0, pts[-1], SEG_S)))
    if pts[-1] - pts[edges[-1]] < SEG_S / 2 and len(edges) > 1:
        edges.pop()
    return list(zip(edges, edges[1:] + [len(pts)]))


def score_all(gt, pred, segs):
    return {"full": track.score(gt, pred, 0, len(gt)),
            "segments": [track.score(gt, pred, lo, hi) for lo, hi in segs]}


def _tc_job(args):
    det, trk, params = args
    pred = TC.run(trk, _S["dets"][det], params, _S["fps"])
    return det, trk, {g: score_all(gt, pred, _S["segs"]) for g, gt in _S["gts"].items()}


def detect_pretrained(video, path):
    """COCO-pretrained YOLO26n on every frame (deployed fast path, conf >= 0.05, as the students)."""
    import cv2
    import detectors as D
    from heldout_detect import STUDENT_CONF, pack
    run = D.yolo(P.YOLO_PRETRAINED, conf=STUDENT_CONF)
    cap, out = cv2.VideoCapture(video), []
    while True:
        ok, img = cap.read()
        if not ok:
            break
        b, sc = run(img); out.append(np.c_[b, sc].reshape(-1, 5))
    data, off = pack(out)
    np.savez_compressed(path, data=data, off=off)
    print(f"pretrained YOLO26n: {len(out)} frames", flush=True)


def run_analytics(tracks, pts, W, H, fps):
    cfg = copy.deepcopy(ecfg.DEFAULTS["analytics"]); cfg["line"] = LINE
    ev = []
    a = Analytics(cfg, [], W, H, fps, on_event=ev.append)
    occ = []
    for (b, ids), ts in zip(tracks, pts):
        a.update(float(ts), b, ids); occ.append(len(ids))
    a.update(float(pts[-1]) + 10, np.zeros((0, 4)), np.zeros(0, int))   # flush: finish every open track
    cross = [(e["ts"], e["direction"]) for e in ev if e["kind"] == "line_cross"]
    dwell = np.array(list(a.finished_dwell))
    return {"occupancy": np.array(occ), "unique_people": a.unique, "crossings": dict(a.cross),
            "crossing_events": cross, "dwell_s_median": float(np.median(dwell)) if len(dwell) else 0.0,
            "dwell_s_p90": float(np.percentile(dwell, 90)) if len(dwell) else 0.0,
            "direction_labels": dict(a.direction_labels)}


def match_events(gt_ev, pr_ev, dt=MATCH_DT):
    """Greedy one-to-one matching of crossing events by time, same direction."""
    used = [False] * len(pr_ev); tp = 0
    for tg, dg in gt_ev:
        best, bj = dt + 1e-9, -1
        for j, (tp_, dp) in enumerate(pr_ev):
            if not used[j] and dp == dg and abs(tp_ - tg) <= best:
                best, bj = abs(tp_ - tg), j
        if bj >= 0:
            used[bj] = True; tp += 1
    P_ = tp / max(len(pr_ev), 1); R = tp / max(len(gt_ev), 1)
    return {"tp": tp, "gt_events": len(gt_ev), "pred_events": len(pr_ev), "precision": P_, "recall": R,
            "F1": 2 * P_ * R / max(P_ + R, 1e-9)}


def main():
    global SEG_S
    ap = argparse.ArgumentParser()
    ap.add_argument("steps", nargs="*", default=["pretrained", "bytetrack", "trackers", "analytics"])
    ap.add_argument("--name", required=True); ap.add_argument("--video", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seg", type=float, default=30.0, help="segment length (s) for the paired comparison")
    ap.add_argument("--line", default=None, help="counting line x1,y1,x2,y2 (normalised); default: the test1 line")
    a = ap.parse_args()
    SEG_S = a.seg
    if a.line:                       # fixed per video before any result is computed
        global LINE
        x1, y1, x2, y2 = map(float, a.line.split(","))
        LINE[:] = [[x1, y1], [x2, y2]]
    out = Path(a.out or f"results/{a.name}")
    pts, dets = load_full(out / "detections_full.npz")
    fps = (len(pts) - 1) / (pts[-1] - pts[0])
    pre_path = out / "detections_yolo26n_pretrained.npz"
    if "pretrained" in a.steps:
        detect_pretrained(a.video, pre_path)
    models = list(STUDENTS)
    if pre_path.exists():
        z = np.load(pre_path); dets[PRETRAINED] = unpack(z["data"], z["off"]); models.append(PRETRAINED)
    track.VIDEO = a.video          # BoT-SORT / McByte read the frames for camera-motion compensation
    segs = segments(pts)
    gt = track.build_gt(fps, dets["teacher"])
    gts = {"ByteTrack-GT": gt, "OC-SORT-GT": TC.gt_from("OC-SORT", OCSORT_GT, fps, dets["teacher"])}
    info = {"frames": len(pts), "duration_s": float(pts[-1]), "mean_fps": fps, "segments": len(segs),
            "segment_s": SEG_S, "gt_tracks": {g: len({int(j) for _, ids in v for j in ids}) for g, v in gts.items()}}
    print(info, flush=True)

    if "bytetrack" in a.steps:
        aa1 = json.load(open("results/tracking/tracking_metrics.json"))
        res = {"info": info}
        tracks = {"GT": gt}
        for s in models:
            src = "YOLO26n" if s == PRETRAINED else s     # the pretrained model borrows the YOLO26n aa1 parameters
            res[s] = {"params_tuned_on_aa1": aa1[src]["best_params"]}
            for tag, params in (("default", DEFAULT_BT), ("aa1_tuned", aa1[src]["best_params"])):
                pred = track.run_tracker(dets[s], params, fps)
                res[s][tag] = score_all(gt, pred, segs)
                if tag == "aa1_tuned":
                    tracks[s] = pred
                m = res[s][tag]["full"]
                print(f"{s:13s} {tag:9s} HOTA {m['HOTA']:.3f} MOTA {m['MOTA']:.3f} IDF1 {m['IDF1']:.3f} "
                      f"IDSW {m['IDSW']} IDs {m['pred_IDs']} (GT {m['gt_IDs']})", flush=True)
            if s in aa1:
                res[s]["aa1_heldout_15-30s"] = aa1[s]["tuned"]["heldout_15-30s"]
        paired = {}
        pairs = [("YOLO26n", "RF-DETR Nano")] + ([(PRETRAINED, "RF-DETR Nano"), (PRETRAINED, "YOLO26n")]
                                                 if PRETRAINED in models else [])
        for x_name, y_name in pairs:
            for tag in ("default", "aa1_tuned"):
                for k in ("HOTA", "MOTA", "IDF1"):
                    x = np.array([m[k] for m in res[x_name][tag]["segments"]])
                    y = np.array([m[k] for m in res[y_name][tag]["segments"]])
                    key = f"{tag} {k}" if (x_name, y_name) == pairs[0] else f"{x_name} vs {y_name} | {tag} {k}"
                    paired[key] = {"mean_diff_first_minus_second": float((x - y).mean()), "first_wins": int((x > y).sum()),
                                   "segments": len(x), "wilcoxon_p": float(wilcoxon(x, y).pvalue)}
        res["paired_segments"] = paired
        print(json.dumps(paired, indent=1), flush=True)
        json.dump(res, open(out / "tracking.json", "w"), indent=2)
        np.savez_compressed(out / "tracks.npz", pts=pts, **{f"{k}/{i}": np.c_[b, ids] for k, v in tracks.items()
                                                            for i, (b, ids) in enumerate(v)})

    if "trackers" in a.steps:
        sel = json.load(open("results/tracking/tracker_comparison.json"))
        jobs = [(det, trk, sel[f"{det} | {trk}"]["params"]) for det in STUDENTS for trk in TC.SPACES]
        _S.update(dets=dets, fps=fps, segs=segs, gts=gts)
        res = {"info": info}
        with Pool(min(12, len(jobs))) as pool:
            for det, trk, m in pool.imap_unordered(_tc_job, jobs):
                res[f"{det} | {trk}"] = {"params_from_aa1": sel[f"{det} | {trk}"]["params"], **m}
                print(f"{det:13s} {trk:10s} HOTA {m['ByteTrack-GT']['full']['HOTA']:.3f} "
                      f"(OC-SORT-GT {m['OC-SORT-GT']['full']['HOTA']:.3f}) IDF1 {m['ByteTrack-GT']['full']['IDF1']:.3f}",
                      flush=True)
        json.dump(res, open(out / "tracker_comparison.json", "w"), indent=2)

    if "analytics" in a.steps:
        z = np.load(out / "tracks.npz")
        import cv2
        cap = cv2.VideoCapture(a.video); W, H = int(cap.get(3)), int(cap.get(4))
        tr = {k: [z[f"{k}/{i}"] for i in range(len(pts))] for k in ("GT", *models) if f"{k}/0" in z}
        res = {"line_norm": LINE, "match_dt_s": MATCH_DT}
        A = {k: run_analytics([(v[:, :4], v[:, 4].astype(int)) for v in tr[k]], pts, W, H, fps) for k in tr}
        for k, r in A.items():
            occ = r.pop("occupancy"); ev = r.pop("crossing_events")
            r.update(occupancy_mean=float(occ.mean()), occupancy_max=int(occ.max()), crossing_events_n=len(ev))
            if k != "GT":
                g = A["GT"]
                err = occ - occ_gt
                r["occupancy_MAE"] = float(np.abs(err).mean()); r["occupancy_bias"] = float(err.mean())
                r["unique_people_rel_err"] = (r["unique_people"] - g["unique_people"]) / g["unique_people"]
                r["crossing_match"] = match_events(ev_gt, ev)
                r["crossing_count_rel_err"] = (len(ev) - len(ev_gt)) / max(len(ev_gt), 1)
            else:
                occ_gt, ev_gt = occ, ev
                r["crossing_events"] = ev
            res[k] = r
        res["occupancy_series"] = {"t": pts[::30].round(3).tolist(),
                                   **{k: [len(v) for v in tr[k]][::30] for k in tr}}
        json.dump(res, open(out / "analytics.json", "w"), indent=2, default=float)
        print(json.dumps({k: v for k, v in res.items() if k not in ("occupancy_series",)}, indent=1, default=float))


if __name__ == "__main__":
    main()
