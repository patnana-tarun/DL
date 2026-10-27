"""Held-out test video, stage 4: the deployed edge runtime (edge_deploy/edgevision) end to end, test only.

Same protocol as research/edge_eval.py `files`, on the whole held-out video: FFmpeg decoder process (scaled to
the default 1280x720, as on the device) -> ONNX Runtime detector -> ByteTrack (aa1-tuned defaults) -> online
analytics with the counting line of heldout_track.LINE. Tracks are rescaled to the source resolution and scored
against the teacher pseudo-GT (heldout_track.py), subsampled at the same frames for stride s.

    python research/heldout/heldout_edge.py --name test1 --video videos/test1.mp4
Writes results/<name>/edge.json (+ per-run reports and track dumps in results/<name>/edge_runs/).
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths as P  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import edge_eval  # noqa: E402
import track  # noqa: E402
from heldout_detect import load_full  # noqa: E402
from heldout_track import LINE  # noqa: E402

RUNS = edge_eval.FILE_RUNS    # same five configurations as the aa1 edge table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True); ap.add_argument("--video", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = Path(a.out or f"results/{a.name}"); edge_eval.RUNS = out / "edge_runs"; edge_eval.RUNS.mkdir(parents=True, exist_ok=True)
    edge_eval.BASE = [o for o in edge_eval.BASE if not o.startswith(("source.uri", "source.duration_s"))] + \
        [f"source.uri={(P.ROOT / a.video).as_posix()}", f"analytics.line={json.dumps(LINE)}"]
    pts, dets = load_full(out / "detections_full.npz")
    fps = (len(pts) - 1) / (pts[-1] - pts[0])
    gt_full = track.build_gt(fps, dets["teacher"])
    analytics = json.load(open(out / "analytics.json"))
    cap = cv2.VideoCapture(a.video); src_w, src_h = cap.get(3), cap.get(4); cap.release()
    res = {}
    for name, (onnx, stride) in RUNS.items():
        t = time.time()
        p, rep = edge_eval.run_pipeline(name, [f"detector.weights={onnx}", f"source.stride={stride}"])
        W, H = p.source.W, p.source.H
        sx, sy = src_w / W, src_h / H
        pred = [(tb * [sx, sy, sx, sy], ids) for _, _, tb, ids, _, _ in p.record]
        gt = gt_full[::stride]
        n = min(len(pred), len(gt))
        m = track.score(gt[:n], pred[:n], 0, n)
        min_frames = max(1, round(track.GT_MIN_LEN / stride))
        res[name] = {"onnx": onnx, "stride": stride, "analysis_fps": round(fps / stride, 2), "frames": n,
                     "frames_gt": len(gt), "full_video": m,
                     "unique_people": edge_eval.unique_ids(pred[:n], min_frames),
                     "gt_unique_people": edge_eval.unique_ids(gt[:n], min_frames),
                     "pipeline_analytics": {k: rep["analytics"][k] for k in ("unique_people", "crossings")},
                     "gt_analytics": {"unique_people": analytics["GT"]["unique_people"], "crossings": analytics["GT"]["crossings"]},
                     "latency_ms": rep["latency_ms"], "provider": rep["detector"]["provider"],
                     "decode_size": [W, H], "wall_s": round(time.time() - t, 1)}
        print(f"{name}: frames {n}/{len(gt)}  HOTA {m['HOTA']:.3f} MOTA {m['MOTA']:.3f} IDF1 {m['IDF1']:.3f}  "
              f"unique {res[name]['unique_people']} (GT {res[name]['gt_unique_people']})  crossings "
              f"{rep['analytics']['crossings']} (GT {analytics['GT']['crossings']})  "
              f"detect p50 {rep['latency_ms']['detect_ms']['p50']:.1f} ms", flush=True)
        np.savez_compressed(edge_eval.RUNS / f"{name}_tracks.npz", **{f"{i}": np.c_[b, ids] for i, (b, ids) in enumerate(pred)})
    ref = json.load(open(out / "tracking.json"))["YOLO26n"]["aa1_tuned"]["full"]
    res["_reference_research_offline"] = {"what": "heldout_track.py, PyTorch FP16 fast path, cv2 decode at 1920x1080, stride 1",
                                         "full_video": ref}
    res["protocol"] = __doc__.split("\n\n")[1].strip()
    json.dump(res, open(out / "edge.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
