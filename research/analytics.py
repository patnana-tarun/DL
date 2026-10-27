"""Crowd analytics from the tracked people (first 30 s, fine-tuned detector + tuned tracker).

Everything is computed from track output (box + ID per frame), so it works for any video
that track.py has processed:
  occupancy      people in frame over time, and unique track IDs seen so far
                 (same >= 0.25 s length filter as the pseudo-GT, so model and GT counts are comparable;
                 an ID is an upper bound on people: a broken-and-restarted track counts twice)
  dwell          how long each person stays in view
  heatmap        where people stand/walk (foot point = bottom-centre of the box)
  flow           mean walking direction and speed per image cell (px/s, image plane)
  line crossing  people crossing a virtual counting line, split by direction
Flow, speed and crossings are in image coordinates: with a handheld/panning camera they include
camera motion, so treat them as indicative, not as world-frame measurements.

    python research/analytics.py [--model YOLO26n] [--out results/tracking] [--video aa1.mp4] [--line 0.5]
Writes <out>/analytics_<model>.json and results/figures/12_crowd_analytics_<model>.png.
"""
import argparse
import json
from pathlib import Path

import cv2
import matplotlib
import numpy as np

import paths  # noqa: F401  (sets the working directory)
import track

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from visualize import COLOR, INK2, SURF, save  # noqa: E402  (shared figure style)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="YOLO26n", choices=track.STUDENTS)
    ap.add_argument("--out", default=str(track.OUT)); ap.add_argument("--video", default=track.VIDEO)
    ap.add_argument("--line", type=float, default=0.5, help="vertical counting line at this fraction of width")
    a = ap.parse_args()
    track.OUT, track.CACHE, track.VIDEO = Path(a.out), Path(a.out) / "detections.npz", a.video
    fps, cache = track.load_cache()
    params = json.load(open(track.OUT / "tracking_metrics.json"))[a.model]["best_params"]
    tracks = track.run_tracker(cache[a.model], params, fps)
    cap = cv2.VideoCapture(a.video); ok, frame = cap.read(); H, W = frame.shape[:2]

    # per-ID foot-point trajectories
    traj = {}
    for i, (b, ids) in enumerate(tracks):
        for (x1, y1, x2, y2), j in zip(b, ids):
            traj.setdefault(int(j), []).append((i, (x1 + x2) / 2, y2))
    raw_ids = len(traj)
    traj = {j: np.array(v) for j, v in traj.items() if len(v) >= track.GT_MIN_LEN}   # same filter as pseudo-GT
    gt_ids = len({int(j) for _, ids in track.build_gt(fps, cache["teacher"]) for j in ids})
    t = np.arange(len(tracks)) / fps
    live = np.array([len(ids) for _, ids in tracks])
    first = sorted(v[0, 0] for v in traj.values())
    unique = np.searchsorted(first, np.arange(len(tracks)), side="right")
    dwell = np.array([(v[-1, 0] - v[0, 0] + 1) / fps for v in traj.values()])

    # line crossings (foot point crossing x = line * W)
    lx = a.line * W; l2r = r2l = 0
    for v in traj.values():
        s = np.sign(v[:, 1] - lx); s = s[s != 0]
        ch = np.diff(s)
        l2r += int((ch > 0).sum()); r2l += int((ch < 0).sum())

    # heatmap + flow field on a coarse grid
    gx, gy = 32, 18
    heat = np.zeros((gy, gx)); vel = np.zeros((gy, gx, 2)); cnt = np.zeros((gy, gx))
    for v in traj.values():
        cx = np.clip((v[:, 1] / W * gx).astype(int), 0, gx - 1); cy = np.clip((v[:, 2] / H * gy).astype(int), 0, gy - 1)
        np.add.at(heat, (cy, cx), 1)
        k = int(fps / 2)   # velocity over 0.5 s windows, smoother than frame-to-frame jitter
        if len(v) > k:
            d = (v[k:, 1:] - v[:-k, 1:]) / ((v[k:, :1] - v[:-k, :1]) / fps)
            np.add.at(vel, (cy[:-k], cx[:-k]), d); np.add.at(cnt, (cy[:-k], cx[:-k]), 1)
    vel /= np.maximum(cnt, 1)[..., None]
    speed = np.linalg.norm(vel, axis=2)[cnt > 0]

    res = {"model": a.model, "seconds": float(t[-1]), "people_in_frame_mean": float(live.mean()),
           "people_in_frame_max": int(live.max()), "unique_track_ids": len(traj),
           "gt_unique_track_ids": gt_ids, "raw_track_ids_unfiltered": raw_ids,
           "min_track_frames": track.GT_MIN_LEN,
           "dwell_s_median": float(np.median(dwell)), "dwell_s_p90": float(np.percentile(dwell, 90)),
           "line_x_px": lx, "crossings_left_to_right": l2r, "crossings_right_to_left": r2l,
           "mean_image_speed_px_s": float(speed.mean()) if len(speed) else 0.0}
    tag = a.model.replace(" ", "_")
    json.dump(res, open(track.OUT / f"analytics_{tag}.json", "w"), indent=2)
    print(json.dumps(res, indent=1))

    c = COLOR["YOLO26n" if a.model.startswith("YOLO") else "RF-DETR Nano"]
    fig = plt.figure(figsize=(14, 7.5))
    g = fig.add_gridspec(2, 3, height_ratios=[1, 1.35])
    ax = fig.add_subplot(g[0, :2])
    ax.plot(t, live, color=c, label="people in frame"); ax.plot(t, unique, color=INK2, ls="--", label="unique track IDs so far")
    ax.set(xlabel="time (s)", ylabel="people", title=f"Occupancy ({a.model} + tuned ByteTrack)"); ax.legend()
    ax = fig.add_subplot(g[0, 2])
    ax.hist(dwell, bins=20, color=c, edgecolor=SURF); ax.set(xlabel="seconds in view", ylabel="people",
                                                             title=f"Dwell time (median {np.median(dwell):.1f} s)")
    ax = fig.add_subplot(g[1, :2])
    hm = cv2.GaussianBlur(cv2.resize(heat, (W, H), interpolation=cv2.INTER_LINEAR), (0, 0), 12)
    ax.imshow(frame[:, :, ::-1])
    ax.imshow(np.ma.masked_less(hm, 0.05 * hm.max()), cmap="magma", alpha=0.6, extent=(0, W, H, 0))
    yy, xx = np.nonzero(cnt > 3)
    ax.quiver((xx + 0.5) * W / gx, (yy + 0.5) * H / gy, vel[yy, xx, 0], -vel[yy, xx, 1], color="white", scale=2500,
              width=0.002)
    ax.axvline(lx, color="white", lw=1.5, ls="--")
    ax.text(lx + 8, 30, f"→ {l2r}   ← {r2l}", color="white", fontsize=10, fontweight="bold")
    ax.axis("off"); ax.set_title("Foot-traffic heatmap, flow, counting line (image plane; frame 0 as backdrop)")
    ax = fig.add_subplot(g[1, 2]); ax.axis("off")
    rows = [("mean / max in frame", f"{res['people_in_frame_mean']:.1f} / {res['people_in_frame_max']}"),
            ("unique track IDs (GT)", f"{res['unique_track_ids']} ({gt_ids})"), ("median / p90 dwell", f"{res['dwell_s_median']:.1f} / {res['dwell_s_p90']:.1f} s"),
            ("line crossings →/←", f"{l2r} / {r2l}"), ("mean image speed", f"{res['mean_image_speed_px_s']:.0f} px/s")]
    for k, (lab, val) in enumerate(rows):
        ax.text(0, 0.9 - k * 0.17, lab, fontsize=10, color=INK2, transform=ax.transAxes)
        ax.text(0, 0.9 - k * 0.17 - 0.07, val, fontsize=16, fontweight="bold", transform=ax.transAxes)
    save(fig, f"12_crowd_analytics_{tag}.png")


if __name__ == "__main__":
    main()
