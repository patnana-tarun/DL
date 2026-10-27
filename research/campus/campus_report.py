"""Stage 3: people counting + attribute analytics + annotated video.

1. Tracks shorter than MIN_S seconds or with low mean confidence are dropped (flicker / false positives).
2. Track stitching: a tracker loses a person behind someone else and restarts them with a new ID. We link
   track A -> B when B starts <= GAP_S after A ends, near where A was heading (constant-velocity
   extrapolation), at a similar box height, and without contradicting gender / dress votes.
   Linking is solved globally with the Hungarian algorithm. "Unique people" = stitched chains.
3. Attributes of a person = mean probe probability over all crops of all tracks in its chain.
4. Line counter: a line across the dominant walking direction; each person counted once per crossing.

    python research/campus/campus_report.py --name video_1 --video videos/video_1.mp4 [--no-video]
Writes results/campus/<name>/{summary.json, people.csv, dashboard.png, annotated.mp4}
"""
import argparse
import csv
import json
import subprocess
from collections import Counter, defaultdict
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths as P  # noqa: E402,F401  (project root as working directory)

import cv2
import matplotlib
import numpy as np
from scipy.optimize import linear_sum_assignment

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from visualize import GRID, INK, INK2, SURF  # noqa: E402,F401  (shared figure style)

MIN_S, MIN_CONF, GAP_S = 0.5, 0.35, 2.5
APP_MAX = 0.26     # max colour-histogram distance for a link; tuned on 38 hand-judged video_1 links
DRESS_BGR = {"uniform": (60, 170, 40), "non_uniform": (40, 90, 235), "unsure": (150, 150, 150)}
C_UNI, C_NON, C_UNK = "#2e9e4f", "#e0602b", "#a3a29d"
C_M, C_F = "#2a78d6", "#c2459a"
FIELDS = ("gender", "dress")


def load_tracks(out, fps):
    rows = np.load(out / "tracks.npz")["rows"]
    tr = {}
    for i in np.unique(rows[:, 1]).astype(int):
        r = rows[rows[:, 1] == i]
        tr[i] = r[np.argsort(r[:, 0])]
    keep = {i: r for i, r in tr.items() if len(r) >= MIN_S * fps and r[:, 6].mean() >= MIN_CONF}
    return rows, tr, keep


def appearance(out, ids):
    """Mean HSV histogram of the torso (18-50 % of the box) and legs (55-90 %), centre half of the width,
    over a track's saved crops. Everyone wears near-identical uniforms, so this only separates
    clearly different clothing; the attribute check does the rest."""
    crops = defaultdict(list)
    for p in (out / "crops").glob("*.jpg"):
        crops[int(p.stem.split("_")[0])].append(p)
    feats = {}
    for i in ids:
        fs = []
        for p in crops.get(i, []):
            hsv = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2HSV); h, w = hsv.shape[:2]
            f = [cv2.calcHist([hsv[int(a * h):int(b * h), w // 4:3 * w // 4]], [0, 1, 2], None, [12, 4, 4],
                              [0, 180, 0, 256, 0, 256]).ravel() for a, b in ((0.18, 0.5), (0.55, 0.9))]
            fs.append(np.concatenate([x / (x.sum() + 1e-9) for x in f]) / 2)
        if fs:
            feats[i] = np.mean(fs, 0)
    return feats


def stitch(keep, attrs, fps, feats):
    """Hungarian linking of track ends to track starts, plus static_links. Returns {track_id: person_id}.
    A link needs: plausible motion, similar size, similar clothing colours, no attribute conflict."""
    ids = sorted(keep)
    ends, starts = [], []
    for i in ids:
        r = keep[i]; k = min(len(r), int(fps / 2))
        c = np.c_[(r[:, 2] + r[:, 4]) / 2, r[:, 5]]           # foot point
        v = (c[-1] - c[-k]) / max(r[-1, 0] - r[-k, 0], 1)     # px / frame over the last 0.5 s
        ends.append((r[-1, 0], c[-1], v, r[-1, 5] - r[-1, 3]))
        starts.append((r[0, 0], c[0], r[0, 5] - r[0, 3]))
    BIG = 1e6
    cost = np.full((len(ids), len(ids)), BIG)
    for a, (fa, pa, va, ha) in enumerate(ends):
        for b, (fb, pb, hb) in enumerate(starts):
            dt = fb - fa
            if a == b or not (0 < dt <= GAP_S * fps):
                continue
            if not 0.6 < hb / ha < 1.6:
                continue
            A, B = attrs.get(ids[a], {}), attrs.get(ids[b], {})
            if any(A.get(f) not in (None, "unsure") and B.get(f) not in (None, "unsure") and A[f] != B[f]
                   for f in FIELDS):
                continue
            fa_, fb_ = feats.get(ids[a]), feats.get(ids[b])
            if fa_ is None or fb_ is None or 1 - np.sqrt(fa_ * fb_).sum() > APP_MAX:
                continue
            d = np.linalg.norm(pa + va * dt - pb) / max(ha, hb)    # distance in body heights
            if d > 0.8:
                continue
            cost[a, b] = d + dt / (GAP_S * fps)
    ra, rb = linear_sum_assignment(cost)
    links = [(ids[a], ids[b]) for a, b in zip(ra, rb) if cost[a, b] < BIG]
    static = static_links(keep, feats, fps)
    parent = {i: i for i in ids}                    # union-find over motion links + static links
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for a, b in links + static:
        parent[root(a)] = root(b)
    roots = sorted({root(i) for i in ids}, key=lambda r: min(keep[i][0, 0] for i in ids if root(i) == r))
    pid = {r: k + 1 for k, r in enumerate(roots)}
    return {i: pid[root(i)] for i in ids}, {"motion": len(links), "static": len(static)}


def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1])); i = ix * iy
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


def static_links(keep, feats, fps):
    """People who sit or stand still (a guard behind plants) flicker in and out of detection and get a
    new ID every few seconds for the whole video.
    Anchors: tracks clearly stationary for >= 5 s (foot point moves < 0.15 body heights, box height
    changes < 12 %, so distant walkers heading for the camera do not qualify).
    A track (anchor, or any short stationary track) joins an anchor when their boxes overlap
    (IoU > 0.6), it lies within 30 s of the anchor, the clothing matches, and where the two coexist
    their boxes coincide (a duplicate ID, not two people side by side)."""
    def still(r, k, hr):
        h = r[:, 5] - r[:, 3]; fx, fy = (r[:, 2] + r[:, 4]) / 2, r[:, 5]
        p10, p90 = np.percentile(h, [10, 90])
        return max(np.ptp(np.percentile(fx, [10, 90])), np.ptp(np.percentile(fy, [10, 90]))) < k * np.median(h) \
            and p90 / p10 < hr
    box = {i: np.median(r[:, 2:6], 0) for i, r in keep.items()}
    anchors = [i for i, r in keep.items() if len(r) >= 5 * fps and still(r, 0.15, 1.12)]
    cands = [i for i, r in keep.items() if still(r, 0.3, 1.15)]
    out = []
    for a in anchors:
        ra = keep[a]
        for b in cands:
            if b == a or (b in anchors and b < a):
                continue
            rb = keep[b]
            gap = max(rb[0, 0] - ra[-1, 0], ra[0, 0] - rb[-1, 0])
            if gap > 30 * fps or _iou(box[a], box[b]) <= 0.6:
                continue
            fa, fb = feats.get(a), feats.get(b)
            if fa is not None and fb is not None and 1 - np.sqrt(fa * fb).sum() > APP_MAX:
                continue
            common, ia, ib = np.intersect1d(ra[:, 0], rb[:, 0], return_indices=True)
            if len(common) and np.median([_iou(ra[j, 2:6], rb[k, 2:6]) for j, k in zip(ia, ib)]) < 0.7:
                continue
            out.append((a, b))
    return out


# probe outputs (campus_probe.py) carry P(positive) per crop; field: (positive, negative)
PROB_FIELDS = {"dress": ("uniform", "non_uniform"), "gender": ("female", "male")}
ABSTAIN = 0.15


def person_attributes(person, raw):
    """Attributes of each person from the crops of all its tracks: mean probe probability (campus_probe.py)."""
    by = defaultdict(list)
    for d in raw:
        if d["track"] in person:
            by[person[d["track"]]].append(d)
    res = {}
    for p, ans in by.items():
        r = {"n_crops": len(ans)}
        for f in FIELDS:
            pos, neg = PROB_FIELDS[f]; q = float(np.mean([a[f + "_p"] for a in ans]))
            r[f], r[f + "_conf"] = ("unsure" if abs(q - 0.5) < ABSTAIN else pos if q > 0.5 else neg), round(max(q, 1 - q), 3)
        res[p] = r
    return res


def crossings(keep, person, W, H):
    """Count line crossings across the dominant motion axis (a line through the image centre).
    Hysteresis: a person is on a side only once the foot point is > 0.3 body heights past the line,
    and a crossing is a change of side, so someone standing or sitting on the line never counts; people
    whose foot point never moves a body height along the axis (box jitter of a static person) are skipped."""
    dx = dy = 0.0
    for r in keep.values():
        dx += abs((r[-1, 2] + r[-1, 4]) / 2 - (r[0, 2] + r[0, 4]) / 2); dy += abs(r[-1, 5] - r[0, 5])
    horiz = dx >= dy
    pos = {}
    for i, r in keep.items():
        pos.setdefault(person[i], []).append(np.c_[r[:, 0], (r[:, 2] + r[:, 4]) / 2 if horiz else r[:, 5], r[:, 5] - r[:, 3]])
    line = W / 2 if horiz else 0.6 * H
    fwd = back = 0; events = []
    for p, segs in pos.items():
        s = np.concatenate(segs); s = s[np.argsort(s[:, 0])]
        if np.ptp(np.percentile(s[:, 1], [5, 95])) < np.median(s[:, 2]):
            continue                                  # never moved a body height along the axis: static
        d = s[:, 1] - line; m = 0.3 * s[:, 2]
        side = np.where(d > m, 1, np.where(d < -m, -1, 0)); nz = side != 0
        side, fr = side[nz], s[nz, 0]
        for k in np.nonzero(np.diff(side))[0]:
            if side[k + 1] > 0:
                fwd += 1
            else:
                back += 1
            events.append((int(fr[k + 1]), p, int(side[k + 1])))
    names = ("left→right", "right→left") if horiz else ("towards camera (down)", "away from camera (up)")
    return {"axis": "vertical line x=W/2" if horiz else "horizontal line y=0.6H", "line_px": line,
            "hysteresis": "0.3 body heights", names[0]: fwd, names[1]: back, "total": fwd + back}, horiz, line, events


def dashboard(out, name, fps, n, keep, person, pa, occ, cross):
    fig = plt.figure(figsize=(15, 9))
    g = fig.add_gridspec(2, 3, height_ratios=[1, 1.1], hspace=0.45, wspace=0.35)
    t = np.arange(n) / fps
    first = np.zeros(n)
    for p in set(person.values()):
        f0 = min(keep[i][0, 0] for i, q in person.items() if q == p); first[int(f0)] += 1
    ax = fig.add_subplot(g[0, :2])
    ax.plot(t, occ, color=C_M, lw=1, label="people in frame")
    ax2 = ax.twinx(); ax2.plot(t, np.cumsum(first), color=INK, lw=1.8, label="unique people so far"); ax2.grid(False)
    ax.set(xlabel="time (s)", ylabel="people in frame", title=f"{name}: occupancy and cumulative unique people")
    ax2.set_ylabel("unique people")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels(); ax.legend(h1 + h2, l1 + l2, loc="upper left", frameon=False)

    ax = fig.add_subplot(g[0, 2]); ax.axis("off")
    known = [r for r in pa.values()]
    lines = [("Unique people", len(set(person.values()))), ("Peak in frame", int(occ.max())),
             ("Mean in frame", f"{occ.mean():.1f}"), ("Line crossings", cross["total"]),
             ("With attributes", len(known))]
    for k, (a, b) in enumerate(lines):
        ax.text(0, 1 - k * 0.2, str(b), fontsize=22, weight="bold", color=INK, transform=ax.transAxes, va="top")
        ax.text(0, 1 - k * 0.2 - 0.1, a, fontsize=10, color=INK2, transform=ax.transAxes, va="top")

    def bars(ax, cnt, order, colors, title):
        vals = [cnt.get(o, 0) for o in order]
        ax.barh(range(len(order)), vals, color=colors); ax.set_yticks(range(len(order)), order); ax.invert_yaxis()
        for k, v in enumerate(vals):
            ax.text(v, k, f" {v} ({v / max(sum(vals), 1):.0%})", va="center", fontsize=9, color=INK2)
        ax.set_title(title); ax.set_xlim(0, max(vals + [1]) * 1.35); ax.grid(axis="y", visible=False)

    bars(fig.add_subplot(g[1, 0]), Counter(r["gender"] for r in known), ["male", "female", "unsure"], [C_M, C_F, C_UNK], "Gender")
    bars(fig.add_subplot(g[1, 1]), Counter(r["dress"] for r in known), ["uniform", "non_uniform", "unsure"], [C_UNI, C_NON, C_UNK], "Dress")
    ax = fig.add_subplot(g[1, 2])
    ct = Counter((r["gender"], r["dress"]) for r in known)
    gs, ds = ["male", "female"], ["uniform", "non_uniform"]
    left = np.zeros(2)
    for d, c in zip(ds, [C_UNI, C_NON]):
        v = np.array([ct[(x, d)] for x in gs]); ax.barh(gs, v, left=left, color=c, label=d); left += v
    ax.invert_yaxis(); ax.legend(frameon=False, fontsize=8); ax.set_title("Dress by gender"); ax.grid(axis="y", visible=False)
    fig.savefig(out / "dashboard.png", dpi=130, bbox_inches="tight", facecolor=SURF); plt.close(fig)


def render(video, out, rows, person, pa, horiz, line, events, fps):
    cap = cv2.VideoCapture(video); W, H = int(cap.get(3)), int(cap.get(4))
    ow, oh = 1280, 720; s = ow / W
    vw = cv2.VideoWriter(str(out / "annotated_raw.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (ow, oh))
    byf = defaultdict(list)
    for r in rows:
        if int(r[1]) in person:
            byf[int(r[0])].append(r)
    ev = sorted(events); k = 0; nf = nb = 0; seen = set()
    f = 0
    while True:
        ok, im = cap.read()
        if not ok:
            break
        im = cv2.resize(im, (ow, oh))
        while k < len(ev) and ev[k][0] <= f:
            nf += ev[k][2] > 0; nb += ev[k][2] < 0; k += 1
        if horiz:
            cv2.line(im, (int(line * s), 0), (int(line * s), oh), (0, 220, 255), 2)
        else:
            cv2.line(im, (0, int(line * s)), (ow, int(line * s)), (0, 220, 255), 2)
        for r in byf.get(f, []):
            p = person[int(r[1])]; seen.add(p); a = pa.get(p, {})
            col = DRESS_BGR.get(a.get("dress", "unsure"), DRESS_BGR["unsure"])
            x1, y1, x2, y2 = (r[2:6] * s).astype(int)
            cv2.rectangle(im, (x1, y1), (x2, y2), col, 2)
            g = {"male": "M", "female": "F"}.get(a.get("gender"), "?")
            d = {"uniform": "uniform", "non_uniform": "non-uniform"}.get(a.get("dress"), "?")
            txt = f"#{p} {g} {d}"
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(im, (x1, y1 - th - 6), (x1 + tw + 4, y1), col, -1)
            cv2.putText(im, txt, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        hud = [f"in frame: {len(byf.get(f, []))}", f"unique people: {len(seen)}",
               f"crossings  +{nf} / -{nb}", f"t = {f / fps:5.1f}s"]
        cv2.rectangle(im, (8, 8), (250, 18 + 22 * len(hud)), (20, 20, 20), -1)
        for j, h in enumerate(hud):
            cv2.putText(im, h, (16, 30 + 22 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        vw.write(im); f += 1
    vw.release()
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(out / "annotated_raw.mp4"),
                    "-c:v", "libx264", "-crf", "26", "-preset", "fast", "-pix_fmt", "yuv420p", str(out / "annotated.mp4")], check=True)
    (out / "annotated_raw.mp4").unlink()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True); ap.add_argument("--video", required=True)
    ap.add_argument("--no-video", action="store_true")
    a = ap.parse_args()
    out = Path("results/campus") / a.name
    z = np.load(out / "tracks.npz"); fps, W, H = float(z["fps"]), int(z["W"]), int(z["H"])
    n = int(z["rows"][:, 0].max()) + 1          # container frame count is approximate (VFR phone video)
    rows, tr, keep = load_tracks(out, fps)
    raw = [json.loads(line) for line in open(out / "attributes_raw.jsonl")] if (out / "attributes_raw.jsonl").exists() else []
    tattr = {int(k): v for k, v in json.load(open(out / "attributes.json")).items()} if (out / "attributes.json").exists() else {}
    person, links = stitch(keep, tattr, fps, appearance(out, keep))
    pa = person_attributes(person, raw)
    occ = np.zeros(n)
    for i, r in keep.items():
        occ[r[:, 0].astype(int)] += 1
    cross, horiz, line, events = crossings(keep, person, W, H)

    people = sorted(set(person.values()))
    known = [pa[p] for p in people if p in pa]
    dur = {p: sum(len(keep[i]) for i, q in person.items() if q == p) / fps for p in people}
    cnt = lambda f: dict(Counter(r[f] for r in known))  # noqa: E731
    summary = {
        "video": a.video, "fps": fps, "frames": n, "seconds": round(n / fps, 1),
        "detector": "YOLO26x @1280", "tracker": "BoT-SORT + ReID + GMC, then offline stitching",
        "attribute_model": "linear probes on RF-DETR Large (DINOv2) backbone features + colour histograms, "
                           "trained on 200 hand-labelled tracks; per person = mean probability over all its crops "
                           "(see results/campus/validation/probe_scores.json for held-out accuracy)",
        "raw_track_ids": len(tr), "tracks_kept": len(keep), "stitch_links": links,
        "unique_people": len(people), "people_with_attributes": len(known),
        "people_in_frame_mean": round(float(occ.mean()), 2), "people_in_frame_max": int(occ.max()),
        "time_in_view_s_median": round(float(np.median(list(dur.values()))), 2),
        "line_crossings": cross,
        "gender": cnt("gender"), "dress": cnt("dress"),
        "dress_by_gender": {f"{g}/{d}": k for (g, d), k in Counter((r["gender"], r["dress"]) for r in known).items()},
    }
    json.dump(summary, open(out / "summary.json", "w"), indent=1)
    with open(out / "people.csv", "w", newline="") as fo:
        w = csv.writer(fo)
        cols = [c for f in FIELDS for c in (f, f + "_conf")]
        w.writerow(["person", "track_ids", "first_s", "last_s", "time_in_view_s", "n_crops"] + cols)
        for p in people:
            ts = [i for i, q in person.items() if q == p]
            f0 = min(keep[i][0, 0] for i in ts) / fps; f1 = max(keep[i][-1, 0] for i in ts) / fps
            r = pa.get(p, {})
            w.writerow([p, " ".join(map(str, ts)), round(f0, 2), round(f1, 2), round(dur[p], 2), r.get("n_crops", 0)]
                       + [r.get(c, "") for c in cols])
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    dashboard(out, a.name, fps, n, keep, person, pa, occ, cross)
    if not a.no_video:
        render(a.video, out, rows, person, pa, horiz, line, events, fps)


if __name__ == "__main__":
    main()
