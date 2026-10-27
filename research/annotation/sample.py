"""Choose the frames humans will annotate (fixed seed, decided before anyone annotates).

det_main   detection frames from the TEST portion of every scene (never training frames):
           aa1 test 255-300 s, campus scene 1 / 2 test splits, held-out test1, stratified per scene by
           people-count tercile x mean brightness half, at least 8 s apart within a scene where possible
track_*    consecutive tracking clips at 5 fps (aa1 15-30 s, the tracking evaluation window; a 30 s test1 segment)
The pseudo-labels only decide which frames are picked (density strata); they are never shown to annotators.

    python research/annotation/sample.py [--per-scene 60] [--clip-s 30]
-> research/annotation/tasks/<task>/{manifest.json, images/}
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths as P  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402

TASKS = Path("research/annotation/tasks")
SCENES = {"aa1": "datasets/coco/test", "scene_v1": "datasets/scene_v1/splits/test",
          "scene_v2": "datasets/scene_v2/splits/test", "test1": "datasets/test1/coco/test"}


def det_task(per_scene, seed=0):
    rng = np.random.default_rng(seed)
    out = TASKS / "det_main"; (out / "images").mkdir(parents=True, exist_ok=True)
    items = []
    for scene, d in SCENES.items():
        f = Path(d) / "_annotations.coco.json"
        if not f.exists():
            print("skip (not built yet):", scene); continue
        coco = json.load(open(f))
        n = np.bincount([a["image_id"] for a in coco["annotations"]], minlength=len(coco["images"]) + 1)
        rows = []
        for im in coco["images"]:
            img = cv2.imread(str(Path(d) / im["file_name"]), cv2.IMREAD_GRAYSCALE)
            rows.append((im, int(n[im["id"]]), float(img.mean()), im.get("t", im.get("frame", 0) / 59.94)))
        cnt = np.array([r[1] for r in rows]); br = np.array([r[2] for r in rows])
        terc = np.digitize(cnt, np.percentile(cnt, [33.3, 66.7])); bright = (br > np.median(br)).astype(int)
        strata = {}
        for i, r in enumerate(rows):
            strata.setdefault((int(terc[i]), int(bright[i])), []).append(i)
        k_each = int(np.ceil(per_scene / max(len(strata), 1)))
        chosen = []
        for key, idx in sorted(strata.items()):
            idx = list(rng.permutation(idx)); pick = []
            for i in idx:                                  # spread in time: >= 8 s from frames already picked
                if all(abs(rows[i][3] - rows[j][3]) >= 8 for j in pick + chosen):
                    pick.append(i)
                if len(pick) == k_each:
                    break
            for i in idx:                                  # fill up if the spacing rule left the stratum short
                if len(pick) == k_each:
                    break
                if i not in pick:
                    pick.append(i)
            chosen += pick
        for i in sorted(chosen[:per_scene], key=lambda j: rows[j][3]):
            im, c, b, t = rows[i]
            name = f"{scene}__{im['file_name']}"
            shutil.copy(Path(d) / im["file_name"], out / "images" / name)
            items.append({"file": name, "scene": scene, "source": str(Path(d) / im["file_name"]),
                          "t": round(t, 3), "stratum": {"density_tercile": int(terc[i]), "bright": int(bright[i])}})
    json.dump({"task": "det_main", "kind": "det", "items": items}, open(out / "manifest.json", "w"), indent=1)
    print("det_main:", len(items), "frames")


def clip_task(name, video, start_s, dur_s, fps_out=5.0):
    out = TASKS / name; (out / "images").mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video); items, next_t = [], start_s
    while True:
        ok, img = cap.read()
        if not ok:
            break
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        if t < next_t - 1e-6:
            continue
        if t >= start_s + dur_s:
            break
        f = f"{name}__{int(round(t * 1000)):07d}.jpg"
        cv2.imwrite(str(out / "images" / f), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        items.append({"file": f, "scene": Path(video).stem, "t": round(t, 3)})
        next_t += 1 / fps_out
    json.dump({"task": name, "kind": "track", "video": video, "fps": fps_out, "items": items},
              open(out / "manifest.json", "w"), indent=1)
    print(name, len(items), "frames")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-scene", type=int, default=60)
    ap.add_argument("--clip-s", type=float, default=30)
    a = ap.parse_args()
    det_task(a.per_scene)
    clip_task("track_aa1", P.VIDEO_AA1, 15.0, min(a.clip_s, 15.0))    # the tracking evaluation window
    clip_task("track_test1", "videos/test1.mp4", 60.0, a.clip_s)
