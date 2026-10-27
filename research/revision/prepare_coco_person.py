"""COCO val2017 person subsets for the forgetting study (Hugging Face parquet mirror of COCO 2017 val).

The two parquet halves of val2017 are used for different things, so no image is both trained on and scored:
  half 0 -> datasets/coco_replay/      replay pool mixed into fine-tuning (person images + ~15 % person-free
                                       negatives), YOLO format, labels = COCO person boxes (human annotations)
  half 1 -> datasets/coco_person_val/  generic person set for checking forgetting and choosing the WiSE-FT blend
                                       (COCO json, the evaluate.py layout: coco/test/_annotations.coco.json)
COCO val2017 was not used to train the COCO-pretrained students (they were trained on train2017).

    python research/revision/prepare_coco_person.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths as P  # noqa: E402,F401

import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

SRC = Path("datasets/coco2017_hf")
HALVES = sorted(SRC.glob("val-*.parquet"))
N_REPLAY_POS, N_REPLAY_NEG = 620, 100        # ~1:1 with the 720 aa1 training frames
N_VAL_POS, N_VAL_NEG = 500, 100
PERSON = 0


def rows(path):
    f = pq.ParquetFile(path)
    for g in range(f.num_row_groups):
        for r in f.read_row_group(g).to_pylist():
            yield r


def pick(path, n_pos, n_neg, seed):
    rs = list(rows(path))
    pos = [r for r in rs if PERSON in r["objects"]["category"]]
    neg = [r for r in rs if PERSON not in r["objects"]["category"]]
    rng = np.random.default_rng(seed)
    return ([pos[i] for i in rng.choice(len(pos), min(n_pos, len(pos)), replace=False)] +
            [neg[i] for i in rng.choice(len(neg), min(n_neg, len(neg)), replace=False)]), len(pos), len(neg)


def persons(r):
    return [b for b, c in zip(r["objects"]["bbox"], r["objects"]["category"]) if c == PERSON]   # xyxy pixels


def main():
    assert len(HALVES) == 2, f"expected two parquet halves in {SRC}"
    info = {}
    # replay pool, YOLO format
    sel, npos, nneg = pick(HALVES[0], N_REPLAY_POS, N_REPLAY_NEG, 0)
    img_dir, lab_dir = Path("datasets/coco_replay/images"), Path("datasets/coco_replay/labels")
    img_dir.mkdir(parents=True, exist_ok=True); lab_dir.mkdir(parents=True, exist_ok=True)
    boxes = 0
    for r in sel:
        stem = f"coco{r['image_id']:012d}"
        (img_dir / f"{stem}.jpg").write_bytes(r["image"]["bytes"])
        W, H = r["width"], r["height"]
        lines = []
        for x1, y1, x2, y2 in persons(r):
            lines.append(f"0 {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
        (lab_dir / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        boxes += len(lines)
    info["replay"] = {"images": len(sel), "person_images_available": npos, "negatives_available": nneg,
                      "boxes": boxes, "source": HALVES[0].name}
    # generic person val, COCO json
    sel, npos, nneg = pick(HALVES[1], N_VAL_POS, N_VAL_NEG, 1)
    split = Path("datasets/coco_person_val/coco/test"); split.mkdir(parents=True, exist_ok=True)
    coco = {"images": [], "annotations": [], "categories": [{"id": 1, "name": "person", "supercategory": "none"}]}
    for r in sel:
        name = f"coco{r['image_id']:012d}.jpg"
        (split / name).write_bytes(r["image"]["bytes"])
        img_id = len(coco["images"]) + 1
        coco["images"].append({"id": img_id, "file_name": name, "width": r["width"], "height": r["height"],
                               "coco_id": r["image_id"]})
        for x1, y1, x2, y2 in persons(r):
            w, h = x2 - x1, y2 - y1
            coco["annotations"].append({"id": len(coco["annotations"]) + 1, "image_id": img_id, "category_id": 1,
                                        "bbox": [x1, y1, w, h], "area": w * h, "iscrowd": 0})
    json.dump(coco, open(split / "_annotations.coco.json", "w"))
    info["person_val"] = {"images": len(sel), "boxes": len(coco["annotations"]), "source": HALVES[1].name}
    json.dump(info, open("datasets/coco_person_val/stats.json", "w"), indent=1)
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
