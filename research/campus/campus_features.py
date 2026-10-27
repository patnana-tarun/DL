"""Appearance features for every person crop, from the DINOv2 backbone inside RF-DETR Large
(weights already on disk from the aa1 study; nothing is downloaded).

Crop -> 448x224 (patch 16 -> 28x14 tokens) -> backbone feature maps of the last two levels
(384 ch each). Each map is mean-pooled globally and over four horizontal bands
(head/chest, torso, hips, legs), since shirt, vest and trousers live at known heights on the body.
Feature = 2 levels x 5 pools x 384 = 3840-d, L2-normalised.
Also stored: colour histograms (HSV 8x3x3 per band, 4 bands = 288-d), since the uniform is a colour cue.

    HF_HUB_OFFLINE=1 python research/campus/campus_features.py --name video_1
Writes results/campus/<name>/features.npz (names, feats)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths as P  # noqa: E402,F401  (project root as working directory)

import cv2
import numpy as np
import torch

BANDS = [(0.0, 1.0), (0.05, 0.3), (0.2, 0.5), (0.45, 0.7), (0.6, 1.0)]
MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])


def load_encoder():
    from rfdetr import RFDETRLarge
    return RFDETRLarge().model.model.backbone[0].encoder.cuda().eval().half()


@torch.no_grad()
def embed(enc, paths, bs=32):
    out = []
    for s in range(0, len(paths), bs):
        ims = []
        for p in paths[s:s + bs]:
            im = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
            ims.append(((cv2.resize(im, (224, 448), interpolation=cv2.INTER_AREA) / 255.0 - MEAN) / STD).transpose(2, 0, 1))
        x = torch.from_numpy(np.stack(ims)).cuda().half()
        maps = enc(x)[-2:]
        f = []
        for m in maps:
            h = m.shape[2]
            for a, b in BANDS:
                f.append(m[:, :, int(a * h):max(int(b * h), int(a * h) + 1)].float().mean((2, 3)))
        f = torch.cat(f, 1)
        out.append(torch.nn.functional.normalize(f, dim=1).cpu().numpy())
    return np.concatenate(out)


def colour(paths):
    out = []
    for p in paths:
        hsv = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2HSV); h, w = hsv.shape[:2]
        f = []
        for a, b in BANDS[1:]:
            r = hsv[int(a * h):int(b * h), w // 5:4 * w // 5]
            x = cv2.calcHist([r], [0, 1, 2], None, [8, 3, 3], [0, 180, 0, 256, 0, 256]).ravel()
            f.append(x / (x.sum() + 1e-9))
        out.append(np.sqrt(np.concatenate(f)))              # Hellinger map
    return np.array(out)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--name", required=True); a = ap.parse_args()
    out = Path("results/campus") / a.name
    paths = sorted((out / "crops").glob("*.jpg"))
    feats = embed(load_encoder(), paths)
    np.savez_compressed(out / "features.npz", names=np.array([p.name for p in paths]), feats=feats.astype(np.float16),
                        colour=colour(paths).astype(np.float16))
    print(a.name, feats.shape)


if __name__ == "__main__":
    main()
