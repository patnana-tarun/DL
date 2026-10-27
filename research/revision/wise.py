"""Weight-space ensembling of the COCO-pretrained and a fine-tuned YOLO26n (WiSE-FT, Wortsman et al. 2022;
WiSE-OD for detectors): theta(a) = (1 - a) * theta_pretrained + a * theta_finetuned, a in [0, 1].

The 80-class variants of train_variants.py share every tensor shape with the pretrained model, so the blend is
exact. The published aa1 model has a 1-class head whose classification branch has a different width; for it
only the shape-compatible tensors (backbone, neck, box branch) are blended and the fine-tuned class branch is
kept ("partial" blend, reported separately).

BatchNorm: blending running statistics linearly is not the statistics of the blended network, so each blend is
also written with re-estimated BN statistics ("bnre"): cumulative averages over a calibration set drawn from
training-time data only (aa1 train frames + COCO replay images), forward passes in train mode, no gradients.

    python research/revision/wise.py --ft runs/revision/ft80/weights/best.pt --tag ft80
    -> runs/revision/wise/<tag>_a<alpha>[_bnre].pt
"""
import argparse
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths as P  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

ALPHAS = [round(a, 2) for a in np.linspace(0, 1, 11)]
CALIB = [("datasets/yolo/train/images", 96), ("datasets/coco_replay/images", 96)]


def load_ckpt(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def blend_state(pre, ft, a):
    """-> (state dict, n blended, n kept from ft)."""
    out, nb, nk = {}, 0, 0
    for k, v in ft.items():
        p = pre.get(k)
        if p is not None and p.shape == v.shape and v.is_floating_point():
            out[k] = ((1 - a) * p.float() + a * v.float()).to(v.dtype); nb += 1
        else:
            out[k] = v.clone(); nk += 1
    return out, nb, nk


def letterbox(img, size):
    H, W = img.shape[:2]; s = size / max(H, W)
    img = cv2.resize(img, (round(W * s), round(H * s)), interpolation=cv2.INTER_LINEAR)
    h, w = img.shape[:2]
    out = np.full((h + (-h) % 32, w + (-w) % 32, 3), 114, np.uint8); out[:h, :w] = img
    return torch.from_numpy(out[:, :, ::-1].copy()).permute(2, 0, 1).float()[None] / 255


@torch.no_grad()
def recalibrate_bn(model, imgsz=1280):
    paths = []
    rng = np.random.default_rng(0)
    for d, n in CALIB:
        files = sorted(Path(d).glob("*.jpg"))
        paths += [files[i] for i in rng.choice(len(files), min(n, len(files)), replace=False)]
    model = model.float().cuda()
    bns = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    for m in bns:
        m.reset_running_stats(); m.momentum = None          # cumulative moving average
    model.eval()
    for m in bns:
        m.train()
    for p in paths:
        model(letterbox(cv2.imread(str(p)), imgsz).cuda())
    for m in bns:
        m.momentum = 0.1
    model.eval()
    return model.half().cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre", default=P.YOLO_PRETRAINED)
    ap.add_argument("--ft", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--alphas", default=",".join(map(str, ALPHAS)))
    ap.add_argument("--bn", default="both", choices=["interp", "bnre", "both"])
    ap.add_argument("--imgsz", type=int, default=1280)
    a = ap.parse_args()
    pre, ft = load_ckpt(a.pre), load_ckpt(a.ft)
    sp = pre["model"].float().state_dict()
    out = Path("runs/revision/wise"); out.mkdir(parents=True, exist_ok=True)
    for al in [float(x) for x in a.alphas.split(",")]:
        st, nb, nk = blend_state(sp, ft["model"].float().state_dict(), al)
        for mode in (("interp", "bnre") if a.bn == "both" else (a.bn,)):
            ck = copy.copy(ft); ck["ema"] = None; ck["optimizer"] = None
            m = copy.deepcopy(ft["model"]).float(); m.load_state_dict(st)
            m = recalibrate_bn(m, a.imgsz) if mode == "bnre" else m.half()
            ck["model"] = m
            ck["wise"] = {"alpha": al, "pre": a.pre, "ft": a.ft, "bn": mode, "blended": nb, "kept_ft": nk}
            name = f"{a.tag}_a{al:.1f}" + ("_bnre" if mode == "bnre" else "")
            torch.save(ck, out / f"{name}.pt")
            print(name, "blended", nb, "kept", nk, flush=True)


if __name__ == "__main__":
    main()
