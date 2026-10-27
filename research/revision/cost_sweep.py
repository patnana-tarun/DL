"""Compute and latency of both students per input size (review item 4: compare at matched input / compute).

GFLOPs: one forward pass at the letterboxed 16:9 input (YOLO: long side x 9/16 rounded up to 32; RF-DETR: square
input of the given side, as rfdetr runs it), torch FlopCounterMode. Latency (--latency): median of the deployed
fast paths (research/detectors.py) over the aa1 test frames on this desktop GPU, after 20 warm-up frames.

    python research/revision/cost_sweep.py [--latency]     -> results/revision/cost_sweep.json
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
import torch  # noqa: E402

SIZES = (384, 512, 640, 768, 960, 1280)


def flops(net, x):
    from torch.utils.flop_counter import FlopCounterMode
    net.requires_grad_(False)
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        net(x)
    return fc.get_total_flops() / 1e9


def main():
    import detectors as D
    from rfdetr import RFDETRNano
    from ultralytics import YOLO
    ap = argparse.ArgumentParser(); ap.add_argument("--latency", action="store_true"); a = ap.parse_args()
    out = {"YOLO26n": {}, "RF-DETR Nano": {}}
    ynet = YOLO("runs/revision/ft80/weights/best.pt").model.cuda().eval()
    out["YOLO26n"]["params_M"] = sum(p.numel() for p in ynet.parameters()) / 1e6
    rf_ckpt = D.rfdetr_best(P.RFDETR_FT_RUN)
    frames = [cv2.imread(str(f)) for f in sorted(Path("datasets/coco/test").glob("*.jpg"))]
    for sz in SIZES:
        h = int(np.ceil(sz * 9 / 16 / 32) * 32)
        out["YOLO26n"][sz] = {"input": [h, sz], "GFLOPs": flops(ynet, torch.zeros(1, 3, h, sz, device="cuda"))}
        m = RFDETRNano(pretrain_weights=rf_ckpt, resolution=sz)
        net = m.model.model.cuda().eval()
        out["RF-DETR Nano"]["params_M"] = sum(p.numel() for p in net.parameters()) / 1e6
        out["RF-DETR Nano"][sz] = {"input": [sz, sz], "GFLOPs": flops(net, torch.zeros(1, 3, sz, sz, device="cuda"))}
        del m, net; torch.cuda.empty_cache()
        if a.latency:
            for fam, run in (("YOLO26n", D.yolo("runs/revision/ft80/weights/best.pt", True, imgsz=sz)),
                             ("RF-DETR Nano", D.rfdetr(rf_ckpt, True, sz))):
                for f in frames[:20]:
                    run(f)
                t = []
                for f in frames:
                    torch.cuda.synchronize(); t0 = time.perf_counter(); run(f); torch.cuda.synchronize()
                    t.append(1000 * (time.perf_counter() - t0))
                out[fam][sz]["latency_ms_p50"] = float(np.median(t))
                del run; torch.cuda.empty_cache()
        print(sz, {k: v[sz] for k, v in out.items()}, flush=True)
    Path("results/revision").mkdir(parents=True, exist_ok=True)
    json.dump(out, open("results/revision/cost_sweep.json", "w"), indent=1)


if __name__ == "__main__":
    main()
