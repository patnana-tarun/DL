"""Person detectors behind one interface: run(bgr_uint8) -> (xyxy [N,4], scores [N]).

`fast=True` is the deployment path found by benchmark_latency.py:
  YOLO26n     : uint8 frame uploaded once, resize/pad/normalise on GPU, fused FP16 net,
                one-to-many head + GPU NMS (the NMS-free head loses ~0.02 AP in crowds).
  RF-DETR Nano: FP16 + TorchScript via rfdetr's `inference()`, GPU-side colour/scale conversion.
`fast=False` is the stock Ultralytics / rfdetr `predict()` path, kept as the reference.
"""
from pathlib import Path

import numpy as np
import torch
import torchvision

import paths as P

PERSON_COCO_YOLO = 0      # Ultralytics COCO-80 index
PERSON_COCO_RFDETR = 1    # rfdetr keeps original COCO-91 ids for pretrained checkpoints


def rfdetr_best(run_dir):
    return str(sorted(Path(run_dir).glob("checkpoint_best_total.pth"))[0])


def yolo(weights, fast=True, imgsz=1280, conf=0.001, iou=0.7, max_det=300, nms=True):
    from ultralytics import YOLO
    model = YOLO(weights)
    single = len(model.names) == 1
    if not fast:
        def run(img):
            r = model.predict(img, imgsz=imgsz, conf=conf, iou=iou, max_det=max_det, nms=nms,
                              classes=None if single else [PERSON_COCO_YOLO], verbose=False)[0]
            return r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()
        return run

    net = model.model
    net.end2end = not nms            # choose the head before fuse() drops the other one
    net = net.fuse().cuda().eval().half()
    cls_row = 4 + (0 if single else PERSON_COCO_YOLO)

    @torch.inference_mode()
    def run(img):
        H, W = img.shape[:2]
        s = imgsz / max(H, W)
        x = torch.from_numpy(img).cuda(non_blocking=True).permute(2, 0, 1)[None].flip(1).half() / 255
        if s != 1:
            x = torch.nn.functional.interpolate(x, scale_factor=s, mode="bilinear", align_corners=False)
        h, w = x.shape[2:]
        x = torch.nn.functional.pad(x, (0, (-w) % 32, 0, (-h) % 32), value=114 / 255)
        y = net(x)
        y = y[0] if isinstance(y, (list, tuple)) else y
        if nms:   # one-to-many head: (1, 4+nc, N) cx cy w h cls... -> person scores -> GPU NMS
            y = y[0].float()
            sc = y[cls_row]
            k = sc > conf
            b, sc = y[:4, k].T, sc[k]
            b = torch.cat([b[:, :2] - b[:, 2:] / 2, b[:, :2] + b[:, 2:] / 2], 1)
            keep = torchvision.ops.nms(b, sc, iou)[:max_det]
            b, sc = b[keep], sc[keep]
        else:     # one-to-one head: (1, 300, 6) x1 y1 x2 y2 score cls
            y = y[0].float()
            y = y[(y[:, 4] > conf) & ((y[:, 5] == PERSON_COCO_YOLO) | single)]
            b, sc = y[:, :4], y[:, 4]
        return (b / s).cpu().numpy(), sc.cpu().numpy()
    return run


def rfdetr(weights, fast=True, resolution=None, conf=0.001):
    from rfdetr import RFDETRNano
    kw = {"pretrain_weights": weights}
    if resolution:
        kw["resolution"] = resolution
    m = RFDETRNano(**kw)
    person = [0, 1] if len(m.class_names) == 1 else [PERSON_COCO_RFDETR]
    if fast:
        m.inference(compile=True, dtype=torch.float16)

    def run(img):
        if fast:
            x = torch.from_numpy(img).cuda(non_blocking=True).permute(2, 0, 1).flip(0).float() / 255
        else:
            x = img[:, :, ::-1].copy()
        d = m.predict(x, threshold=conf, include_source_image=False)
        k = np.isin(d.class_id, person)
        return d.xyxy[k], d.confidence[k]
    return run


def model_zoo(fast=True):
    """Every model compared in the study, as lazy factories."""
    return {
        "YOLO26n (pretrained)": lambda: yolo(P.YOLO_PRETRAINED, fast),
        "YOLO26n (fine-tuned)": lambda: yolo(P.YOLO_FT, fast),
        "RF-DETR Nano (pretrained)": lambda: rfdetr(P.RFDETR_PRETRAINED, fast),
        "RF-DETR Nano (fine-tuned)": lambda: rfdetr(rfdetr_best(P.RFDETR_FT_RUN), fast, 640),
    }
