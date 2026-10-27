"""Person detector: BGR uint8 frame -> (xyxy float32 [N,4] in frame pixels, scores float32 [N]).

Backends
  onnxruntime  the exported ONNX graph; TensorRT (FP16, engine cached on disk) -> CUDA -> CPU, whichever
               is available. No PyTorch needed, which keeps the Jetson install small.
  ultralytics  Ultralytics YOLO with .pt / .onnx / .engine weights (needs PyTorch).

The exported YOLO26n uses the one-to-many head + NMS: the study found the NMS-free head loses ~0.02 AP
in crowds (results/latency.json). Both output layouts are decoded anyway:
  (1, 4+nc, N)  cx cy w h + class scores, needs NMS
  (1, K, 6)     x1 y1 x2 y2 score class, NMS-free
"""
import json
import logging
import time
from pathlib import Path

import cv2
import numpy as np

from .config import resolve

log = logging.getLogger("edgevision.detector")
PERSON = 0                       # COCO-80 person index; single-class models use 0 as well


def letterbox(img, out_hw, pad_value=114):
    """Resize keeping aspect ratio, pad (centred) to out_hw. -> padded, scale, (pad_x, pad_y)."""
    H, W = img.shape[:2]
    oh, ow = out_hw
    s = min(oh / H, ow / W)
    nh, nw = round(H * s), round(W * s)
    if (nh, nw) != (H, W):
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    px, py = (ow - nw) / 2, (oh - nh) / 2
    top, left = int(round(py - 0.1)), int(round(px - 0.1))
    out = cv2.copyMakeBorder(img, top, oh - nh - top, left, ow - nw - left, cv2.BORDER_CONSTANT,
                             value=(pad_value,) * 3)
    return out, s, (left, top)


def nms(boxes, scores, iou_thr):
    """Greedy NMS. boxes xyxy [N,4]; returns kept indices, highest score first."""
    order = np.argsort(-scores, kind="stable")
    x1, y1, x2, y2 = boxes.T
    area = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        r = order[1:]
        iw = (np.minimum(x2[i], x2[r]) - np.maximum(x1[i], x1[r])).clip(0)
        ih = (np.minimum(y2[i], y2[r]) - np.maximum(y1[i], y1[r])).clip(0)
        inter = iw * ih
        iou = inter / (area[i] + area[r] - inter + 1e-9)
        order = r[iou <= iou_thr]
    return np.array(keep, dtype=np.int64)


def decode(out, conf, iou, max_det, person=PERSON):
    """Raw network output -> (xyxy, scores) in network-input pixels."""
    out = np.asarray(out, dtype=np.float32)
    if out.ndim == 3:
        out = out[0]
    if out.shape[-1] == 6 and out.shape[0] >= out.shape[1]:          # (K, 6) NMS-free
        cls_ok = (out[:, 5] == person) if out[:, 5].max(initial=0) > 0 else np.ones(len(out), bool)
        k = (out[:, 4] > conf) & cls_ok
        return out[k, :4], out[k, 4]
    if out.shape[0] < out.shape[1]:                                   # (4+nc, N) -> (N, 4+nc)
        out = out.T
    nc = out.shape[1] - 4
    sc = out[:, 4 + (person if nc > 1 else 0)]
    k = sc > conf
    b, sc = out[k, :4], sc[k]
    b = np.c_[b[:, :2] - b[:, 2:] / 2, b[:, :2] + b[:, 2:] / 2]
    if len(b):
        keep = nms(b, sc, iou)[:max_det]
        b, sc = b[keep], sc[keep]
    return b, sc


class OnnxDetector:
    def __init__(self, cfg):
        import onnxruntime as ort
        self.cfg = cfg
        path = resolve(cfg["weights"])
        avail = ort.get_available_providers()
        want = {"tensorrt": "TensorrtExecutionProvider", "cuda": "CUDAExecutionProvider",
                "cpu": "CPUExecutionProvider"}
        providers = []
        for p in cfg["providers"]:
            name = want[p]
            if name not in avail:
                continue
            if p == "tensorrt":
                cache = resolve(cfg["trt_cache"]); Path(cache).mkdir(parents=True, exist_ok=True)
                providers.append((name, {"trt_fp16_enable": bool(cfg["fp16"]), "trt_engine_cache_enable": True,
                                         "trt_engine_cache_path": cache, "trt_timing_cache_enable": True,
                                         "trt_timing_cache_path": cache}))
            else:
                providers.append(name)
        if not providers:
            providers = ["CPUExecutionProvider"]
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        t = time.time()
        self.sess = ort.InferenceSession(path, so, providers=providers)
        self.inp = self.sess.get_inputs()[0]
        self.hw = tuple(int(x) for x in self.inp.shape[2:4])
        self.fp16_input = "float16" in self.inp.type
        meta = self.sess.get_modelmeta().custom_metadata_map
        self.names = meta.get("names", "")
        log.info("onnxruntime %s, providers %s, input %s, loaded in %.1fs", ort.__version__,
                 self.sess.get_providers(), self.hw, time.time() - t)
        self.provider = self.sess.get_providers()[0]

    def __call__(self, img):
        x, s, (px, py) = letterbox(img, self.hw)
        x = np.ascontiguousarray(x[:, :, ::-1].transpose(2, 0, 1))[None]
        x = x.astype(np.float16 if self.fp16_input else np.float32) / 255.0
        out = self.sess.run(None, {self.inp.name: x})[0]
        b, sc = decode(out, self.cfg["conf"], self.cfg["iou"], self.cfg["max_det"])
        if len(b):
            b = (b - [px, py, px, py]) / s
            H, W = img.shape[:2]
            b = b.clip([0, 0, 0, 0], [W, H, W, H])
        return b.astype(np.float32), sc.astype(np.float32)


class UltralyticsDetector:
    def __init__(self, cfg):
        from ultralytics import YOLO
        self.cfg = cfg
        w = resolve(cfg["weights"])
        self.model = YOLO(w, task="detect")
        meta = Path(w).with_suffix(".json")
        self.imgsz = json.loads(meta.read_text())["imgsz"] if meta.exists() else None
        self.single = len(self.model.names) == 1
        self.provider = f"ultralytics:{Path(w).suffix[1:]}"

    def __call__(self, img):
        r = self.model.predict(img, imgsz=self.imgsz or 1280, conf=self.cfg["conf"], iou=self.cfg["iou"],
                               max_det=self.cfg["max_det"], half=bool(self.cfg["fp16"]),
                               classes=None if self.single else [PERSON], verbose=False)[0]
        return r.boxes.xyxy.cpu().numpy().astype(np.float32), r.boxes.conf.cpu().numpy().astype(np.float32)


def build(cfg):
    return OnnxDetector(cfg) if cfg["backend"] == "onnxruntime" else UltralyticsDetector(cfg)
