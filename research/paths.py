"""Project paths shared by every research script.

Importing this module makes the project root the working directory, so the scripts can be launched
from anywhere (`python research/evaluate.py`) and every relative path below means the same thing.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "research"))

VIDEO_AA1 = "videos/aa1.mp4"
YOLO_PRETRAINED = "models/pretrained/yolo26n.pt"      # COCO-pretrained student
YOLO_TEACHER = "models/pretrained/yolo26x.pt"         # teacher (pseudo-labels) and campus detector
RFDETR_PRETRAINED = "rf-detr-nano.pth"                # resolved by rfdetr from its weight cache
YOLO_FT = "runs/yolo26n/weights/best.pt"              # fine-tuned YOLO26n (aa1)
RFDETR_FT_RUN = "runs/rfdetr_nano_v2"                 # fine-tuned RF-DETR Nano (aa1, cosine LR)
DEPLOY = ROOT / "edge_deploy"                         # self-contained edge runtime (copied to the Jetson)
