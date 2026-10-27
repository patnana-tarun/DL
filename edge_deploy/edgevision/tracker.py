"""Multi-object tracking: ByteTrack from the `trackers` library, with the same pre-filter as the study
(score cut + class-agnostic NMS before association), so the tuned parameters transfer unchanged.

ByteTrack's lost-track buffer is defined in 30-fps frames (seconds = buffer / 30), and when a timestamp is
passed the motion model uses the real time step. So tracking stays consistent at any analysis frame rate
and through dropped frames on a live stream.
"""
import numpy as np
import supervision as sv
from trackers import ByteTrackTracker

PRE_KEYS = ("min_conf", "nms_iou")


class Tracker:
    def __init__(self, cfg, fps):
        p = dict(cfg)
        self.min_conf = float(p.pop("min_conf", 0.0))
        self.nms_iou = float(p.pop("nms_iou", 0.0))
        self.t = ByteTrackTracker(frame_rate=float(fps), **p)

    def update(self, xyxy, conf, ts=None):
        """-> (xyxy [M,4] float32, track ids [M] int) of confirmed tracks in this frame."""
        k = conf >= self.min_conf
        d = sv.Detections(xyxy=np.asarray(xyxy, np.float32)[k], confidence=np.asarray(conf, np.float32)[k],
                          class_id=np.zeros(int(k.sum()), int))
        if self.nms_iou and len(d):
            d = d.with_nms(threshold=self.nms_iou, class_agnostic=True)
        o = self.t.update(d, timestamp=ts) if ts is not None else self.t.update(d)
        if not len(o):
            return np.zeros((0, 4), np.float32), np.zeros(0, int)
        keep = o.tracker_id >= 0
        return o.xyxy[keep].astype(np.float32), o.tracker_id[keep].astype(int)
