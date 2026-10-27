"""Draw tracks, the counting line, zones and a status bar on a frame (in place on a copy)."""
import cv2
import numpy as np


def _color(tid):
    h = (tid * 37) % 180
    c = cv2.cvtColor(np.uint8([[[h, 200, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return int(c[0]), int(c[1]), int(c[2])


def draw(img, boxes, ids, snap, fps=None, alert=None):
    out = img.copy()
    H, W = out.shape[:2]
    for name, poly in (snap.get("zones_px") or {}).items():
        cv2.polylines(out, [poly.astype(np.int32)], True, (255, 200, 0), 2, cv2.LINE_AA)
        cv2.putText(out, name, tuple(poly[0].astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2, cv2.LINE_AA)
    (x1, y1), (x2, y2) = snap["line"]
    p1, p2 = (int(x1 * W), int(y1 * H)), (int(x2 * W), int(y2 * H))
    cv2.line(out, p1, p2, (255, 255, 255), 2, cv2.LINE_AA)
    c = snap["crossings"]
    lab = snap.get("crossing_directions", {})
    txt = f"{lab.get('a_to_b', 'a>b')}: {c['a_to_b']}   {lab.get('b_to_a', 'b>a')}: {c['b_to_a']}"
    tx = min(max(8, (p1[0] + p2[0]) // 2 + 8), W - 330)
    cv2.putText(out, txt, (tx, max(70, min(p1[1], p2[1]) + 70)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(out, txt, (tx, max(70, min(p1[1], p2[1]) + 70)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    for (a, b, cc, d), tid in zip(boxes.astype(int), ids):
        col = _color(int(tid))
        cv2.rectangle(out, (a, b), (cc, d), col, 2)
        cv2.putText(out, f"#{tid}", (a, max(12, b - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2, cv2.LINE_AA)
    bar = (f"in view {snap['occupancy']}   unique {snap['unique_people']}   "
           f"t {snap['t']:.1f}s" + (f"   {fps:.1f} fps" if fps else ""))
    cv2.rectangle(out, (0, 0), (W, 34), (20, 20, 20), -1)
    cv2.putText(out, bar, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    if alert:
        (tw, _), _ = cv2.getTextSize(alert, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.rectangle(out, (W - tw - 24, 0), (W, 34), (30, 30, 200), -1)
        cv2.putText(out, alert, (W - tw - 12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return out
