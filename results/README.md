# YOLO26n vs RF-DETR Nano on aa1.mp4: detection, tracking, model selection and the edge pipeline

All numbers were regenerated on 2026-10-07/08 with the current scripts, in the project venv
(torch 2.14 + CUDA 13, ultralytics 8.4.164, rfdetr 1.11.0, trackers 2.6.1, onnxruntime 1.30).
They match the previous run (2026-09-29) to the third decimal. Re-run everything with
`research/run_all.sh`; logs are in `results/logs/`.

> **Caveat:** all ground truth comes from teacher models (RF-DETR Large + YOLO26x ensemble), not human
> annotators. The metrics measure agreement with a much larger teacher ensemble, not accuracy against
> human-verified truth.

## Pipeline

| Step | Script | What it does |
|---|---|---|
| 1. Data | `prepare_data.py` | Samples 1,082 frames (every 0.25 s) from 30 s to 5:00 of `videos/aa1.mp4` (1280×720, 59.94 fps). Labels people with an RF-DETR Large + YOLO26x (TTA) ensemble fused by WBF, keeping fused score ≥ 0.40. Splits by time: train 30–210 s (720 frames, 12,024 boxes), valid 210–255 s (180 / 3,371), test 255–300 s (182 / 3,441). |
| 2. Train | `train_yolo.py`, `train_rfdetr.py` | YOLO26n: 150 epochs at 1280 px, AdamW, cosine LR. RF-DETR Nano: 80 epochs at 640 px, cosine LR, EMA, effective batch 16. An earlier 60-epoch step-LR RF-DETR run reached the same test AP (0.724 vs 0.725, p = 0.58), so only the cosine run is kept. |
| 3. Evaluate | `evaluate.py` | Scores every model on train, valid and test with one protocol. The operating threshold is the best-F1 threshold on **valid**, frozen for test. Includes bootstrap 95% CIs (B = 1000), paired tests and an overfitting diagnosis. |
| 4. Cross-check | `crosscheck.py` | Recomputes the headline metrics with independent code (see below). |
| 5. Latency | `benchmark_latency.py` | Times every inference variant end to end on the 182 test frames and re-scores each variant. |
| 6. Label bias | `teacher_bias.py` | Re-scores test predictions against each teacher's labels on its own. |
| 7. Track | `track.py`, `track_compare.py` | ByteTrack on the first 30 s, which no model saw in training. Parameters are grid-searched on 0–15 s (8,688 configs per model) and reported on the unseen 15–30 s. Then 6 trackers × 2 pseudo-GT linkings. |
| 8. Analytics | `analytics.py` | Occupancy, unique track IDs, dwell, heatmap, flow, line crossings. |
| 9. Select + export | `select_model.py` | Decision matrix → deploy model → ONNX at three input sizes → accuracy verified through the edge runtime. |
| 10. Edge pipeline | `edge_eval.py`, `edge_demo.py` | The deployed runtime (`edge_deploy/edgevision`) on file and live RTSP streams, scored against the same pseudo-GT, plus a recorded closed-loop session with the VLM. |
| 11. VLM | `vlm_eval.py` | Qwen3-VL-2B: count grounding, zero-shot attributes, tool-call reliability, latency (`results/vlm/`). |
| 12. Figures | `visualize.py` | `results/figures/*.png`. |
| 13. Held-out video | `heldout/*.py` | Test-only evaluation on `videos/test1.mp4` with every threshold and parameter frozen from aa1: see **`test1/README.md`**. The aa1 ranking reverses there (pretrained YOLO26n 0.684 > RF-DETR-N FT 0.535 > YOLO26n FT 0.412 AP50-95). |

## Detection: test split (255–300 s)

| Model | mAP50-95 [95% CI] | mAP50 | mAP75 | AP small | AR@300 | F1 @ valid thr | LAMR ↓ | Params | GFLOPs | Latency (deployed) |
|---|---|---|---|---|---|---|---|---|---|---|
| YOLO26n pretrained | 0.760 [0.751, 0.769] | 0.947 | 0.901 | 0.496 | 0.830 | 0.882 | 0.436 | 2.6 M | 14.4 | — |
| **YOLO26n fine-tuned** | **0.825 [0.818, 0.834]** | **0.963** | **0.939** | **0.552** | **0.875** | **0.891** | **0.354** | 2.5 M | 13.6 | **3.68 ms** |
| RF-DETR Nano pretrained | 0.382 [0.366, 0.401] | 0.705 | 0.357 | 0.199 | 0.514 | 0.658 | 0.711 | 30.5 M | 31.9 | — |
| RF-DETR Nano fine-tuned | 0.725 [0.714, 0.738] | 0.943 | 0.838 | 0.298 | 0.808 | 0.885 | 0.411 | 30.5 M | 97.3 | 7.89 ms |

- **YOLO26n vs RF-DETR (both fine-tuned):** +0.099 AP50-95, paired bootstrap 95% CI [0.092, 0.107],
  p < 0.001 (no resample out of 1000 favoured RF-DETR).
- **Overfitting** (train − test gap minus the pretrained model's gap): YOLO26n +0.066 (train 0.920,
  valid 0.854, test 0.825), so it fits the training frames noticeably, yet still has the best valid and
  test scores. RF-DETR −0.026, with no excess gap.
- **Counting at the valid-F1 threshold:** YOLO26n's threshold is low (0.24), so it over-counts on test
  (count MAE 3.0 people per frame vs 1.5 for RF-DETR). For counting, use a higher threshold. The edge
  pipeline counts tracks, not raw detections (see the edge section).

### Independent cross-check (`crosscheck.py` → `crosscheck_detection.json`, `crosscheck_tracking.json`): 54/54 pass

| Check | Result |
|---|---|
| From-scratch 101-point AP (own matcher) vs pycocotools, 4 models × 3 splits | max difference 0.0012 AP |
| Optimal (Hungarian) vs greedy matching, F1 at the frozen threshold | max difference 0.0009 |
| Operating threshold re-derived as best F1 on valid | identical for all 4 models |
| Ultralytics' own validator vs our protocol (YOLO26n, test) | 0.832 vs 0.825 AP50-95 (different interpolation and NMS path) |
| Label counts: COCO JSON = YOLO txt = `stats.json` | 12,024 / 3,371 / 3,441 boxes |
| Every bootstrap CI contains its estimate; p < 0.05 ⇔ CI excludes 0 | all |
| Tracker re-run reproduces stored HOTA/MOTA/IDF1 | exact |
| py-motmetrics vs `trackers.eval` (MOTA / IDF1 / ID switches) | 0.770 / 0.805 / 41 vs 0.770 / 0.805 / 42 (YOLO); identical for RF-DETR |

### Label bias (`teacher_bias.json`): part of the detection gap depends on the teacher

| Model | vs fused labels | vs RF-DETR Large only | vs YOLO26x only |
|---|---|---|---|
| YOLO26n fine-tuned | 0.825 | 0.726 | 0.773 |
| RF-DETR Nano fine-tuned | 0.725 | **0.731** | 0.630 |

Each student scores best against its own family's teacher. Against RF-DETR-Large-only labels the two
are tied; read the 0.10 AP gap as "under these fused labels", not as an absolute ranking.

## Latency (`latency.json`, median per 1280×720 frame, RTX 4060 Laptop, batch 1, pre + post included)

| Variant | Median | AP50-95 |
|---|---|---|
| YOLO26n Ultralytics predict FP32 (baseline) | 5.31 ms | 0.827 |
| **YOLO26n GPU pre-proc + fused FP16 + GPU NMS (deployed)** | **3.68 ms** | **0.825** |
| YOLO26n NMS-free head + GPU pre-proc FP16 | 3.44 ms | 0.808 (the NMS-free head loses ~0.02 AP in crowds) |
| RF-DETR Nano predict FP32 (baseline) | 21.33 ms | 0.728 |
| **RF-DETR Nano FP16 + TorchScript + GPU pre-proc (deployed)** | **7.89 ms** | **0.725** |

## Tracking: ByteTrack, held-out 15–30 s

| Model | Params | HOTA | MOTA | IDF1 | DetA | AssA | ID switches | IDs (GT = 112) |
|---|---|---|---|---|---|---|---|---|
| YOLO26n | default | 0.715 | 0.731 | 0.774 | 0.713 | 0.719 | 40 | 115 |
| YOLO26n | tuned | 0.746 | 0.770 | 0.805 | 0.737 | 0.759 | 42 | 138 |
| RF-DETR Nano | default | 0.644 | 0.560 | 0.721 | 0.623 | 0.670 | 69 | 123 |
| RF-DETR Nano | tuned | 0.757 | 0.788 | 0.829 | 0.732 | 0.786 | 60 | 145 |

Best parameters (chosen on 0–15 s): YOLO26n min_conf 0.2, NMS 0.6, activation 0.7, high_conf 0.4, buffer
30, IoU 0.4, confirm after 3 frames. RF-DETR min_conf 0.5, NMS 0.6, activation 0.7, high_conf 0.4, buffer
15, IoU 0.4, confirm after 1 frame. Default ByteTrack settings clearly favour YOLO26n (+0.07 HOTA).
In this single search the two are tied within noise. The broader comparison below shows a small YOLO26n edge.

### Six trackers × two GT linkings (`tracking/tracker_comparison.json`), held-out HOTA

| Tracker | YOLO26n (ByteTrack-GT) | RF-DETR (ByteTrack-GT) | YOLO26n (OC-SORT-GT) | RF-DETR (OC-SORT-GT) |
|---|---|---|---|---|
| SORT | 0.749 | 0.734 | 0.653 | 0.629 |
| ByteTrack | **0.758** | 0.739 | 0.653 | 0.633 |
| OC-SORT | **0.758** | 0.738 | 0.648 | 0.629 |
| C-BIoU | 0.738 | 0.741 | 0.659 | 0.641 |
| BoT-SORT | 0.737 | **0.747** | 0.656 | 0.635 |
| McByte | 0.751 | 0.743 | **0.661** | **0.643** |

YOLO26n wins 10 of 12 pairings by about 0.01–0.02 HOTA. Absolute HOTA depends on how the GT was
linked (~0.75 vs ~0.65), so compare models only under the same GT.

## Crowd analytics, first 30 s (`tracking/analytics_<model>.json`)

| | Pseudo-GT | YOLO26n | RF-DETR Nano |
|---|---|---|---|
| Unique track IDs (≥ 0.25 s) | 169 | 176 (+4%) | 181 (+7%) |
| People in frame, mean / max | 15.4 / 21 | 16.2 / 23 | 14.7 / 20 |
| Dwell, median / p90 | — | 1.5 / 6.1 s | 1.3 / 4.9 s |

Unique track IDs are an upper bound on unique people (a broken-and-restarted track counts twice). Line
crossings and px/s speeds are image-plane values on a handheld camera, so treat them as indicative only.

## Model selection and export (`model_selection.json`)

| Criterion (weight) | YOLO26n | RF-DETR Nano |
|---|---|---|
| Test AP50-95 (0.30) | **0.825** | 0.725 |
| Worst-teacher AP, label-bias robustness (0.10) | **0.726** | 0.630 |
| Mean HOTA, 6 trackers × 2 GTs (0.25) | **0.702** (wins 10/12) | 0.688 |
| Deployed latency (0.20) | **3.68 ms** | 7.89 ms |
| GFLOPs (0.10) / params (0.05) | **13.6 / 2.5 M** | 97.3 / 30.5 M |
| **Weighted score** | **1.00** | 0.71 |

YOLO26n also wins under accuracy-only, speed-only, tracking-only, equal and bias-aware weightings
(`winner_robust: true`). It is exported to ONNX (one-to-many head + external NMS, the deployed variant)
and re-scored on the test split through the edge runtime's own detector (`edgevision/detector.py`, ONNX Runtime):

| ONNX input | AP50-95 | AP50 | AP small | Use |
|---|---|---|---|---|
| 736×1280 | **0.827** (PyTorch 0.825: parity ✓) | 0.963 | 0.552 | desktop, Orin NX / AGX |
| 544×960 | 0.794 | 0.959 | 0.505 | Orin Nano (default in `configs/jetson.yaml`) |
| 384×640 | 0.682 | 0.926 | 0.295 | only for very constrained devices: small people are lost |

## Edge pipeline, end to end (`edge/edge_eval.json`)

The deployed runtime (FFmpeg decoder process → ONNX detector → ByteTrack → online analytics) is run on
the same held-out first 30 s and scored against the same pseudo-GT. Frame stride s keeps original frames
0, s, 2s, …, which models a lower analysis rate on the Jetson.

| ONNX input | Analysis rate | HOTA held-out | MOTA | IDF1 | Unique people (GT) |
|---|---|---|---|---|---|
| 736×1280 | 59.9 fps | **0.746** | 0.758 | 0.806 | 181 (169) |
| 736×1280 | 30 fps | 0.743 | 0.753 | 0.806 | 164 (165) |
| 736×1280 | 15 fps | 0.735 | 0.736 | 0.796 | 160 (168) |
| 544×960 | 15 fps | 0.726 | 0.754 | 0.794 | 144 (168) |
| 384×640 | 15 fps | 0.699 | 0.718 | 0.785 | 125 (168) |

- At full rate, the deployed pipeline reproduces the research pipeline (0.746 HOTA, `track.py`), although
  it decodes with FFmpeg instead of OpenCV and runs the ONNX graph instead of PyTorch FP16.
- Dropping to 15 fps costs only 0.011 HOTA. The 544×960 input (Orin Nano default) costs 0.02 HOTA in total.
  384×640 loses small people (−0.047 HOTA, unique people −26%).
- Tracker + analytics add about 1.3 ms + 0.2 ms per frame. Detector latencies here are CPU ONNX Runtime on a
  loaded machine and are not representative of TensorRT on a Jetson.
- **Live RTSP** (video published in real time at 59.94 fps through MediaMTX, read as `rtsp://`, GPU
  backend): 1,602 frames analysed, 197 stale frames dropped (11%) instead of falling behind; detector
  p50 14 ms with the VLM sharing the GPU.

### Closed-loop session with the VLM (`edge/demo/`, `research/edge_demo.py`)

aa1 is published as a live RTSP camera and analysed at 15 fps with the web server and Qwen3-VL-2B. A scripted
operator uses only the HTTP API, as the UI does. Results:

- "Alert me when more than N people are in view for 2 s": the model called `set_alert(occupancy > N, 2 s)`.
- The rule fired, and the VLM attached a scene report (`alert_context`).
- "How busy has it been over the last minute?": the model called `get_history(1)`.
- "Make the counting line a horizontal line across the middle": the model's arguments were wrong 3 times.
  The intent guardrail refused each attempt, and the rule-based fallback set the line correctly.
- "Remove all alerts": done by the model.

Every answer to a control request ends with a system-generated `[system] Changed: ...` line. Median chat
latency on the desktop is about 2.5 s.

Earlier runs of this session exposed the failures that led to the guardrails: a vertical line set for
"horizontal", a claimed removal with no tool call, footers copied from history, and ordinary scenes flagged
as risky. See `vlm/README.md` (E3b, E5) for the measured effect of the redesign.

## Outputs

- `figures/`: 01 dataset, 02 PR/F1, 03 detection metrics, 04 size and speed, 05 training curves,
  07 tracking, 08 grid sensitivity, 09 split generalisation, 10 latency variants, 11 CIs. 06 (qualitative) and
  12 (crowd analytics heatmaps) are drawn over video frames that show people, so they are generated locally by
  `visualize.py` and `analytics.py` and not kept in git.
- `metrics.json` (test), `metrics_splits.json`, `significance.json`, `latency.json`, `teacher_bias.json`,
  `crosscheck_*.json`, `model_selection.json`, `pr_data.npz`, `preds/*.json.gz` (raw detections).
- `tracking/`: `tracking_metrics.json`, `grid_search.json`, `tracker_comparison.json`, `pseudo_gt_info.json`,
  `analytics_*.json`, `detections.npz` (cache), annotated videos.
- `edge/`: `edge_eval.json`, `runs/` (per-run reports and track dumps), `demo/`.
- `vlm/`: VLM experiments. `campus/`: campus case study (`campus/README.md`). `test1/`: held-out video (`test1/README.md`).
- Weights: `runs/yolo26n/weights/best.pt`, `runs/rfdetr_nano_v2/checkpoint_best_total.pth`; deployed ONNX in
  `edge_deploy/models/`.
