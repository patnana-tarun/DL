# Held-out video `videos/test1.mp4`: test-only evaluation of the aa1 models

`test1.mp4` was used **only for testing**. No weights were trained, and no threshold, tracker parameter or
counting rule was fitted on it. Every model is an aa1 checkpoint, the operating thresholds are the aa1-valid ones
(`results/metrics_splits.json`), and the tracker parameters are the ones selected on aa1
(`results/tracking/{tracking_metrics,tracker_comparison}.json`). Generated 2026-10-22/23 with the project venv.

> **Caveat (as for aa1):** ground truth comes from the same teacher ensemble (RF-DETR Large + YOLO26x with TTA, WBF,
> fused score ≥ 0.40), not from human annotators. The visual audit below checks how far the labels can be trusted.

## The video

| | |
|---|---|
| Content | Tree-lined campus road, fixed camera, students walking on the footpaths and the road, some cyclists |
| Format | 1920×1080 H.264, variable frame rate (16.7 / 22.2 ms frames, mean 57.0 fps), 297.6 s, 16,970 frames |
| Detection set | 1,191 frames, one every 0.25 s of PTS, read sequentially (VFR), JPEG q95 as for aa1 |
| Labels | 9,828 boxes, 8.3 people per frame (max 16), median height 99 px (p10 47, p90 261), 16% COCO-small |
| Tracking GT | teacher boxes on all 16,970 frames, linked by ByteTrack (aa1 `GT_PARAMS`), tracks ≥ 15 frames: 299 identities (OC-SORT-linked GT: 189) |

Compared with aa1 (crowded Times Square, handheld 720p), this scene has fewer and smaller people, a static camera,
different clothing (uniforms), dense foliage, bins, benches and lamp posts.

## Protocol

| Step | Script | Notes |
|---|---|---|
| 1. Label + detect | `research/heldout/heldout_detect.py` | One sequential pass over every frame: teacher (fused + each teacher's own boxes) and both fine-tuned students (conf ≥ 0.05). Sampled frames go through a JPEG round trip and are re-labelled on the JPEG, as for aa1. Chunked and resumable (~70 min on the RTX 4060). |
| 2. Detection | `heldout_eval.py` | All 4 models (deployed fast paths). COCO AP/AR, LAMR, P/R/F1/count at the **frozen aa1-valid threshold**. CIs: **moving-block bootstrap** (10 s blocks = 40 frames, B = 1000), because frames 0.25 s apart are correlated. The i.i.d. frame bootstrap is also stored. Paired tests on the same resamples. Recall by box height. Label bias (each teacher alone). Teacher-vs-teacher agreement. |
| 3. Tracking + analytics | `heldout_track.py` | ByteTrack (aa1-tuned and default), 6 trackers with the aa1-selected parameters, 2 GT linkings, 10 × 30 s segments for a paired Wilcoxon test. The analytics use the deployed `edgevision.analytics.Analytics` definitions with frame PTS. The counting line (horizontal segment, x 0.25–0.80 W at y = 0.75 H, across the left footpath and the road) was fixed from one frame **before any result was computed**. |
| 4. Edge runtime | `heldout_edge.py` | Deployed pipeline (FFmpeg → ONNX Runtime → ByteTrack → analytics), five configurations as in the aa1 edge table. |
| 4b. Resize diagnostic | `heldout_resize.py` | Re-scores YOLO26n FT/pre and the deployed ONNX detector after 1080p → 720p downscaling with bilinear / bicubic / area / FFmpeg. |
| 5. Label audit | `heldout_audit.py` | 40 random frames: every model–label disagreement (IoU < 0.1) of YOLO26n FT / pre and RF-DETR-N FT, plus 80 random labels, judged from crops (`audit/sheet_*.jpg`, `audit/verdicts.tsv`). |
| 6. Figures | `heldout_figures.py` | `figures/01–06`. |

Reproduce: `research/heldout/run_heldout.sh videos/test1.mp4 test1`. The audit verdicts in `audit/verdicts.tsv` are
the hand step. Unit tests: `research/tests/test_heldout.py`.

## Detection (1,191 frames, 9,828 labels)

| Model | AP50-95 [95% block CI] | aa1 test | Δ | AP50 | AP75 | AP small | R 32–64 px | F1 @ aa1 thr [CI] | FPPI | LAMR ↓ | Count MAE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **YOLO26n pretrained** | **0.684** [0.658, 0.706] | 0.760 | −0.076 | **0.914** | **0.790** | **0.353** | **0.403** | **0.840** [0.809, 0.865] (thr 0.36) | **0.23** | **0.299** | 1.95 |
| YOLO26n fine-tuned (aa1) | 0.412 [0.361, 0.453] | 0.825 | **−0.413** | 0.602 | 0.456 | 0.046 | 0.048 | 0.645 [0.582, 0.688] (thr 0.24) | 1.04 | 0.597 | 2.94 |
| RF-DETR Nano pretrained | 0.371 [0.326, 0.410] | 0.382 | −0.011 | 0.669 | 0.357 | 0.048 | 0.252 | 0.637 [0.590, 0.681] (thr 0.31) | 2.58 | 0.601 | 1.82 |
| RF-DETR Nano fine-tuned (aa1) | 0.535 [0.490, 0.573] | 0.725 | −0.190 | 0.820 | 0.583 | 0.138 | 0.253 | 0.767 [0.725, 0.800] (thr 0.46) | 0.56 | 0.446 | 2.29 |

Paired block bootstrap (AP50-95, B = 1000; p < 0.001 means no resample crossed 0):

| Comparison | Δ AP50-95 [95% CI] | Δ F1 [95% CI] |
|---|---|---|
| YOLO26n FT − RF-DETR-N FT | −0.123 [−0.157, −0.087], p < 0.001 | −0.123 [−0.161, −0.089] |
| YOLO26n FT − YOLO26n pretrained | **−0.272** [−0.311, −0.238], p < 0.001 | −0.195 [−0.247, −0.157] |
| RF-DETR-N FT − RF-DETR-N pretrained | +0.163 [+0.148, +0.178], p < 0.001 | +0.130 [+0.109, +0.150] |
| YOLO26n pre − RF-DETR-N pre | +0.312 [+0.288, +0.338], p < 0.001 | +0.202 [+0.173, +0.233] |

**The aa1 ranking reverses.** The aa1 winner (YOLO26n fine-tuned, 0.825) is the worst fine-tuned model here, and it
falls below its own COCO-pretrained weights. RF-DETR Nano's fine-tuning still transfers (+0.163 over pretrained).
The best detector on this scene is the **pretrained YOLO26n**.

Why:
- **Small people.** Recall of people 32–64 px tall (2,643 labels) is 0.048 for YOLO26n FT vs 0.403 pretrained,
  0.253 RF-DETR-N FT. From 128 px up, all four models reach 0.84–0.98 recall (`figures/02_transfer.png`).
- **False positives on scene objects.** YOLO26n FT makes 1.04 FP per frame (pretrained: 0.23). In the audited
  frames, 21 of its 41 FPs are green dustbins, benches, tree trunks and lamp-post bases (objects that do not occur
  in aa1), 3 are people the teachers missed, and 17 overlap a label (IoU ≥ 0.1) but are poorly localised or duplicate boxes.
- This matches the aa1 overfitting diagnosis: YOLO26n FT had an excess train–test gap of +0.066 on aa1 and RF-DETR
  none (−0.026). The diagnosis predicted which student would transfer worse.
- Calibration is not the cause. The best-F1 threshold on this video ("oracle", a diagnostic only, never used)
  gives YOLO26n FT F1 0.646 vs 0.645 at the frozen threshold, and RF-DETR-N FT 0.784 vs 0.767.

### Label bias and teacher agreement

| Model | vs fused labels (9,828) | vs RF-DETR Large only (9,150) | vs YOLO26x only (12,247) |
|---|---|---|---|
| YOLO26n pretrained | **0.684** | **0.621** | **0.594** |
| YOLO26n fine-tuned | 0.412 | 0.405 | 0.331 |
| RF-DETR Nano pretrained | 0.371 | 0.421 | 0.291 |
| RF-DETR Nano fine-tuned | 0.535 | 0.568 | 0.427 |

The ranking pretrained YOLO26n > RF-DETR-N FT > YOLO26n FT holds under all three label sets. The teachers agree less
on this scene than on aa1: RF-DETR Large scores 0.527 AP50-95 against YOLO26x's labels (0.614 the other way round).
YOLO26x labels 34% more boxes than RF-DETR Large; 62% of the extra boxes are people under 64 px.

### Label audit (`audit/audit.json`)

40 random frames (seed 0). Each model–label disagreement of YOLO26n FT, YOLO26n pre and RF-DETR-N FT was judged
from a crop with context: detections with IoU < 0.1 to every label, and labels with IoU < 0.1 to every detection at
the frozen threshold. Detections at IoU 0.1–0.5 count as localisation and are not audited. Disagreements are merged
across models (IoU > 0.5 = one region). 80 random labels were also judged.

| Region type | n | Real person | Rate [Wilson 95% CI] | Not a person |
|---|---|---|---|---|
| Random label | 80 | 79 | 98.8% [93.3, 99.8] | 1 (green dustbin) |
| Label missed by a model | 122 | 121 | 99.2% [95.5, 99.9] | 1 (green dustbin) |
| Detection far from every label | 36 | 9 | 25.0% [13.8, 41.1] | 27 (dustbins, benches, tree trunks, one 738 px group box) |

- Labels are precise, and when a model misses a label it is almost always a real person: median height 57 px (p10–p90:
  40–92 px), in the far half of the road, or partly behind a tree.
- 9 people were missed by the teachers: cut by the frame edge, behind a tree, or inside a group.
- Crediting every audited label error to the models changes their F1 on these frames by at most 0.013
  (YOLO26n FT 0.606 → 0.617, pre 0.831 → 0.840, RF-DETR-N FT 0.772 → 0.784). **The reversal is not a pseudo-label
  artefact.**
- The judge was an AI model looking at crops, not an independent human annotator. The verdicts are
  in `audit/verdicts.tsv` and the crops in `audit/sheet_*.jpg`, so anyone can re-check them.

## Tracking (all 16,970 frames, 299 pseudo-GT identities)

ByteTrack, scored on the whole video (`tracking.json`):

| Detector | Params | HOTA | DetA | AssA | MOTA | IDF1 | IDSW | MT / ML | IDs |
|---|---|---|---|---|---|---|---|---|---|
| YOLO26n FT | default | 0.404 | 0.401 | 0.409 | 0.371 | 0.467 | 209 | 50 / 163 | 138 |
| YOLO26n FT | aa1-tuned | 0.440 | 0.413 | 0.470 | 0.463 | 0.507 | 168 | 40 / 179 | 199 |
| RF-DETR-N FT | default | 0.409 | 0.475 | 0.358 | 0.419 | 0.477 | 811 | 118 / 100 | 164 |
| RF-DETR-N FT | aa1-tuned | 0.496 | 0.513 | 0.483 | 0.593 | 0.570 | 397 | 63 / 142 | 372 |
| **YOLO26n pretrained** | default | 0.552 | **0.610** | 0.503 | **0.672** | 0.609 | 156 | 114 / 116 | 106 |
| **YOLO26n pretrained** | YOLO26n aa1-tuned | **0.581** | 0.590 | **0.574** | 0.663 | **0.651** | **96** | 81 / 149 | 152 |

For reference, aa1 held-out 15–30 s HOTA was 0.746 (YOLO26n FT) and 0.757 (RF-DETR-N FT). The pretrained YOLO26n
was never tuned for tracking: it uses the parameters tuned on aa1 for the fine-tuned YOLO26n, or the defaults.

Paired over the 10 × 30 s segments (Wilcoxon signed-rank, aa1 parameters):
- RF-DETR-N FT beats YOLO26n FT in 8/10 segments: ΔHOTA +0.055, p = 0.010; ΔMOTA +0.135, p = 0.004. With default
  parameters the difference is not significant (p = 0.43).
- Pretrained YOLO26n beats RF-DETR-N FT in 10/10 segments (ΔHOTA +0.099, p = 0.002) and YOLO26n FT in 10/10
  (+0.154, p = 0.002).

Six trackers with the aa1-selected parameters (`tracker_comparison.json`), whole-video HOTA:

| Tracker | YOLO26n FT (ByteTrack-GT) | RF-DETR-N FT (ByteTrack-GT) | YOLO26n FT (OC-SORT-GT) | RF-DETR-N FT (OC-SORT-GT) |
|---|---|---|---|---|
| SORT | 0.395 | 0.499 | 0.391 | 0.499 |
| ByteTrack | 0.414 | 0.509 | 0.423 | 0.506 |
| OC-SORT | 0.386 | 0.509 | 0.378 | 0.506 |
| C-BIoU | 0.425 | 0.515 | 0.427 | 0.506 |
| BoT-SORT | 0.388 | **0.522** | 0.396 | **0.512** |
| McByte | 0.403 | 0.498 | 0.398 | 0.504 |

RF-DETR-N FT wins 12/12 tracker × GT pairings by 0.08–0.13 HOTA. On aa1, YOLO26n won 10/12 by 0.01–0.02.

The dip in HOTA around 120–150 s (`figures/04_tracking.png`) is a stretch with almost nobody in view (0–2
people), where single misses dominate the score.

## Analytics (`analytics.json`, deployed definitions, counting line fixed in advance)

| | Pseudo-GT | YOLO26n FT | RF-DETR-N FT | YOLO26n pre |
|---|---|---|---|---|
| People in frame, mean / max | 8.0 / 17 | 4.1 / 12 | 5.2 / 13 | 5.5 / 14 |
| Occupancy MAE (bias) per frame | — | 3.84 (−3.84) | 2.75 (−2.73) | **2.45** (−2.44) |
| Line crossings (bottom→top / top→bottom) | 44 (39 / 5) | 27 (25 / 2) | 31 (28 / 3) | **39** (35 / 4) |
| Crossing events matched (±1 s, same direction): P / R / F1 | — | 0.93 / 0.57 / 0.70 | 1.00 / 0.70 / 0.83 | 0.95 / 0.84 / **0.89** |
| Unique people (tracks ≥ 0.25 s) | 301 | 186 (−38%) | 304 (+1%) | 150 (−50%) |
| Median dwell (s) | 4.4 | 3.3 | 2.3 | 7.1 |

- Every model **undercounts** occupancy, because the pseudo-GT includes distant people that the students do not
  detect.
- **Line crossings are the robust counter.** People cross the line near the camera, where they are large. The
  pretrained YOLO26n recovers 84% of the crossing events with 95% precision.
- **Unique track IDs are not a usable metric here.** RF-DETR's 304 vs 301 is right only because two errors cancel:
  fragmentation (397 ID switches, short dwell) adds IDs, and missed distant people remove them. The pretrained
  YOLO26n keeps identities longest (96 switches, longest dwell) but misses distant people, so it reports the fewest.

## Edge runtime (`edge.json`)

The deployed runtime (FFmpeg decoder → ONNX Runtime, CPU → ByteTrack with the aa1 defaults → analytics on the same
counting line) runs the **fine-tuned YOLO26n** export, so it inherits that detector's weakness on this scene. The
pseudo-GT is subsampled at the same frames for each stride.

| ONNX input | Analysis rate | HOTA | MOTA | IDF1 | Unique (GT) | Crossings (GT 44) | Detect p50 (CPU) |
|---|---|---|---|---|---|---|---|
| 736×1280 | 57.0 fps (every frame) | 0.430 | 0.444 | 0.495 | 184 (299) | 28 | 34 ms |
| 736×1280 | 28.5 fps | 0.445 | 0.443 | 0.518 | 159 (297) | 25 | 35 ms |
| 736×1280 | 14.3 fps | 0.452 | 0.438 | 0.533 | 143 (295) | 26 | 35 ms |
| 544×960 (Orin Nano default) | 14.3 fps | 0.334 | 0.145 | 0.378 | 118 (295) | 24 | 20 ms |
| 384×640 | 14.3 fps | 0.254 | 0.061 | 0.246 | 72 (295) | 19 | 11 ms |
| Offline research path (PyTorch FP16, every frame) | 57.0 fps | 0.440 | 0.463 | 0.507 | | 27 | |

- The full-rate runtime is within 0.010 HOTA of the offline path. Analysing fewer frames costs nothing here.
- The smaller Jetson inputs cost much more than on aa1 (544×960: −0.12 HOTA here vs −0.02 on aa1), because this
  scene's people are small and the fine-tuned model already struggles with them.
- Latencies are CPU ONNX Runtime on a loaded desktop, not TensorRT on a Jetson.

### Runtime bug found by this test (fixed, `edge_vfr_bug.json`)

The first run gave full-rate HOTA **0.362**, below the stride-2 and stride-4 runs, which made no sense. The
deployed decoder turned out to deliver **17,854 frames instead of 16,970**. Without `source.fps`,
`edgevision/stream.py` added `-fps_mode passthrough` only for stride > 1. At stride 1, FFmpeg's rawvideo output
fell back to constant frame rate and padded this variable-frame-rate video to 60 fps with duplicated frames. File
timestamps were also index / mean fps, which on this video is up to **12.5 s** off the real PTS, because the 45 fps
stretches are clustered.

Fix (2026-10-23): `-fps_mode passthrough` whenever `source.fps` is 0, and file frames get their real media
timestamps from ffprobe (`stream.probe_pts` / `stream.media_ts`, unit-tested). Results after the fix:
- full-rate HOTA 0.362 → 0.430;
- strides 2 and 4 unchanged;
- the aa1 edge table (constant frame rate) re-run and **bit-identical**;
- 283/283 edge tests pass.

The pre-fix log is `logs/heldout_edge_before_vfr_fix.log`.

A resize-filter diagnostic ruled out the other suspect (`resize_sensitivity.json`, `heldout_resize.py`). Scaling
1080p → 720p with bilinear, bicubic, area or FFmpeg's own scaler changes AP50-95 by at most 0.013, for both
YOLO26n models and for the ONNX export. The ONNX export matches PyTorch on this video too (0.408 vs 0.412).

## What this means for the project

1. **The in-domain model selection does not transfer to a new scene.** `select_model.py` picked YOLO26n FT on aa1
   evidence, and the choice holds for the aa1 camera. On a new camera without site-specific labels, the
   COCO-pretrained YOLO26n is the better detector (+0.27 AP50-95, +0.14 HOTA). RF-DETR-N FT is second.
2. **Practical options for a new site:** (a) deploy the pretrained YOLO26n (no labels needed); (b) re-run the
   teacher → student pipeline on footage from that camera (it is automated, and `heldout_detect.py` already
   produces the labels); (c) fine-tune on several scenes, so the nano model cannot specialise to one background.
3. **The overfitting diagnosis is predictive.** The model with the excess train–test gap on aa1 is the one that
   transferred badly.

## Files

- `detection.json`: per-model metrics, CIs, recall by height, label bias, teacher agreement, frozen thresholds.
- `significance.json`: paired tests (block and i.i.d. bootstrap). `pr_data.npz`: PR curves. `preds/*.json.gz`: raw detections.
- `detections_full.npz`: teacher + student boxes for every frame. `detections_yolo26n_pretrained.npz`: the
  pretrained YOLO26n on every frame. `parts/`: the chunked pass.
- `tracking.json`, `tracker_comparison.json`, `tracks.npz` (GT and model tracks, per frame), `analytics.json`.
- `edge.json`, `edge_runs/`: deployed runtime reports, SQLite stores and track dumps. `edge_vfr_bug.json`: the
  runtime bug found and fixed. `resize_sensitivity.json`: downscaling-filter diagnostic.
- `audit/`: regions, contact sheets, verdicts, scores.
- `figures/`: 01 qualitative, 02 transfer + recall by height, 03 PR, 04 tracking, 05 analytics, 06 edge.
- `logs/`: run logs.
- The dataset is in `datasets/test1/`: `frames/` (1,191 JPEGs), `coco/test/_annotations.coco.json`, `teachers.json`
  (per-frame boxes of each teacher), `stats.json`.
