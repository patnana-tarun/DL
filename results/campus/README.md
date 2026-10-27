# Campus video analytics: video_1.mp4 and video_2.mp4

People tracking, people counting, gender, and uniform vs non-uniform, for two
~200 s phone videos of the Amrita campus (1920×1080, variable frame rate ≈ 52 and 49 fps).
Generated 2026-10-01. No VLM was used: the attribute models are trained on hand labels, using only
weights already on disk.

## Pipeline

| Step | Script | What it does |
|---|---|---|
| 1. Detect + track | `campus_track.py` | YOLO26x at 1280 px (the teacher from the aa1 study; more accurate on a new domain than the fine-tuned YOLO26n, which was fitted to Times Square). BoT-SORT with ReID + motion compensation, 3 s buffer. Keeps the 6 best crops per track (tall, confident, not cut by the frame edge, not overlapped by another person). |
| 2. Features | `campus_features.py` | DINOv2 backbone of RF-DETR Large (already on disk). Per crop: band-pooled whole-body embedding and HSV colour histograms. |
| 3. Attributes | `campus_probe.py` | One logistic-regression head per attribute, trained on 200 hand-labelled tracks. The feature set and regularisation per head are chosen by 5-fold CV on the training tracks only. |
| 4. Count + report | `campus_report.py` | Drops flicker tracks, rejoins broken tracks into people, counts, aggregates attributes per person (mean probability over all its crops), and writes the dashboard, CSV and annotated video. |

Step 3 originally had a Qwen3-VL-8B alternative; it was never run (download cancelled) and is superseded by the on-device Qwen3-VL-2B evaluation in `results/vlm/`.

## Results

| | video_1 | video_2 |
|---|---|---|
| Unique people (estimate, see caveats) | **257** | **307** |
| People in frame: mean / peak | 8.5 / 16 | 8.5 / 18 |
| Line crossings (vertical line at the image centre) | 73 (71 right→left, 2 left→right) | 92 (87 right→left, 5 left→right) |
| Median time in view | 3.6 s | 3.1 s |
| Raw track IDs → kept tracks → people | 431 → 307 → 257 | 670 → 458 → 307 |
| Gender (male / female / unsure) | 143 / 78 / 23 | 117 / 144 / 25 |
| Dress (uniform / non-uniform / unsure) | 191 / 38 / 15 | 263 / 15 / 8 |

Attribute counts are over people with at least one usable crop (244 and 286). People seen only far away
or cut off by the frame edge have no crop and no attributes.

Per video: `summary.json`, `people.csv` (one row per person: track IDs, first/last seen, time in view, every
attribute with its confidence), `dashboard.png`, `annotated.mp4` (1280×720; box colour: green = uniform,
orange = non-uniform, grey = unsure; label = person #, gender, dress; HUD counters).

## Accuracy (held-out)

Hand labels are in `validation/`. **Test set**: 50 random tracks per video (`*_labels.tsv`). **Training set**:
another 100 tracks per video (`*_train_labels.tsv`), disjoint from the test set. Anything the crops did not
show clearly is labelled "?" and not scored. The model abstains (reports "unsure") when its probability is
within 0.35–0.65. Coverage is the share of scorable test people it answers.

"Per person" is the final output: test tracks are mapped to their stitched person and scored once each.
This matters because the seated guard alone owns 7 of the 100 random test tracks.

| Attribute | Per person, accuracy [95% CI] | Coverage | Per track | Training CV |
|---|---|---|---|---|
| Uniform vs non-uniform | **0.90** [0.80, 0.95] (61/68) | 93% | 0.83 (60/72) | 0.93 |
| Gender | **0.87** [0.76, 0.93] (58/67) | 88% | 0.80 (57/71) | 0.93 |

Full numbers, including each test error: `validation/probe_scores.json`.

## Caveats

- **Unique people is an estimate, not an exact count.** Tracks break when someone is occluded. Broken tracks
  are rejoined only when motion, box size, clothing colours and the predicted gender/dress all agree; a
  motion-only rule joined different people about a third of the time. Distant people are too blurry to
  confirm every rejoin, so a person seen far away and then close up can still be counted twice. **Line
  crossings are the more robust people counter.**
- **Stationary people.** A security guard seated behind plants in video_2 was split into 38 tracks. A
  static-person pass merges these (a still ≥5 s anchor, same spot, same clothes), and people who never move
  a body height are excluded from line crossings.
- **Gender** is predicted from appearance: the uniform style is gendered here (beige vest = women's
  uniform). Treat it as an aggregate statistic, not an identity claim about any individual.
- **No ID-card analytics.** ID card detection and lanyard colour were removed (2026-10-01). The hand-label
  files still have `id_card` / `lanyard` columns, but nothing reads them.
- **Glare.** video_2 is filmed from ground level into the sun: more abstentions and lower per-track accuracy.
- **Hand labels were made by an AI model from crops.** They are not independent annotator ground truth.
- **Frame seeking.** The videos have variable frame rate, so `cv2` frame seeking is inexact. All scripts
  read frames sequentially.

## Reproduce

```
.venv/bin/python research/campus/campus_track.py --video videos/video_1.mp4        # and video_2
HF_HUB_OFFLINE=1 .venv/bin/python research/campus/campus_features.py --name video_1 # and video_2
.venv/bin/python research/campus/campus_probe.py
.venv/bin/python research/campus/campus_report.py --name video_1 --video videos/video_1.mp4   # and video_2
```

Re-run 2026-10-08 (probe + report from the cached tracks and features): identical counts and accuracies.
A zero-shot comparison with the Qwen3-VL-2B model of the edge system is in `results/vlm/` (experiment E2).
