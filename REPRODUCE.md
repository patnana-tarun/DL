# Reproducing every number, table and figure

Everything in the paper is generated from `results/**/*.json` by `paper/make_numbers.py` (macros) and
`paper/make_figures.py` (figures); `bash paper/build.sh` rebuilds the PDF. This file maps each result to the
command that produces it. Run Python as `env -u PYTHONPATH .venv/bin/python` (keeps foreign site-packages out).

## 0. Environment and integrity

| Item | How |
|---|---|
| Python packages (exact versions) | `requirements.txt` (pinned); full `pip freeze` in `results/environment.json` |
| Edge runtime packages | `edge_deploy/requirements-*.txt`, `edge_deploy/Dockerfile.jetson` |
| Checksums of videos, weights, datasets, labels | `MANIFEST.sha256` — verify with `sha256sum -c MANIFEST.sha256` |
| Regenerate manifest + environment record | `python research/revision/manifest.py` |
| Seeds | training seed 0; bootstrap / sampling seeds 0; VLM temperature 0 (greedy) |
| Determinism | Ultralytics `deterministic=False` (cuDNN autotune): retraining reproduces AP to about ±0.005, not bit-exactly; evaluation, bootstrap, tracking and analytics are seeded or deterministic (a re-run of the held-out tracking and analytics produced byte-identical JSON) |
| Hardware used | RTX 4060 Laptop 8 GB, i7-13650HX, 24 GB RAM. No Jetson measurements are reported |

Inputs that are not generated: `videos/*.mp4` (see the paper's data statement for provenance and licence),
`models/pretrained/yolo26{n,x}.pt` (Ultralytics), RF-DETR checkpoints (downloaded by `rfdetr`), Qwen3-VL-2B GGUF
(`edge_deploy/scripts/03_download_models.sh`), COCO val2017 (Hugging Face mirror `detection-datasets/coco`, files
`data/val-0000{0,1}-of-00002-*.parquet`, into `datasets/coco2017_hf/`).

## 1. Original study (aa1, held-out video, edge, VLM) — `research/run_all.sh`

| Result | Command (logs in `results/logs/`) |
|---|---|
| Pseudo-labels, splits | `python research/prepare_data.py` |
| Students | `python research/train_yolo.py`, `python research/train_rfdetr.py` |
| Detection table, CIs, significance | `python research/evaluate.py`; independent re-check `python research/crosscheck.py detection` |
| Latency variants | `python research/benchmark_latency.py` |
| Label bias | `python research/teacher_bias.py` |
| Tracking, 6 trackers x 2 GT linkings | `python research/track.py detect tune render`, `python research/track_compare.py`, `python research/crosscheck.py tracking` |
| Decision matrix + ONNX export | `python research/select_model.py` |
| Streamed runtime | `python research/edge_eval.py` |
| VLM E1–E5 | `python research/vlm_eval.py` (VLM server: `edge_deploy/scripts/start_vlm.sh`) |
| Held-out video test1 (test only) | `research/heldout/run_heldout.sh videos/test1.mp4 test1` |
| Campus case study | `research/campus/*.py` (see `results/campus/README.md`) |

## 2. Revision experiments — `research/revision/run_revision.sh`

| Result | Command | Output |
|---|---|---|
| Block-bootstrap sensitivity (2.5/5/10 s, Politis–White, corrected p) | `python research/revision/temporal_ci.py` | `results/revision/temporal_ci.json` |
| Campus scenes: teacher labels for every frame | `python research/heldout/heldout_detect.py --video videos/video_N.mp4 --name scene_vN` | `datasets/scene_vN/`, `results/scene_vN/` |
| Campus temporal splits | `python research/revision/make_scene_splits.py scene_v1 scene_v2` | `datasets/scene_vN/{yolo,splits}/` |
| COCO replay pool + person validation set | `python research/revision/prepare_coco_person.py` | `datasets/coco_replay/`, `datasets/coco_person_val/` |
| Label-threshold diagnostic + "ignore" labels | `python research/revision/aa1_ignore.py` | `results/revision/label_threshold.json` |
| Fine-tuning variants | `python research/revision/train_variants.py ft80 replay freeze res640 multiscene v1 v2 ignore` | `runs/revision/<variant>/` |
| WiSE-FT blends (raw / re-estimated BN) | `python research/revision/wise.py --ft runs/revision/ft80/weights/best.pt --tag ft80` | `runs/revision/wise/` |
| Every model on every scene | `python research/revision/multiscene_eval.py --set base|wise|variants|ressweep` | `results/revision/multiscene/*.json` |
| Shift-aware model selection | `python research/revision/select_robust.py` | `results/revision/selection_robust.json` |
| Export of the selected model for the runtime (ONNX, parity check on aa1 and test1) | `python research/revision/export_selected.py` | `results/revision/export_selected.json`, `edge_deploy/models/yolo26n_wise_person_*.onnx` |
| Campus-scene tracking (frozen aa1 parameters) | `python research/heldout/heldout_track.py ... --name scene_vN --line 0.5,0.0,0.5,1.0` | `results/scene_vN/tracking.json` |
| VLM safety suite E6 | `python research/revision/vlm_safety_eval.py` (VLM server up) | `results/revision/vlm_safety.json` |
| Desktop contention benchmark (not reported in the paper; hardware validation is future work) | `edge_deploy/tools/bench.py` (see `run_revision.sh`) | `results/revision/bench_desktop.json` |
| Human ground truth (needs two annotators) | `research/annotation/PROTOCOL.md` | `results/revision/human_eval_det_main.json` |

## 3. On a Jetson (future work; not part of the paper)

```
cd edge_deploy && scripts/00_check_system.sh && scripts/01_install_deps.sh && scripts/02_build_llamacpp.sh
scripts/03_download_models.sh && scripts/04_warmup_detector.sh && scripts/start_vlm.sh &
python tools/bench.py --config configs/jetson.yaml --video ../videos/aa1.mp4 --duration 1800 --repeats 3
```
Record the nvpmodel mode by name, `jetson_clocks` state, L4T release and ambient temperature (the harness stores
what it can read). Put the JSON in `results/jetson/` and rebuild the paper.
