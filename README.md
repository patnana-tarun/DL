# Pseudo-Labels, Scene Shift and Tool-Using VLMs

**A reliability study of an edge-oriented crowd-analytics pipeline.** Code, results and paper.

*Korukonda L K M Prem Chand · Patnana Tarun · K Somasundaram*

This repository turns unlabelled pedestrian video into a closed-loop crowd-analytics system and then asks
how far each stage can be trusted:

1. **Pseudo-labelling.** A teacher ensemble (RF-DETR-L + YOLO26x, TTA, weighted box fusion) labels one camera.
2. **Students.** Two real-time detectors, **YOLO26n** and **RF-DETR Nano**, are fine-tuned on those labels and
   compared for detection, tracking, latency and robustness to new scenes.
3. **Runtime (EdgeVision).** The selected detector, exported to ONNX, runs in a streaming pipeline:
   FFmpeg input (file, RTSP/UDP/SRT, USB or CSI camera) → ByteTrack → deterministic analytics and alert rules → SQLite.
4. **VLM operator.** A local **Qwen3-VL-2B** (llama.cpp) reports on the scene, verifies alerts, answers questions
   and reconfigures the analytics through tool calls, behind a deterministic **policy layer**.

```
            offline  (research/)                                   online  (edge_deploy/)
 video ─► teacher ensemble ─► pseudo-labels ─► fine-tune        camera / RTSP / file ─► FFmpeg decoder
          RF-DETR-L + YOLO26x   (WBF, temporal   YOLO26n,                    │
          (TTA)                  splits)         RF-DETR Nano                ▼
                                                    │              YOLO26n (ONNX / TensorRT)
                                 evaluate · cross-check · track             │
                                 latency · label bias · scene shift         ▼
                                                    │              ByteTrack ─► analytics + alerts ─► SQLite
                                                    ▼                   │            ▲      │
                                             select_model.py ── ONNX ───┘   policy   │      │ numbers + frame
                                                                            layer ───┤      ▼
                                                                operator ◄─────────► Qwen3-VL-2B
                                                                (web UI / terminal)   (llama.cpp)
```

## Key findings

All numbers are AP50-95 unless stated; confidence intervals and per-scene tables are in [`results/`](results/).

| Finding | Evidence |
|---|---|
| Single-camera fine-tuning helps in-domain but hurts everywhere else | YOLO26n: aa1 **0.760 → 0.825**, held-out video **0.684 → 0.412**, COCO people **0.533 → 0.198** |
| Part of the loss is a size-blind label threshold | the 0.40 cut drops 96 % / 83 % of teacher boxes under 32 px / 32–64 px tall |
| Weight-space ensembling (WiSE) recovers it without training | blend chosen on *other* cameras: aa1 **0.804**, held-out **0.678** |
| The CNN's in-domain lead was an input-size effect | at matched size RF-DETR Nano ties YOLO26n in-domain (0.821 vs 0.820) and transfers far better (0.718 vs 0.331); YOLO26n is 25–44× cheaper in FLOPs |
| Frame-level bootstrap CIs are too narrow on video | circular-block intervals are 2.9–3.8× wider; no conclusion changes |
| The 2B VLM is not safe on its own; the policy layer is | 300-case adversarial suite: unsafe-request rejection **0.42 → 0.90** at unchanged task success; sign-in-frame prompt injection 10/10 → 0/10 |

Timings are from a desktop GPU (RTX 4060 Laptop); on-device Jetson validation is future work.

## Repository layout

| Path | Contents |
|---|---|
| [`research/`](research/) | The study, run in order by `run_all.sh`: data prep → training → evaluation → cross-checks → latency → label bias → tracking → analytics → model selection → edge/VLM evaluation |
| [`research/revision/`](research/revision/) | Revision experiments: multi-scene evaluation, fine-tuning variants, WiSE blends, shift-aware selection, block bootstrap, VLM safety suite (`run_revision.sh`) |
| [`research/heldout/`](research/heldout/) | Test-only evaluation on a held-out video (`run_heldout.sh`) |
| [`research/campus/`](research/campus/) | Campus case study: tracking, counting, attribute probes |
| [`research/annotation/`](research/annotation/) | Blind human-annotation protocol, browser tool and agreement analysis |
| [`edge_deploy/`](edge_deploy/) | **EdgeVision**, the deployable runtime: `edgevision/` package, configs, ONNX detectors, scripts, tests. See its [README](edge_deploy/README.md) |
| [`paper/`](paper/) | IEEE paper (`main.tex`, `main.pdf`, `supplement.pdf`). Every number is generated from `results/` by `make_numbers.py` |
| [`results/`](results/) | Every metric, figure and log behind the paper, with a README per study |
| [`docs/`](docs/) | Literature review, research notes, development log |
| [`REPRODUCE.md`](REPRODUCE.md) | Command for every table and figure |
| [`MANIFEST.sha256`](MANIFEST.sha256) | Checksums of all inputs and artefacts |

## Getting started

Requires Python 3.12, an NVIDIA GPU with CUDA 13 for training, and FFmpeg.

```bash
git clone git@github.com:patnana-tarun/DL.git && cd DL
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

# tests
.venv/bin/python -m pytest research/tests
cd edge_deploy && ../.venv/bin/python -m pytest -m "not slow" && cd ..
```

On a machine with ROS installed, prefix Python commands with `env -u PYTHONPATH`.

### Live demo (desktop)

Stream a video like an IP camera, run the pipeline and the VLM, then open <http://localhost:8000>:

```bash
cd edge_deploy
scripts/03_download_models.sh                                     # Qwen3-VL-2B GGUF
scripts/start_vlm.sh &                                            # llama-server on :8080
../.venv/bin/python tools/stream_videos.py ../videos/aa1.mp4 &    # rtsp://127.0.0.1:8554/cam1 (MediaMTX)
../.venv/bin/python -m edgevision --config configs/desktop.yaml
../.venv/bin/python tools/chat_cli.py                             # optional terminal chat
```

### Reproducing the paper

```bash
research/run_all.sh                       # original study
research/revision/run_revision.sh         # revision experiments
bash paper/build.sh                       # numbers, figures and PDF from results/ (needs tectonic)
```

[`REPRODUCE.md`](REPRODUCE.md) maps each table and figure to its command.

### Jetson Orin

[`edge_deploy/README.md`](edge_deploy/README.md) is a step-by-step guide: JetPack 6, dependencies, building
llama.cpp with CUDA, downloading the VLM, connecting a camera, the web UI, and running at boot with systemd.

## Data and weights

Large, downloadable or privacy-sensitive files are not in the repository. It contains the code, configs, paper,
every metric and figure, and the deployable ONNX detectors in `edge_deploy/models/`. To reproduce from scratch:

| Path | How to get it |
|---|---|
| `videos/*.mp4` | Input videos; not redistributed (see the paper's data statement). Checksums in `MANIFEST.sha256` |
| `models/pretrained/yolo26{n,x}.pt`, `weights/` | Ultralytics release weights, downloaded by `ultralytics` on first use |
| `datasets/` | Regenerated from the videos (`research/prepare_data.py`, `research/heldout/heldout_detect.py`); COCO val2017 as in `REPRODUCE.md` |
| `runs/` | Training runs: `research/train_*.py`, `research/revision/train_variants.py`, `research/revision/wise.py` |
| `edge_deploy/models/vlm/` | `edge_deploy/scripts/03_download_models.sh` |
| `edge_deploy/third_party/`, `edge_deploy/tools/bin/` | llama.cpp (`scripts/02_build_llamacpp.sh`, or `llama-server` on `PATH`) and MediaMTX (GitHub releases, or `mediamtx` on `PATH`) |
| `results/**/preds/`, `*.npz`, `*.db`, `results/logs/revision/` | Raw prediction dumps and long logs, regenerated by the scripts |
| `results/campus/*/crops/`, `research/annotation/tasks/*/images/` | Images of people, withheld for privacy; regenerated from the videos |

## Status

- Ground truth is a teacher-ensemble pseudo-GT, cross-checked with independent metric code and audited visually.
  Human annotation follows [`research/annotation/PROTOCOL.md`](research/annotation/PROTOCOL.md) and is pending.
- On-device Jetson measurements are future work; the benchmark harness is `edge_deploy/tools/bench.py`.

## Citation

```bibtex
@misc{premchand2026pseudolabels,
  title  = {Pseudo-Labels, Scene Shift and Tool-Using {VLMs}: A Reliability Study of an
            Edge-Oriented Crowd-Analytics Pipeline},
  author = {Prem Chand, Korukonda L K M and Tarun, Patnana and Somasundaram, K},
  year   = {2026},
  note   = {\url{https://github.com/patnana-tarun/DL}}
}
```
