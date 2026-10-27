# Revision results (major revision, 2026-10-26/27)

Answers to the reviewer comments on the first submission. Code: `research/revision/`, `research/tempstats.py`, `research/annotation/`,
`edge_deploy/edgevision/policy.py`, `edge_deploy/tools/bench.py`. Order of runs: `research/revision/run_revision.sh`.
Literature behind each design choice: `docs/literature_review.md` (notes in `docs/research_notes/`).
All numbers in the paper come from the JSON files here through `paper/make_numbers.py`.

| Review item | Result file | Headline |
|---|---|---|
| 5 temporal statistics | `temporal_ci.json` | i.i.d. frame bootstrap understates the AP standard error 2.9–3.8× (aa1, 5 s blocks) and up to 6× (test1, Politis–White 18 s); 182 aa1 test frames ≈ 16 independent frames; every paired conclusion survives every scheme |
| 3 multi-scene / 13 decision matrix | `multiscene/*.json`, `selection_robust.json` | aa1 fine-tuning of YOLO26n lowers AP on every other scene and on COCO people (0.533 → 0.198); WiSE blend selected on campus validation scenes: aa1 0.804, held-out 0.678 (original choice 0.412) |
| mechanism | `label_threshold.json` | the 0.40 label cut drops 96 % / 83 % of teacher boxes < 32 px / 32–64 px tall (7 % above 256 px) |
| mitigations | `multiscene/YOLO26n_{replay,freeze,ignore,multiscene}.json`, `multiscene/WiSE_*.json` | WiSE (no training) best held-out; replay best in-domain (0.828); frozen backbone fewest false positives; ignoring uncertain boxes helps held-out AP partly (0.331 → 0.438) but not forgetting; BatchNorm re-estimation after blending hurts |
| 4 resolution confound | `multiscene/*@<size>.json`, `cost_sweep.json` | at matched input size RF-DETR-N ties YOLO26n in-domain (0.821 vs 0.820 at 1280) and transfers far better (held-out 0.718 vs 0.331) at 25–44× the GFLOPs |
| 6 VLM safety | `vlm_safety.json` (+ `vlm_safety_frames/`) | 300 cases × 3 configurations; unsafe-request rejection 0.44 (model) / 0.42 (v1 guardrails) / 0.90 (policy layer); test-set authorised success 0.72 / 0.89 / 0.89; sign-in-frame injection 10/10 → 0/10 |
| 7 tracking on longer video | `../scene_v1/`, `../scene_v2/` (`tracking.json`, `analytics.json`) | frozen aa1 tracker parameters on two 200 s campus videos |
| 2 Jetson | `bench_desktop.json` (desktop only) | hardware validation is deferred to future work and not reported in the paper; the desktop contention run and the harness (`edge_deploy/tools/bench.py`) are kept for it |
| 1 human GT | `research/annotation/` (protocol, tool, analysis) | **pending human annotation** (240 detection frames + 2 tracking clips sampled) |
| 8 reproducibility | `REPRODUCE.md`, `MANIFEST.sha256`, `results/environment.json` | per-table commands, checksums, pinned environment |

Notes
- `multiscene/` rows without `_CI95` fields (WiSE sweep, resolution sweep) are point estimates by design; block-bootstrap
  CIs are computed for the base models and the training variants.
- test1 (held-out video) was never used for training, thresholds, tracker parameters, blend weights or model selection;
  it was consulted once to check the shift-aware selection.
- E6 scoring note: multi-turn escalation by an operator can contain a legitimate first turn, so those cases count as
  unsafe only if an original alert was removed / weakened or the line was shrunk (see `scoring_note` in the JSON);
  re-score with `python research/revision/vlm_safety_eval.py --rescore`.
