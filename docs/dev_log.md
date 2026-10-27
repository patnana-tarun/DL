# Build progress (handoff file — read this first when resuming)

Request (2026-10-07): (1) finish the project end-to-end, re-verify all metrics; (2) remove unnecessary
files/packages/versions; (3) FFmpeg-stream-based analysis + self-contained Jetson Orin deployment folder with
beginner steps; (4) closed loop: train/compare YOLO26n vs RF-DETR -> select -> analytics -> Qwen VLM, user
chats with the VLM on the Jetson; (5) IEEE double-column paper.

Rules: run Python as `env -u PYTHONPATH .venv/bin/python` (ROS paths pollute PYTHONPATH). **Never delete
files** — the user approves deletions; candidates go in the "Cleanup list" below. Network is slow (~1 MB/s).

## Layout (new)
- `research/` study scripts (paths.py sets cwd = project root), `research/campus/` campus case study,
  `research/run_all.sh` reproduces everything, `research/crosscheck.py` independent metric checks,
  `research/select_model.py` decision matrix + ONNX export + parity check.
- `edge_deploy/` self-contained runtime copied to the Jetson: `edgevision/` package, `configs/`,
  `models/` (ONNX + `vlm/*.gguf`), `scripts/`, `tools/` (stream_videos.py, chat_cli.py, bin/mediamtx),
  `third_party/llama.cpp-vulkan-x64` (desktop VLM server), `tests/`.
- `paper/` IEEE paper. `models/pretrained/` yolo26n.pt, yolo26x.pt. `results/logs/` run logs.

## Status — ALL DONE 2026-10-08; final summary given; only the cleanup list awaits user approval
- [x] Re-ran evaluate.py (4 models; RF-DETR v1 dropped). Numbers reproduce (YOLO 0.825, RF 0.725).
- [x] crosscheck detection: 50/50 pass (results/crosscheck_detection.json).
- [x] latency + teacher_bias re-run.
- [x] track.py, track_compare, analytics, visualize re-run: identical to published numbers.
- [x] crosscheck tracking 4/4 pass (results/crosscheck_tracking.json).
- [x] research/edge_eval.py files: deployed pipeline HOTA 0.746 at stride 1 (= research 0.746); 0.735 @15 fps.
- [x] select_model re-run (robust YOLO26n), edge_eval live RTSP (1602 analysed / 197 dropped), campus probe+reports identical.
- [x] research/edge_demo.py recorded (results/edge/demo/): every change took effect; guardrails added after
      demo failures: intent check (refuses contradicting line args), rule-based fallback for unambiguous commands,
      system footer listing actual changes, history without footer, calibrated risk prompt, describe-on-alert.
      Final: E3b sel 0.91 / args 0.83 / FP 0.06 (fallback not needed on E3 set); E4 re-run; 280 edge tests pass.
- [x] Docs updated (README/architecture addenda, chat_cli --token, results/README demo); 282 edge + 25 research tests pass.
- [x] Paper: E3b, E5, guardrails, demo integrated; rebuilt by main session (8 pp, 0 TBD).
- [x] results/README.md + campus README updated (VLM section points to results/vlm/README.md written by agent C).
- [x] Paper rebuilt with final numbers and checked.
- [x] select_model.py: YOLO26n wins under all weightings; ONNX 736x1280/544x960/384x640 exported, parity ok.
- [x] edgevision package written (stream, detector, tracker, analytics, store, vlm, commands, server, UI).
- [x] Qwen3-VL-2B GGUF + mmproj downloaded; llama-server works (start: edge_deploy/scripts/start_vlm.sh).
- [x] Sub-agent A: Jetson README + scripts done (edge_deploy/README.md, docs/architecture.md, scripts/00-05, systemd/, requirements-*.txt, tools/mediamtx.yml, Dockerfile.jetson).
- [x] Fixed edgevision issues found by agent A (all items below; 273 edge + 25 research tests pass):
      vlm.image in DEFAULTS; live fps refined from arrivals + duration-based unique count in live mode;
      edge_eval store.path absolute (move edge_deploy/results/edge/runs/*.db -> results/edge/runs/);
      /unalert requires arg; /api/heatmap.png; optional server.token auth; GStreamer decoder option
      (nvv4l2decoder for RTSP on Jetson); set_alert value float(); cancel queued VLM jobs on stop;
      start_vlm.sh --no-webui -> --no-ui. Then re-run tests.
- [x] Sub-agent B: tests written (6 runtime bugs fixed); ap_from empty-detections bug fixed in research/evaluate.py.
- [x] Sub-agent C: VLM evaluation E1-E4 done (results/vlm/README.md). Findings: VLM copies any count given
      (even +10 wrong); frame attached to tool-selection call -> 75% spurious tool calls (text-only 6%).
- [x] Evaluation-driven agent redesign in vlm.py DONE; E3b: sel 0.91, args 0.76, FP 0.06; E5: verdicts
      unreliable (informed 1.00/0.00, blind 0.42/0.13) -> default verify_mode=describe. results/vlm/README.md updated.
      Original plan was: two-stage ask (stage 1 text-only tool
      routing, stage 2 image answer without tools); clearer metric/op tool descriptions; verify prompt that does
      not leak the measured value. Then ask agent C to re-run E3 on the deployed SceneAgent.ask path and add E5
      (alert verification: true vs fabricated alerts, value-revealed vs blind prompt).
- [x] Sub-agent D: paper built and updated.
- [x] Top-level README.md, requirements.txt.

## Cleanup — DONE 2026-10-08 (approved by the user)
Removed: __pycache__/, research/campus/campus_attributes.py (its prompt is now inlined in research/vlm_eval.py),
runs/rfdetr_nano/ (v1), runs/yolo26n last.pt + training plots, rfdetr tfevents, uncompressed results/preds/*.json,
results/test_predictions.json, work/, requirements_frozen.txt, campus validation helper scripts/lists;
uninstalled bitsandbytes, accelerate, imageio-ffmpeg. 279 MB freed. Verified afterwards: research imports,
figures, crosscheck 50/50, 25 research + 282 edge tests pass.

## Held-out video test1.mp4 (2026-10-22/23) — request: "use videos/test1.mp4 only for testing, publication standard"
Test-only: no training / tuning / threshold search on it; thresholds from aa1-valid, tracker params from aa1.
Code: `research/heldout/` (heldout_detect -> heldout_eval -> heldout_track [pretrained bytetrack trackers analytics]
-> heldout_edge -> heldout_audit sheets/score -> heldout_figures; heldout_resize = filter diagnostic;
`run_heldout.sh`), tests `research/tests/test_heldout.py` (7). Results + full write-up: `results/test1/README.md`.
Data: `datasets/test1/` (1,191 JPEG frames, COCO labels, per-teacher boxes). Paper: new Sec. "Held-out scene"
(Tables VI/VII, fig_heldout), abstract/intro/discussion/conclusion updated; `make_numbers.heldout()`, `make_figures.fig_heldout()`.
- [x] Teacher + students on all 16,970 frames (VFR, sequential), 9,828 labels / 299 GT identities.
- [x] Detection: ranking reverses. YOLO26n pre 0.684 > RF-DETR-N FT 0.535 > YOLO26n FT 0.412 > RF pre 0.371 (block CIs,
      all paired p<0.001). YOLO26n FT loses small people (R32-64 0.05 vs 0.40) + FPs on bins/benches/trees. Holds
      under both single-teacher label sets.
- [x] Label audit (40 frames, 238 regions, judged by an AI model from crops): labels 79/80 people; missed labels 121/122
      people; 9/36 far detections are real people. F1 shift <= 0.013 -> not a label artefact.
- [x] Tracking (frozen params): HOTA YOLO26n pre 0.581 > RF FT 0.496 > YOLO FT 0.440; RF wins 12/12 tracker x GT.
      Crossings F1 0.89 / 0.83 / 0.70. Unique-ID counts unreliable (errors cancel).
- [x] Paper rebuilt: 10 pages (was 8).
- [x] Edge runtime on test1: the test exposed a VFR BUG in edge_deploy/edgevision/stream.py (stride 1 without
      source.fps -> ffmpeg rawvideo CFR-resampled to 60 fps, 17,854 vs 16,970 frames; file ts = idx/mean fps, up to
      12.5 s off). FIXED: -fps_mode passthrough whenever fps=0 + real PTS via ffprobe (stream.probe_pts/media_ts,
      new unit test). test1 full-rate HOTA 0.362 -> 0.430 (offline 0.440); aa1 edge table re-run: bit-identical;
      283 edge + 32 research tests pass. Resize-filter diagnostic: <= 0.013 AP, not the cause (resize_sensitivity.json).
- [x] Reproducibility: heldout_track bytetrack+analytics re-run -> byte-identical JSON. Paper: 10 pages, 0 TBD,
      no overfull boxes.

## Major revision (2026-10-26) — request: "go through minor_review.txt, rework the project for the best outcome"
Review = major revision (Jetson not measured, pseudo-GT only, 1 extra scene, resolution confound, i.i.d. CIs,
VLM safety untested, repro package). Literature report: `docs/literature_review.md`
(notes in `docs/research_notes/`).
Code (all new, nothing deleted): `research/revision/` (+ `research/tempstats.py`, `research/annotation/`),
`edge_deploy/edgevision/policy.py`, `edge_deploy/tools/bench.py`. Queue: `research/revision/run_revision.sh`
(started automatically after the campus labelling; logs `results/logs/revision/`; DO NOT edit it while it runs).
- [x] Temporal stats: `temporal_ci.py` (CBB/MBB 2.5/5/10 s, Politis-White, (k+1)/(B+1) p) -> results/revision/temporal_ci.json
- [x] Policy layer (RBAC, validation, intent gate, /confirm, rate limits, hash-chained audit, stage-1 isolation,
      data retention) + 32 tests; 315 edge tests pass.
- [x] VLM safety suite E6 (300 cases x model_only / guardrails / policy) `vlm_safety_eval.py` -> results/revision/vlm_safety.json
- [x] Campus scenes labelled (scene_v1/v2, heldout_detect.py) -> multi-scene splits `make_scene_splits.py`
- [x] COCO val2017 (HF parquet) -> replay pool (half 0) + person val (half 1) `prepare_coco_person.py`
- [x] Training variants (80-class head so WiSE is exact): ft80 replay multiscene freeze res640 ignore v1 v2
      (Ultralytics 8.4 nests relative project dirs under runs/detect/: runs/revision/<v> are symlinks to them)
- [x] WiSE sweeps (raw BN best; BN re-estimation hurts), multi-scene eval, resolution sweep + cost_sweep.py, scene tracking
- [x] Shift-aware selection `select_robust.py`: candidates = aa1-adapted models, OOD = campus valid scenes ->
      WiSE ft80 a0.3 (aa1 0.804, test1 0.678 vs original choice 0.412)
- [x] E6 safety: unsafe rejected 0.44 / 0.42 / 0.90 (model / v1 guardrails / policy); sign-in-frame attack 10/10 -> 0/10
- [ ] Human GT: protocol + blind tool + agreement/adjudication/eval ready (`research/annotation/`, 240 det frames +
      2 clips sampled); ANNOTATION BY HUMANS PENDING (user). Jetson run PENDING (no board): `edge_deploy/tools/bench.py`.
- [x] Paper rewritten around reliability: paper/main.tex = \input of paper/sections/*.tex (17 pp incl. appendices
      and refs; no page limit, so the former supplement is now the appendix, sections/appendix.tex; supplement.tex kept,
      not built); submitted version kept as paper/main_v1_submitted.tex. Authors filled (2026-10-27).
- [x] Hardware validation (Jetson + desktop contention bench) removed from the paper by user request (2026-10-27):
      stated as future work in conclusion; sections/jetson.tex is a short scope note. bench_desktop.json kept.
- [ ] AUTHORS must fill video provenance / licence (% AUTHORS comment in sections/privacy.tex).
- Summary of all revision results: results/revision/README.md; per-table commands: REPRODUCE.md.
