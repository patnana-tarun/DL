# VLM stage evaluation: Qwen3-VL-2B in the closed loop

Generated 2026-10-08 by `research/vlm_eval.py` (`env -u PYTHONPATH .venv/bin/python research/vlm_eval.py all`).
The protocol for each experiment is in the script's docstring and in the `protocol` field of each JSON file.

**Setup.** Qwen3-VL-2B-Instruct, `Qwen3VL-2B-Instruct-Q4_K_M.gguf` (SHA-256 `089d75c5…`) plus
`mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf` (SHA-256 `f9a68fab…`). Served by llama.cpp `llama-server` 0.6.0-dev
(build 11476, commit 988190680, Vulkan x64 prebuilt) with `edge_deploy/scripts/start_vlm.sh` (`-ngl 99 -c 8192
--jinja -fa auto`), on an RTX 4060 Laptop GPU. Every call goes through the deployed client
(`edgevision.vlm.VLMClient`, `encode_image`, `SYSTEM`, `TOOLS`). Requests run one at a time at temperature 0.
Model hashes, server build, prompts and every request and response are stored in the JSON files.

**Determinism.** E1 was run three times. Runs 2 and 3 match on all 182 × 6 answers. In run 1 a few `b_sys`
answers differed (MAE 3.19 vs 3.16), so greedy decoding on this server is almost, but not exactly,
reproducible. E2 and E3 gave the same metrics on both of their runs.

## E1: Count grounding (`e1_counting.json`, `fig_e1_counting.png`)

Data: the 182 aa1 test frames. Pseudo-GT counts range from 13 to 24 people (mean 18.9). Images are sent at
`image_max_side` 896. The detector is the deployed YOLO26n ONNX (736×1280) on CPU, at conf 0.24 (the valid
best-F1 threshold read from `results/metrics.json`). Parse failures: 0% in every condition.

| Condition | MAE [95% CI] | RMSE | Bias | Exact | Within ±2 | Pearson r |
|---|---|---|---|---|---|---|
| (a) detector alone | 2.97 [2.60, 3.37] | 4.04 | +2.84 | 0.13 | 0.52 | 0.48 |
| (b) VLM, image only | 7.91 [7.45, 8.36] | 8.50 | −7.91 | 0.02 | 0.06 | −0.35 |
| (b_sys) VLM, image + SYSTEM, no measurements | 3.16 [2.82, 3.50] | 3.96 | −2.29 | 0.12 | 0.48 | 0.19 |
| (c) VLM, image + SYSTEM + detector count | 2.97 [2.60, 3.37] | 4.04 | +2.84 | 0.13 | 0.52 | 0.48 |
| (d) VLM, text only + detector count | 2.97 [2.60, 3.37] | 4.04 | +2.84 | 0.13 | 0.52 | 0.48 |
| (c_off) VLM, image + SYSTEM + **wrong** count (detector + 10) | 12.84 [12.43, 13.25] | 13.16 | +12.84 | 0 | 0 | 0.48 |
| reference: constant 19 (median test GT, an oracle) | 1.87 [1.69, 2.08] | 2.36 | +0.09 | 0.15 | 0.74 | n/a |

- **Grounding fixes VLM counting, but only by copying.** Paired difference MAE(b) − MAE(c) = +4.93 people per
  frame, 95% CI [4.32, 5.61], p < 0.001. In (c), (d) and (c_off) the VLM's answer equals the count it was
  given on 182/182 frames, even when that count is 10 too high and the image disagrees. The VLM does not
  check the number against the image. It is exactly as good as the detector it is fed, and inherits the
  detector's bias: at conf 0.24, YOLO26n over-counts by +2.8.
- **On its own, the VLM cannot count this crowd.** Its image-only answers take only the values 10, 12 or 13
  (plus one 1 and one 7). It undercounts by 8 people, and the answers correlate negatively with GT.
- **Caution with `b_sys`.** Its low MAE is not counting. With the SYSTEM prompt the model answers "about 15"
  (109 frames) or "about 20" (66 frames), as the prompt asks when it estimates. Those round numbers happen
  to sit near this split's mean (r = 0.19).
- **The test split has little variation in counts.** A constant answer of 19 beats every condition. Use E1 to
  compare conditions, not to measure counting accuracy in absolute terms. MAE by crowd size: there are no
  frames with GT ≤ 10. For GT 11–20 (136 frames), MAE is a/c 3.39, b 6.79, b_sys 2.77. For GT > 20
  (46 frames), MAE is a/c 1.74, b 11.20, b_sys 4.41.

## E2: Zero-shot attributes vs trained linear probes (`e2_attributes.json`)

84 labelled campus test tracks have crops. The VLM gets the 3 best crops per track (246 calls) with the
uniform prompt from `campus_attributes.py`, then the answers are majority-voted. Scoring is per track and
matches `campus_probe.evaluate`. The probe's per-track outputs in `attributes.json` reproduce
`probe_scores.json` exactly. 100% of replies were valid JSON. The VLM never answered "unsure" on any crop.

| Attribute | Scope | VLM acc [95% CI] (k/n) | VLM coverage | Probe acc [95% CI] (k/n) | Probe coverage |
|---|---|---|---|---|---|
| Gender | both | 0.79 [0.69, 0.86] (63/80) | 0.99 | 0.80 [0.70, 0.88] (57/71) | 0.88 |
| Gender | video_1 | 0.79 [0.65, 0.89] (34/43) | 0.98 | 0.83 [0.67, 0.92] (29/35) | 0.80 |
| Gender | video_2 | 0.78 [0.63, 0.89] (29/37) | 1.00 | 0.78 [0.62, 0.88] (28/36) | 0.97 |
| Uniform | both | 0.84 [0.74, 0.90] (66/79) | 1.00 | 0.83 [0.73, 0.90] (60/72) | 0.91 |
| Uniform | video_1 | 0.90 [0.78, 0.96] (38/42) | 1.00 | 0.89 [0.75, 0.96] (33/37) | 0.88 |
| Uniform | video_2 | 0.76 [0.60, 0.87] (28/37) | 1.00 | 0.77 [0.61, 0.88] (27/35) | 0.95 |

- **No significant difference** between the VLM and the probe. McNemar test on the tracks both methods
  answered: gender 70 tracks, 11 vs 12 discordant, p = 1.0. Uniform 72 tracks, 2 vs 2 discordant, p = 1.0.
  Because the VLM almost never abstains, it gets more tracks right in absolute terms. With abstentions
  counted as wrong: gender 0.78 vs 0.70, uniform 0.84 vs 0.76.
- **Both methods are close to the majority-class baseline, and the VLM is skewed.** Uniform: the baseline
  ("always uniform") is 0.77, and balanced accuracy is 0.64 for the VLM and 0.65 for the probe. Both find
  only 5 of the 18 non-uniform tracks. Gender: the baseline is 0.59. The VLM answered "male" on 202 of 246
  crops and found 15 of 33 female tracks (balanced accuracy 0.73). The probe found 20 of 33 (balanced
  accuracy 0.80).
- The VLM is zero-shot and uses 3 crops. The probe was trained on 200 hand-labelled tracks and averages up
  to 6 crops.

## E3: Tool-calling reliability (`e3_tools.json`)

45 operator utterances: 29 should call one tool, 16 should call none. Each is sent with the real SYSTEM
prompt, a realistic measurement context in `SceneAgent.context()` format, and TOOLS, in one round.
Condition `with_frame` adds the camera frame, as `SceneAgent.ask` does. Condition `text_only` does not.

| Metric | with frame (deployed) | text only |
|---|---|---|
| Tool-selection accuracy (all 45) [95% CI] | 0.62 [0.48, 0.75] | 0.91 [0.79, 0.97] |
| Selection accuracy on the 29 tool utterances | 0.83 | 0.90 |
| Argument exact match (29 tool utterances) [95% CI] | 0.55 [0.38, 0.72] | 0.62 [0.44, 0.77] |
| Argument match, lenient (> = ≥, kind and name not checked) | 0.66 | 0.76 |
| False-positive tool calls on the 16 no-tool utterances | **0.75** (12/16) | 0.06 (1/16) |
| Missed tool calls | 0 | 0.07 |
| Valid JSON arguments | 1.00 (41 calls) | 1.00 (28 calls) |

- **With the camera frame attached, the model calls tools on chit-chat and on questions about the scene.**
  12 of 16 no-tool prompts triggered a call, for example "Describe the scene" → `get_history(10)` and
  "How many people are in view right now?" → `get_history(10)`. Without the image this drops to 1 of 16.
  In the deployed loop these calls are read-only (`get_history` / `get_recent_events`), so the cost is an
  extra round and latency, not a wrong configuration change.
- **By tool, without the image:** `remove_alert` 4/4, `get_history` 3/3, `set_counting_line` 5/5 selected
  and 4/5 arguments right. A "horizontal line at 75%" came back as a degenerate point, (0, 0.75)–(0, 0.75).
  `set_alert` is the weak spot: 11/13 selected but only 5/13 with the right arguments. Typical errors are
  "over 15" → `>= 15`; "stays longer than 2 minutes" → `occupancy >= 2, for_s 120`; "40 crossings in a
  minute" → `crossings_total`; "loiters" → `loitering <= 5`. `get_recent_events` drops `kind`, and turns
  "loitering in the past 15 min" into `get_history(15)`.
- The server parsed every tool call. None had to be recovered from `<tool_call>` text.

## E4: Latency (`e4_latency.json`, `fig_e4_latency.png`)

The workload is the deployed periodic scene report: SYSTEM + context + OBSERVE prompt, max 256 tokens. Each
row is 10 frames after 2 warm-up calls, with the prompt cache disabled. These are **desktop numbers (RTX 4060
Laptop, Vulkan), not Jetson.**

| Request | Prompt tokens | Prefill ms (server) | Prefill tok/s | Decode tok/s | Output tokens | E2E median | E2E p90 |
|---|---|---|---|---|---|---|---|
| Image, max side 448 | 932 | 216 | 4,312 | 137 | 152 | 1.32 s | 1.51 s |
| Image, max side 672 | 1,072 | 266 | 4,034 | 136 | 164 | 1.48 s | 1.84 s |
| Image, max side 896 (default) | 1,268 | 346 | 3,669 | 134 | 163 | 1.56 s | 1.77 s |
| Image, max side 1280 | 1,740 | 582 | 2,989 | 129 | 162 | 1.96 s | 2.12 s |
| Text-only chat | 762 | 132 | 5,782 | 140 | 19 | 0.27 s | 0.27 s |
| Tool-call turn | 1,406 | 234 | 6,007 | 135 | 55 | 0.64 s | 0.64 s |

- **Decoding dominates image requests** (about 1.1–1.2 s for about 160 tokens). Going from 448 to 1280 px
  adds about 800 image tokens and 0.37 s of prefill.
- The very first request of a new prompt shape can take several seconds (Vulkan warm-up). This is excluded
  by the warm-up calls.
- Client-side JPEG encoding takes about 2 ms.
- `prefill_ms` is the server's figure for the whole prompt. In llama.cpp's multimodal path this includes the
  vision encoder.
- nvidia-smi showed 85–92% GPU utilisation around the run, so other GPU work may have overlapped. The
  numbers are indicative.

## E3b: Tool calling through the redesigned agent (`e3b_tools_deployed.json`)

E3 showed that attaching the frame to the tool-selection request makes the model call read-only tools on
most plain questions. `SceneAgent.ask` was therefore changed to two stages:

1. **Routing.** Text only: the question, the history and the measurements, with the tools offered. Up to
   3 rounds of tool calls are executed.
2. **Answer.** The live frame, the question and a note of the executed calls, with no tools.

The `set_alert` schema also got metric and operator descriptions with examples. The same 45 utterances were
sent through the real agent, scoring the tool calls it actually **executed**, with a fresh agent per
utterance and temperature 0:

| Metric | E3 single stage + frame (old) | E3 text only | **E3b deployed two-stage** |
|---|---|---|---|
| Tool-selection accuracy (45) | 0.62 | 0.91 | **0.91** [0.79, 0.97] |
| Argument exact match (29 tool utterances) | 0.55 | 0.62 | **0.76** [0.58, 0.88] |
| Argument match, lenient | 0.66 | 0.76 | **0.86** |
| False tool calls on 16 no-tool questions | 0.75 | 0.06 | **0.06** |
| Missed tool calls | — | — | 0.03 |
| Median turn latency (desktop) | — | — | 1.26 s (2.7 requests per turn) |

The redesign removes the spurious tool calls and raises argument accuracy by 0.21. The cost is one extra
text-only request per turn, about 0.3 s on the desktop.

## E5: Can the VLM verify an alert? (`e5_verify.json`)

Data: 60 test frames, each with a TRUE occupancy alert (threshold = GT − 5, measured = GT) and a FABRICATED
one (threshold = GT + 8, claimed measured = GT + 10, as if the detector over-counted). `SceneAgent.verify`
was run with a yes/no verdict prompt in two modes:

| Mode | Confirms true alerts (sensitivity) | Rejects fabricated alerts (specificity) | Balanced accuracy |
|---|---|---|---|
| informed: sees the measured value and the measurements | 1.00 [0.94, 1.00] | **0.00** [0.00, 0.06] | 0.50 |
| blind: image only | 0.42 [0.30, 0.54] | 0.13 [0.07, 0.24] | 0.28 |

- **Informed mode** confirms everything, the same copy behaviour as E1.
- **Blind mode** is worse than chance. A 2B model cannot judge "more than N people" from the image (E1
  image-only MAE ≈ 8).
- An earlier run of this experiment had a harness bug: the measurement context was a fixed snapshot (14
  people) that contradicted the fabricated alerts. That run showed an artificial specificity of 0.88. The
  numbers above use context consistent with each alert, as in deployment.

## Design decisions taken from E1–E5

- **Counts, crossings and alert rules come only from the detector and tracker.** The VLM never overrides a
  number. It is told to use the measurements, and it does (E1).
- **On an alert, the VLM describes rather than verifies.** The default `vlm.verify_mode: describe`
  attaches a VLM scene report to the alert (`alert_context` event) and gives no yes/no verdict. The verdict
  modes (`blind`, `informed`) are kept as opt-in research options with the E5 numbers above.
- **Chat uses two stages:** text-only tool routing, then an answer with the image (E3 → E3b).
- **The VLM's value** is in what the numbers cannot say: what people are doing, describing anomalies, and
  natural-language access to and control of the analytics. Its attribute judgements are comparable to
  trained probes (E2).

## Limitations

- **Pseudo-GT counts.** E1's ground truth is the teacher ensemble, not human counts, and the test split
  covers a narrow range (13–24 people). The ≤ 10 stratum is empty.
- **Assistant-made labels.** E2's hand labels were made by an AI model from the crops, not by
  independent annotators. With 79–81 scorable tracks, the confidence intervals are about ±0.1.
- **E5 covers count-threshold alerts only.** Verification of qualitative events (falls, fights) was not
  tested.
- **Hand-written utterance set.** E3 has 45 utterances written for this test. Acceptable arguments were
  fixed before the runs, but the set is small and is not a benchmark.
- **One model, one quantisation.** Results are for a single model size and quantisation (2B, Q4_K_M) and
  one server build, at temperature 0. The 4B/8B models were not evaluated.
- **Desktop GPU.** E4 latency is from a desktop GPU with possible contention. Jetson Orin latency will be
  higher and has to be measured on the device.
