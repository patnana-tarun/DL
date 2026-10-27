# On-device benchmarking methodology for NVIDIA Jetson Orin (detector + ByteTrack + small VLM)

Scope: how to measure a concurrent YOLO-nano (TensorRT / ONNX Runtime) + ByteTrack + small-VLM (Qwen3-VL-2B via llama.cpp CUDA) pipeline on Orin Nano 8GB / Orin NX / AGX Orin, plus published reference numbers to quote as external estimates. Research date: 2026-10-26. Every number below is someone else's measurement, not ours. Each one comes with its JetPack version, power mode and precision where the source gives them. When a source leaves one out, the entry says "not stated".

---

## Q1. Tooling: tegrastats, jtop, nvpmodel/jetson_clocks, power rails, unified memory, trtexec, Nsight

### Takeaway
On Orin Nano and NX, the on-module INA3221 reports three rails: VDD_IN (total module power), VDD_CPU_GPU_CV and VDD_SOC. tegrastats logs them in mW, both instantaneous and averaged, at a configurable interval. These rails are what J/frame should be computed from: use VDD_IN for module energy. Power modes are set with `nvpmodel`, and their IDs and clocks differ per module, so the paper must report the exact mode ID and name. The thermal and over-current throttle thresholds are documented and can be checked during sustained runs.

### Cited Findings
**tegrastats fields (Jetson Linux r36.4 docs)**
- `RAM X/Y (lfb NxZ)`: RAM used and total in MB, plus the largest free block size and count. `SWAP X/Y (cached Z)`. `IRAM` is video-engine-local memory. `CPU [X%,...]@Z` gives per-core load relative to the current frequency, and a powered-down core shows `off`. `EMC X%@Y` is external memory controller bandwidth % and frequency. `GR3D_FREQ X%@[Y1,Y2]` is GPU activity %, with one frequency per GPC (AGX Orin has two GPCs, each with its own frequency controller). Also VIC_FREQ, APE, `MTS fg/bg`, temperatures as `NAME@XXC` from thermal zones, rails as `VDDX cur/avg` in mW, and NVENC/NVDEC/NVDLA stats. — [NVIDIA Jetson Linux r36.4, Tegrastats Utility](https://docs.nvidia.com/jetson/archives/r36.4/DeveloperGuide/AT/JetsonLinuxDevelopmentTools/TegrastatsUtility.html)
- Options: `--interval <ms>` (default 1000 ms), `--logfile <file>`, `--verbose`, `--stop`. — [Tegrastats Utility](https://docs.nvidia.com/jetson/archives/r36.4/DeveloperGuide/AT/JetsonLinuxDevelopmentTools/TegrastatsUtility.html)

**Power rails / INA3221 (Jetson Linux r36.4, Platform Power and Performance)**
- Orin NX and Orin Nano: one INA3221 at I2C 0x40, sysfs `/sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon<x>`. Ch1 `VDD_IN` is total module power, Ch2 `VDD_CPU_GPU_CV` covers the CPU, GPU and CV cores, and Ch3 `VDD_SOC` is the SoC core. — [NVIDIA r36.4 Platform Power & Performance (Orin)](https://docs.nvidia.com/jetson/archives/r36.4/DeveloperGuide/SD/PlatformPowerAndPerformance/JetsonOrinNanoSeriesJetsonOrinNxSeriesAndJetsonAgxOrinSeries.html)
- AGX Orin: two INA3221s. 0x40 carries `VDD_GPU_SOC`, `VDD_CPU_CV`, `VIN_SYS_5V0` and a SUMMATION channel. 0x41 carries `VDDQ_VDD2_1V8AO`. NVIDIA warns not to write any INA3221 sysfs node because it can damage the device. — [same](https://docs.nvidia.com/jetson/archives/r36.4/DeveloperGuide/SD/PlatformPowerAndPerformance/JetsonOrinNanoSeriesJetsonOrinNxSeriesAndJetsonAgxOrinSeries.html)
- Community caveats: one Orin Nano user saw VDD_IN, VDD_CPU_GPU_CV and VDD_SOC fail to add up to the configured 15 W budget ([NVIDIA forum](https://forums.developer.nvidia.com/t/jetson-orin-nano-power-gui/258454)). VDD_IN is total module power, so the sub-rails are not additive components of it. A user also reported that the INA3221 / tegrastats sampling is too slow to capture fast power transients compared with an external analyser ([NVIDIA forum](https://forums.developer.nvidia.com/t/externally-measuring-current-draw-power/372739)). On an Orin NX, a reported channel-4 sum did not match expectations ([NVIDIA forum](https://forums.developer.nvidia.com/t/jetson-orin-nx-power-management-parameters/280460)).

**nvpmodel power modes (r36.4; CPU/GPU max MHz)**
- Orin Nano 8GB: 15W = ID 0 (default; 6 CPUs @1510.4, GPU 4 TPC @624.75), 7W = ID 1 (4 CPUs @960, GPU 2 TPC @408), MaxN = ID 2 (6 CPUs @1728, GPU @1020). MaxN (the "Super", 25 W) mode is "available only when flashing with the jetson-orin-nano-devkit-super configurations". With the devkit-super flash, 15W gives 1497.6 CPU / 612 GPU. — [r36.4 Platform Power & Performance](https://docs.nvidia.com/jetson/archives/r36.4/DeveloperGuide/SD/PlatformPowerAndPerformance/JetsonOrinNanoSeriesJetsonOrinNxSeriesAndJetsonAgxOrinSeries.html)
- Orin NX 16GB: MaxN = ID 0 (8 CPUs @1984, GPU 4 TPC @918), 10W = ID 1, 15W = ID 2 (default; 4 CPUs @1420.8, GPU 2 TPC @612), 25W = ID 3 (8 CPUs @1497.6, GPU 4 TPC @408). Orin NX 8GB: MaxN GPU @765, 20W = ID 3. — [same](https://docs.nvidia.com/jetson/archives/r36.4/DeveloperGuide/SD/PlatformPowerAndPerformance/JetsonOrinNanoSeriesJetsonOrinNxSeriesAndJetsonAgxOrinSeries.html)
- AGX Orin 32GB: MaxN = ID 0 (8 CPUs @2188.8, 7 TPC @930.75), 15W = ID 1, 30W = ID 2 (default), 40W = ID 3. AGX Orin 64GB uses the same ID layout with 15W / 30W or 35W (default) / 50W or 60W. — [same](https://docs.nvidia.com/jetson/archives/r36.4/DeveloperGuide/SD/PlatformPowerAndPerformance/JetsonOrinNanoSeriesJetsonOrinNxSeriesAndJetsonAgxOrinSeries.html)
- Note: the mode IDs are not consistent across modules. ID 0 is 15W on the Nano but MaxN on the NX and AGX, so "`nvpmodel -m 0`" means different things on different boards.

**Throttling thresholds (r36.4)**
- Thermal (AGX Orin, Orin NX/Nano, non-industrial): SW throttle at 99.0 °C, HW throttle at 103.0 °C, SW shutdown at 104.5 °C, HW shutdown at 105.0 °C. Clock throttling is done through the `cpufreq_cooling` / `devfreq_cooling` devices, with the fan as the first line of defense. — [same](https://docs.nvidia.com/jetson/archives/r36.4/DeveloperGuide/SD/PlatformPowerAndPerformance/JetsonOrinNanoSeriesJetsonOrinNxSeriesAndJetsonAgxOrinSeries.html)
- Over-current (SOCTHERM_OC): Orin Nano 8GB throttles at VDD_IN average 15 W (OC2, CPU/GPU to 50%) and instantaneous 20 W (OC3, 87.5%). The devkit-super flash raises this to a 25 W average / 25 W instantaneous budget. Orin NX 16GB: 25 W average (OC2, 50%) and 30 W instantaneous (OC3). AGX Orin 64GB: 60 W TDP, with a 65 W instantaneous limit triggering OC3 at 50%. — [same](https://docs.nvidia.com/jetson/archives/r36.4/DeveloperGuide/SD/PlatformPowerAndPerformance/JetsonOrinNanoSeriesJetsonOrinNxSeriesAndJetsonAgxOrinSeries.html)
- Fan profiles: the Orin NX/Nano default is `quiet`, with `cool` as the alternative. The AGX Orin default is `cool`. — [same](https://docs.nvidia.com/jetson/archives/r36.4/DeveloperGuide/SD/PlatformPowerAndPerformance/JetsonOrinNanoSeriesJetsonOrinNxSeriesAndJetsonAgxOrinSeries.html)

**jetson-stats / jtop**
- Python API: `from jtop import jtop; with jtop() as jetson: while jetson.ok(): print(jetson.stats)`. `jetson.ok()` "will provide the proper update frequency". The README says it "Works with all NVIDIA Jetpack", including Ubuntu 24.04. Advanced usage (power, temperature and memory objects, logging) is documented at rnext.it. — [rbonghi/jetson_stats GitHub](https://github.com/rbonghi/jetson_stats); [advanced usage](https://rnext.it/jetson_stats/advanced-usage.html)

**Ultralytics' recommended setup**
- The Ultralytics Jetson guide recommends `sudo nvpmodel -m 0` (described as MAX power) and `sudo jetson_clocks` before benchmarking, but does not say these were used for its published tables. — [Ultralytics NVIDIA Jetson guide](https://docs.ultralytics.com/guides/nvidia-jetson/)

### Inferences
- **Recommended logging stack:** run `tegrastats --interval 100 --logfile run.log` (or jtop at a similar rate) throughout every run, and record the timestamp of each frame from the application. Energy per frame is ∫VDD_IN dt ÷ frames processed over the same window. Report VDD_CPU_GPU_CV separately as the compute-rail share. Don't add the sub-rails together to get a total; VDD_IN already is the total.
- Because the INA3221 updates slowly, energy should be integrated over long windows (≥60 s) and not attributed to individual frames. Per-frame J is a window average.
- **Unified memory:** the tegrastats `RAM` field is the only memory budget, because the CPU and GPU share it. `nvidia-smi`-style per-GPU memory accounting does not apply. Log RAM, SWAP and the `lfb` largest-free-block value, since fragmentation matters (see Q4).
- trtexec and Nsight Systems: I did not fetch primary docs for these in this pass. The protocol should still use `trtexec --loadEngine=... --warmUp=... --duration=... --percentile=50,95,99 --useSpinWait` for isolated engine latency, and `nsys profile` with CUDA and OS-runtime traces to show kernel interleaving between the TRT and llama.cpp processes. Flag names should be checked against the installed TensorRT version (see Gaps).

### Gaps
- I did not fetch primary NVIDIA docs for `jetson_clocks` (it pins CPU/GPU/EMC to the maximum of the current nvpmodel mode; this is from memory and not cited), trtexec flags, or Nsight Systems on Jetson.
- The VDD_IN accuracy and calibration spec for the INA3221 is not in the sources found. An external USB-C or barrel-jack power meter is advisable to cross-check.
- The jtop API attribute list (`jetson.power`, `jetson.temperature`, `jetson.memory`) was not visible in the fetched README.

---

## Q2. Methodology: warm-up, sustained runs, tail latency, frame drops, J/frame, concurrency, MLPerf reference

### Takeaway
MLPerf Inference Edge rules give a defensible template. Report the 90th-percentile latency for SingleStream and the 99th for MultiStream, use a minimum 600 s run, and run the same code for accuracy and performance. Our protocol should go further: a 30–60 min sustained run to expose thermal and over-current throttling, p50/p95/p99 latency, frame-drop rate against a fixed-rate source, and window-integrated J/frame. Measure detector-only, VLM-only and concurrent configurations in the same power mode.

### Cited Findings
- MLPerf Inference Edge scenarios: SingleStream uses a "90%-ile early-stopping latency estimate", with the next query issued once the previous one completes, 1 sample/query and a minimum duration of 600 s. MultiStream reports the "99%-ile early-stopping latency estimate" with 8 samples/query over 600 s. Offline reports measured throughput with ≥24,576 samples in one query (or the dataset size if smaller), over 600 s. — [MLCommons inference_rules.adoc](https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc)
- Object detection on Edge requires SingleStream, MultiStream and Offline. Language processing requires SingleStream and Offline. — [MLCommons inference rules](https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc)
- "The same code must be run for both the accuracy and performance LoadGen modes". Caching of queries or intermediate results is prohibited. Performance runs sample from the test set with replacement. — [MLCommons inference rules](https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc)
- Inferred results are allowed: Offline ≈ 1000 / mean SingleStream latency (ms). — [MLCommons inference rules](https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc)
- Ultralytics' Jetson tables exclude pre- and post-processing ("Inference time does not include pre/post-processing"). They are model-only latencies, not end-to-end pipeline latencies. — [Ultralytics Jetson guide](https://docs.ultralytics.com/guides/nvidia-jetson/)
- Edge VLM energy is dominated by decode. On a Jetson Orin NX and an RTX 3070, average power stays within about 5% across input conditions, so energy tracks time. Each output token costs about 11–39× the wall-clock time of an input token. Pruning all visual tokens saves at most about 10% of energy, while capping output length saves up to 97%. — [Zhan et al., "Seeing is Free, Speaking is Not", arXiv:2607.09520 (ACM MM '26)](https://arxiv.org/abs/2607.09520)
- Cross-platform VLM benchmarking: HoliBench found that on PaliGemma-3B the AGX Orin was slower than a datacenter GPU but used less energy per inference, so the latency and energy rankings were inverted. — [HoliBench, arXiv:2609.12412](https://arxiv.org/pdf/2609.12412) (seen via search snippet only; details not verified)
- VLA work on AGX Orin measures latency, throughput and memory across power modes from 15 W to 50 W, which is a precedent for sweeping nvpmodel modes. — [Taherin et al., arXiv:2509.11480](https://arxiv.org/html/2509.11480v2)

### Inferences (proposed protocol, our design, not sourced)
1. **Fix and record the configuration:** module and RAM size, carrier, JetPack/L4T version (`cat /etc/nv_tegra_release`), CUDA/TensorRT/llama.cpp commit, nvpmodel mode **name and ID** (`nvpmodel -q`), jetson_clocks on/off, fan profile, ambient temperature, storage (NVMe vs SD), and headless vs GUI.
2. **Warm-up:** build the TRT engines offline, then discard the first N frames or roughly 60 s (CUDA context, TRT lazy init, llama.cpp KV allocation and mmproj load). Report cold-start time (model load, first frame, first VLM token) as a separate metric.
3. **Configurations (same power mode):** (A) detector + ByteTrack only; (B) VLM only, with a fixed prompt set, fixed `n_predict` and fixed image size; (C) A + B concurrent at the paper's real VLM trigger rate; (D) optionally C with the VLM at a lower OS priority (`nice`), or with the detector pinned to cores via `taskset`.
4. **Input:** a fixed-rate source (e.g. a 30 fps file decoded at real time, with no faster-than-real-time reads). A frame counts as dropped if its processing has not started before the next frame arrives, or if it exceeds a deadline (e.g. 33.3 ms). Report drop rate, p50/p95/p99/max end-to-end latency (decode→track output), and detector-only GPU latency (trtexec) separately.
5. **VLM metrics:** image-encode (mmproj/ViT) ms, prefill ms (TTFT), decode tok/s and total latency per query. Fix the output-token cap, since output length dominates energy (arXiv:2607.09520).
6. **Sustained:** 30–60 min per configuration after warm-up, with tegrastats at 100–1000 ms. Plot FPS, latency, GR3D_FREQ, temperature and VDD_IN against time. Report the first 5 min against the last 5 min and flag any throttle events (temperature ≥99 °C, or VDD_IN reaching the OC thresholds in Q1).
7. **Energy:** J/frame = ∫VDD_IN over the window ÷ frames processed. Also report J per VLM query and idle baseline power.
8. **Repeats:** ≥3 runs, reporting mean ± std of p50/p95/p99. Tail-percentile CIs can come from bootstrap resampling, or from the MLPerf-style early-stopping estimate.

### Gaps
- MLPerf power-measurement rules (PTDaemon, the separate submission-rules doc) were not fetched.
- No published 30–60 min thermal-throttling study of YOLO + VLM concurrency on Orin was found.

---

## Q3. Published reference numbers (external estimates)

### Takeaway
On Orin Nano Super, Orin NX 16GB and AGX Orin 64GB, YOLO-nano TensorRT FP16 model-only latency at 640 is about 4.6 / 4.1 / 2.6 ms (Ultralytics, YOLO26 nano rows). NVIDIA TensorRT-Edge-LLM reports Qwen3-VL-2B (INT4 AWQ LLM + FP16 ViT, batch 1, JetPack 7.2) on Orin Nano 8GB at ViT 151 ms, prefill 191 ms and decode 36.9 tok/s, using about 4.4 GB of GPU memory. I found no controlled llama.cpp GGUF Qwen-VL numbers for Orin Nano.

### Cited Findings
**YOLO detector (Ultralytics Jetson guide, current version; model = YOLO26 nano per the page's table heading; 640 input; "Inference time does not include pre/post-processing"; JetPack and power mode not stated for the runs; mAP50-95 on the Ultralytics default benchmark set, likely coco128, not COCO val)** — [Ultralytics NVIDIA Jetson guide](https://docs.ultralytics.com/guides/nvidia-jetson/)

| Device (Ultralytics ver.) | ONNX (ms) | TRT FP32 | TRT FP16 | TRT INT8 | mAP FP16 / INT8 |
|---|---|---|---|---|---|
| Orin Nano Super (8.4.33) | 15.76 | 7.53 | 4.57 | 3.80 | 0.4800 / 0.4490 |
| Orin NX 16GB (8.4.33) | 14.18 | 7.01 | 4.13 | 3.49 | 0.4789 / 0.4489 |
| AGX Orin 64GB (8.4.32) | 9.87 | 4.18 | 2.62 | 2.30 | 0.4789 / 0.4640 |
| AGX Thor (8.4.7) | 6.58 | 1.90 | 1.39 | 1.52 | 0.4797 / 0.4273 |

- INT8 costs about 3 mAP points on Nano and NX in these tables (0.480 → 0.449), with only about 17% latency gain over FP16. — derived from the [Ultralytics tables](https://docs.ultralytics.com/guides/nvidia-jetson/)
- The page currently shows only YOLO26 rows. The YOLO11n and v8n rows from earlier versions are no longer shown. — [Ultralytics guide](https://docs.ultralytics.com/guides/nvidia-jetson/)
- Community: YOLOv8n at 640, Orin Nano Super, MAXN_SUPER, JetPack 6.2.2, TensorRT INT8 ≈ 286 FPS (3.49 ms GPU latency) vs ≈ 48 FPS for PyTorch FP32. — [hokwangchoi/jetson-orin-nano-benchmarks](https://github.com/hokwangchoi/jetson-orin-nano-benchmarks)
- Community: YOLO11n TRT FP16 on Orin Nano (Super), JetPack 6.2, reported at 7–13 ms per inference ([NVIDIA forum](https://forums.developer.nvidia.com/t/expected-fps-for-yolox-s-tensorrt-engine-on-jetson-orin-nano-jetpack-6-2/329875)) and about 11 ms for a YOLO11n TRT engine (JetPack 6.2, devkit) ([Ultralytics community](https://community.ultralytics.com/t/yolov11-c-example-for-jetson-orin-nano-jetpack6-2/785)). These probably include pipeline overhead, unlike the 4.57 ms model-only figure.
- DeepStream: YOLOv8s on Orin NX 16GB, INT8 ≈ 181 FPS. — [Ultralytics DeepStream guide](https://docs.ultralytics.com/de/guides/deepstream-nvidia-jetson/)

**Small VLMs: NVIDIA TensorRT-Edge-LLM v0.9.0 (JetPack 7.2 listed for all devices; LLM INT4 AWQ, ViT FP16; power mode not reported; output length not reported; prefill ms ≈ TTFT proxy)** — [TensorRT-Edge-LLM performance benchmarks](https://nvidia.github.io/TensorRT-Edge-LLM/0.9.1/user_guide/performance/performance-benchmarks.html)

| Device | Model | Batch | Prefill ms (input tok) | ViT ms (ViT tok) | Decode tok/s | GPU MB |
|---|---|---|---|---|---|---|
| Orin Nano 8GB | Qwen3-VL-2B | 1 | 190.9 (292) | 151.1 (265) | 36.9 | 4,380 |
| Orin Nano 8GB | Qwen3.5-2B (VLM) | 1 | 360.4 (296) | 143.2 (265) | 29.9 | 4,621 |
| Orin Nano 8GB | Qwen3.5-0.8B | 1 | 251.9 (296) | 49.2 (265) | 55.3 | 2,603 |
| Orin NX 16GB | Qwen3-VL-2B | 1 | 103.1 (292) | 82.1 (265) | 60.1 | 4,344 |
| Orin NX 16GB | Qwen3-VL-4B | 1 | 244.6 | 83.7 | 31.2 | 5,903 |
| AGX Orin 64GB | Qwen3-VL-2B | 1 | 50.0 (292) | 39.0 (265) | 98.4 | 8,140 |
| AGX Orin 64GB | Qwen3-VL-2B | 8 | 355.8 | 256.8 | 372.3 | 10,246 |

- Text-only LLMs, same source, Orin Nano 8GB: Qwen3-1.7B decodes at 37.4 tok/s (3,234 MB) and Qwen3-0.6B at 72.8 tok/s. — [TensorRT-Edge-LLM](https://nvidia.github.io/TensorRT-Edge-LLM/0.9.1/user_guide/performance/performance-benchmarks.html)
- The source notes that v0.8.0 had a uniform performance regression that v0.9.0 fixes. — [same](https://nvidia.github.io/TensorRT-Edge-LLM/0.9.1/user_guide/performance/performance-benchmarks.html)

**Jetson AI Lab (archived/deprecated page; MLC INT4 for LLMs; units not labelled, presumably tok/s for LLM/VLM and FPS for ViT; JetPack and power mode not stated; "Original" vs "Super" Orin Nano)** — [Jetson AI Lab benchmarks (archive)](https://jetson-ai-lab.com/archive/benchmarks.html)
- VLMs, Orin Nano original → Super: Qwen2-VL 2B 2.8 → 4.4; SmolVLM 2B 8.1 → 12.9; PaliGemma2 3B 13.7 → 21.6; InternVL2.5 4B 2.5 → 5.1; VILA 1.5 3B 0.7 → 1.06; LLaVA 1.6 7B 0.412 → 0.57.
- LLMs: SmolLM2 41 → 64.5; Llama 3.2 3B 27.7 → 43.07; Qwen2.5 7B 14.2 → 21.75.
- ViT: clip-vit-base-patch32 196 → 314; Grounding DINO 4.11 → 6.23; SAM2 base 4.42 → 6.34.
- Caution: the VLM values (e.g. Qwen2-VL 2B at 2.8 → 4.4) are far below the TRT-Edge-LLM decode rates for a similar size class. They probably measure something different, such as end-to-end image queries/s or a different metric. The page does not define the units, so they should not be quoted as tok/s.

**Other**
- Qwen2.5-VL-3B on AGX Orin 64GB: a user got about 30 tok/s, while NVIDIA quoted 216 tok/s and later measured 225.65 output tok/s. The gap likely comes from concurrency/batching and quantization settings, which shows how much "tok/s" depends on setup. — [NVIDIA forum](https://forums.developer.nvidia.com/t/the-token-speed-of-qwen-2-5-vl-3b-model-is-very-lower-on-jeston-agx-orin/345073)
- llama.cpp + LLaVA-1.5-13B 4-bit GGUF with mmproj on AGX Orin: about 10 tok/s. — [Jetson AI Lab LLaVA tutorial](https://www.jetson-ai-lab.com/tutorial_llava.html)
- Orin Nano Super, Qwen2.5 3B under Ollama (llama.cpp backend): 22.6 tok/s. This is a non-controlled review. — [computingforgeeks](https://computingforgeeks.com/edge-llm-jetson-orin-vs-ai-hat-2-vs-hailo-10h/)
- RF-DETR on Jetson: no source found in this pass.

### Inferences
- **Expected detector budget on Orin Nano Super (external estimate):** about 4–5 ms model-only FP16 at 640 suggests ≥200 FPS model-only. A real pipeline with decode, preprocessing, NMS-free postprocessing (YOLO26) and ByteTrack will be lower. Community end-to-end reports of 7–13 ms for YOLO11n suggest roughly 75–140 FPS. Thirty fps for a single stream at 640 has large headroom.
- **Expected VLM cost on Orin Nano 8GB (external estimate, TRT-Edge-LLM):** one Qwen3-VL-2B query with about 265 image tokens and a 32-token answer ≈ 151 ms ViT + about 191 ms prefill (the prefill figure probably already covers the visual tokens) + 32/36.9 s ≈ 0.87 s decode. That is roughly 1.1–1.3 s per query when running alone. llama.cpp is typically slower than TRT-Edge-LLM, so our numbers may be worse, which needs measuring.
- Scaling between devices for Qwen3-VL-2B decode is Nano 36.9 → NX 60.1 → AGX 98.4 tok/s. That is about ×1.6 per step, consistent with memory bandwidth limits.

### Gaps
- No controlled llama.cpp (GGUF Q4/Q8 + mmproj) Qwen2-VL, Qwen2.5-VL or Qwen3-VL numbers on Orin Nano were found. A Chinese-language guide tested Qwen2.5-VL-3B Q4_K_M with mmproj-f16 on Orin Nano Super, but its numbers were not visible ([CSDN](https://deepseek.csdn.net/6a9c19b348977663a5dd4b96.html), not fetched).
- No YOLO results above 640 (e.g. 960/1280) on Orin were found.
- No RF-DETR Jetson numbers were found.
- The Ultralytics tables don't state power mode or JetPack, so they should be cited as "Ultralytics-reported, configuration unspecified".
- MLPerf Edge results for Orin were not fetched.

---

## Q4. Memory budget: 2B VLM (Q4/Q8 GGUF + mmproj) + TensorRT YOLO on 8 GB Orin Nano

### Takeaway
Qwen3-VL-2B in INT4 uses about 4.4 GB of GPU memory under TRT-Edge-LLM on Orin Nano 8GB. A 2B VLM plus a nano TRT detector is plausible within 8 GB of shared memory when headless. Several Orin Nano reports, however, describe NvMap allocation failures (error 12) caused by fragmentation and an r36.4.7 security-patch limit, including when PyTorch and llama.cpp share the GPU. Peak RAM and the largest free block need to be measured, not assumed.

### Cited Findings
- Qwen3-VL-2B (INT4 AWQ + FP16 ViT) uses 4,380 MB GPU memory on Orin Nano 8GB, and Qwen3.5-2B uses 4,621 MB. — [TensorRT-Edge-LLM benchmarks](https://nvidia.github.io/TensorRT-Edge-LLM/0.9.1/user_guide/performance/performance-benchmarks.html)
- Qwen3-VL-2B in **FP16** via Transformers ran on Orin Nano but "uses nearly all the RAM". Qwen3-VL-4B was not confirmed working. NVIDIA staff recommended the Jetson AI Lab RAM optimisations (disable the desktop GUI, stop services, mount swap) and said they had tested Qwen2.5-VL-3B on Orin Nano. — [NVIDIA forum: Orin Nano Qwen3-VL-4B](https://forums.developer.nvidia.com/t/orin-nano-qwen3-vl-4b/348122)
- `NvMapMemAllocInternalTagged: 1075072515 error 12` / "unable to allocate CUDA0 buffer" on Orin Nano: NVIDIA staff tied this to r36.4.7. The security patch for CVE-2025-33182/33177 "led to some limitations in the allocable memory". Staff said on 2025-12-18 that "We have fixed the memory issue internally". — [NVIDIA forum](https://forums.developer.nvidia.com/t/orin-nano-qwen3-vl-4b/348122); [Ollama errors orin nano](https://forums.developer.nvidia.com/t/ollama-errors-orin-nano/345290/28)
- Fragmentation: one Ollama report had about 3.8 GB RAM free but only very small contiguous blocks available. — [NVIDIA forum](https://forums.developer.nvidia.com/t/ollama-llm-inference-problems-on-jetson-orin-nano-cuda-memory-allocation-failure-and-cpu-memory-error/363574)
- Sharing the GPU between PyTorch and llama.cpp on Orin Nano 8GB Super (JetPack 6.2.2) produced an NVML assertion in PyTorch's CUDACachingAllocator, plus an OOM on a 929 MB weight buffer attributed to NvMap fragmentation. `PYTORCH_NO_CUDA_NVML=1` did not help. — [NVIDIA forum](https://forums.developer.nvidia.com/t/pytorch-cudacachingallocator-nvml-assertion-when-sharing-cuda-context-with-llama-cpp-on-orin-nano-8-gb-jetpack-6-2-2/370049)
- A community llama.cpp Docker guide recommends keeping context ≤2048 on Orin Nano because memory is shared. — [zahamm/llama-cpp-jetson (Docker Hub)](https://hub.docker.com/r/zahamm/llama-cpp-jetson)
- A Chinese-language guide warns that omitting `--mmproj` lets text queries work while image queries fail. — [CSDN llama.cpp Jetson 8GB guide](https://deepseek.csdn.net/6a9c19b348977663a5dd4b96.html) (snippet only)

### Inferences
- Budget sketch, not sourced and to be verified on the board: Ubuntu headless plus the CUDA context(s) take about 1–1.5 GB. A YOLO-nano TRT engine plus its activations is small (tens to low hundreds of MB). A Qwen3-VL-2B Q4_K_M GGUF plus an F16 mmproj, KV for a ≤2048-token context, and compute buffers likely totals 2–3.5 GB. The TRT-Edge-LLM figure of 4.4 GB is a conservative upper reference. This should fit, but with limited headroom on 8 GB. Q8 adds roughly 1 GB.
- **Run the detector and VLM as separate processes rather than sharing one PyTorch CUDA context with llama.cpp**, given the reported allocator conflicts. Avoid PyTorch in the deployed detector path (use the TRT runtime or ONNX Runtime TRT EP). Load the VLM first, then build/load the TRT engine, to reduce fragmentation (hypothesis to test).
- Report the exact L4T revision, since r36.4.7 has known allocation limits.

### Gaps
- No primary source gives measured llama.cpp RSS for Qwen3-VL-2B GGUF + mmproj on Orin. Exact GGUF and mmproj file sizes were not checked here (they can be read from the Hugging Face model card).
- Whether the 2025-12 NvMap fix shipped in a public L4T release (e.g. r36.5) was not confirmed.

---

## Q5. Known issues: GPU contention, CUDA MPS on Jetson, scheduling priorities, nvv4l2decoder

### Takeaway
CUDA MPS is officially supported on Jetson since CUDA 12.5 / JetPack 6.1. Forum reports show the server does not auto-start the way it does on x86, and that static SM partitioning fails on Orin Nano. MPS should therefore be treated as an optional experiment, not a requirement. Without it, TRT and llama.cpp time-slice the single iGPU, so detector tail latency during VLM prefill and decode is the key concurrency metric.

### Cited Findings
- NVIDIA staff: MPS "has been supported on Jetson since CUDA 12.5 so it's available in JetPack 6.1". Release notes say MPS is "now supported on L4T and embedded-Linux Tegra platforms". Earlier staff replies had said MPS was not supported on Tegra and Orin. — [NVIDIA forum: Jetson Orin with JetPack 6.1 support MPS?](https://forums.developer.nvidia.com/t/jetson-orin-with-jetpack6-1-support-mps/317978/6); [MPS on AGX Orin? (older)](https://forums.developer.nvidia.com/t/mps-on-agx-orin/219157)
- On Orin NX with JetPack 6.2, `nvidia-cuda-mps-control -d` does not auto-launch the MPS server when CUDA apps start, unlike x86. — [NVIDIA forum: Proper use of CUDA MPS?](https://forums.developer.nvidia.com/t/proper-use-of-cuda-mps/337315)
- On Orin Nano, MPS runs, but enabling static partitioning per the docs makes the MPS daemon fail to launch. — [NVIDIA forum](https://forums.developer.nvidia.com/t/static-partitioning-failed-with-nvidia-mps-on-jetson-orin-nano/363234)
- The edge VLM energy study found that power is roughly constant and decode dominates time. Long VLM decode phases are therefore the main periods of sustained GPU occupancy that compete with the detector. — [arXiv:2607.09520](https://arxiv.org/abs/2607.09520)

### Inferences
- **Concurrency controls to test** (our design): (i) no MPS, default time-slicing; (ii) CUDA stream priorities in the detector, since TRT accepts a user stream that can be created with high priority (this only helps within a single process); (iii) a cap on llama.cpp's `n_predict` and a lower VLM trigger rate; (iv) MPS with `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` for the VLM client, if MPS starts on the board; (v) CPU isolation (`taskset`/`nice`) for ByteTrack and decode. Report detector p99 and drop rate for each.
- Hardware decode: use GStreamer `nvv4l2decoder` (NVDEC) with `nvvidconv` into NVMM buffers, or DeepStream, so H.264/H.265 decode stays off the CPU and the GPU SMs. Log NVDEC load in tegrastats. No source was fetched for this in this pass; the Ultralytics DeepStream guide is a related reference ([Ultralytics DeepStream guide](https://docs.ultralytics.com/guides/deepstream-nvidia-jetson)).

### Gaps
- No published measurement was found of detector latency inflation when a VLM shares an Orin iGPU (TRT + llama.cpp). This is a novel measurement our harness would contribute.
- The current state of MPS in the CUDA-for-Tegra app note was not confirmed from a primary doc.
- No primary source was fetched for the nvv4l2decoder throughput limits on Orin Nano. Note that the Orin Nano has no NVENC; an NVDEC decoder is present, but its details are unverified here.
- No Jetson-specific guidance on GPU scheduling priority between processes was found.
