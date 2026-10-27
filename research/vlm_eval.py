"""Quantitative evaluation of the vision-language model (VLM) stage of the closed loop.

Model under test: Qwen3-VL-2B-Instruct (Q4_K_M GGUF + Q8_0 mmproj) served by llama.cpp `llama-server`
(Vulkan build, OpenAI-compatible API at http://127.0.0.1:8080/v1, started separately by
edge_deploy/scripts/start_vlm.sh). All calls go through the deployed client (edge_deploy/edgevision/vlm.py:
VLMClient, encode_image, SYSTEM, TOOLS, parse_json), at temperature 0 (greedy, deterministic) and one
request at a time. Every output records the model files (size + SHA-256), the server build and the prompts.

  e1  Count grounding: does feeding the detector's analytics to the VLM fix VLM counting?
      Data: the 182 test frames of the aa1 study (datasets/coco/test, 255-300 s); ground truth per frame =
      number of pseudo-GT boxes (teacher ensemble, not human). Conditions:
        a      detector alone: edge_deploy OnnxDetector, models/yolo26n_person_736x1280.onnx, CPU provider,
               conf = valid-split best-F1 threshold of YOLO26n (fine-tuned) read from results/metrics.json,
               NMS IoU 0.7 (deployed default) -> number of boxes
        b      VLM, image only (no system prompt): COUNT_Q
        b_sys  VLM, image + the deployed SYSTEM prompt, no measurements (isolates the system prompt)
        c      VLM, image + SYSTEM + a context line with the detector's count of condition a
        d      VLM, text only (SYSTEM + context, no image): control for "the VLM just copies the number"
        c_off  as c, but the context states a deliberately wrong count (detector count + 10): a simulated
               detector failure, testing whether the VLM checks the number against the image
      Images are sent as the deployed client sends them (encode_image, image_max_side 896, JPEG q85).
      The first integer in the reply is the prediction; no integer = parse failure (excluded from error
      metrics, reported as a rate). Metrics per condition: MAE, RMSE, mean signed error (bias), exact-match
      rate, within-+-2 rate, Pearson r with GT, parse-failure rate; bootstrap 95% CI for MAE (B=1000, seed 0,
      resampling frames); paired bootstrap of the MAE difference between conditions on the frames both
      parsed (b vs c is the headline comparison); MAE stratified by crowd size (GT <=10, 11-20, >20); rate
      at which the VLM answer equals the count it was given; a constant-median reference.
  e2  Zero-shot attribute recognition vs the trained linear probes (campus case study).
      Labels: results/campus/validation/video_{1,2}_labels.tsv (the 100 held-out test tracks; "?" = not
      determinable, excluded). For each labelled test track the 3 best crops (highest score suffix of
      results/campus/<video>/crops/<track>_<frame>_<score>.jpg) are sent one by one with the closed-
      vocabulary campus uniform prompt (CAMPUS_PROMPT below),
      upscaled like the original campus script (min height 448 px, longest side 896 px). A track's label = majority vote over the crops
      ignoring "unsure"; a tie or no informative vote -> "unsure" (abstain). Scoring is campus_probe.evaluate's:
      accuracy over answered tracks, Wilson 95% CI, coverage = answered / labelled, per video and both.
      Probe numbers: results/campus/validation/probe_scores.json (per-track test); per-track probe outputs
      from results/campus/<video>/attributes.json (checked to reproduce probe_scores.json) for an exact
      McNemar test on the tracks both methods answered.
  e3  Tool-calling reliability for the closed loop. A fixed list of operator utterances (UTTERANCES): 29
      that should call exactly one tool with specific arguments and 16 that should not call a tool. Each is
      sent as one round with the real SYSTEM prompt + a realistic measurement context (same format as
      SceneAgent.context()) + TOOLS, max_tokens 384, twice: with a camera frame (as SceneAgent.ask sends
      it) and text only. Tool calls are read like SceneAgent.ask does (server-parsed tool_calls, else
      <tool_call> text). Metrics: tool-selection accuracy (exactly the expected tool, or no call), argument
      exact-match accuracy (numbers within 1e-6, counting-line coordinates within +-0.02, endpoints in
      either order; arguments the utterance does not determine are not checked; names compared
      case-insensitively with spaces/underscores unified), false-positive tool-call rate on no-tool
      utterances, JSON-valid-arguments rate over all emitted calls, and a confusion table. A lenient
      argument score (> and >= treated alike, event kind and rule name not checked) is reported alongside.
  e4  Latency profile of the VLM server on this machine (RTX 4060 Laptop, Vulkan backend; desktop numbers,
      not Jetson). Image queries (SYSTEM + context + the deployed OBSERVE scene-report prompt,
      max_tokens 256) at image_max_side 448 / 672 / 896 / 1280: 2 warm-up calls, then 10 different test
      frames each; text-only chat and tool-call turns likewise (2 + 10). The prompt cache is disabled
      (cache_prompt=false) so every call pays its full prefill. Reported from the server's `timings`:
      prompt tokens, prefill ms, prefill tokens/s, decode tokens/s; client side: end-to-end latency
      median / p90 and image encode time. Other GPU jobs may share the GPU (recorded via nvidia-smi).

    env -u PYTHONPATH .venv/bin/python research/vlm_eval.py {e1,e2,e3,e4,all}
Writes results/vlm/e1_counting.json, e2_attributes.json, e3_tools.json, e4_latency.json,
fig_e1_counting.png, fig_e4_latency.png; logs to results/logs/vlm_eval_<exp>.log.
"""
import argparse
import hashlib
import json
import logging
import math
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import paths as P

import cv2
import numpy as np

sys.path.insert(0, str(P.DEPLOY))
from edgevision.vlm import OBSERVE, SYSTEM, TOOLS, VLMClient, encode_image, parse_json, text_tool_calls  # noqa: E402

OUT = Path("results/vlm")
LOGS = Path("results/logs")
DATA = Path("datasets/coco/test")
CAMPUS = Path("results/campus")
VIDEOS = ["video_1", "video_2"]
ENDPOINT, MODEL = "http://127.0.0.1:8080/v1", "qwen3-vl-2b"
SERVER_BIN = P.DEPLOY / "third_party/llama.cpp-vulkan-x64/llama-server"
VLM_FILES = [P.DEPLOY / "models/vlm/Qwen3VL-2B-Instruct-Q4_K_M.gguf",
             P.DEPLOY / "models/vlm/mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf"]
ONNX = "models/yolo26n_person_736x1280.onnx"           # relative to edge_deploy/
IMAGE_MAX_SIDE = 896                                    # deployed default (edgevision/config.py)
TEMPERATURE = 0.0
B, SEED = 1000, 0
COUNT_Q = "How many people are visible in this image? Reply with only an integer."
log = logging.getLogger("vlm_eval")


# ---------------------------------------------------------------- common
def setup_log(exp):
    LOGS.mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)
    for h in list(log.handlers):
        log.removeHandler(h)
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S")
    for h in (logging.FileHandler(LOGS / f"vlm_eval_{exp}.log", mode="w"), logging.StreamHandler(sys.stdout)):
        h.setFormatter(fmt); log.addHandler(h)


def client():
    return VLMClient({"endpoint": ENDPOINT, "model": MODEL, "timeout_s": 300})


def wait_server(c, max_wait_s=900):
    t = time.time()
    while not c.healthy():
        if time.time() - t > max_wait_s:
            raise RuntimeError(f"VLM server at {ENDPOINT} down for {max_wait_s} s")
        log.info("VLM server not reachable, retrying in 15 s"); time.sleep(15)


def chat(c, messages, tools=None, max_tokens=64):
    """VLMClient.chat at temperature 0 with retry while the server is unreachable."""
    for attempt in range(5):
        try:
            return c.chat(messages, tools=tools, max_tokens=max_tokens, temperature=TEMPERATURE)
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as e:
            log.info("request failed (%s), waiting for the server", e); wait_server(c)
    raise RuntimeError("VLM request failed 5 times")


def sha256(p, chunk=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def server_info():
    def get(path):
        try:
            with urllib.request.urlopen(ENDPOINT.rsplit("/v1", 1)[0] + path, timeout=10) as r:
                return json.loads(r.read())
        except (urllib.error.URLError, OSError, ValueError):
            return {}
    props = get("/props")
    try:
        ver = subprocess.run([str(SERVER_BIN), "--version"], capture_output=True, text=True, timeout=30,
                             env={"LD_LIBRARY_PATH": str(SERVER_BIN.parent)})
        version = (ver.stdout + ver.stderr).strip().splitlines()
    except (OSError, subprocess.TimeoutExpired) as e:
        version = [f"unavailable: {e}"]
    meta = (get("/v1/models").get("data") or [{}])[0].get("meta", {})
    return {"endpoint": ENDPOINT, "model_alias": MODEL, "server_binary": str(SERVER_BIN.relative_to(P.ROOT)),
            "server_version": version, "build_info": props.get("build_info"),
            "model_path_reported": props.get("model_path"), "n_ctx": meta.get("n_ctx") or props.get(
                "default_generation_settings", {}).get("n_ctx"), "total_slots": props.get("total_slots"),
            "n_params": meta.get("n_params"), "ftype": meta.get("ftype") or props.get("model_ftype"),
            "launch": "edge_deploy/scripts/start_vlm.sh (-ngl 99 -c 8192 --jinja -fa auto, Vulkan NVIDIA device)",
            "model_files": [{"file": str(p.relative_to(P.ROOT)), "bytes": p.stat().st_size, "sha256": sha256(p)}
                            for p in VLM_FILES],
            "client": "edgevision.vlm.VLMClient (edge_deploy), temperature 0, sequential requests"}


def gpu_snapshot():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total",
                            "--format=csv,noheader"], capture_output=True, text=True, timeout=20)
        a = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                            "--format=csv,noheader"], capture_output=True, text=True, timeout=20)
        return {"gpu": r.stdout.strip(), "compute_apps": a.stdout.strip().splitlines()}
    except (OSError, subprocess.TimeoutExpired):
        return None


def image_msg(text, img=None, max_side=IMAGE_MAX_SIDE):
    content = [{"type": "text", "text": text}]
    if img is not None:
        content.insert(0, {"type": "image_url", "image_url": {"url": encode_image(img, max_side)}})
    return {"role": "user", "content": content}


def strip_images(messages):
    """Request log without the base64 payload (replaced by its length and hash)."""
    out = []
    for m in messages:
        if isinstance(m.get("content"), list):
            c = []
            for part in m["content"]:
                if part.get("type") == "image_url":
                    u = part["image_url"]["url"]
                    c.append({"type": "image_url", "image_url": f"<jpeg data-uri, {len(u)} chars, "
                                                               f"sha1 {hashlib.sha1(u.encode()).hexdigest()[:12]}>"})
                else:
                    c.append(part)
            m = {**m, "content": c}
        out.append(m)
    return out


def wilson(k, n, z=1.96):
    """Same as campus_probe.wilson."""
    if n == 0:
        return [None, None]
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [round(float((c - h) / d), 3), round(float((c + h) / d), 3)]


def r3(x):
    return None if x is None or (isinstance(x, float) and not math.isfinite(x)) else round(float(x), 4)


def dump(obj, name):
    with open(OUT / name, "w") as f:
        json.dump(obj, f, indent=1, default=str)
    log.info("wrote %s", OUT / name)


# ---------------------------------------------------------------- figures (shared style)
def plt_style():
    import visualize as V               # sets the shared rcParams (matplotlib Agg)
    return V.plt, V


def save_fig(fig, name):
    fig.tight_layout(); fig.savefig(OUT / name, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig); log.info("wrote %s", OUT / name)


# ================================================================ E1 counting
COND = {
    "a": "detector alone (YOLO26n ONNX, CPU)",
    "b": "VLM, image only",
    "b_sys": "VLM, image + SYSTEM (no measurements)",
    "c": "VLM, image + SYSTEM + detector count",
    "d": "VLM, text only (SYSTEM + detector count)",
    "c_off": "VLM, image + SYSTEM + WRONG count (detector + 10)",
}
OFFSET = 10            # c_off: simulated detector failure, does the VLM notice the image disagrees?


def context_line(n):
    return f"Measurements: occupancy={n} (person detector: people in the current frame)."


def first_int(text):
    m = re.search(r"\d+", text or "")
    return int(m.group(0)) if m else None


def count_metrics(pred, gt):
    pred = np.array([np.nan if p is None else p for p in pred], float); gt = np.asarray(gt, float)
    ok = ~np.isnan(pred)
    e = pred[ok] - gt[ok]
    res = {"n": int(len(gt)), "n_parsed": int(ok.sum()), "parse_failure_rate": r3(1 - ok.mean())}
    if not ok.any():
        return res
    r = np.corrcoef(pred[ok], gt[ok])[0, 1] if pred[ok].std() > 0 and gt[ok].std() > 0 else float("nan")
    res.update({"MAE": r3(np.abs(e).mean()), "RMSE": r3(np.sqrt((e ** 2).mean())), "bias": r3(e.mean()),
                "exact_match": r3((e == 0).mean()), "within_2": r3((np.abs(e) <= 2).mean()), "pearson_r": r3(r),
                "pred_mean": r3(pred[ok].mean()), "gt_mean": r3(gt[ok].mean())})
    rng = np.random.default_rng(SEED); ae = np.abs(e)
    boot = [ae[rng.integers(0, len(ae), len(ae))].mean() for _ in range(B)]
    res["MAE_CI95"] = [r3(x) for x in np.percentile(boot, [2.5, 97.5])]
    return res


def paired(pa, pb, gt):
    """MAE(pa) - MAE(pb) on frames where both parsed; paired bootstrap over frames."""
    idx = [i for i in range(len(gt)) if pa[i] is not None and pb[i] is not None]
    ea = np.array([abs(pa[i] - gt[i]) for i in idx], float); eb = np.array([abs(pb[i] - gt[i]) for i in idx], float)
    d = ea - eb
    rng = np.random.default_rng(SEED)
    boot = np.array([d[j].mean() for j in (rng.integers(0, len(d), len(d)) for _ in range(B))])
    return {"n_frames": len(idx), "MAE_first": r3(ea.mean()), "MAE_second": r3(eb.mean()),
            "delta_MAE": r3(d.mean()), "CI95": [r3(x) for x in np.percentile(boot, [2.5, 97.5])],
            "p_two_sided": r3(min(1.0, 2 * min((boot <= 0).mean(), (boot >= 0).mean()))),
            "frames_first_better": int((d < 0).sum()), "frames_second_better": int((d > 0).sum()),
            "ties": int((d == 0).sum())}


def e1():
    from edgevision.detector import OnnxDetector
    c = client(); wait_server(c)
    conf = float(json.load(open("results/metrics.json"))["YOLO26n (fine-tuned)"]["conf"])
    det_cfg = {"backend": "onnxruntime", "weights": ONNX, "providers": ["cpu"], "fp16": False, "conf": conf,
               "iou": 0.7, "max_det": 300, "trt_cache": "models/trt_cache"}
    det = OnnxDetector(det_cfg)
    assert det.provider == "CPUExecutionProvider", det.provider
    coco = json.load(open(DATA / "_annotations.coco.json"))
    n_gt = Counter(a["image_id"] for a in coco["annotations"])
    images = sorted(coco["images"], key=lambda im: im["frame"])
    log.info("E1: %d frames, detector conf %.3f (results/metrics.json), provider %s", len(images), conf, det.provider)
    rows = []
    for k, im in enumerate(images):
        img = cv2.imread(str(DATA / im["file_name"]))
        t = time.time(); boxes, _ = det(img); det_ms = (time.time() - t) * 1000
        n_det = int(len(boxes)); ctx = context_line(n_det)
        msgs = {
            "b": [image_msg(COUNT_Q, img)],
            "b_sys": [{"role": "system", "content": SYSTEM}, image_msg(COUNT_Q, img)],
            "c": [{"role": "system", "content": SYSTEM + "\n\n" + ctx}, image_msg(COUNT_Q, img)],
            "d": [{"role": "system", "content": SYSTEM + "\n\n" + ctx}, {"role": "user", "content": COUNT_Q}],
            "c_off": [{"role": "system", "content": SYSTEM + "\n\n" + context_line(n_det + OFFSET)},
                      image_msg(COUNT_Q, img)],
        }
        row = {"image_id": im["id"], "file": im["file_name"], "frame": im["frame"], "gt": n_gt[im["id"]],
               "a": n_det, "detector_ms": round(det_ms, 1)}
        for cond, m in msgs.items():
            msg, info = chat(c, m, max_tokens=16)
            text = msg.get("content") or ""
            row[cond] = first_int(text)
            row[cond + "_raw"] = text
            row[cond + "_latency_s"] = info["latency_s"]
            row[cond + "_prompt_tokens"] = info["prompt_tokens"]
        rows.append(row)
        if k % 20 == 0 or k == len(images) - 1:
            log.info("  %3d/%d %s gt %d | a %s b %s b_sys %s c %s d %s", k + 1, len(images), im["file_name"],
                     row["gt"], row["a"], row["b"], row["b_sys"], row["c"], row["d"])

    gt = [r["gt"] for r in rows]
    metrics = {cond: count_metrics([r[cond] for r in rows], gt) for cond in COND}
    pairs = {f"{x} vs {y}": paired([r[x] for r in rows], [r[y] for r in rows], gt)
             for x, y in (("b", "c"), ("b", "b_sys"), ("b_sys", "c"), ("a", "c"), ("c", "d"), ("c_off", "c"))}
    bins = {"gt<=10": (0, 10), "gt 11-20": (11, 20), "gt>20": (21, 10 ** 6)}
    strat = {}
    for bn, (lo, hi) in bins.items():
        sel = [r for r in rows if lo <= r["gt"] <= hi]
        strat[bn] = {"n_frames": len(sel)}
        for cond in COND:
            e = [r[cond] - r["gt"] for r in sel if r[cond] is not None]
            strat[bn][cond] = {"MAE": r3(np.mean(np.abs(e))) if e else None, "bias": r3(np.mean(e)) if e else None,
                               "n_parsed": len(e)}
    copy = {cond: r3(np.mean([r[cond] == r["a"] for r in rows if r[cond] is not None])) for cond in ("b", "b_sys", "c", "d")}
    copy["c_off (answer == stated wrong count)"] = r3(np.mean([r["c_off"] == r["a"] + OFFSET for r in rows
                                                               if r["c_off"] is not None]))
    med = float(np.median(gt))
    const = {"constant = median test GT (oracle reference, uses the test labels)": {
        "value": med, **{k: v for k, v in count_metrics([med] * len(gt), gt).items() if k != "pearson_r"}}}
    out = {"experiment": "E1 count grounding", "protocol": __doc__.split("  e2 ")[0].split("  e1 ")[1].strip(),
           "conditions": COND,
           "settings": {"data": str(DATA), "n_frames": len(rows), "gt": "number of pseudo-GT boxes per frame "
                        "(RF-DETR Large + YOLO26x-TTA ensemble, WBF, score >= 0.40)",
                        "gt_range": [min(gt), max(gt)], "detector": {**det_cfg, "conf_source":
                        "results/metrics.json['YOLO26n (fine-tuned)']['conf'] (valid best-F1)", "provider": det.provider},
                        "image_max_side": IMAGE_MAX_SIDE, "temperature": TEMPERATURE, "max_tokens": 16,
                        "question": COUNT_Q, "system_prompt": SYSTEM,
                        "context_line_template": context_line("<n>"),
                        "parse": "first run of digits in the reply", "bootstrap": {"B": B, "seed": SEED},
                        "server": server_info()},
           "metrics": metrics, "paired_MAE_differences": pairs, "stratified_by_gt": strat,
           "rate_vlm_answer_equals_detector_count": copy, "reference_baselines": const, "rows": rows}
    dump(out, "e1_counting.json")
    for cond in COND:
        m = metrics[cond]
        log.info("  %-6s MAE %s %s RMSE %s bias %s exact %s within2 %s r %s parse-fail %s", cond, m.get("MAE"),
                 m.get("MAE_CI95"), m.get("RMSE"), m.get("bias"), m.get("exact_match"), m.get("within_2"),
                 m.get("pearson_r"), m["parse_failure_rate"])
    for k, v in pairs.items():
        log.info("  %s: dMAE %s CI %s p %s (n %d)", k, v["delta_MAE"], v["CI95"], v["p_two_sided"], v["n_frames"])
    fig_e1(out)


def fig_e1(out):
    plt, V = plt_style()
    rows, M = out["rows"], out["metrics"]
    col = {"a": V.COLOR["YOLO26n"], "b": V.COLOR["RF-DETR Nano"], "b_sys": "#f2a57e", "c": "#1f9e74",
           "d": V.COLOR["Teacher"], "c_off": "#b23a48"}
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1.1, 1]})
    gt = np.array([r["gt"] for r in rows])
    lo, hi = 0, max(gt.max(), max((r[k] or 0) for r in rows for k in ("a", "b", "c"))) + 2
    ax.plot([lo, hi], [lo, hi], color=V.INK2, lw=1, ls=":", label="y = x")
    rng = np.random.default_rng(0)
    for k, mk in (("b", "o"), ("c", "s"), ("a", "^")):
        p = np.array([np.nan if r[k] is None else r[k] for r in rows], float)
        j = rng.uniform(-0.18, 0.18, len(gt))
        ax.scatter(gt + j, p + rng.uniform(-0.18, 0.18, len(gt)), s=16, marker=mk, color=col[k], alpha=0.65,
                   edgecolor="none", label=f"({k}) {COND[k]}  MAE {M[k]['MAE']:.2f}")
    ax.set_xlim(gt.min() - 2, gt.max() + 2); ax.set_ylim(lo, hi)
    ax.set_xlabel("pseudo-GT people per frame"); ax.set_ylabel("predicted count")
    ax.set_title("Per-frame counts, 182 test frames"); ax.legend(fontsize=7.5, loc="lower right")
    ks = list(COND)
    for i, k in enumerate(ks):
        m = M[k]; l, h = m["MAE_CI95"]
        bx.barh(i, m["MAE"], color=col[k], height=0.6, edgecolor=V.SURF)
        bx.plot([l, h], [i, i], color=V.INK, lw=1.5)
        bx.text(h + 0.15, i, f"{m['MAE']:.2f} [{l:.2f}, {h:.2f}]", va="center", fontsize=8, color=V.INK2)
    bx.set_yticks(range(len(ks)), [f"({k}) {COND[k]}" for k in ks], fontsize=8)
    bx.invert_yaxis(); bx.set_xlabel("count MAE (people / frame), 95% bootstrap CI")
    bx.set_title("Count error by condition"); bx.grid(axis="y", visible=False); bx.set_axisbelow(True)
    bx.set_xlim(0, max(M[k]["MAE_CI95"][1] for k in ks) * 1.45)
    save_fig(fig, "fig_e1_counting.png")


# ================================================================ E2 attributes
FIELDS = {"gender": ["male", "female", "unsure"], "dress": ["uniform", "non_uniform", "unsure"]}
HEADS = {"dress": ("uniform", "non_uniform"), "gender": ("female", "male")}      # (positive, negative) as campus_probe
PROBE_KEY = {"dress": "uniform", "gender": "gender"}                            # head names in probe_scores.json


CAMPUS_PROMPT = 'This is a crop of one person from a video of an Indian university campus.\n\nThe student uniform is:\n- men: a plain white or off-white shirt with khaki / beige trousers.\n- women: a sleeveless beige / khaki overcoat (vest) worn over a white or cream kurta, with white or khaki bottoms.\nAnything else (jeans, T-shirts, coloured or printed shirts or kurtas, sarees, other outfits) is not the uniform.\n\nAnswer with only this JSON, choosing each value from the listed options:\n{"gender": "male|female|unsure",\n "dress": "uniform|non_uniform|unsure"}\nUse "unsure" only if the crop does not show enough.'
# (verbatim copy of PROMPT from the retired research/campus/campus_attributes.py, the closed-vocabulary
#  campus uniform prompt; kept here so E2 stays reproducible)


def campus_prompt():
    return CAMPUS_PROMPT


def load_crop(p, min_h=448, max_side=896):
    """Upscale small crops (height >= min_h, longest side <= max_side, PIL bicubic) -> BGR array for encode_image."""
    from PIL import Image
    im = Image.open(p).convert("RGB")
    w, h = im.size
    s = max(min_h / h, 1.0)
    s = min(s, max_side / max(w * s, h * s) * s)
    im = im.resize((max(32, int(w * s)), max(32, int(h * s))), Image.BICUBIC)
    return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR), im.size


def parse_attr(text):
    d = parse_json(text)
    ok = isinstance(d, dict)
    d = d if ok else {}
    out = {}
    for k, opts in FIELDS.items():
        v = str(d.get(k, "")).lower().strip().replace("-", "_").replace(" ", "_")
        out[k] = v if v in opts else "unsure"
    return out, ok and all(str(d.get(k, "")).lower().strip().replace("-", "_").replace(" ", "_") in FIELDS[k]
                           for k in FIELDS)


def vote(answers, field):
    c = Counter(a[field] for a in answers if a[field] != "unsure")
    if not c:
        return "unsure", 0.0
    top = c.most_common()
    if len(top) > 1 and top[0][1] == top[1][1]:
        return "unsure", 0.5
    return top[0][0], round(top[0][1] / sum(c.values()), 3)


def load_labels():
    lab = {}
    for v in VIDEOS:
        rows = [ln.rstrip("\n").split("\t") for ln in open(CAMPUS / "validation" / f"{v}_labels.tsv")
                if not ln.startswith("#")]
        for r in rows[1:]:
            d = dict(zip(rows[0], r)); lab[(v, d["track"])] = d
    return lab


def score(pred, lab, field):
    """campus_probe.evaluate scoring for one method. pred: {(video, track): label|'unsure'}."""
    pos, neg = HEADS[field]
    res = {}
    for scope in VIDEOS + ["both"]:
        k = n = lab_n = 0; errs = []; cm = Counter(); labs = Counter()
        for key, d in lab.items():
            if (scope != "both" and key[0] != scope) or key not in pred or d[field] not in (pos, neg):
                continue
            lab_n += 1; labs[d[field]] += 1
            cm[f"true={d[field]} pred={pred[key]}"] += 1
            if pred[key] == "unsure":
                continue
            n += 1; k += int(pred[key] == d[field])
            if pred[key] != d[field]:
                errs.append(f"{key[0]}#{key[1]} true={d[field]} pred={pred[key]}")
        res[scope] = {"accuracy": round(k / n, 3) if n else None, "ci95": wilson(k, n), "correct": k, "answered": n,
                      "labelled": lab_n, "coverage": round(n / lab_n, 3) if lab_n else None,
                      "accuracy_abstain_as_wrong": round(k / lab_n, 3) if lab_n else None,
                      "balanced_accuracy_answered": r3(np.mean([cm[f"true={c} pred={c}"] / max(1, sum(
                          cm[f"true={c} pred={o}"] for o in (pos, neg))) for c in (pos, neg)])) if n else None,
                      "majority_class_baseline": r3(max(labs.values()) / lab_n) if lab_n else None,
                      "confusion": dict(cm), "errors": errs}
    return res


def mcnemar(a_pred, b_pred, lab, field):
    """Exact McNemar on tracks both methods answered (a = probe, b = VLM)."""
    pos, neg = HEADS[field]
    both = [k for k, d in lab.items() if d[field] in (pos, neg) and a_pred.get(k, "unsure") != "unsure"
            and b_pred.get(k, "unsure") != "unsure"]
    ac = [a_pred[k] == lab[k][field] for k in both]; bc = [b_pred[k] == lab[k][field] for k in both]
    n01 = sum(a and not b for a, b in zip(ac, bc)); n10 = sum(b and not a for a, b in zip(ac, bc))
    n = n01 + n10
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(min(n01, n10) + 1)) / 2 ** n) if n else 1.0
    return {"n_both_answered": len(both), "probe_accuracy_on_these": r3(np.mean(ac)) if both else None,
            "vlm_accuracy_on_these": r3(np.mean(bc)) if both else None, "probe_right_vlm_wrong": n01,
            "vlm_right_probe_wrong": n10, "p_exact_two_sided": r3(p)}


def e2(max_crops=3):
    c = client(); wait_server(c)
    prompt = campus_prompt()
    lab = load_labels()
    probe_scores = json.load(open(CAMPUS / "validation" / "probe_scores.json"))
    probe_pred = {}
    for v in VIDEOS:
        attrs = json.load(open(CAMPUS / v / "attributes.json"))
        for t, r in attrs.items():
            probe_pred[(v, t)] = {f: r[f] for f in FIELDS}
    crops = defaultdict(list)
    for v in VIDEOS:
        for p in (CAMPUS / v / "crops").glob("*.jpg"):
            crops[(v, p.stem.split("_")[0])].append(p)
    keys = [k for k, d in lab.items() if k in crops and any(d[f] in HEADS[f] for f in FIELDS)]
    log.info("E2: %d labelled test tracks with crops (of %d), up to %d crops each", len(keys), len(lab), max_crops)
    crop_rows, track_rows, vlm_pred = [], [], {f: {} for f in FIELDS}
    for i, key in enumerate(keys):
        best = sorted(crops[key], key=lambda p: (-int(p.stem.split("_")[2]), p.name))[:max_crops]
        answers = []
        for p in best:
            img, size = load_crop(p)
            msg, info = chat(c, [image_msg(prompt, img, max_side=896)], max_tokens=64)
            text = msg.get("content") or ""
            a, valid = parse_attr(text)
            answers.append(a)
            crop_rows.append({"video": key[0], "track": key[1], "crop": p.name, "sent_size_wh": list(size), **a,
                              "valid_json": valid, "text": text, "latency_s": info["latency_s"],
                              "prompt_tokens": info["prompt_tokens"]})
        tr = {"video": key[0], "track": key[1], "n_crops": len(best), "label_gender": lab[key]["gender"],
              "label_dress": lab[key]["dress"]}
        for f in FIELDS:
            pred, share = vote(answers, f)
            vlm_pred[f][key] = pred
            tr[f"vlm_{f}"], tr[f"vlm_{f}_vote_share"] = pred, share
            tr[f"probe_{f}"] = probe_pred.get(key, {}).get(f, "unsure")
        track_rows.append(tr)
        if i % 10 == 0 or i == len(keys) - 1:
            log.info("  %3d/%d %s#%s label %s/%s vlm %s/%s probe %s/%s", i + 1, len(keys), *key, tr["label_gender"],
                     tr["label_dress"], tr["vlm_gender"], tr["vlm_dress"], tr["probe_gender"], tr["probe_dress"])

    results, check = {}, {}
    for f in FIELDS:
        pp = {k: probe_pred[k][f] for k in keys if k in probe_pred}
        vlm_s, probe_s = score(vlm_pred[f], lab, f), score(pp, lab, f)
        stored = probe_scores[PROBE_KEY[f]]["test_per_track"]
        check[f] = all(probe_s[s]["correct"] == stored[s]["correct"] and probe_s[s]["answered"] == stored[s]["answered"]
                       and probe_s[s]["labelled"] == stored[s]["labelled"] for s in VIDEOS + ["both"])
        results[f] = {"vlm": vlm_s, "probe_stored": {s: {k: v for k, v in r.items()} for s, r in stored.items()},
                      "probe_recomputed_from_attributes_json": probe_s,
                      "mcnemar_both_answered": {s: mcnemar(pp, vlm_pred[f], {k: d for k, d in lab.items()
                                                if s == "both" or k[0] == s}, f) for s in VIDEOS + ["both"]}}
    crop_stats = {f: dict(Counter(r[f] for r in crop_rows)) for f in FIELDS}
    out = {"experiment": "E2 zero-shot attributes vs linear probes",
           "protocol": __doc__.split("  e3 ")[0].split("  e2 ")[1].strip(),
           "settings": {"labels": [f"results/campus/validation/{v}_labels.tsv" for v in VIDEOS],
                        "max_crops_per_track": max_crops, "crop_choice": "highest score suffix (campus_track crop quality)",
                        "upscale": "height >= 448 px, longest side <= 896 px, bicubic",
                        "prompt": prompt, "system_prompt": None, "temperature": TEMPERATURE, "max_tokens": 64,
                        "vote": "majority over crops ignoring unsure; tie or no informative vote -> unsure (abstain)",
                        "probe": "campus_probe.py linear probes (DINOv2 features, mean prob over up to 6 crops, "
                                 "abstain when |p-0.5| < 0.15)",
                        "probe_attributes_json_reproduces_probe_scores": check, "server": server_info()},
           "results": results, "crop_answer_counts": crop_stats,
           "crop_valid_json_rate": r3(np.mean([r["valid_json"] for r in crop_rows])),
           "tracks": track_rows, "crops": crop_rows}
    dump(out, "e2_attributes.json")
    for f in FIELDS:
        for s in VIDEOS + ["both"]:
            vs, ps = results[f]["vlm"][s], results[f]["probe_stored"][s]
            log.info("  %-6s %-7s VLM acc %s %s cov %s (%d/%d) | probe acc %s %s cov %s | McNemar p %s", f, s,
                     vs["accuracy"], vs["ci95"], vs["coverage"], vs["correct"], vs["answered"], ps["accuracy"],
                     ps["ci95"], ps["coverage"], results[f]["mcnemar_both_answered"][s]["p_exact_two_sided"])
    log.info("  probe attributes.json reproduces probe_scores.json: %s; crop answers %s", check, crop_stats)


# ================================================================ E3 tool calling
SNAPSHOT = {
    "t": 1843.6, "frames": 27654, "occupancy": 14, "crowd_level": "moderate", "unique_people": 412,
    "crossings": {"a_to_b": 96, "b_to_a": 88}, "crossing_directions": {"a_to_b": "left to right",
                                                                      "b_to_a": "right to left"},
    "crossings_last_min": 7, "occupancy_last_1min": {"mean": 13.6, "max": 17},
    "occupancy_last_5min": {"mean": 12.9, "max": 19}, "trend": "steady", "active_tracks": 14,
    "longest_present": [{"track": 3981, "dwell_s": 41.2}, {"track": 4017, "dwell_s": 22.5},
                        {"track": 4033, "dwell_s": 12.0}],
    "dwell_finished_median_s": 4.3, "dwell_finished_p90_s": 13.8, "mean_speed_px_s": 61.7, "zones": {},
    "loitering": 0, "line": [[0.5, 0.0], [0.5, 1.0]],
    "rules": [{"name": "crowding", "metric": "occupancy", "op": ">", "value": 20, "for_s": 3, "cooldown_s": 60},
              {"name": "evening_quiet", "metric": "occupancy", "op": "<", "value": 2, "for_s": 120, "cooldown_s": 60}],
    "recent_events": [
        {"ts": 1838.9, "kind": "line_cross", "track": 4021, "direction": "b_to_a", "a_to_b": 96, "b_to_a": 87},
        {"ts": 1840.2, "kind": "person_exit", "track": 3990, "dwell_s": 6.1},
        {"ts": 1841.0, "kind": "line_cross", "track": 4030, "direction": "a_to_b", "a_to_b": 96, "b_to_a": 87},
        {"ts": 1842.7, "kind": "line_cross", "track": 4002, "direction": "b_to_a", "a_to_b": 96, "b_to_a": 88}],
}
HISTORY10 = {"minutes": 10, "samples": 600, "occupancy_mean": 12.4, "occupancy_max": 21, "new_unique_people": 133,
             "crossings": 61}
CONTEXT = (f"Measurements at t={SNAPSHOT['t']} s (JSON): {json.dumps(SNAPSHOT, separators=(',', ':'))}\n"
           f"Last 10 minutes: {json.dumps(HISTORY10, separators=(',', ':'))}")


def A(**kw):
    return kw


# (utterance, expected tool or None, [acceptable argument specs]); only listed keys are checked.
UTTERANCES = [
    ("notify me if over 15 people stay for 10 seconds", "set_alert",
     [A(metric="occupancy", op=">", value=15, for_s=10)]),
    ("Alert me when there are more than 25 people in view.", "set_alert", [A(metric="occupancy", op=">", value=25)]),
    ("Set an alarm if occupancy reaches 30 or more.", "set_alert",
     [A(metric="occupancy", op=">=", value=30), A(metric="occupancy", op=">", value=29)]),
    ("Tell me if the area stays empty for a minute.", "set_alert",
     [A(metric="occupancy", op="<", value=1, for_s=60), A(metric="occupancy", op="<=", value=0, for_s=60)]),
    ("warn when someone loiters", "set_alert",
     [A(metric="loitering", op=">", value=0), A(metric="loitering", op=">=", value=1)]),
    ("Create an alert when anyone stays longer than 2 minutes.", "set_alert",
     [A(metric="max_dwell_s", op=">", value=120), A(metric="max_dwell_s", op=">=", value=120)]),
    ("If more than 40 people cross the line in a minute, alert me.", "set_alert",
     [A(metric="crossings_last_min", op=">", value=40)]),
    ("Alert if the total number of crossings goes above 500.", "set_alert",
     [A(metric="crossings_total", op=">", value=500)]),
    ("Let me know when unique visitors exceed 1000.", "set_alert", [A(metric="unique_people", op=">", value=1000)]),
    ("Raise an alert when fewer than 3 people are present for 30 seconds.", "set_alert",
     [A(metric="occupancy", op="<", value=3, for_s=30)]),
    ("Warn me if occupancy is at least 20 for 5 seconds, and call the rule crowd20.", "set_alert",
     [A(metric="occupancy", op=">=", value=20, for_s=5, name="crowd20")]),
    ("Ping me if someone's dwell time goes over 45 seconds.", "set_alert",
     [A(metric="max_dwell_s", op=">", value=45)]),
    ("Watch for crowding: more than 18 people for 15 s.", "set_alert",
     [A(metric="occupancy", op=">", value=18, for_s=15)]),
    ("Remove the crowding alert.", "remove_alert", [A(name="crowding")]),
    ("Delete all alert rules.", "remove_alert", [A(name="all"), A(name="*")]),
    ("Turn off the evening_quiet rule.", "remove_alert", [A(name="evening_quiet")]),
    ("Clear every alert.", "remove_alert", [A(name="all"), A(name="*")]),
    ("Put a horizontal counting line across the middle of the image.", "set_counting_line",
     [A(x1=0, y1=0.5, x2=1, y2=0.5)]),
    ("Move the counting line to a vertical line at one third of the width.", "set_counting_line",
     [A(x1=1 / 3, y1=0, x2=1 / 3, y2=1)]),
    ("Make the counting line vertical through the centre.", "set_counting_line", [A(x1=0.5, y1=0, x2=0.5, y2=1)]),
    ("Set the counting line horizontally at 75% of the image height.", "set_counting_line",
     [A(x1=0, y1=0.75, x2=1, y2=0.75)]),
    ("Change the counting line to run from the top-left corner to the bottom-right corner.", "set_counting_line",
     [A(x1=0, y1=0, x2=1, y2=1)]),
    ("how busy was it in the last 30 minutes?", "get_history", [A(minutes=30)]),
    ("Give me occupancy statistics for the past hour.", "get_history", [A(minutes=60)]),
    ("How did the crowd level change over the last 2 hours?", "get_history", [A(minutes=120)]),
    ("any alerts in the last 10 minutes?", "get_recent_events", [A(minutes=10, kind="alert")]),
    ("Show me the line crossings from the last 5 minutes.", "get_recent_events", [A(minutes=5, kind="line_cross")]),
    ("Has anyone been loitering in the past 15 minutes?", "get_recent_events", [A(minutes=15, kind="loitering")]),
    ("List all events from the last 20 minutes.", "get_recent_events", [A(minutes=20, kind=None)]),
    # ---- no tool expected
    ("What are people doing?", None, []),
    ("Hi there!", None, []),
    ("Thanks, that's all for now.", None, []),
    ("How many people are in view right now?", None, []),
    ("How many unique people have been seen so far?", None, []),
    ("How many people have crossed the counting line?", None, []),
    ("Is the crowd getting bigger or smaller?", None, []),
    ("Describe the scene.", None, []),
    ("Is anyone running?", None, []),
    ("What is the weather like?", None, []),
    ("Which alert rules are active?", None, []),
    ("Is it crowded here?", None, []),
    ("What does unique_people mean?", None, []),
    ("Are there any vehicles in the frame?", None, []),
    ("What is the median dwell time?", None, []),
    ("Can you tell me who the man in the red shirt is?", None, []),
]
LINE_TOL, NUM_TOL = 0.02, 1e-6


def norm_name(s):
    return re.sub(r"[\s_\-]+", "_", str(s).strip().lower())


def lenient(spec):
    """Lenient variant: strict and non-strict inequalities count as the same, the event kind and the rule name
    are not checked (numbers, metric and line coordinates still are)."""
    return {k: v for k, v in spec.items() if k not in ("op", "kind", "name")}


def arg_match(tool, args, spec):
    """True if `args` satisfies one acceptable spec. kind=None means absent or 'any'."""
    if not isinstance(args, dict):
        return False
    tol = LINE_TOL if tool == "set_counting_line" else NUM_TOL

    def ok(k, want, a):
        got = a.get(k)
        if want is None:
            return got in (None, "any", "")
        if isinstance(want, (int, float)):
            try:
                return abs(float(got) - want) <= tol
            except (TypeError, ValueError):
                return False
        return got is not None and norm_name(got) == norm_name(want)

    if all(ok(k, w, args) for k, w in spec.items()):
        return True
    if tool == "set_counting_line":                       # same line, endpoints swapped
        sw = {"x1": args.get("x2"), "y1": args.get("y2"), "x2": args.get("x1"), "y2": args.get("y1")}
        return all(ok(k, w, sw) for k, w in spec.items())
    return False


def e3():
    c = client(); wait_server(c)
    coco = json.load(open(DATA / "_annotations.coco.json"))
    im = sorted(coco["images"], key=lambda i: i["frame"])[len(coco["images"]) // 2]
    frame = cv2.imread(str(DATA / im["file_name"]))
    system = SYSTEM + "\n\n" + CONTEXT
    n_tool = sum(u[1] is not None for u in UTTERANCES)
    log.info("E3: %d utterances (%d tool, %d no-tool), frame %s", len(UTTERANCES), n_tool, len(UTTERANCES) - n_tool,
             im["file_name"])
    out = {"experiment": "E3 tool-calling reliability", "protocol": __doc__.split("  e4 ")[0].split("  e3 ")[1].strip(),
           "settings": {"system_prompt": system, "tools": TOOLS, "temperature": TEMPERATURE, "max_tokens": 384,
                        "frame": im["file_name"], "image_max_side": IMAGE_MAX_SIDE, "rounds": 1,
                        "tolerances": {"numeric": NUM_TOL, "counting_line": LINE_TOL},
                        "utterances": [{"text": u, "expected_tool": t, "acceptable_args": s} for u, t, s in UTTERANCES],
                        "server": server_info()},
           "conditions": {}}
    for cond in ("with_frame", "text_only"):
        rows = []
        for u, exp_tool, specs in UTTERANCES:
            user = image_msg(u, frame) if cond == "with_frame" else {"role": "user", "content": u}
            msgs = [{"role": "system", "content": system}, user]
            msg, info = chat(c, msgs, tools=TOOLS, max_tokens=384)
            calls = msg.get("tool_calls") or []
            source = "server" if calls else None
            if not calls:
                calls = text_tool_calls(msg.get("content")); source = "text" if calls else None
            parsed = []
            for cl in calls:
                fn = cl.get("function", {})
                raw = fn.get("arguments")
                try:
                    args = raw if isinstance(raw, dict) else json.loads(raw or "{}"); valid = isinstance(args, dict)
                except (json.JSONDecodeError, TypeError):
                    args, valid = raw, False
                parsed.append({"name": fn.get("name"), "arguments_raw": raw, "arguments": args, "json_valid": valid})
            names = [p["name"] for p in parsed]
            pred = "none" if not names else names[0] if len(names) == 1 else "multiple:" + "+".join(names)
            sel_ok = (exp_tool is None and not names) or (exp_tool is not None and names == [exp_tool])
            args_ok = bool(sel_ok and exp_tool is not None and parsed[0]["json_valid"]
                           and any(arg_match(exp_tool, parsed[0]["arguments"], s) for s in specs))
            args_len = bool(sel_ok and exp_tool is not None and parsed[0]["json_valid"]
                            and any(arg_match(exp_tool, parsed[0]["arguments"], lenient(s)) for s in specs))
            rows.append({"utterance": u, "expected_tool": exp_tool, "predicted": pred, "selection_correct": sel_ok,
                         "args_correct": args_ok if exp_tool else None,
                         "args_correct_lenient": args_len if exp_tool else None, "calls": parsed, "call_source": source,
                         "content": msg.get("content"), "latency_s": info["latency_s"],
                         "prompt_tokens": info["prompt_tokens"], "completion_tokens": info["completion_tokens"],
                         "request": {"messages": strip_images(msgs), "tools": "TOOLS (see settings)",
                                     "temperature": TEMPERATURE, "max_tokens": 384},
                         "response_message": msg})
            log.info("  [%s] %-80s exp %-18s got %-24s sel %s args %s", cond, u[:80], exp_tool, pred, sel_ok,
                     rows[-1]["args_correct"])
        tool_rows = [r for r in rows if r["expected_tool"]]
        none_rows = [r for r in rows if not r["expected_tool"]]
        all_calls = [cl for r in rows for cl in r["calls"]]
        conf = defaultdict(Counter)
        for r in rows:
            conf[r["expected_tool"] or "none"][r["predicted"]] += 1
        per_tool = {}
        for t in sorted({r["expected_tool"] for r in tool_rows}):
            rr = [r for r in tool_rows if r["expected_tool"] == t]
            per_tool[t] = {"n": len(rr), "selection_acc": r3(np.mean([r["selection_correct"] for r in rr])),
                           "args_acc": r3(np.mean([r["args_correct"] for r in rr]))}
        sel_tool = [r["selection_correct"] for r in tool_rows]
        m = {"n_utterances": len(rows), "n_tool": len(tool_rows), "n_no_tool": len(none_rows),
             "tool_selection_accuracy_all": r3(np.mean([r["selection_correct"] for r in rows])),
             "tool_selection_accuracy_all_ci95": wilson(sum(r["selection_correct"] for r in rows), len(rows)),
             "tool_selection_accuracy_tool_utterances": r3(np.mean(sel_tool)),
             "argument_exact_match_accuracy": r3(np.mean([r["args_correct"] for r in tool_rows])),
             "argument_exact_match_ci95": wilson(sum(r["args_correct"] for r in tool_rows), len(tool_rows)),
             "argument_match_lenient_accuracy": r3(np.mean([r["args_correct_lenient"] for r in tool_rows])),
             "argument_accuracy_given_correct_tool": r3(np.mean([r["args_correct"] for r in tool_rows
                                                                if r["selection_correct"]])) if any(sel_tool) else None,
             "false_positive_tool_call_rate": r3(np.mean([r["predicted"] != "none" for r in none_rows])),
             "missed_tool_call_rate": r3(np.mean([r["predicted"] == "none" for r in tool_rows])),
             "json_valid_arguments_rate": r3(np.mean([cl["json_valid"] for cl in all_calls])) if all_calls else None,
             "n_calls_emitted": len(all_calls),
             "calls_parsed_from_text": sum(r["call_source"] == "text" for r in rows),
             "median_latency_s": r3(np.median([r["latency_s"] for r in rows])),
             "per_tool": per_tool, "confusion": {k: dict(v) for k, v in conf.items()}}
        out["conditions"][cond] = {"metrics": m, "rows": rows}
        log.info("  [%s] selection acc %s (tool utts %s), args acc %s, FP rate %s, JSON valid %s", cond,
                 m["tool_selection_accuracy_all"], m["tool_selection_accuracy_tool_utterances"],
                 m["argument_exact_match_accuracy"], m["false_positive_tool_call_rate"], m["json_valid_arguments_rate"])
    dump(out, "e3_tools.json")


# ================================================================ E4 latency
def timed_post(c, messages, tools=None, max_tokens=256):
    """One request through VLMClient's transport with the prompt cache off (cold prefill every call)."""
    payload = {"model": c.model, "messages": messages, "max_tokens": max_tokens, "temperature": TEMPERATURE,
               "stream": False, "cache_prompt": False}
    if tools:
        payload["tools"] = tools
    for attempt in range(5):
        try:
            t = time.time(); r = c._post("/chat/completions", payload); lat = time.time() - t
            break
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as e:
            log.info("request failed (%s), waiting for the server", e); wait_server(c)
    else:
        raise RuntimeError("VLM request failed 5 times")
    tm, u = r.get("timings", {}), r.get("usage", {})
    msg = r["choices"][0]["message"]
    return {"latency_s": round(lat, 4), "prompt_tokens": u.get("prompt_tokens"),
            "completion_tokens": u.get("completion_tokens"), "prompt_n": tm.get("prompt_n"),
            "cache_n": tm.get("cache_n"), "prefill_ms": tm.get("prompt_ms"), "prefill_tok_s": tm.get("prompt_per_second"),
            "decode_n": tm.get("predicted_n"), "decode_ms": tm.get("predicted_ms"),
            "decode_tok_s": tm.get("predicted_per_second"), "tool_call": bool(msg.get("tool_calls")),
            "reply": (msg.get("content") or "")[:200]}


def summarise(rows):
    def q(k, p):
        v = [r[k] for r in rows if r.get(k) is not None]
        return r3(np.percentile(v, p)) if v else None
    return {"n": len(rows), "prompt_tokens_median": q("prompt_tokens", 50), "completion_tokens_median": q("completion_tokens", 50),
            "prefill_ms_median": q("prefill_ms", 50), "prefill_tok_s_median": q("prefill_tok_s", 50),
            "decode_tok_s_median": q("decode_tok_s", 50), "decode_ms_median": q("decode_ms", 50),
            "e2e_s_median": q("latency_s", 50), "e2e_s_p90": q("latency_s", 90),
            "encode_ms_median": q("encode_ms", 50)}


def e4(sizes=(448, 672, 896, 1280), n=10, warm=2):
    c = client(); wait_server(c)
    coco = json.load(open(DATA / "_annotations.coco.json"))
    images = sorted(coco["images"], key=lambda i: i["frame"])
    system = SYSTEM + "\n\n" + CONTEXT
    gpu_before = gpu_snapshot()
    log.info("E4: sizes %s, %d frames each after %d warm-up; GPU %s", sizes, n, warm, gpu_before)
    groups, rows = {}, []
    for s in sizes:
        rr = []
        for k in range(warm + n):
            img = cv2.imread(str(DATA / images[(k * 13) % len(images)]["file_name"]))
            t = time.time(); url = encode_image(img, s); enc = (time.time() - t) * 1000
            msgs = [{"role": "system", "content": system},
                    {"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}},
                                                 {"type": "text", "text": OBSERVE}]}]
            r = timed_post(c, msgs, max_tokens=256)
            r.update({"condition": f"image {s}", "image_max_side": s, "warmup": k < warm, "encode_ms": round(enc, 2),
                      "sent_wh": [round(img.shape[1] * min(1, s / max(img.shape[:2]))),
                                  round(img.shape[0] * min(1, s / max(img.shape[:2])))]})
            rows.append(r)
            if k >= warm:
                rr.append(r)
        groups[f"image {s}"] = summarise(rr)
        log.info("  image %4d: %s", s, groups[f"image {s}"])
    for name, msgs, tools in (
            ("text chat", [{"role": "system", "content": system},
                           {"role": "user", "content": "How many people have crossed the line so far, and is it busy?"}], None),
            ("tool call", [{"role": "system", "content": system},
                           {"role": "user", "content": "notify me if over 15 people stay for 10 seconds"}], TOOLS)):
        rr = []
        for k in range(warm + n):
            r = timed_post(c, msgs, tools=tools, max_tokens=384 if tools else 256)
            r.update({"condition": name, "warmup": k < warm}); rows.append(r)
            if k >= warm:
                rr.append(r)
        groups[name] = summarise(rr)
        log.info("  %s: %s", name, groups[name])
    out = {"experiment": "E4 VLM latency profile", "protocol": __doc__.split("    env -u")[0].split("  e4 ")[1].strip(),
           "settings": {"machine": "RTX 4060 Laptop GPU (8 GB), llama.cpp Vulkan backend; desktop, NOT Jetson",
                        "image_prompt": OBSERVE, "system_prompt": system, "temperature": TEMPERATURE,
                        "max_tokens": {"image": 256, "text chat": 256, "tool call": 384}, "cache_prompt": False,
                        "warmup_calls": warm, "measured_calls": n, "frames": "test frames, every 13th, 1280x720 JPEG",
                        "gpu_before": gpu_before, "gpu_after": gpu_snapshot(), "server": server_info(),
                        "notes": "prefill_ms is the server's timings.prompt_ms for the whole prompt (in llama.cpp's "
                                 "multimodal path this includes running the vision encoder on the image chunk); "
                                 "e2e is wall time of the HTTP request, client-side JPEG encode excluded "
                                 "(encode_ms reported separately)."},
           "summary": groups, "rows": rows}
    dump(out, "e4_latency.json")
    fig_e4(out)


def fig_e4(out):
    plt, V = plt_style()
    S = out["summary"]; names = list(S)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 4.2))
    ax.set_axisbelow(True); bx.set_axisbelow(True)
    x = np.arange(len(names))
    pre = np.array([S[k]["prefill_ms_median"] / 1000 for k in names])
    dec = np.array([S[k]["decode_ms_median"] / 1000 for k in names])
    ax.bar(x, pre, width=0.6, color=V.COLOR["YOLO26n"], edgecolor=V.SURF, label="prefill (server)")
    ax.bar(x, dec, width=0.6, bottom=pre, color=V.COLOR["RF-DETR Nano"], edgecolor=V.SURF, label="decode (server)")
    ax.scatter(x, [S[k]["e2e_s_p90"] for k in names], marker="_", s=300, color=V.INK, zorder=3, label="end-to-end p90")
    for i, k in enumerate(names):
        ax.text(i, S[k]["e2e_s_p90"], f"{S[k]['e2e_s_median']:.2f} s", ha="center", va="bottom", fontsize=8,
                color=V.INK2)
    ax.set_xticks(x, [k.replace("image ", "image\n") + (" px" if k.startswith("image") else "") for k in names],
                  fontsize=8.5)
    ax.set_ylabel("seconds per request"); ax.set_title("Latency per request (median; label = e2e median)")
    ax.legend(fontsize=8, loc="upper left"); ax.grid(axis="x", visible=False); ax.margins(y=0.15)
    bx.bar(x, [S[k]["prompt_tokens_median"] for k in names], width=0.6, color=V.COLOR["Teacher"], edgecolor=V.SURF)
    for i, k in enumerate(names):
        bx.text(i, S[k]["prompt_tokens_median"], f"{S[k]['prompt_tokens_median']:.0f} tok\n"
                f"{S[k]['decode_tok_s_median']:.0f} tok/s dec", ha="center", va="bottom", fontsize=7.5, color=V.INK2)
    bx.set_xticks(x, [k.replace("image ", "image\n") for k in names], fontsize=8.5)
    bx.set_ylabel("prompt tokens"); bx.set_title("Prompt size and decode speed"); bx.grid(axis="x", visible=False)
    bx.margins(y=0.2)
    fig.suptitle("Qwen3-VL-2B Q4_K_M, llama.cpp Vulkan, RTX 4060 Laptop (desktop, not Jetson)", fontsize=9,
                 color=V.INK2, x=0.01, ha="left")
    save_fig(fig, "fig_e4_latency.png")


# ================================================================ E3b tool calling through the deployed agent
def deployed_agent(frame, **over):
    """A real edgevision SceneAgent (two-stage ask) on an in-memory store, with the E3 measurement context."""
    import copy
    from edgevision.analytics import Analytics
    from edgevision.config import DEFAULTS
    from edgevision.store import Store
    from edgevision.vlm import SceneAgent
    cfg = copy.deepcopy(DEFAULTS["vlm"]); cfg.update(temperature=TEMPERATURE, image="raw", **over)
    an = Analytics(copy.deepcopy(DEFAULTS["analytics"]), SNAPSHOT["rules"], frame.shape[1], frame.shape[0], 15)
    agent = SceneAgent(cfg, an, Store(":memory:"), lambda: (frame, frame, SNAPSHOT["t"]))
    agent.context = lambda: CONTEXT                       # same measurements text as E3
    return agent


def e3b():
    """E3 utterances through SceneAgent.ask (stage 1 text-only routing + tools, stage 2 image answer), scored on
    the tool calls that took effect. "deployed" = model calls + deterministic guardrails (intent check that refuses
    contradicting line arguments, rule-based fallback for unambiguous commands); "model_only" = the model's own
    successful calls. A fresh agent per utterance (rules and line reset)."""
    c = client(); wait_server(c)
    coco = json.load(open(DATA / "_annotations.coco.json"))
    im = sorted(coco["images"], key=lambda i: i["frame"])[len(coco["images"]) // 2]
    frame = cv2.imread(str(DATA / im["file_name"]))
    rows = []
    for u, exp_tool, specs in UTTERANCES:
        agent = deployed_agent(frame)
        calls = []
        orig = agent.client.chat

        def counted(*a, _orig=orig, **k):
            calls.append(1); return _orig(*a, **k)
        agent.client.chat = counted
        t = time.time(); r = agent.ask(u); lat = time.time() - t
        def judge(acts):
            """Score the calls that took effect (a refused call, e.g. by the intent guardrail, changed nothing;
            read-only calls always count)."""
            eff = [x for x in acts if not (isinstance(x["result"], dict) and x["result"].get("ok") is False)
                   or x["tool"] in ("get_history", "get_recent_events")]
            names = sorted({x["tool"] for x in eff})
            sel = (exp_tool is None and not eff) or (exp_tool is not None and names == [exp_tool])
            first = next((x for x in eff if x["tool"] == exp_tool), None)
            args = first["args"] if first else None
            return (sel, bool(sel and exp_tool and any(arg_match(exp_tool, args, sp) for sp in specs)),
                    bool(sel and exp_tool and any(arg_match(exp_tool, args, lenient(sp)) for sp in specs)))
        acts = r["actions"]
        sel_ok, args_ok, args_len = judge(acts)                       # deployed: model + guardrails
        m_sel, m_args, m_len = judge([x for x in acts if x.get("source") == "model"])   # model calls only
        rows.append({"utterance": u, "expected_tool": exp_tool,
                     "executed": [{"tool": x["tool"], "args": x["args"], "source": x.get("source"),
                                   "ok": not (isinstance(x["result"], dict) and x["result"].get("ok") is False)}
                                  for x in acts],
                     "selection_correct": sel_ok, "args_correct": args_ok if exp_tool else None,
                     "args_correct_lenient": args_len if exp_tool else None,
                     "model_only": {"selection_correct": m_sel, "args_correct": m_args if exp_tool else None,
                                    "args_correct_lenient": m_len if exp_tool else None},
                     "answer": r["answer"], "latency_s": round(lat, 3), "requests": len(calls)})
        log.info("  %-80s exp %-18s got %-30s sel %s args %s  %.2fs/%d req", u[:80], exp_tool,
                 [(x["tool"], x["source"][0], x["ok"]) for x in rows[-1]["executed"]], sel_ok,
                 rows[-1]["args_correct"], lat, len(calls))
    tool_rows = [r for r in rows if r["expected_tool"]]; none_rows = [r for r in rows if not r["expected_tool"]]
    m = {"n_utterances": len(rows),
         "tool_selection_accuracy_all": r3(np.mean([r["selection_correct"] for r in rows])),
         "tool_selection_accuracy_all_ci95": wilson(sum(r["selection_correct"] for r in rows), len(rows)),
         "tool_selection_accuracy_tool_utterances": r3(np.mean([r["selection_correct"] for r in tool_rows])),
         "argument_exact_match_accuracy": r3(np.mean([r["args_correct"] for r in tool_rows])),
         "argument_exact_match_ci95": wilson(sum(r["args_correct"] for r in tool_rows), len(tool_rows)),
         "argument_match_lenient_accuracy": r3(np.mean([r["args_correct_lenient"] for r in tool_rows])),
         "false_positive_tool_call_rate": r3(np.mean([bool(r["executed"]) for r in none_rows])),
         "missed_tool_call_rate": r3(np.mean([not r["executed"] for r in tool_rows])),
         "median_latency_s": r3(np.median([r["latency_s"] for r in rows])),
         "median_latency_no_tool_s": r3(np.median([r["latency_s"] for r in none_rows])),
         "mean_requests_per_turn": r3(np.mean([r["requests"] for r in rows])),
         "fallback_actions": sum(1 for r in rows for x in r["executed"] if x["source"] == "fallback"),
         "model_only": {
             "tool_selection_accuracy_all": r3(np.mean([r["model_only"]["selection_correct"] for r in rows])),
             "argument_exact_match_accuracy": r3(np.mean([r["model_only"]["args_correct"] for r in tool_rows])),
             "argument_match_lenient_accuracy": r3(np.mean([r["model_only"]["args_correct_lenient"] for r in tool_rows])),
             "false_positive_tool_call_rate": r3(np.mean([not r["model_only"]["selection_correct"] for r in none_rows]))}}
    old = json.load(open(OUT / "e3_tools.json"))["conditions"]
    out = {"experiment": "E3b tool calling through the deployed two-stage SceneAgent.ask",
           "protocol": e3b.__doc__, "settings": {"frame": im["file_name"], "temperature": TEMPERATURE,
                                                 "context": "E3 CONTEXT", "server": server_info()},
           "metrics": m, "compare_single_stage": {k: old[k]["metrics"] for k in ("with_frame", "text_only")},
           "rows": rows}
    log.info("E3b: %s", json.dumps(m))
    dump(out, "e3b_tools_deployed.json")


# ================================================================ E5 alert verification
def e5(n_frames=60):
    """Alert verification by the VLM (SceneAgent.verify) on true vs fabricated occupancy alerts.
    TRUE alert:  rule occupancy > GT-5, measured value GT          (the situation is real)
    FALSE alert: rule occupancy > GT+8, claimed measured value GT+10 (as if the detector over-counted)
    Modes: blind (image only, no measured value, no measurements context) vs informed (old prompt: measured
    value + live measurements, whose occupancy equals the alert's measured value, as in deployment).
    Raw frames are sent (no boxes drawn). GT = pseudo-GT person count."""
    c = client(); wait_server(c)
    coco = json.load(open(DATA / "_annotations.coco.json"))
    ims = sorted(coco["images"], key=lambda i: i["frame"])
    ims = ims[:: max(1, len(ims) // n_frames)][:n_frames]
    gt = Counter(a["image_id"] for a in coco["annotations"])
    res = {}
    for mode in ("blind", "informed"):
        rows = []
        for im in ims:
            frame = cv2.imread(str(DATA / im["file_name"]))
            g = gt[im["id"]]
            for kind, thr, val in (("true", g - 5, g), ("false", g + 8, g + 10)):
                agent = deployed_agent(frame, verify_mode=mode)
                snap = dict(SNAPSHOT, occupancy=val, crowd_level="high" if val > 15 else "moderate",
                            occupancy_last_1min={"mean": float(val), "max": val}, active_tracks=val)
                ctx = (f"Measurements at t={snap['t']} s (JSON): {json.dumps(snap, separators=(',', ':'))}\n"
                       f"Last 10 minutes: {json.dumps(HISTORY10, separators=(',', ':'))}")
                agent.context = lambda ctx=ctx: ctx       # measurements consistent with the alert, as in deployment
                t = time.time()
                r = agent.verify({"rule": "crowding", "metric": "occupancy", "op": ">", "threshold": thr, "value": val})
                rows.append({"image": im["file_name"], "gt": g, "alert": kind, "threshold": thr, "claimed_value": val,
                             "confirmed": r["confirmed"], "reason": r["reason"], "latency_s": round(time.time() - t, 3)})
        tp = [r["confirmed"] for r in rows if r["alert"] == "true"]
        tn = [not r["confirmed"] for r in rows if r["alert"] == "false"]
        sens, spec = float(np.mean(tp)), float(np.mean(tn))
        res[mode] = {"metrics": {"n_true": len(tp), "n_false": len(tn),
                                 "sensitivity_confirm_true": r3(sens), "sensitivity_ci95": wilson(sum(tp), len(tp)),
                                 "specificity_reject_false": r3(spec), "specificity_ci95": wilson(sum(tn), len(tn)),
                                 "balanced_accuracy": r3((sens + spec) / 2),
                                 "median_latency_s": r3(np.median([r["latency_s"] for r in rows]))},
                     "rows": rows}
        log.info("E5 %s: %s", mode, json.dumps(res[mode]["metrics"]))
    out = {"experiment": "E5 alert verification", "protocol": e5.__doc__,
           "note": "a 2B VLM counts crowds poorly (E1 image-only MAE ~8), so count-threshold verification from the "
                   "image alone is expected to be hard; the informed prompt reveals the claimed value",
           "settings": {"frames": len(ims), "temperature": TEMPERATURE, "server": server_info()}, "modes": res}
    dump(out, "e5_verify.json")


# ================================================================ main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="exp", required=True)
    for k, h in (("e1", "count grounding"), ("e2", "zero-shot attributes vs probes"), ("e3", "tool calling"),
                 ("e4", "latency"), ("e3b", "tool calling through the deployed agent"),
                 ("e5", "alert verification"), ("all", "e1-e5"), ("figs", "re-draw figures from saved JSON")):
        sub.add_parser(k, help=h)
    a = ap.parse_args()
    exps = {"e1": e1, "e2": e2, "e3": e3, "e4": e4, "e3b": e3b, "e5": e5}
    if a.exp == "figs":
        setup_log("figs")
        fig_e1(json.load(open(OUT / "e1_counting.json"))); fig_e4(json.load(open(OUT / "e4_latency.json")))
        return
    for k in (list(exps) if a.exp == "all" else [a.exp]):
        setup_log(k); t = time.time()
        log.info("== %s", k)
        exps[k]()
        log.info("== %s done in %.1f min", k, (time.time() - t) / 60)


if __name__ == "__main__":
    main()
