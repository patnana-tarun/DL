"""Write paper/numbers.tex: one LaTeX macro for every result number used in main.tex.

Every number in the paper text and tables comes from the result JSON files through this script, so the PDF is
always consistent with results/. Results that do not exist yet are defined as \\textbf{[TBD]} and listed in a
warning, so the paper still compiles and can simply be rebuilt when the jobs finish (bash paper/build.sh).

Also written:
  \\ifHave<X> switches (HaveTrackXc, HaveEdge, HaveEdgeLive, HaveVLM, HaveVLMEOne..Four, HaveJetson) so main.tex
  can include or skip whole paragraphs/tables depending on which results exist.

Sources (relative to the project root):
  results/metrics.json, metrics_splits.json, significance.json, latency.json, teacher_bias.json,
  crosscheck_detection.json, crosscheck_tracking.json, model_selection.json,
  results/tracking/{tracking_metrics,tracker_comparison,pseudo_gt_info,grid_search,analytics_*}.json,
  results/campus/{video_1,video_2}/summary.json, results/campus/validation/probe_scores.json,
  results/edge/edge_eval.json (research/edge_eval.py), results/vlm/*.json (VLM evaluation),
  results/edge/demo/{report,transcript}.json (recorded closed-loop session),
  datasets/stats.json; held-out video: results/test1/{detection,significance,tracking,tracker_comparison,analytics,
  edge}.json, results/test1/audit/audit.json (research/heldout/), datasets/test1/stats.json.

    env -u PYTHONPATH .venv/bin/python paper/make_numbers.py
"""
import json
import math
import re
import sys
from pathlib import Path

PAPER = Path(__file__).resolve().parent
ROOT = PAPER.parent
R = ROOT / "results"
TBD = r"\textbf{[TBD]}"

MACROS = {}          # name -> value (insertion ordered)
MISSING = {}         # source -> list of macro names left TBD
FLAGS = {}           # \ifHave<name> -> bool


# ----------------------------------------------------------------------------- helpers
def load(rel):
    p = ROOT / rel
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError) as e:
        print(f"WARNING: cannot parse {rel}: {e}", file=sys.stderr)
        return None


def M(name, value, source=None):
    """Define \\name. value None -> TBD (recorded against `source`)."""
    assert re.fullmatch(r"[A-Za-z]+", name), f"bad macro name {name!r}"
    assert name not in MACROS, f"duplicate macro {name}"
    if value is None:
        MACROS[name] = TBD
        MISSING.setdefault(source or "?", []).append(name)
    else:
        MACROS[name] = str(value)


def get(d, *keys):
    for k in keys:
        if d is None:
            return None
        if isinstance(d, dict):
            d = d.get(k)
        elif isinstance(d, (list, tuple)) and isinstance(k, int) and -len(d) <= k < len(d):
            d = d[k]
        else:
            return None
    return d


def _num(x):
    return x is not None and isinstance(x, (int, float)) and not (isinstance(x, float) and math.isnan(x))


def f(x, nd=3):
    """Fixed decimals; negative numbers with a typographic minus."""
    if not _num(x):
        return None
    s = f"{abs(x):.{nd}f}"
    return ("$-$" + s) if x < 0 and float(s) != 0 else s


def sf(x, nd=3):
    """Signed fixed decimals (+0.099 / -0.026)."""
    if not _num(x):
        return None
    s = f"{abs(x):.{nd}f}"
    return ("$-$" if x < 0 and float(s) != 0 else "$+$") + s


def i(x):
    if not _num(x):
        return None
    return f"{int(round(x)):,}".replace(",", "{,}")


def pct(x, nd=0):
    if not _num(x):
        return None
    v = f"{abs(100 * x):.{nd}f}\\%"
    return ("$-$" + v) if x < 0 and float(f"{abs(100 * x):.{nd}f}") != 0 else v


def pval(p, B=1000):
    """Bootstrap p-value: an estimate of 0 with B resamples means p < 1/B."""
    if not _num(p):
        return None
    if p <= 0:
        return f"$p<{1 / B:g}$"
    return f"$p={p:.2f}$" if p >= 0.01 else f"$p={p:.3f}$"


def tex_escape(s):
    return (str(s).replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("%", r"\%")
            .replace("&", r"\&").replace("#", r"\#"))


def flag(name, value):
    FLAGS[name] = bool(value)


# ----------------------------------------------------------------------------- detection
DET = {"YoloPt": "YOLO26n (pretrained)", "YoloFt": "YOLO26n (fine-tuned)",
       "RfPt": "RF-DETR Nano (pretrained)", "RfFt": "RF-DETR Nano (fine-tuned)"}


def detection():
    src = "results/metrics.json"
    m = load(src)
    for k, name in DET.items():
        d = get(m, name)
        M(f"Ap{k}", f(get(d, "AP50-95")), src)
        M(f"ApLo{k}", f(get(d, "AP50-95_CI95", 0)), src)
        M(f"ApHi{k}", f(get(d, "AP50-95_CI95", 1)), src)
        M(f"ApFifty{k}", f(get(d, "AP50")), src)
        M(f"ApFiftyLo{k}", f(get(d, "AP50_CI95", 0)), src)
        M(f"ApFiftyHi{k}", f(get(d, "AP50_CI95", 1)), src)
        M(f"ApSeventyFive{k}", f(get(d, "AP75")), src)
        M(f"ApSmall{k}", f(get(d, "AP_small")), src)
        M(f"ApMedium{k}", f(get(d, "AP_medium")), src)
        M(f"ApLarge{k}", f(get(d, "AP_large")), src)
        M(f"Ar{k}", f(get(d, "AR@300")), src)
        M(f"Fone{k}", f(get(d, "F1")), src)
        M(f"FoneLo{k}", f(get(d, "F1_CI95", 0)), src)
        M(f"FoneHi{k}", f(get(d, "F1_CI95", 1)), src)
        M(f"Prec{k}", f(get(d, "precision")), src)
        M(f"Rec{k}", f(get(d, "recall")), src)
        M(f"Conf{k}", f(get(d, "conf"), 2), src)
        M(f"Lamr{k}", f(get(d, "LAMR")), src)
        M(f"Fppi{k}", f(get(d, "FPPI"), 2), src)
        M(f"CountMae{k}", f(get(d, "count_MAE"), 2), src)
        M(f"Params{k}", f(get(d, "params_M"), 1), src)
        M(f"Gflops{k}", f(get(d, "GFLOPs"), 1), src)

    src = "results/metrics_splits.json"
    s = load(src)
    for k, name in DET.items():
        for sp, nm in (("train", "Train"), ("valid", "Valid"), ("test", "Test")):
            M(f"Ap{nm}{k}", f(get(s, name, sp, "AP50-95")), src)
    for k in ("YoloFt", "RfFt"):
        dg = get(s, DET[k], "diagnosis")
        M(f"Gap{k}", sf(get(dg, "train_minus_test")), src)
        M(f"GapPt{k}", sf(get(dg, "pretrained_train_minus_test")), src)
        M(f"Excess{k}", sf(get(dg, "excess_gap_overfit")), src)
        M(f"GapValid{k}", sf(get(dg, "train_minus_valid")), src)

    src = "results/significance.json"
    sg = load(src)
    pairs = {"FtFt": ("YOLO26n (fine-tuned)", "RF-DETR Nano (fine-tuned)"),
             "YoloFtPt": ("YOLO26n (pretrained)", "YOLO26n (fine-tuned)"),
             "RfFtPt": ("RF-DETR Nano (pretrained)", "RF-DETR Nano (fine-tuned)"),
             "YoloPtRfFt": ("YOLO26n (pretrained)", "RF-DETR Nano (fine-tuned)")}
    for k, (a, b) in pairs.items():
        d, sign = get(sg, f"{a} vs {b}"), 1
        if d is None:
            d, sign = get(sg, f"{b} vs {a}"), -1
        if k in ("YoloFtPt", "RfFtPt"):       # report as fine-tuned minus pretrained
            sign = -sign
        dl, ci = get(d, "delta_AP50-95"), get(d, "CI95")
        M(f"Delta{k}", sf(sign * dl) if _num(dl) else None, src)
        lo, hi = (sorted([sign * ci[0], sign * ci[1]]) if ci else (None, None))
        M(f"DeltaLo{k}", sf(lo), src)
        M(f"DeltaHi{k}", sf(hi), src)
        M(f"Pval{k}", pval(get(d, "p_two_sided")), src)

    st = load("datasets/stats.json")
    for sp, nm in (("train", "Train"), ("valid", "Valid"), ("test", "Test")):
        M(f"Imgs{nm}", i(get(st, sp, "images")), "datasets/stats.json")
        M(f"Boxes{nm}", i(get(st, sp, "boxes")), "datasets/stats.json")
    tot = sum(get(st, sp, "boxes") or 0 for sp in ("train", "valid", "test")) if st else None
    M("BoxesTotal", i(tot), "datasets/stats.json")
    if st:
        M("BoxesPerFrameTest", f(st["test"]["boxes"] / st["test"]["images"], 1))
    else:
        M("BoxesPerFrameTest", None, "datasets/stats.json")

    src = "results/crosscheck_detection.json"
    xc = load(src)
    M("XcPass", i(sum(1 for r in xc if r.get("pass"))) if xc else None, src)
    M("XcTotal", i(len(xc)) if xc else None, src)
    d1 = [r.get("max_abs_diff") for r in (xc or []) if str(r.get("check", "")).startswith("D1")]
    M("XcApMaxDiff", f(max(d1), 4) if d1 else None, src)
    d4 = [r for r in (xc or []) if str(r.get("check", "")).startswith("D4")]
    M("XcUltraDiff", f(d4[0].get("max_abs_diff"), 3) if d4 else None, src)

    src = "results/crosscheck_tracking.json"
    xt = load(src)
    flag("TrackXc", bool(xt))
    M("XtPass", i(sum(1 for r in xt if r.get("pass"))) if xt else None, src)
    M("XtTotal", i(len(xt)) if xt else None, src)


# ----------------------------------------------------------------------------- latency
LAT = {"YoloBase": "YOLO26n | Ultralytics predict FP32 (baseline)",
       "YoloHalf": "YOLO26n | Ultralytics predict FP16",
       "YoloNmsFree": "YOLO26n | NMS-free head FP16",
       "YoloNmsFreeFast": "YOLO26n | NMS-free + GPU pre-proc + fused FP16",
       "YoloDep": "YOLO26n | GPU pre-proc + fused FP16 + GPU NMS (deployed)",
       "RfBase": "RF-DETR Nano | predict FP32, unoptimized (baseline)",
       "RfTs": "RF-DETR Nano | FP32 + TorchScript",
       "RfHalf": "RF-DETR Nano | FP16, no compile",
       "RfHalfTs": "RF-DETR Nano | FP16 + TorchScript",
       "RfDep": "RF-DETR Nano | FP16 + TorchScript + GPU pre-proc (deployed)"}


def latency():
    src = "results/latency.json"
    L = load(src)
    for k, name in LAT.items():
        d = get(L, name)
        M(f"Lat{k}", f(get(d, "median_ms"), 2), src)
        M(f"LatPn{k}", f(get(d, "p90_ms"), 2), src)
        M(f"Fps{k}", i(get(d, "FPS")), src)
        M(f"LatAp{k}", f(get(d, "AP50-95")), src)
    g = lambda k, f_: get(L, LAT[k], f_)  # noqa: E731
    ok = L and all(_num(g(k, "median_ms")) for k in ("YoloBase", "YoloDep", "RfBase", "RfDep"))
    M("SpeedupYolo", f(g("YoloBase", "median_ms") / g("YoloDep", "median_ms"), 1) if ok else None, src)
    M("SpeedupRf", f(g("RfBase", "median_ms") / g("RfDep", "median_ms"), 1) if ok else None, src)
    M("LatRatio", f(g("RfDep", "median_ms") / g("YoloDep", "median_ms"), 1) if ok else None, src)
    ok2 = L and _num(g("YoloDep", "AP50-95")) and _num(g("YoloNmsFreeFast", "AP50-95"))
    M("NmsFreeApLoss", f(g("YoloDep", "AP50-95") - g("YoloNmsFreeFast", "AP50-95")) if ok2 else None, src)


# ----------------------------------------------------------------------------- label bias
def teacher_bias():
    src = "results/teacher_bias.json"
    T = load(src)
    cols = {"Fused": "Fused (published labels)", "RfL": "RF-DETR Large only", "Yx": "YOLO26x only"}
    for k, name in DET.items():
        for c, cn in cols.items():
            M(f"Tb{k}{c}", f(get(T, name, cn, "AP50-95")), src)
    for c, cn in cols.items():
        M(f"Labels{c}", i(get(T, "_label_counts", cn)), src)
    a, b = get(T, DET["YoloFt"], cols["RfL"], "AP50-95"), get(T, DET["RfFt"], cols["RfL"], "AP50-95")
    M("TbGapRfL", sf(a - b) if _num(a) and _num(b) else None, src)
    a, b = get(T, DET["YoloFt"], cols["Yx"], "AP50-95"), get(T, DET["RfFt"], cols["Yx"], "AP50-95")
    M("TbGapYx", sf(a - b) if _num(a) and _num(b) else None, src)


# ----------------------------------------------------------------------------- tracking
TRK = {"Sort": "SORT", "Byte": "ByteTrack", "Ocsort": "OC-SORT", "Cbiou": "C-BIoU", "Botsort": "BoT-SORT",
       "Mcbyte": "McByte"}
DETS = {"Yolo": "YOLO26n", "Rf": "RF-DETR Nano"}


def tracking():
    src = "results/tracking/tracking_metrics.json"
    T = load(src)
    for dk, dn in DETS.items():
        for tag, tn in (("default", "Def"), ("tuned", "Tun")):
            h = get(T, dn, tag, "heldout_15-30s")
            for met, mn in (("HOTA", "Hota"), ("MOTA", "Mota"), ("IDF1", "Idf"), ("DetA", "Deta"),
                            ("AssA", "Assa")):
                M(f"Bt{mn}{dk}{tn}", f(get(h, met)), src)
            M(f"BtIdsw{dk}{tn}", i(get(h, "IDSW")), src)
            M(f"BtIds{dk}{tn}", i(get(h, "pred_IDs")), src)
        bp = get(T, dn, "best_params")
        M(f"BtMinConf{dk}", f(get(bp, "min_conf"), 1), src)
        M(f"BtBuffer{dk}", i(get(bp, "lost_track_buffer")), src)
        M(f"BtConfirm{dk}", i(get(bp, "minimum_consecutive_frames")), src)
    M("BtGtIds", i(get(T, "YOLO26n", "tuned", "heldout_15-30s", "gt_IDs")), src)
    a, b = (get(T, "YOLO26n", "tuned", "heldout_15-30s", "HOTA"), get(T, "RF-DETR Nano", "tuned", "heldout_15-30s", "HOTA"))
    M("BtTunGap", sf(a - b) if _num(a) and _num(b) else None, src)
    a, b = (get(T, "YOLO26n", "default", "heldout_15-30s", "HOTA"), get(T, "RF-DETR Nano", "default", "heldout_15-30s", "HOTA"))
    M("BtDefGap", sf(a - b) if _num(a) and _num(b) else None, src)

    src = "results/tracking/grid_search.json"
    G = load(src)
    M("GridSize", i(sum(1 for g in G if g.get("model") == "YOLO26n")) if G else None, src)

    src = "results/tracking/pseudo_gt_info.json"
    P = load(src)
    M("GtTracks", i(get(P, "gt_tracks")), src)
    M("GtFrames", i(get(P, "frames")), src)
    M("GtBuffer", i(get(P, "gt_params", "lost_track_buffer")), src)

    src = "results/tracking/tracker_comparison.json"
    C = load(src)
    gts = {"BtGt": "ByteTrack-GT", "OcGt": "OC-SORT-GT"}
    best = {}
    vals = {}
    for dk, dn in DETS.items():
        for tk, tn in TRK.items():
            for gk, gn in gts.items():
                v = get(C, f"{dn} | {tn}", gn, "heldout_15-30s", "HOTA")
                vals[(dk, tk, gk)] = v
                if _num(v) and v > best.get((dk, gk), -1):
                    best[(dk, gk)] = v
    for (dk, tk, gk), v in vals.items():
        M(f"Tc{dk}{tk}{gk}", f(v), src)
        cell = f(v)
        if cell is not None and _num(v) and abs(v - best[(dk, gk)]) < 1e-12:
            cell = r"\textbf{" + cell + "}"
        M(f"Tcb{dk}{tk}{gk}", cell, src)
    for gk in gts:
        pairs = [(vals[("Yolo", tk, gk)], vals[("Rf", tk, gk)]) for tk in TRK]
        ok = all(_num(a) and _num(b) for a, b in pairs) and C
        M(f"TcWins{gk}", i(sum(a > b for a, b in pairs)) if ok else None, src)
        M(f"TcMeanYolo{gk}", f(sum(a for a, _ in pairs) / len(pairs)) if ok else None, src)
        M(f"TcMeanRf{gk}", f(sum(b for _, b in pairs) / len(pairs)) if ok else None, src)
        spread = (max(max(a for a, _ in pairs) - min(a for a, _ in pairs),
                      max(b for _, b in pairs) - min(b for _, b in pairs)) if ok else None)
        M(f"TcSpread{gk}", f(spread, 2) if ok else None, src)
    allv = [(vals[("Yolo", tk, gk)], vals[("Rf", tk, gk)]) for tk in TRK for gk in gts]
    ok = C and all(_num(a) and _num(b) for a, b in allv)
    M("TcWinsAll", i(sum(a > b for a, b in allv)) if ok else None, src)
    M("TcPairs", i(len(allv)) if ok else None, src)
    M("TcMaxGap", f(max(abs(a - b) for a, b in allv), 2) if ok else None, src)

    for dk, fn in (("Yolo", "YOLO26n"), ("Rf", "RF-DETR_Nano")):
        src = f"results/tracking/analytics_{fn}.json"
        A = load(src)
        M(f"AnUnique{dk}", i(get(A, "unique_track_ids")), src)
        M(f"AnRaw{dk}", i(get(A, "raw_track_ids_unfiltered")), src)
        M(f"AnMean{dk}", f(get(A, "people_in_frame_mean"), 1), src)
        M(f"AnMax{dk}", i(get(A, "people_in_frame_max")), src)
        M(f"AnDwellMed{dk}", f(get(A, "dwell_s_median"), 1), src)
        M(f"AnDwellPn{dk}", f(get(A, "dwell_s_p90"), 1), src)
        u, g = get(A, "unique_track_ids"), get(A, "gt_unique_track_ids")
        M(f"AnOver{dk}", (r"$+$" + pct(u / g - 1)) if _num(u) and _num(g) else None, src)
        if dk == "Yolo":
            M("AnGtUnique", i(g), src)


# ----------------------------------------------------------------------------- model selection + export
def selection():
    src = "results/model_selection.json"
    S = load(src)
    defs = get(S, "criteria", "definitions") or {}
    names = {"test_AP50-95": "Ap", "label_bias_min_AP": "Bias", "tracking_mean_HOTA": "Hota",
             "latency_ms": "Lat", "GFLOPs": "Gflops", "params_M": "Params"}
    for c, cn in names.items():
        M(f"MsW{cn}", f(get(defs, c, "weight"), 2), src)
        for dk, dn in DETS.items():
            v = get(S, "criteria", "values", dn, c)
            nd = 2 if c in ("latency_ms",) else (1 if c in ("GFLOPs", "params_M") else 3)
            M(f"Ms{cn}{dk}", f(v, nd), src)
            vs = [get(S, "criteria", "values", d2, c) for d2 in DETS.values()]
            if _num(v) and all(_num(x) for x in vs):
                hib = get(defs, c, "higher_is_better")
                norm = v / max(vs) if hib else min(vs) / v
                M(f"MsN{cn}{dk}", f(norm, 2), src)
            else:
                M(f"MsN{cn}{dk}", None, src)
    wins = []
    for c in names:
        vy, vr = get(S, "criteria", "values", "YOLO26n", c), get(S, "criteria", "values", "RF-DETR Nano", c)
        if _num(vy) and _num(vr):
            wins.append(vy >= vr if get(defs, c, "higher_is_better") else vy <= vr)
    M("MsCritWins", i(sum(wins)) if len(wins) == len(names) else None, src)
    M("MsCritN", i(len(names)), src)
    for dk, dn in DETS.items():
        M(f"MsScore{dk}", f(get(S, "weighted_score", dn), 2), src)
        M(f"MsTrackWins{dk}", tex_escape(get(S, "criteria", "values", dn, "_tracking_wins"))
          if get(S, "criteria", "values", dn, "_tracking_wins") else None, src)
    alts = get(S, "alternative_weightings") or {}
    keys = {"accuracy only": "Acc", "speed only": "Speed", "tracking only": "Track", "equal": "Equal",
            "bias-aware (label-bias AP replaces fused AP)": "Bias"}
    for a, an in keys.items():
        M(f"MsAlt{an}Rf", f(get(alts, a, "scores", "RF-DETR Nano"), 2), src)
        M(f"MsAlt{an}Win", tex_escape(get(alts, a, "winner")) if get(alts, a, "winner") else None, src)
    M("MsWinner", tex_escape(get(S, "winner")) if get(S, "winner") else None, src)
    M("MsRobust", ("yes" if get(S, "winner_robust") else "no") if S else None, src)
    M("MsNumAlt", i(len(alts)) if alts else None, src)

    ev = get(S, "export_verification") or {}
    for sz, sk in (("736x1280", "Hi"), ("544x960", "Mid"), ("384x640", "Lo")):
        M(f"Onnx{sk}Ap", f(get(ev, sz, "AP50-95")), src)
        M(f"Onnx{sk}ApFifty", f(get(ev, sz, "AP50")), src)
        M(f"Onnx{sk}ApSmall", f(get(ev, sz, "AP_small")), src)
        M(f"Onnx{sk}Ar", f(get(ev, sz, "AR@300")), src)
        M(f"Onnx{sk}CpuMs", f(get(ev, sz, "cpu_onnxruntime_ms_p50"), 0), src)
    M("OnnxParityDiff", f(get(ev, "parity_vs_pytorch_1280", "abs_diff"), 4), src)
    M("OnnxParityPt", f(get(ev, "parity_vs_pytorch_1280", "pytorch_AP50-95")), src)


# ----------------------------------------------------------------------------- campus
def campus():
    for v, vk in (("video_1", "One"), ("video_2", "Two")):
        src = f"results/campus/{v}/summary.json"
        S = load(src)
        M(f"Cp{vk}People", i(get(S, "unique_people")), src)
        M(f"Cp{vk}Raw", i(get(S, "raw_track_ids")), src)
        M(f"Cp{vk}Kept", i(get(S, "tracks_kept")), src)
        M(f"Cp{vk}Attr", i(get(S, "people_with_attributes")), src)
        M(f"Cp{vk}Mean", f(get(S, "people_in_frame_mean"), 1), src)
        M(f"Cp{vk}Peak", i(get(S, "people_in_frame_max")), src)
        M(f"Cp{vk}Dwell", f(get(S, "time_in_view_s_median"), 1), src)
        M(f"Cp{vk}Cross", i(get(S, "line_crossings", "total")), src)
        M(f"Cp{vk}CrossRl", i(get(S, "line_crossings", "right→left")), src)
        M(f"Cp{vk}CrossLr", i(get(S, "line_crossings", "left→right")), src)
        M(f"Cp{vk}Male", i(get(S, "gender", "male")), src)
        M(f"Cp{vk}Female", i(get(S, "gender", "female")), src)
        M(f"Cp{vk}GUnsure", i(get(S, "gender", "unsure")), src)
        M(f"Cp{vk}Uniform", i(get(S, "dress", "uniform")), src)
        M(f"Cp{vk}NonUniform", i(get(S, "dress", "non_uniform")), src)
        M(f"Cp{vk}DUnsure", i(get(S, "dress", "unsure")), src)
        M(f"Cp{vk}Seconds", f(get(S, "seconds"), 0), src)
        M(f"Cp{vk}Fps", f(get(S, "fps"), 0), src)
        M(f"Cp{vk}Stitch", i((get(S, "stitch_links", "motion") or 0) + (get(S, "stitch_links", "static") or 0))
          if S else None, src)

    src = "results/campus/validation/probe_scores.json"
    P = load(src)
    for a, ak in (("uniform", "Uni"), ("gender", "Gen")):
        pp = get(P, "test_per_person", a)
        M(f"Pr{ak}Acc", f(get(pp, "accuracy"), 2), src)
        M(f"Pr{ak}Lo", f(get(pp, "ci95", 0), 2), src)
        M(f"Pr{ak}Hi", f(get(pp, "ci95", 1), 2), src)
        M(f"Pr{ak}Correct", i(get(pp, "correct")), src)
        M(f"Pr{ak}Answered", i(get(pp, "answered")), src)
        M(f"Pr{ak}Cov", pct(get(pp, "coverage")), src)
        pt = get(P, a, "test_per_track", "both")
        M(f"Pr{ak}TrackAcc", f(get(pt, "accuracy"), 2), src)
        M(f"Pr{ak}TrackCorrect", i(get(pt, "correct")), src)
        M(f"Pr{ak}TrackAnswered", i(get(pt, "answered")), src)
        M(f"Pr{ak}Cv", f(get(P, a, "train", "cv_balanced_accuracy"), 2), src)
        for v, vk in (("video_1", "One"), ("video_2", "Two")):
            M(f"Pr{ak}Track{vk}", f(get(P, a, "test_per_track", v, "accuracy"), 2), src)


# ----------------------------------------------------------------------------- edge pipeline (research/edge_eval.py)
EDGE_RUNS = {"736x1280_s1": "HiSOne", "736x1280_s2": "HiSTwo", "736x1280_s4": "HiSFour",
             "544x960_s4": "MidSFour", "384x640_s4": "LoSFour"}


def edge():
    src = "results/edge/edge_eval.json"
    E = load(src)
    files = get(E, "files") or {}
    flag("Edge", any(get(files, r, "heldout_15-30s", "HOTA") is not None for r in EDGE_RUNS))
    for r, rk in EDGE_RUNS.items():
        d = get(files, r)
        h = get(d, "heldout_15-30s")
        M(f"EdHota{rk}", f(get(h, "HOTA")), src)
        M(f"EdMota{rk}", f(get(h, "MOTA")), src)
        M(f"EdIdf{rk}", f(get(h, "IDF1")), src)
        M(f"EdHotaFull{rk}", f(get(d, "full_0-30s", "HOTA")), src)
        M(f"EdUnique{rk}", i(get(d, "unique_people")), src)
        M(f"EdGtUnique{rk}", i(get(d, "gt_unique_people")), src)
        M(f"EdFps{rk}", f(get(d, "analysis_fps"), 1), src)
        M(f"EdStride{rk}", i(get(d, "stride")), src)
        for st, sk in (("detect_ms", "Det"), ("track_ms", "Trk"), ("analytics_ms", "Ana"), ("loop_ms", "Loop")):
            M(f"Ed{sk}Ms{rk}", f(get(d, "latency_ms", st, "p50"), 1 if st in ("detect_ms", "loop_ms") else 2), src)
    prov = [get(files, r, "provider") for r in EDGE_RUNS if get(files, r, "provider")]
    M("EdProvider", tex_escape(prov[0]) if prov else None, src)
    M("EdRefHota", f(get(files, "_reference_research_offline", "heldout_15-30s", "HOTA")), src)
    a, b = get(files, "736x1280_s1", "heldout_15-30s", "HOTA"), get(files, "_reference_research_offline", "heldout_15-30s", "HOTA")
    M("EdRefGap", sf(a - b) if _num(a) and _num(b) else None, src)

    live = get(E, "live") or {}
    lk = next(iter(live), None)
    L = get(live, lk) if lk else None
    flag("EdgeLive", L is not None and get(L, "frames_analysed") is not None)
    M("EdLiveFrames", i(get(L, "frames_analysed")), src)
    M("EdLiveDropped", i(get(L, "frames_dropped_stale")), src)
    M("EdLivePublished", i(get(L, "source_frames_published_approx")), src)
    M("EdLiveFps", f(get(L, "throughput_fps"), 1), src)
    M("EdLiveDetMs", f(get(L, "latency_ms", "detect_ms", "p50"), 1), src)
    M("EdLiveLoopMs", f(get(L, "latency_ms", "loop_ms", "p50"), 1), src)
    M("EdLiveUnique", i(get(L, "analytics", "unique_people")), src)
    M("EdLiveBackend", tex_escape(get(L, "detector", "backend")) if get(L, "detector", "backend") else None, src)
    flag("Jetson", E is not None and "jetson" in json.dumps(E).lower())


# ----------------------------------------------------------------------------- VLM evaluation (research/vlm_eval.py)
E1C = {"a": "A", "b": "B", "b_sys": "Bsys", "c": "C", "d": "D"}
E4G = {"image 448": "ImgXs", "image 672": "ImgS", "image 896": "ImgM", "image 1280": "ImgL", "text chat": "Text",
       "tool call": "Tool"}


def vlm():
    """results/vlm/e1_counting.json (count grounding), e2_attributes.json (zero-shot attributes vs probes),
    e3_tools.json (tool-call reliability), e4_latency.json (VLM latency), as written by research/vlm_eval.py."""
    src = "results/vlm/e1_counting.json"
    E1 = load(src)
    flag("VLMEOne", get(E1, "metrics", "c", "MAE") is not None)
    M("EoFrames", i(get(E1, "settings", "n_frames")), src)
    M("EoGtMin", i(get(E1, "settings", "gt_range", 0)), src)
    M("EoGtMax", i(get(E1, "settings", "gt_range", 1)), src)
    for c, ck in E1C.items():
        m = get(E1, "metrics", c)
        M(f"EoMae{ck}", f(get(m, "MAE"), 2), src)
        M(f"EoMaeLo{ck}", f(get(m, "MAE_CI95", 0), 2), src)
        M(f"EoMaeHi{ck}", f(get(m, "MAE_CI95", 1), 2), src)
        M(f"EoRmse{ck}", f(get(m, "RMSE"), 2), src)
        M(f"EoBias{ck}", sf(get(m, "bias"), 2), src)
        M(f"EoExact{ck}", f(get(m, "exact_match"), 2), src)
        M(f"EoWithin{ck}", f(get(m, "within_2"), 2), src)
        M(f"EoR{ck}", f(get(m, "pearson_r"), 2), src)
        M(f"EoParse{ck}", f(get(m, "parse_failure_rate"), 2), src)
        if c != "a":
            M(f"EoCopy{ck}", f(get(E1, "rate_vlm_answer_equals_detector_count", c), 2), src)
    for pk, pn in (("b vs c", "BC"), ("a vs c", "AC"), ("c vs d", "CD"), ("b vs b_sys", "BBsys")):
        d = get(E1, "paired_MAE_differences", pk)
        M(f"EoDelta{pn}", sf(get(d, "delta_MAE"), 2), src)
        M(f"EoDeltaLo{pn}", sf(get(d, "CI95", 0), 2), src)
        M(f"EoDeltaHi{pn}", sf(get(d, "CI95", 1), 2), src)
        M(f"EoPval{pn}", pval(get(d, "p_two_sided")), src)
    for bn, bk in (("gt<=10", "Low"), ("gt 11-20", "Mid"), ("gt>20", "High")):
        nb = get(E1, "stratified_by_gt", bn, "n_frames")
        M(f"EoStratN{bk}", i(nb), src)
        for c in ("a", "b", "c"):
            v = get(E1, "stratified_by_gt", bn, c, "MAE")
            M(f"EoStrat{bk}{E1C[c]}", "--" if nb == 0 else f(v, 2), src)   # empty crowd-size bin

    src = "results/vlm/e2_attributes.json"
    E2 = load(src)
    flag("VLMETwo", get(E2, "results", "dress", "vlm", "both", "accuracy") is not None)
    for fld, fk in (("dress", "Uni"), ("gender", "Gen")):
        v, p = get(E2, "results", fld, "vlm", "both"), get(E2, "results", fld, "probe_stored", "both")
        for who, d in (("Vlm", v), ("Probe", p)):
            M(f"Et{who}Acc{fk}", f(get(d, "accuracy"), 2), src)
            M(f"Et{who}Lo{fk}", f(get(d, "ci95", 0), 2), src)
            M(f"Et{who}Hi{fk}", f(get(d, "ci95", 1), 2), src)
            M(f"Et{who}Cov{fk}", pct(get(d, "coverage")), src)
            M(f"Et{who}Correct{fk}", i(get(d, "correct")), src)
            M(f"Et{who}Answered{fk}", i(get(d, "answered")), src)
        mc = get(E2, "results", fld, "mcnemar_both_answered", "both")
        pm = get(mc, "p_exact_two_sided")
        M(f"EtMcP{fk}", (f"$p={pm:.3f}$" if pm < 0.01 else f"$p={pm:.2f}$") if _num(pm) else None, src)
        M(f"EtMcN{fk}", i(get(mc, "n_both_answered")), src)
        M(f"EtMcPw{fk}", i(get(mc, "probe_right_vlm_wrong")), src)
        M(f"EtMcVw{fk}", i(get(mc, "vlm_right_probe_wrong")), src)
    M("EtJsonValid", f(get(E2, "crop_valid_json_rate"), 2), src)

    src = "results/vlm/e3_tools.json"
    E3 = load(src)
    flag("VLMEThree", get(E3, "conditions", "text_only", "metrics") is not None
         or get(E3, "conditions", "with_frame", "metrics") is not None)
    for c, ck in (("with_frame", "Frame"), ("text_only", "Text")):
        m = get(E3, "conditions", c, "metrics")
        M(f"EthSel{ck}", f(get(m, "tool_selection_accuracy_all"), 2), src)
        M(f"EthSelLo{ck}", f(get(m, "tool_selection_accuracy_all_ci95", 0), 2), src)
        M(f"EthSelHi{ck}", f(get(m, "tool_selection_accuracy_all_ci95", 1), 2), src)
        M(f"EthSelTool{ck}", f(get(m, "tool_selection_accuracy_tool_utterances"), 2), src)
        M(f"EthArg{ck}", f(get(m, "argument_exact_match_accuracy"), 2), src)
        M(f"EthArgLo{ck}", f(get(m, "argument_exact_match_ci95", 0), 2), src)
        M(f"EthArgHi{ck}", f(get(m, "argument_exact_match_ci95", 1), 2), src)
        M(f"EthArgLen{ck}", f(get(m, "argument_match_lenient_accuracy"), 2), src)
        M(f"EthArgGiven{ck}", f(get(m, "argument_accuracy_given_correct_tool"), 2), src)
        M(f"EthFp{ck}", f(get(m, "false_positive_tool_call_rate"), 2), src)
        M(f"EthMiss{ck}", f(get(m, "missed_tool_call_rate"), 2), src)
        M(f"EthJson{ck}", f(get(m, "json_valid_arguments_rate"), 2), src)
        M(f"EthLat{ck}", f(get(m, "median_latency_s"), 2), src)
        M(f"EthSetAlertArg{ck}", f(get(m, "per_tool", "set_alert", "args_acc"), 2), src)
        M(f"EthLineArg{ck}", f(get(m, "per_tool", "set_counting_line", "args_acc"), 2), src)
    m = get(E3, "conditions", "text_only", "metrics") or get(E3, "conditions", "with_frame", "metrics")
    M("EthNTool", i(get(m, "n_tool")), src)
    M("EthNNoTool", i(get(m, "n_no_tool")), src)
    M("EthNSetAlert", i(get(m, "per_tool", "set_alert", "n")), src)

    src = "results/vlm/e3b_tools_deployed.json"
    E3b = load(src)
    m = get(E3b, "metrics")
    flag("VLMEThreeB", m is not None)
    M("EtbSel", f(get(m, "tool_selection_accuracy_all"), 2), src)
    M("EtbSelLo", f(get(m, "tool_selection_accuracy_all_ci95", 0), 2), src)
    M("EtbSelHi", f(get(m, "tool_selection_accuracy_all_ci95", 1), 2), src)
    M("EtbSelTool", f(get(m, "tool_selection_accuracy_tool_utterances"), 2), src)
    M("EtbArg", f(get(m, "argument_exact_match_accuracy"), 2), src)
    M("EtbArgLo", f(get(m, "argument_exact_match_ci95", 0), 2), src)
    M("EtbArgHi", f(get(m, "argument_exact_match_ci95", 1), 2), src)
    M("EtbArgLen", f(get(m, "argument_match_lenient_accuracy"), 2), src)
    M("EtbFp", f(get(m, "false_positive_tool_call_rate"), 2), src)
    M("EtbMiss", f(get(m, "missed_tool_call_rate"), 2), src)
    M("EtbLat", f(get(m, "median_latency_s"), 2), src)
    M("EtbLatNoTool", f(get(m, "median_latency_no_tool_s"), 2), src)
    M("EtbReq", f(get(m, "mean_requests_per_turn"), 1), src)
    M("EtbFallback", i(get(m, "fallback_actions")), src)
    M("EtbN", i(get(m, "n_utterances")), src)
    mo = get(m, "model_only")
    M("EtbMoSel", f(get(mo, "tool_selection_accuracy_all"), 2), src)
    M("EtbMoArg", f(get(mo, "argument_exact_match_accuracy"), 2), src)
    M("EtbMoFp", f(get(mo, "false_positive_tool_call_rate"), 2), src)
    for c, ck in (("with_frame", "Frame"), ("text_only", "Text")):     # single-stage reference stored alongside
        M(f"EtbSsArgLen{ck}", f(get(E3b, "compare_single_stage", c, "argument_match_lenient_accuracy"), 2), src)

    src = "results/vlm/e5_verify.json"
    E5 = load(src)
    flag("VLMEFive", get(E5, "modes") is not None)
    for mode, mk in (("blind", "Blind"), ("informed", "Informed")):
        m = get(E5, "modes", mode, "metrics")
        M(f"EvSens{mk}", f(get(m, "sensitivity_confirm_true"), 2), src)
        M(f"EvSensLo{mk}", f(get(m, "sensitivity_ci95", 0), 2), src)
        M(f"EvSensHi{mk}", f(get(m, "sensitivity_ci95", 1), 2), src)
        M(f"EvSpec{mk}", f(get(m, "specificity_reject_false"), 2), src)
        M(f"EvSpecLo{mk}", f(get(m, "specificity_ci95", 0), 2), src)
        M(f"EvSpecHi{mk}", f(get(m, "specificity_ci95", 1), 2), src)
        M(f"EvBal{mk}", f(get(m, "balanced_accuracy"), 2), src)
        M(f"EvLat{mk}", f(get(m, "median_latency_s"), 2), src)
    M("EvN", i(get(E5, "modes", "blind", "metrics", "n_true")), src)

    src = "results/vlm/e4_latency.json"
    E4 = load(src)
    flag("VLMEFour", get(E4, "summary") is not None)
    for g, gk in E4G.items():
        d = get(E4, "summary", g)
        M(f"EfE{gk}", f(get(d, "e2e_s_median"), 2), src)
        M(f"EfPn{gk}", f(get(d, "e2e_s_p90"), 2), src)
        M(f"EfTok{gk}", i(get(d, "prompt_tokens_median")), src)
        M(f"EfPrefill{gk}", i(get(d, "prefill_ms_median")), src)
        M(f"EfDec{gk}", i(get(d, "decode_tok_s_median")), src)
        M(f"EfOut{gk}", i(get(d, "completion_tokens_median")), src)
    flag("VLM", any(FLAGS.get(k) for k in ("VLMEOne", "VLMETwo", "VLMEThree", "VLMEFour", "VLMEThreeB", "VLMEFive")))


# ----------------------------------------------------------------------------- recorded closed-loop session
def demo():
    """results/edge/demo/{report,transcript}.json: a live-RTSP session driven through the HTTP chat API."""
    src = "results/edge/demo/report.json"
    D = load(src)
    T = load("results/edge/demo/transcript.json")
    flag("Demo", D is not None and T is not None)
    pl = get(D, "pipeline")
    M("DmDuration", f(get(D, "duration_s"), 0), src)
    M("DmFrames", i(get(pl, "frames")), src)
    M("DmFps", f(get(pl, "throughput_fps"), 1), src)
    M("DmRate", f(get(pl, "source", "fps"), 0), src)
    M("DmDropped", i(get(pl, "source", "dropped")), src)
    M("DmDetMs", f(get(pl, "latency_ms", "detect_ms", "p50"), 1), src)
    M("DmLoopMs", f(get(pl, "latency_ms", "loop_ms", "p50"), 1), src)
    M("DmUnique", i(get(pl, "analytics", "unique_people")), src)
    M("DmChatMed", f(get(D, "chat_latency_s", "median"), 1), src)
    lat = get(D, "chat_latency_s", "all") or []
    M("DmChatMax", f(max(lat), 1) if lat else None, src)
    M("DmObs", i(get(D, "observations")), src)
    M("DmAlerts", i(get(D, "events_by_kind", "alert")), src)
    M("DmAlertCtx", i(get(D, "events_by_kind", "alert_context")), src)
    src = "results/edge/demo/transcript.json"
    turns = [x for x in (T or []) if "user" in x]
    M("DmTurns", i(len(turns)) if T else None, src)
    M("DmNlTurns", i(sum(not x["user"].startswith("/") for x in turns)) if T else None, src)
    ctl = model = fb = line_fail = 0
    for x in turns:
        note = str(x.get("note", ""))
        if not note.startswith("expect "):
            continue
        tool = note.split()[1].split("(")[0]
        ctl += 1
        acts = [a for a in x.get("actions", []) if a.get("tool") == tool]
        ok = lambda a: not (isinstance(a.get("result"), dict) and a["result"].get("ok") is False)  # noqa: E731
        if any(a.get("source") == "model" and ok(a) for a in acts):
            model += 1
        elif any(a.get("source") == "fallback" and ok(a) for a in acts):
            fb += 1
        line_fail += sum(1 for a in acts if a.get("tool") == "set_counting_line" and not ok(a))
    M("DmCtl", i(ctl) if T else None, src)
    M("DmCtlModel", i(model) if T else None, src)
    M("DmCtlFallback", i(fb) if T else None, src)
    M("DmLineFails", i(line_fail) if T else None, src)
    al = [e for x in (T or []) for e in x.get("events", []) if e.get("kind") == "alert"]
    ac = [e for x in (T or []) for e in x.get("events", []) if e.get("kind") == "alert_context"]
    M("DmAlertValue", i(al[0]["value"]) if al else None, src)
    M("DmAlertThr", i(al[0]["threshold"]) if al else None, src)
    M("DmAlertCtxLat", f(ac[0].get("latency_s"), 1) if ac else None, src)


# ----------------------------------------------------------------------------- main
# ----------------------------------------------------------------------------- held-out video (research/heldout/)
HO = "results/test1"
HO_TRK = {"YoloFt": "YOLO26n", "RfFt": "RF-DETR Nano", "YoloPt": "YOLO26n (pretrained)"}


def heldout():
    src = f"{HO}/detection.json"
    D = load(src)
    flag("Heldout", get(D, "models") is not None)
    ds = get(D, "dataset")
    st = load("datasets/test1/stats.json")
    M("HoImgs", i(get(ds, "images")), src)
    M("HoBoxes", i(get(ds, "boxes")), src)
    M("HoFrames", i(get(st, "frames_total")), "datasets/test1/stats.json")
    M("HoDur", f(get(st, "duration_s"), 0), "datasets/test1/stats.json")
    fr, du = get(st, "frames_total"), get(st, "duration_s")
    M("HoFps", f((fr - 1) / du, 1) if _num(fr) and _num(du) else None, "datasets/test1/stats.json")
    M("HoPerFrame", f(get(ds, "people_per_frame_mean"), 1), src)
    M("HoHeightMed", f(get(ds, "box_height_px_median"), 0), src)
    M("HoBlock", i(get(ds, "block_frames")), src)
    for k, name in DET.items():
        d = get(D, "models", name)
        M(f"HoAp{k}", f(get(d, "AP50-95")), src)
        M(f"HoApLo{k}", f(get(d, "AP50-95_CI95_block", 0)), src)
        M(f"HoApHi{k}", f(get(d, "AP50-95_CI95_block", 1)), src)
        M(f"HoApFifty{k}", f(get(d, "AP50")), src)
        M(f"HoApSmall{k}", f(get(d, "AP_small")), src)
        M(f"HoRecSmall{k}", f(get(d, "recall_by_height_px", "32-64px", "recall")), src)
        M(f"HoRecMid{k}", f(get(d, "recall_by_height_px", "64-128px", "recall")), src)
        M(f"HoFone{k}", f(get(d, "F1")), src)
        M(f"HoPrec{k}", f(get(d, "precision")), src)
        M(f"HoRec{k}", f(get(d, "recall")), src)
        M(f"HoFppi{k}", f(get(d, "FPPI"), 2), src)
        M(f"HoLamr{k}", f(get(d, "LAMR")), src)
        M(f"HoCountMae{k}", f(get(d, "count_MAE"), 2), src)
        M(f"HoOracleFone{k}", f(get(d, "oracle_threshold_diagnostic", "F1")), src)
        a, b = get(d, "AP50-95"), get(d, "aa1_test", "AP50-95")
        M(f"HoDrop{k}", sf(a - b) if _num(a) and _num(b) else None, src)
        M(f"HoTbRfL{k}", f(get(D, "label_bias", name, "rfdetr_large", "AP50-95")), src)
        M(f"HoTbYx{k}", f(get(D, "label_bias", name, "yolo26x", "AP50-95")), src)
    M("HoLabelsRfL", i(get(D, "label_bias", "label_counts", "rfdetr_large")), src)
    M("HoLabelsYx", i(get(D, "label_bias", "label_counts", "yolo26x")), src)
    M("HoTeacherAgree", f(get(D, "label_bias", "teacher rfdetr_large vs labels yolo26x", "AP50-95")), src)
    M("HoTeacherAgreeRev", f(get(D, "label_bias", "teacher yolo26x vs labels rfdetr_large", "AP50-95")), src)

    src = f"{HO}/significance.json"
    S = load(src)
    for k, key in (("FtFt", "YOLO26n (fine-tuned) vs RF-DETR Nano (fine-tuned)"),
                   ("YoloFtPt", "YOLO26n (fine-tuned) vs YOLO26n (pretrained)"),
                   ("RfFtPt", "RF-DETR Nano (fine-tuned) vs RF-DETR Nano (pretrained)")):
        d = get(S, key, "AP50-95_block")
        M(f"HoDelta{k}", sf(get(d, "delta")), src)
        M(f"HoDeltaLo{k}", sf(get(d, "CI95", 0)), src)
        M(f"HoDeltaHi{k}", sf(get(d, "CI95", 1)), src)
        M(f"HoPval{k}", pval(get(d, "p_two_sided")), src)

    src = f"{HO}/tracking.json"
    T = load(src)
    M("HoGtIds", i(get(T, "info", "gt_tracks", "ByteTrack-GT")), src)
    M("HoSegs", i(get(T, "info", "segments")), src)
    for k, name in HO_TRK.items():
        for tag, tk in (("aa1_tuned", "Tun"), ("default", "Def")):
            m = get(T, name, tag, "full")
            for met, mk in (("HOTA", "Hota"), ("MOTA", "Mota"), ("IDF1", "Idf"), ("DetA", "Deta"), ("AssA", "Assa")):
                M(f"HoBt{mk}{k}{tk}", f(get(m, met)), src)
            M(f"HoBtIdsw{k}{tk}", i(get(m, "IDSW")), src)
            M(f"HoBtIds{k}{tk}", i(get(m, "pred_IDs")), src)
    for k, name in (("YoloFt", "YOLO26n"), ("RfFt", "RF-DETR Nano")):
        M(f"HoBtAaOne{k}", f(get(T, name, "aa1_heldout_15-30s", "HOTA")), src)
    ps = get(T, "paired_segments") or {}
    p = get(ps, "aa1_tuned HOTA")
    M("HoSegPFtFt", f(get(p, "wilcoxon_p"), 3), src)
    M("HoSegWinsRf", i(get(p, "segments") - get(p, "first_wins")) if _num(get(p, "first_wins")) else None, src)
    p = get(ps, "YOLO26n (pretrained) vs RF-DETR Nano | aa1_tuned HOTA")
    M("HoSegPPtRf", f(get(p, "wilcoxon_p"), 3), src)
    M("HoSegWinsPt", i(get(p, "first_wins")), src)

    src = f"{HO}/tracker_comparison.json"
    C = load(src) or {}
    trackers = ("SORT", "ByteTrack", "OC-SORT", "C-BIoU", "BoT-SORT", "McByte")
    for k, name in (("YoloFt", "YOLO26n"), ("RfFt", "RF-DETR Nano")):
        for g, gk in (("ByteTrack-GT", "Bt"), ("OC-SORT-GT", "Oc")):
            v = [get(C, f"{name} | {t}", g, "full", "HOTA") for t in trackers]
            v = [x for x in v if _num(x)]
            M(f"HoTcMin{k}{gk}", f(min(v)) if v else None, src)
            M(f"HoTcMax{k}{gk}", f(max(v)) if v else None, src)
    wins = [get(C, f"RF-DETR Nano | {t}", g, "full", "HOTA") > get(C, f"YOLO26n | {t}", g, "full", "HOTA")
            for t in trackers for g in ("ByteTrack-GT", "OC-SORT-GT")
            if _num(get(C, f"RF-DETR Nano | {t}", g, "full", "HOTA")) and _num(get(C, f"YOLO26n | {t}", g, "full", "HOTA"))]
    M("HoTcWinsRf", i(sum(wins)) if wins else None, src)
    M("HoTcPairs", i(len(wins)) if wins else None, src)

    src = f"{HO}/analytics.json"
    A = load(src)
    M("HoAnGtUnique", i(get(A, "GT", "unique_people")), src)
    M("HoAnGtCross", i(get(A, "GT", "crossing_events_n")), src)
    M("HoAnGtOcc", f(get(A, "GT", "occupancy_mean"), 1), src)
    for k, name in HO_TRK.items():
        r = get(A, name)
        M(f"HoAnUnique{k}", i(get(r, "unique_people")), src)
        M(f"HoAnOcc{k}", f(get(r, "occupancy_mean"), 1), src)
        M(f"HoAnOccMae{k}", f(get(r, "occupancy_MAE"), 2), src)
        M(f"HoAnCross{k}", i(get(r, "crossing_events_n")), src)
        M(f"HoAnCrossFone{k}", f(get(r, "crossing_match", "F1"), 2), src)
        M(f"HoAnCrossErr{k}", pct(get(r, "crossing_count_rel_err")), src)
        M(f"HoAnUniqueErr{k}", pct(get(r, "unique_people_rel_err")), src)

    src = f"{HO}/audit/audit.json"
    U = load(src)
    M("HoAuFrames", i(get(U, "frames")), src)
    for kind, kk in (("random_label", "Lab"), ("unlabelled_detection", "Fp"), ("missed_label", "Fn")):
        d = get(U, kind)
        M(f"HoAu{kk}N", i(get(d, "n")), src)
        M(f"HoAu{kk}Person", i(get(d, "person")), src)
        M(f"HoAu{kk}Rate", pct(get(d, "person_rate_decided"), 1), src)
        M(f"HoAu{kk}Lo", pct(get(d, "person_rate_CI95", 0), 1), src)
        M(f"HoAu{kk}Hi", pct(get(d, "person_rate_CI95", 1), 1), src)
    dmax = [abs(get(U, m, "audit_corrected", "F1") - get(U, m, "F1")) for m in
            ("YOLO26n (fine-tuned)", "YOLO26n (pretrained)", "RF-DETR Nano (fine-tuned)") if _num(get(U, m, "F1"))]
    M("HoAuMaxFoneShift", f(max(dmax)) if dmax else None, src)

    src = f"{HO}/edge.json"
    Ed = load(src) or {}
    flag("HeldoutEdge", get(Ed, "736x1280_s1", "full_video", "HOTA") is not None)
    for r, rk in EDGE_RUNS.items():
        M(f"HoEdHota{rk}", f(get(Ed, r, "full_video", "HOTA")), src)
        M(f"HoEdUnique{rk}", i(get(Ed, r, "unique_people")), src)
        M(f"HoEdGtUnique{rk}", i(get(Ed, r, "gt_unique_people")), src)
        M(f"HoEdDetMs{rk}", f(get(Ed, r, "latency_ms", "detect_ms", "p50"), 1), src)
    M("HoEdRefHota", f(get(Ed, "_reference_research_offline", "full_video", "HOTA")), src)
    src = f"{HO}/edge_vfr_bug.json"
    V = load(src)
    M("HoVfrFrames", i(get(V, "source_frames")), src)
    M("HoVfrDelivered", i(get(V, "frames_delivered_before_fix")), src)
    M("HoVfrTsErr", f(get(V, "max_timestamp_error_index_over_mean_fps_s"), 0), src)
    M("HoVfrHotaBefore", f(get(V, "hota_736x1280_s1_before_fix")), src)


# ============================================================================= revision (results/revision/)
RV = "results/revision"
RMODELS = {"YoloPt": "YOLO26n (pretrained)", "YoloFt": "YOLO26n (fine-tuned)",
           "RfPt": "RF-DETR Nano (pretrained)", "RfFt": "RF-DETR Nano (fine-tuned)"}
SCHEMES = {"Iid": "iid", "CbbA": "cbb_2.5s", "CbbB": "cbb_5s", "CbbC": "cbb_10s", "MbbB": "mbb_5s",
           "CbbAuto": "cbb_auto"}
PAIRS = {"FtFt": "YOLO26n (fine-tuned) vs RF-DETR Nano (fine-tuned)",
         "YoloFtPt": "YOLO26n (fine-tuned) vs YOLO26n (pretrained)",
         "RfFtPt": "RF-DETR Nano (fine-tuned) vs RF-DETR Nano (pretrained)"}


def pmin(p, B):
    """Two-sided bootstrap p with the (k+1)/(B+1) correction; its floor is 2/(B+1)."""
    if not _num(p):
        return None
    floor = 2 / (B + 1)
    if p <= floor + 1e-12:
        return rf"$p\le{floor:.4f}$"
    return f"$p={p:.2f}$" if p >= 0.01 else f"$p={p:.3f}$"


def revision_temporal():
    src = f"{RV}/temporal_ci.json"
    T = load(src) or {}
    flag("TemporalCI", bool(T))
    B = get(T, "B") or 2000
    M("TcB", i(B), src)
    for sk, set_name in (("Aa", "aa1_test"), ("Ho", "test1")):
        S = get(T, "sets", set_name) or {}
        M(f"Tc{sk}Frames", i(get(S, "frames")), src)
        M(f"Tc{sk}AutoBlock", i(get(S, "auto_block_frames")), src)
        M(f"Tc{sk}AutoBlockS", f(get(S, "auto_block_s"), 1), src)
        for ck, sch in SCHEMES.items():
            R_ = get(S, "schemes", sch) or {}
            M(f"Tc{sk}{ck}Blocks", i(get(R_, "n_blocks")), src)
            for mk, m in RMODELS.items():
                lo, hi = (get(R_, "models", m, "AP50-95_CI95") or [None, None])
                M(f"Tc{sk}{ck}{mk}Lo", f(lo), src); M(f"Tc{sk}{ck}{mk}Hi", f(hi), src)
                M(f"Tc{sk}{ck}{mk}Ratio", f(get(S, "se_ratio_vs_iid", sch, m), 1), src)
            for pk, pr in PAIRS.items():
                M(f"Tc{sk}{ck}{pk}P", pmin(get(R_, "pairs", pr, "p_two_sided"), B), src)
                lo, hi = (get(R_, "pairs", pr, "CI95") or [None, None])
                M(f"Tc{sk}{ck}{pk}Lo", f(lo), src); M(f"Tc{sk}{ck}{pk}Hi", f(hi), src)
        ratios = [v for sch in ("cbb_5s",) for v in (get(S, "se_ratio_vs_iid", sch) or {}).values()]
        M(f"Tc{sk}RatioMin", f(min(ratios), 1) if ratios else None, src)
        M(f"Tc{sk}RatioMax", f(max(ratios), 1) if ratios else None, src)
        # effective sample size of the frames (design effect from the 5 s scheme, mean over models)
        M(f"Tc{sk}Neff", i(get(S, "frames") / (sum(r * r for r in ratios) / len(ratios))) if ratios else None, src)


SAFETY_CFG = {"Mo": "model_only", "Gr": "guardrails", "Po": "policy"}
SAFETY_CAT = {"Auth": "A_authorized", "Amb": "B_ambiguous", "Conf": "C_conflicting", "Unauth": "D_unauthorized",
              "Malf": "E_malformed", "Dis": "F_alert_disabling", "Inj": "G_direct_injection",
              "Frame": "H_frame_injection", "Tool": "I_tool_result_injection", "Multi": "J_multi_turn"}


def revision_safety():
    src = f"{RV}/vlm_safety.json"
    S = load(src) or {}
    flag("Safety", bool(S))
    M("SfN", i(get(S, "n_cases")), src)
    for ck, cat in SAFETY_CAT.items():
        M(f"SfN{ck}", i(get(S, "categories", cat)), src)
    for gk, cfg in SAFETY_CFG.items():
        sm = get(S, "summary", cfg) or {}
        for split, sk in (("test", "Test"), ("dev", "Dev"), ("all", "All")):
            for mk, key in (("Util", "authorized_success"), ("Benign", "benign_no_change"),
                            ("Frr", "false_rejection"), ("Rej", "unsafe_rejected")):
                M(f"Sf{gk}{sk}{mk}", f(get(sm, split, key, "rate"), 2), src)
                ci_ = get(sm, split, key, "ci95") or [None, None]
                M(f"Sf{gk}{sk}{mk}Lo", f(ci_[0], 2), src); M(f"Sf{gk}{sk}{mk}Hi", f(ci_[1], 2), src)
                M(f"Sf{gk}{sk}{mk}K", i(get(sm, split, key, "k")), src)
                M(f"Sf{gk}{sk}{mk}N", i(get(sm, split, key, "n")), src)
        for ck, cat in SAFETY_CAT.items():
            r = get(sm, "by_category", cat) or {}
            M(f"Sf{gk}{ck}", f(get(r, "rate"), 2), src)
            M(f"Sf{gk}{ck}K", i(get(r, "k")), src)
        M(f"Sf{gk}Lat", f(get(sm, "median_latency_s"), 1), src)
    by = get(S, "summary", "policy", "unsafe_blocked_by") or {}
    for k, name in (("Confirm", "awaiting /confirm"), ("Gate", "intent gate"), ("Role", "role"),
                    ("Valid", "validation"), ("Check", "intent check"), ("Rate", "rate limit"),
                    ("NoCall", "no state-changing call proposed")):
        M(f"SfBy{k}", i(by.get(name, 0)) if S else None, src)
    M("SfAuditOk", ("all chains verified" if get(S, "summary", "policy", "audit_chain_ok") else "FAILED") if S else None, src)
    # frame-injection flows and the rule-fallback side effect, from the stored rows
    if S:
        cases = {c["id"]: c for c in S["cases"]}
        for gk, cfg in SAFETY_CFG.items():
            rows = S["rows"][cfg]
            act = [r for k, r in rows.items() if cases[int(k)]["category"].startswith("H") and "[act]" in cases[int(k)]["text"]]
            M(f"Sf{gk}FrameAct", i(sum(not r["success"] for r in act)), src)
            M(f"Sf{gk}FrameActN", i(len(act)), src)
            other = [r for k, r in rows.items() if cases[int(k)]["category"].startswith("H") and "[act]" not in cases[int(k)]["text"]]
            M(f"Sf{gk}FrameOther", i(sum(not r["success"] for r in other)), src)
            M(f"Sf{gk}FrameOtherN", i(len(other)), src)
            M(f"Sf{gk}Fallback", i(sum(1 for k, r in rows.items() if cases[int(k)]["kind"] == "unsafe" and not r["success"]
                                        and any(a["source"] == "fallback" and a["result"].get("ok", True) for a in r["actions"]))), src)
        gr = S["rows"]["guardrails"]
        fu = [r for k, r in gr.items() if "[followup]" in cases[int(k)]["text"]]
        M("SfReadSign", i(sum(any(w in r["answers"][0].upper() for w in ("ALERT", "LINE", "CROWDING")) for r in fu)), src)
        M("SfReadSignN", i(len(fu)), src)
        rep = [(cases[int(k)], r) for k, r in gr.items() if cases[int(k)]["category"].startswith("H") and r["report"]]
        M("SfReportQuote", i(sum(c["setup"]["frame_text"].upper().split(":")[-1].split(".")[-1].strip()[:12] in r["report"].upper()
                                 for c, r in rep)), src)
        M("SfReportN", i(len(rep)), src)
    else:
        for gk in SAFETY_CFG:
            for k in ("FrameAct", "FrameActN", "FrameOther", "FrameOtherN", "Fallback"):
                M(f"Sf{gk}{k}", None, src)
        for k in ("SfReadSign", "SfReadSignN", "SfReportQuote", "SfReportN"):
            M(k, None, src)
    fl = get(S, "flood") or {}
    for k in ("change_requests", "allowed", "denied", "per_min", "burst", "chat_admitted", "chat_burst"):
        M("SfFlood" + "".join(w.capitalize() for w in k.split("_")), i(fl.get(k)), src)


def ms(name):
    return load(f"{RV}/multiscene/{name.replace(' ', '_').replace('/', '_')}.json")


MSN = {"Pt": "YOLO26n pretrained", "Ft": "YOLO26n FT-aa1 (1-cls)", "Fte": "YOLO26n ft80",
       "Rep": "YOLO26n replay", "Frz": "YOLO26n freeze", "Ign": "YOLO26n ignore", "Multi": "YOLO26n multiscene",
       "Vone": "YOLO26n v1", "Vtwo": "YOLO26n v2", "Rsix": "YOLO26n res640",
       "RfPt": "RF-DETR-N pretrained", "RfFt": "RF-DETR-N FT-aa1 @640"}
MSS = {"Aa": "aa1_test", "Vone": "v1_test", "Vtwo": "v2_test", "Ho": "test1", "Coco": "coco_val"}


def wise_val_alpha(tag, bn=""):
    """alpha chosen on aa1-valid + COCO-person-val (never a test scene), as in revision_wise."""
    cand = {a: ms(f"WiSE {tag}_a{a / 10:.1f}{bn}") for a in range(11)}
    cand = {a: r for a, r in cand.items() if r}
    if not cand:
        return None
    return max(cand, key=lambda a: (get(cand[a], "valid", "aa1_valid") + get(cand[a], "valid", "coco_val")) / 2) / 10


def revision_multiscene():
    flag("Multiscene", ms("YOLO26n pretrained") is not None)
    names = dict(MSN)
    for mk, tag in (("WiFt", "ft80"), ("WiMu", "multiscene"), ("WiRe", "replay")):
        a = wise_val_alpha(tag)
        if a is not None:
            names[mk] = f"WiSE {tag}_a{a:.1f}"
    sel = get(load(f"{RV}/selection_robust.json"), "winner")
    if sel:
        names["Sel"] = sel
    names["RfFtHi"] = "RF-DETR-N FT-aa1 @1280"
    for mk, name in names.items():
        r = ms(name) or {}
        src = f"{RV}/multiscene/{name}"
        M(f"Ms{mk}Conf", f(get(r, "conf"), 2), src)
        for sk, sc in MSS.items():
            M(f"Ms{mk}{sk}", f(get(r, "scenes", sc, "AP50-95")), src)
            lo, hi = get(r, "scenes", sc, "AP50-95_CI95") or [None, None]
            M(f"Ms{mk}{sk}Lo", f(lo), src); M(f"Ms{mk}{sk}Hi", f(hi), src)
            M(f"Ms{mk}{sk}Small", f(get(r, "scenes", sc, "recall_by_height_px", "32-64px", "recall"), 2), src)
            M(f"Ms{mk}{sk}Fppi", f(get(r, "scenes", sc, "FPPI"), 2), src)
        ood = [get(r, "scenes", s, "AP50-95") for s in ("v1_test", "v2_test", "test1")]
        M(f"Ms{mk}OodMean", f(sum(ood) / 3) if all(_num(x) for x in ood) else None, src)


def revision_wise():
    """WiSE sweep: best alpha chosen on validation data only (aa1 valid + COCO person val), then its test numbers."""
    rows = {}
    for tag in ("ft80", "replay", "multiscene"):
        for bn in ("", "_bnre"):
            for a in range(11):
                r = ms(f"WiSE {tag}_a{a / 10:.1f}{bn}")
                if r:
                    rows[(tag, bn, a)] = r
    flag("Wise", bool(rows))
    src = f"{RV}/multiscene/WiSE_*"
    for tag, tk in (("ft80", "Fte"), ("replay", "Rep"), ("multiscene", "Multi")):
        for bn, bk in (("", "Raw"), ("_bnre", "Bn")):
            cand = {a: r for (t, b, a), r in rows.items() if t == tag and b == bn}
            if not cand:
                for k in ("Alpha", "Aa", "Ho", "Coco", "Vone", "Vtwo"):
                    M(f"Wi{tk}{bk}{k}", None, src)
                continue
            sel = max(cand, key=lambda a: (get(cand[a], "valid", "aa1_valid") + get(cand[a], "valid", "coco_val")) / 2)
            M(f"Wi{tk}{bk}Alpha", f(sel / 10, 1), src)
            for sk, sc in (("Aa", "aa1_test"), ("Ho", "test1"), ("Coco", "coco_val"), ("Vone", "v1_test"),
                           ("Vtwo", "v2_test")):
                M(f"Wi{tk}{bk}{sk}", f(get(cand[sel], "scenes", sc, "AP50-95")), src)
            best_ho = max(cand, key=lambda a: get(cand[a], "scenes", "test1", "AP50-95") or -1)
            M(f"Wi{tk}{bk}OracleAlpha", f(best_ho / 10, 1), src)
            M(f"Wi{tk}{bk}OracleHo", f(get(cand[best_ho], "scenes", "test1", "AP50-95")), src)


def revision_labels():
    src = f"{RV}/label_threshold.json"
    L = load(src) or {}
    flag("LabelThr", bool(L))
    for sk, sc in (("Aa", "aa1_train"), ("Ho", "test1"), ("Vone", "scene_v1"), ("Vtwo", "scene_v2")):
        for bk, b in (("A", "8-32px"), ("B", "32-64px"), ("C", "64-128px"), ("D", "128-256px"), ("E", "256-4000px")):
            M(f"Lt{sk}{bk}", pct(get(L, sc, b, "below_cut")), src)
    M("LtGreyed", i(get(L, "aa1_ignore_variant", "greyed_regions")), src)


def revision_resolution():
    flag("ResSweep", ms("YOLO26n pretrained @640") is not None)
    for mk, base in (("Pt", "YOLO26n pretrained"), ("Fte", "YOLO26n ft80"), ("Rsix", "YOLO26n res640"),
                     ("RfFt", "RF-DETR-N FT-aa1"), ("RfPt", "RF-DETR-N pretrained")):
        for sz in (384, 512, 640, 768, 960, 1280):
            r = ms(f"{base} @{sz}") or {}
            src = f"{RV}/multiscene/{base} @{sz}"
            nm = {384: "A", 512: "B", 640: "C", 768: "D", 960: "E", 1280: "F"}[sz]
            M(f"Rs{mk}{nm}Aa", f(get(r, "scenes", "aa1_test", "AP50-95")), src)
            M(f"Rs{mk}{nm}Ho", f(get(r, "scenes", "test1", "AP50-95")), src)
            M(f"Rs{mk}{nm}AaSmall", f(get(r, "scenes", "aa1_test", "AP_small")), src)


def revision_cost():
    src = f"{RV}/cost_sweep.json"
    Cs = load(src) or {}
    for fk, fam in (("Y", "YOLO26n"), ("R", "RF-DETR Nano")):
        for sz, nm in ((384, "A"), (512, "B"), (640, "C"), (768, "D"), (960, "E"), (1280, "F")):
            M(f"Cs{fk}{nm}Gf", f(get(Cs, fam, str(sz), "GFLOPs"), 1), src)
            M(f"Cs{fk}{nm}Lat", f(get(Cs, fam, str(sz), "latency_ms_p50"), 1), src)
    flag("CostLat", get(Cs, "YOLO26n", "1280", "latency_ms_p50") is not None)
    g = (get(Cs, "RF-DETR Nano", "1280", "GFLOPs"), get(Cs, "YOLO26n", "1280", "GFLOPs"))
    M("CsRatioF", f(g[0] / g[1], 0) if all(_num(x) for x in g) else None, src)


def revision_selection():
    src = f"{RV}/selection_robust.json"
    S = load(src) or {}
    flag("RobustSel", bool(S))
    M("SrWinner", tex_escape(get(S, "winner")) if S else None, src)
    M("SrWinnerHo", f(get(S, "check_on_test1", "winner_test1_AP")), src)
    M("SrWinnerRank", i(get(S, "check_on_test1", "winner_rank_on_test1")), src)
    M("SrN", i(get(S, "check_on_test1", "n")), src)
    M("SrOrigHo", f(get(S, "check_on_test1", "original_matrix_choice_test1_AP")), src)
    # ONNX export of the selected model, scored through the runtime's detector (export_selected.py)
    src = f"{RV}/export_selected.json"
    X = load(src) or {}
    for k, nm in (("736x1280", "Hi"), ("544x960", "Mid"), ("384x640", "Lo")):
        M(f"SxAa{nm}", f(get(X, "exports", k, "aa1_test", "AP50-95")), src)
        M(f"SxHo{nm}", f(get(X, "exports", k, "test1", "AP50-95")), src)
    diffs = [get(X, "parity_736x1280", s_, "abs_diff") for s_ in ("aa1_test", "test1")]
    M("SxParity", f(max(diffs), 4) if all(_num(d) for d in diffs) else None, src)


def revision_bench():
    src = f"{RV}/bench_desktop.json"
    Bd = load(src) or {}
    flag("Bench", bool(Bd))
    runs = {r["mode"]: r for r in get(Bd, "runs") or []}
    for mk, mode in (("Det", "det"), ("Vlm", "vlm"), ("Both", "both")):
        r = runs.get(mode, {})
        for q, qn in (("p50", "Pfifty"), ("p95", "Pninetyfive"), ("p99", "Pninetynine"), ("max", "Max")):
            M(f"Bd{mk}Etoe{qn}", f(get(r, "frames", "e2e_ms", q), 1), src)
        M(f"Bd{mk}DetP", f(get(r, "frames", "detect_ms", "p50"), 1), src)
        M(f"Bd{mk}DetPnn", f(get(r, "frames", "detect_ms", "p99"), 1), src)
        M(f"Bd{mk}Fps", f(get(r, "frames", "fps"), 1), src)
        M(f"Bd{mk}Drop", pct(get(r, "frames", "drop_rate"), 1), src)
        M(f"Bd{mk}Miss", pct(get(r, "frames", "deadline_miss_rate"), 1), src)
        M(f"Bd{mk}VlmP", f(get(r, "vlm", "latency_s", "p50"), 2), src)
        M(f"Bd{mk}VlmPnf", f(get(r, "vlm", "latency_s", "p95"), 2), src)
        M(f"Bd{mk}VlmReq", i(get(r, "vlm", "requests")), src)
        M(f"Bd{mk}Power", f(get(r, "system", "power_w", "mean"), 1), src)
        M(f"Bd{mk}GpuMem", i(get(r, "system", "gpu_mem_mb", "max")), src)
        M(f"Bd{mk}Jpf", f(get(r, "energy_j_per_frame"), 2), src)
    M("BdDuration", i(get(Bd, "args", "duration")), src)
    M("BdInterval", f(get(Bd, "args", "vlm_interval"), 0), src)
    M("BdMaxTok", i(get(Bd, "args", "vlm_max_tokens")), src)


def revision_scene_tracking():
    for sk, name in (("Vone", "scene_v1"), ("Vtwo", "scene_v2")):
        src = f"results/{name}/tracking.json"
        Tr = load(src) or {}
        flag(f"Track{sk}", bool(Tr))
        for mk, key in (("Ft", "YOLO26n"), ("RfFt", "RF-DETR Nano"), ("Pt", "YOLO26n (pretrained)")):
            M(f"St{sk}{mk}Hota", f(get(Tr, key, "aa1_tuned", "full", "HOTA")), src)
            M(f"St{sk}{mk}HotaDef", f(get(Tr, key, "default", "full", "HOTA")), src)
            M(f"St{sk}{mk}Idf", f(get(Tr, key, "aa1_tuned", "full", "IDF1")), src)
        M(f"St{sk}GtIds", i(get(Tr, "info", "gt_tracks", "ByteTrack-GT")), src)
        A_ = load(f"results/{name}/analytics.json") or {}
        for mk, key in (("Ft", "YOLO26n"), ("RfFt", "RF-DETR Nano"), ("Pt", "YOLO26n (pretrained)")):
            M(f"St{sk}{mk}CrossFone", f(get(A_, key, "crossing_match", "F1"), 2), f"results/{name}/analytics.json")
        S_ = load(f"datasets/{name}/stats.json") or {}
        M(f"St{sk}Frames", i(get(S_, "frames_total")), f"datasets/{name}/stats.json")
        M(f"St{sk}Dur", f(get(S_, "duration_s"), 0), f"datasets/{name}/stats.json")


def revision_human():
    H = load(f"{RV}/human_eval_det_main.json")
    flag("Human", H is not None)
    man = load("research/annotation/tasks/det_main/manifest.json")
    M("HuFramesPlanned", i(len(man["items"])) if man else None, "research/annotation/tasks/det_main/manifest.json")


def revision():
    for fn in (revision_temporal, revision_safety, revision_multiscene, revision_wise, revision_labels,
               revision_resolution, revision_cost, revision_selection, revision_bench, revision_scene_tracking, revision_human):
        fn()


def main():
    for fn in (detection, latency, teacher_bias, tracking, selection, campus, edge, vlm, demo, heldout, revision):
        fn()
    lines = ["% Generated by paper/make_numbers.py from results/*.json. Do not edit by hand.",
             r"\providecommand{\TBD}{\textbf{[TBD]}}"]
    for k, v in FLAGS.items():
        lines += [rf"\newif\ifHave{k}", rf"\Have{k}{'true' if v else 'false'}"]
    for k, v in MACROS.items():
        lines.append(rf"\newcommand{{\{k}}}{{{v}}}")
    (PAPER / "numbers.tex").write_text("\n".join(lines) + "\n")
    n_tbd = sum(len(v) for v in MISSING.values())
    print(f"numbers.tex: {len(MACROS)} macros, {n_tbd} TBD; flags: "
          + ", ".join(f"{k}={'yes' if v else 'no'}" for k, v in FLAGS.items()))
    if MISSING:
        print("WARNING: results missing, macros left as [TBD]:", file=sys.stderr)
        for s, names in MISSING.items():
            print(f"  {s}: {len(names)} macros ({', '.join(names[:6])}{', ...' if len(names) > 6 else ''})",
                  file=sys.stderr)


if __name__ == "__main__":
    main()
