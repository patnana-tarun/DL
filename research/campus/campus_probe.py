"""Attribute classifiers on the backbone features (campus_features.py): a linear probe per attribute.

Heads (binary logistic regression, class-balanced; L2 strength and feature set (backbone only, or
backbone + colour histograms) picked by 5-fold CV on the training tracks, grouped by track):
  uniform   uniform vs non_uniform
  gender    male vs female
Train: the 200 hand-labelled training tracks (validation/*_train_labels.tsv).
Test : the 100 random hand-labelled tracks (validation/*_labels.tsv), never used for training or tuning.
A track's probability = mean of its crops' probabilities; |p - 0.5| < ABSTAIN -> "unsure".

    python research/campus/campus_probe.py
Writes results/campus/validation/probe_scores.json and, per video, attributes_raw.jsonl + attributes.json
(same format campus_report.py reads).
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths as P  # noqa: E402,F401  (project root as working directory)

import numpy as np
import torch

V = Path("results/campus/validation")
VIDEOS = ["video_1", "video_2"]
ABSTAIN = 0.15            # track prob within 0.35-0.65 -> unsure
HEADS = {                 # head: (label column, positive, negative(s), output field, pos value, neg value)
    "uniform": ("dress", ["uniform"], ["non_uniform"], "dress", "uniform", "non_uniform"),
    "gender": ("gender", ["female"], ["male"], "gender", "female", "male"),
}


def load_labels(suffix):
    lab = {}
    for v in VIDEOS:
        rows = [line.rstrip("\n").split("\t") for line in open(V / f"{v}{suffix}") if not line.startswith("#")]
        head = rows[0]
        for r in rows[1:]:
            d = dict(zip(head, r)); lab[(v, d["track"])] = d
    return lab


def target(d, head):
    col, pos, neg = HEADS[head][:3]
    return 1 if d[col] in pos else 0 if d[col] in neg else None


def fit(X, y, lam, w=None):
    X = torch.tensor(X, dtype=torch.float32); y = torch.tensor(y, dtype=torch.float32)
    w = torch.ones_like(y) if w is None else torch.tensor(w, dtype=torch.float32)
    W = torch.zeros(X.shape[1], requires_grad=True); b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], max_iter=300, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = (w * torch.nn.functional.binary_cross_entropy_with_logits(X @ W + b, y, reduction="none")).sum() / w.sum() \
            + lam * (W * W).sum()
        loss.backward(); return loss
    opt.step(closure)
    return W.detach().numpy(), float(b)


def prob(model, X):
    W, b = model
    return 1 / (1 + np.exp(-(X @ W + b)))


def crops_of(feats, keys):
    """-> X, y, group index for all crops of the labelled tracks."""
    X, g = [], []
    for k, key in enumerate(keys):
        for f in feats[key]:
            X.append(f); g.append(k)
    return np.array(X), np.array(g)


def train_head(feats, lab, head):
    keys = [k for k, d in lab.items() if k in feats and target(d, head) is not None]
    yt = np.array([target(lab[k], head) for k in keys])
    X, g = crops_of(feats, keys); y = yt[g]
    wcls = {c: len(y) / (2 * max((y == c).sum(), 1)) for c in (0, 1)}
    w = np.array([wcls[c] for c in y]) / np.bincount(g)[g]       # class-balanced, each track weighs the same
    rng = np.random.default_rng(0); fold = rng.permutation(len(keys)) % 5
    best = None
    for lam in (1e-3, 1e-2, 1e-1, 1.0):
        acc, nll = [], []
        for f in range(5):
            tr = fold[g] != f
            m = fit(X[tr], y[tr], lam, w[tr])
            for k in np.nonzero(fold == f)[0]:
                p = np.clip(prob(m, X[g == k]).mean(), 1e-4, 1 - 1e-4)
                acc.append((p > 0.5) == yt[k]); nll.append(-np.log(p if yt[k] else 1 - p))
        # balanced accuracy (both classes count equally), ties broken by log-loss; no abstention here
        bal = np.mean([np.mean([a for a, t in zip(acc, yt) if t == c]) for c in (0, 1)])
        if best is None or (bal, -np.mean(nll)) > best[:2]:
            best = (bal, -np.mean(nll), lam)
    return fit(X, y, best[2], w), {"lambda": best[2], "cv_balanced_accuracy": round(float(best[0]), 3),
                                   "n_tracks": len(keys), "n_pos": int(yt.sum()), "n_neg": int(len(yt) - yt.sum())}


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [round((c - h) / d, 3), round((c + h) / d, 3)]


def load_features():
    """-> {"dino": feats, "dino+colour": feats} (each {(video, track): [standardised crop vectors]}), names."""
    sets, names = {"dino": {}, "dino+colour": {}}, {}
    for v in VIDEOS:
        z = np.load(f"results/campus/{v}/features.npz")
        F = z["feats"].astype(np.float32); C = np.concatenate([F, z["colour"].astype(np.float32)], 1)
        for n, f, c in zip(z["names"], F, C):
            key = (v, n.split("_")[0])
            sets["dino"].setdefault(key, []).append(f); sets["dino+colour"].setdefault(key, []).append(c)
            names.setdefault(key, []).append(str(n))
    for s, feats in sets.items():
        allf = np.concatenate([np.array(f) for f in feats.values()])
        mu, sd = allf.mean(0), allf.std(0) + 1e-6              # standardise (unsupervised, all crops)
        sets[s] = {k: (np.array(f) - mu) / sd for k, f in feats.items()}
    return sets, names


def evaluate(model, feats, test, h):
    res = {}
    for scope in VIDEOS + ["both"]:
        k = n = lab_n = 0; errs = []
        for key, d in test.items():
            if (scope != "both" and key[0] != scope) or key not in feats:
                continue
            t = target(d, h)
            if t is None:
                continue
            lab_n += 1
            p = prob(model, feats[key]).mean()
            if abs(p - 0.5) < ABSTAIN:
                continue
            n += 1; k += int((p > 0.5) == t)
            if (p > 0.5) != t:
                errs.append(f"{key[0]}#{key[1]} true={t} p={p:.2f}")
        res[scope] = {"accuracy": round(k / n, 3) if n else None, "ci95": wilson(k, n), "correct": k,
                      "answered": n, "labelled": lab_n, "coverage": round(n / lab_n, 3) if lab_n else None,
                      "errors": errs}
    return res


def main():
    sets, names = load_features()
    train, test = load_labels("_train_labels.tsv"), load_labels("_labels.tsv")
    assert not set(train) & set(test)

    models, use, report = {}, {}, {"protocol": __doc__.split("\n\n")[1]}
    for h in HEADS:
        # feature set per head chosen by training CV only (balanced accuracy, then log-loss)
        cands = {s: train_head(sets[s], train, h) for s in sets}
        use[h] = max(cands, key=lambda s: (cands[s][1]["cv_balanced_accuracy"], -cands[s][1]["lambda"]))
        models[h], info = cands[use[h]]
        info = {"features": use[h], **info,
                "cv_by_features": {s: c[1]["cv_balanced_accuracy"] for s, c in cands.items()}}
        res = evaluate(models[h], sets[use[h]], test, h)
        report[h] = {"train": info, "test_per_track": res}
        print(h, info, {s: (r["accuracy"], f'{r["correct"]}/{r["answered"]}', f'cov {r["coverage"]}') for s, r in res.items()})

    # apply to every crop of every track
    for v in VIDEOS:
        out = Path("results/campus") / v
        per_track = {}
        with open(out / "attributes_raw.jsonl", "w") as fo:
            for key in sorted((k for k in names if k[0] == v), key=lambda k: int(k[1])):
                P = {h: prob(models[h], sets[use[h]][key]) for h in HEADS}; X = P["gender"]
                for j, n in enumerate(names[key]):
                    d = {"crop": n, "track": int(key[1]), "frame": int(n.split("_")[1])}
                    for h, (_, _, _, field, pos, neg) in HEADS.items():
                        p = P[h][j]; d[field] = "unsure" if abs(p - 0.5) < ABSTAIN else (pos if p > 0.5 else neg)
                        d[field + "_p"] = round(float(p), 3)
                    fo.write(json.dumps(d) + "\n")
                r = {"n_crops": len(X)}
                for h, (_, _, _, field, pos, neg) in HEADS.items():
                    p = float(P[h].mean())
                    r[field] = "unsure" if abs(p - 0.5) < ABSTAIN else (pos if p > 0.5 else neg)
                    r[field + "_conf"] = round(max(p, 1 - p), 3)
                per_track[key[1]] = r
        json.dump(per_track, open(out / "attributes.json", "w"), indent=1)
        print(v, "tracks labelled:", len(per_track))
    report["test_per_person"] = per_person(test)
    json.dump(report, open(V / "probe_scores.json", "w"), indent=1)


def per_person(test):
    """Accuracy of the final per-person output (what campus_report reports): test tracks are mapped to
    their stitched person, the person's attributes are the vote over all its crops, and a person is
    scored once even if several of its tracks were sampled (the seated guard has 7 test tracks)."""
    import campus_report as R
    FIELD = {"uniform": ("dress", "dress"), "gender": ("gender", "gender")}
    out = {}
    for h, (col, field) in FIELD.items():
        k = n = lab_n = 0
        for v in VIDEOS:
            o = Path("results/campus") / v; fps = float(np.load(o / "tracks.npz")["fps"])
            _, _, keep = R.load_tracks(o, fps)
            tattr = {int(a): b for a, b in json.load(open(o / "attributes.json")).items()}
            person, _ = R.stitch(keep, tattr, fps, R.appearance(o, keep))
            pa = R.person_attributes(person, [json.loads(x) for x in open(o / "attributes_raw.jsonl")])
            seen = {}
            for (vv, t), d in test.items():
                if vv != v or target(d, h) is None or int(t) not in person:
                    continue
                seen.setdefault(person[int(t)], set()).add(target(d, h))
            for p, labs in seen.items():
                if len(labs) != 1:
                    continue                                   # conflicting hand labels: ID switch, skip
                t = labs.pop(); lab_n += 1
                pred = pa.get(p, {}).get(field, "unsure")
                if pred == "unsure":
                    continue
                pos = HEADS[h][4]
                n += 1; k += int((pred == pos) == bool(t))
        out[h] = {"accuracy": round(k / n, 3) if n else None, "ci95": wilson(k, n), "correct": k, "answered": n,
                  "labelled_people": lab_n, "coverage": round(n / lab_n, 3) if lab_n else None}
        print("per person", h, out[h])
    return out


if __name__ == "__main__":
    main()
