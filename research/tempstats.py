"""Resampling helpers for temporally correlated video frames (shared by the revision analyses).

Frames sampled 0.25 s apart from one video are not independent, so an i.i.d. frame bootstrap understates
uncertainty. Everything here resamples whole contiguous blocks of frames:

  moving block (MBB, Kuensch 1989)      blocks start anywhere in [0, n - b]
  circular block (CBB, Politis & Romano 1992)  blocks wrap around, so every frame is equally likely
  stationary (SB, Politis & Romano 1994)  geometric block lengths with mean b
  optimal_block_length                  Politis & White (2004) with the Patton, Politis & White (2009)
                                        correction, on a per-frame series
  boot_p                                two-sided bootstrap p-value with the (k + 1) / (B + 1) finite-sample
                                        correction (Davison & Hinkley 1997; Phipson & Smyth 2010), so
                                        p is never 0 and its floor is 2 / (B + 1)

Experimental unit = the frame; the resampling unit = a block of b consecutive frames. A block bootstrap over
frames of ONE video still measures only frame-sampling uncertainty within that video, never scene-to-scene
variation (that needs independent videos, see revision/multiscene_eval.py).
"""
import numpy as np


def mbb(n, b, rng):
    starts = rng.integers(0, n - b + 1, int(np.ceil(n / b)))
    return np.concatenate([np.arange(s, s + b) for s in starts])[:n]


def cbb(n, b, rng):
    starts = rng.integers(0, n, int(np.ceil(n / b)))
    return (np.concatenate([np.arange(s, s + b) for s in starts])[:n]) % n


def stationary(n, b, rng):
    out, p = [], 1.0 / b
    while len(out) < n:
        s, L = rng.integers(0, n), rng.geometric(p)
        out.extend((s + np.arange(L)) % n)
    return np.array(out[:n])


SCHEMES = {"mbb": mbb, "cbb": cbb, "sb": stationary}


def resamples(n, b, B, scheme="cbb", seed=0):
    """B index arrays of length n. b <= 1 -> i.i.d. frame bootstrap."""
    rng = np.random.default_rng(seed)
    if b <= 1:
        return [rng.integers(0, n, n) for _ in range(B)]
    return [SCHEMES[scheme](n, min(b, n), rng) for _ in range(B)]


def _acov(x, k):
    x = x - x.mean()
    return np.dot(x[:len(x) - k], x[k:]) / len(x)


def optimal_block_length(x):
    """Politis & White (2004) / Patton et al. (2009) automatic block length for one series.

    Returns {"sb": b_stationary, "cb": b_circular, "m_hat": lag where autocorrelation becomes negligible}.
    """
    x = np.asarray(x, float); n = len(x)
    if n < 8 or x.std() == 0:
        return {"sb": 1.0, "cb": 1.0, "m_hat": 0}
    k_n = max(5, int(np.ceil(np.sqrt(np.log10(n)))))
    m_max = int(np.ceil(np.sqrt(n))) + k_n
    b_max = np.ceil(min(3 * np.sqrt(n), n / 3))
    c = 2.0 * np.sqrt(np.log10(n) / n)
    g0 = _acov(x, 0)
    rho = np.array([_acov(x, k) / g0 for k in range(m_max + 1)])
    m_hat = None
    for m in range(1, m_max - k_n + 1):            # first lag after which k_n consecutive |rho| are insignificant
        if np.all(np.abs(rho[m + 1:m + 1 + k_n]) < c):
            m_hat = m; break
    if m_hat is None:
        m_hat = m_max
    M = min(2 * m_hat, m_max)
    ks = np.arange(-M, M + 1)
    t = np.abs(ks) / max(M, 1)
    lam = np.where(t <= 0.5, 1.0, np.where(t <= 1.0, 2 * (1 - t), 0.0))   # flat-top lag window
    R = np.array([_acov(x, abs(k)) for k in ks])
    G = np.sum(lam * np.abs(ks) * R)
    g = np.sum(lam * R)                                                    # 2*pi * spectral density at 0
    if g <= 0:
        return {"sb": 1.0, "cb": 1.0, "m_hat": int(m_hat)}
    b_sb = (2 * G ** 2 / (2 * g ** 2)) ** (1 / 3) * n ** (1 / 3)
    b_cb = (2 * G ** 2 / ((4 / 3) * g ** 2)) ** (1 / 3) * n ** (1 / 3)
    return {"sb": float(min(max(b_sb, 1), b_max)), "cb": float(min(max(b_cb, 1), b_max)), "m_hat": int(m_hat)}


def boot_p(d):
    """Two-sided p for H0: delta = 0 from bootstrap replicates d of the paired difference (CI inversion),
    with the (k + 1) / (B + 1) correction."""
    d = np.asarray(d); B = len(d)
    lo = (np.sum(d <= 0) + 1) / (B + 1); hi = (np.sum(d >= 0) + 1) / (B + 1)
    return float(min(1.0, 2 * min(lo, hi)))


def ci(x, level=0.95):
    a = (1 - level) / 2
    return [float(v) for v in np.percentile(x, [100 * a, 100 * (1 - a)])]


def wilson(k, n, z=1.959964):
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return [0.0, 1.0]
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(max(0, c - h)), float(min(1, c + h))]
