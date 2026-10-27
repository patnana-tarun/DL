"""Block resampling, corrected bootstrap p-values and block-length selection (research/tempstats.py)."""
import numpy as np

import tempstats as T


def test_circular_blocks_wrap_and_are_contiguous():
    idx = T.cbb(10, 4, np.random.default_rng(1))
    assert len(idx) == 10 and idx.min() >= 0 and idx.max() < 10
    for blk in (idx[:4], idx[4:8]):
        assert all((b - a) % 10 == 1 for a, b in zip(blk[:-1], blk[1:]))


def test_iid_when_block_is_one():
    r = T.resamples(50, 1, 3, seed=0)
    assert len(r) == 3 and all(len(x) == 50 for x in r)


def test_boot_p_never_zero_and_floor():
    B = 999
    assert T.boot_p(np.ones(B)) == 2 / (B + 1)                  # all replicates > 0 -> floor, not 0
    assert T.boot_p(np.r_[np.ones(500), -np.ones(499)]) == 1.0


def test_block_length_grows_with_autocorrelation():
    rng = np.random.default_rng(0)
    white = rng.normal(size=400)
    ar = np.zeros(400)
    for t in range(1, 400):
        ar[t] = 0.9 * ar[t - 1] + rng.normal()
    assert T.optimal_block_length(ar)["cb"] > 3 * T.optimal_block_length(white)["cb"]


def test_wilson_bounds():
    lo, hi = T.wilson(0, 50)
    assert lo == 0 and 0.06 < hi < 0.08                          # 0/50 -> upper bound ~7.1 %
