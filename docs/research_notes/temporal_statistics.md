# Temporally-Aware Statistical Inference for Detector/Tracker Evaluation on Video Frames

Setup these notes are written for: the detection test set is n = 182 frames sampled every 0.25 s from one video (about 45.5 s of footage). The reviewer asks for (i) a block bootstrap with 2.5 / 5 / 10 s blocks, which is L = 10 / 20 / 40 frames; (ii) a precise account of how "p < 0.001" was obtained from B = 1,000 resamples; and (iii) explicit definitions of the experimental units. All arithmetic for n = 182 below was computed in this session.

---

## 1. Moving block (Künsch 1989), circular block (Politis & Romano 1992) and stationary bootstrap (Politis & Romano 1994): differences and when to use which

### Takeaway
All three methods resample contiguous runs of frames, so short-range autocorrelation inside each run is kept. MBB uses overlapping blocks of fixed length L. CBB is MBB with the series wrapped onto a circle, so frames near the ends are not under-sampled. SB uses random block lengths drawn from a geometric distribution with mean L, also wrapped. After the Nordman (2008) / Patton-Politis-White (2009) correction, fixed-length overlapping blocks (MBB/CBB) are asymptotically more efficient than SB (MSE ratio CB/SB = (2/3)^(2/3) ≈ 0.763). For a reviewer who asked for specific block lengths in seconds, the cleanest primary method is the **circular block bootstrap at fixed L = 10/20/40 frames**, with SB at the same mean lengths as a robustness check.

### Cited Findings
- Künsch (1989), "The jackknife and the bootstrap for general stationary observations", *Annals of Statistics* 17, 1217–1241, is the standard reference for the fixed-block (moving block) bootstrap. R's `boot::tsboot(sim="fixed")` builds each replicate by "taking blocks of length `l` from the original time series and putting them end-to-end". — [boot::tsboot docs](https://rdrr.io/cran/boot/man/tsboot.html)
- Stationary bootstrap: Politis & Romano (1994), *JASA* 89(428), 1303–1313. Blocks are resampled as in the fixed scheme, except that "the block lengths are generated from a geometric distribution with mean `l`". — [boot::tsboot docs](https://rdrr.io/cran/boot/man/tsboot.html); [UCY repository record of the paper](https://gnosis.library.ucy.ac.cy/handle/7/57533)
- In `tsboot`, `endcorr` (end correction) is automatically TRUE for `sim="geom"`. — [boot::tsboot docs](https://rdrr.io/cran/boot/man/tsboot.html)
- Lahiri (1999), "Theoretical comparisons of block bootstrap methods", *Ann. Statist.* 27(1), 386–404. To first order, all block-bootstrap estimators have the same bias, but their variances can differ. "Overlapping blocks [are] preferred over nonoverlapping blocks", and random block lengths "typically [lead] to mean-squared errors larger than those for nonrandom block lengths". — [Project Euclid](https://projecteuclid.org/euclid.aos/1018031117)
- Correction: Nordman (2008) found an error in Lahiri's (1999) variance calculation for the stationary bootstrap. Patton, Politis & White (2009) then corrected Politis & White (2004): the SB variance constant is D_SB = 2g²(0), and the asymptotic relative efficiency is ARE_CB/SB = lim MSE_opt,CB / MSE_opt,SB = (2/3)^(2/3) ≈ 0.7631. In other words, the circular bootstrap at its optimal block length has about 24% lower asymptotic MSE for variance estimation than SB at its optimal mean block length. — [Patton, Politis & White 2009, Econometric Reviews 28(4):372–375](https://public.econ.duke.edu/~ap172/Patton_Politis_White_2009.pdf)
- Simulations in that correction (AR(1), N = 200 and 800): the finite-sample attainable relative efficiency FARE_CB/SB ranged from 0.63 to 0.98. In every case CB had lower MSE than SB when both used estimated block sizes. — [Patton, Politis & White 2009](https://public.econ.duke.edu/~ap172/Patton_Politis_White_2009.pdf)
- Python: `arch.bootstrap` provides `MovingBlockBootstrap`, `CircularBlockBootstrap` and `StationaryBootstrap`. For the stationary bootstrap, the block-size argument is the *average* block size. — [arch StationaryBootstrap docs](https://arch.readthedocs.io/en/latest/bootstrap/generated/arch.bootstrap.StationaryBootstrap.html)
- The arch docs say the IID bootstrap is a poor choice for persistent data, and that "a time-series bootstrap with an appropriately chosen block size should be used". — [arch confidence-interval docs](https://arch.readthedocs.io/en/latest/bootstrap/confidence-intervals.html)

### Inferences
- **Mechanics, for writing the Methods section.** Index frames t = 1..n.
  - *MBB:* there are n − L + 1 overlapping blocks B_j = (X_j, …, X_{j+L−1}). Draw k = ⌈n/L⌉ start indices j uniformly from {1, …, n − L + 1}, concatenate, and truncate to n.
  - *CBB:* the same, but indices are taken mod n (X_{n+i} ≡ X_i). All n start positions are allowed, so every frame appears in exactly L of the n blocks and edge frames are not under-represented.
  - *SB:* start at a uniform random index. At each step, with probability p = 1/L jump to a new uniform index, otherwise continue to the next frame (circularly). Block lengths are Geometric(p) with mean L.
- **When to use which.**
  - Use CBB (or MBB) when you want to report results at fixed, interpretable block durations, as the reviewer requested, and for efficiency (Lahiri 1999, corrected per Patton et al. 2009).
  - Use SB when you want a resampled series that is itself stationary, or as a smoothness check, since random lengths blur the hard cutoff of a fixed L. Run it at mean lengths 10/20/40 as a robustness row.
  - Use the non-overlapping block bootstrap (Carlstein-type) only for its simplicity. Lahiri (1999) shows overlapping blocks are preferred.
- **Practical caveat specific to n = 182.** The number of blocks needed to rebuild one series is ⌈182/L⌉ = 19, 10 and 5 for L = 10, 20 and 40.
  - With 5 blocks of 10 s, the bootstrap distribution is built from only about 4.55 "effective" independent pieces. Intervals will be wide and somewhat unstable, and should be reported as indicative only.
  - CBB still has n = 182 distinct start positions, so the resampling distribution is not degenerate. But the information content is roughly that of 4–5 independent observations.
- **Recommendation for the paper:**
  - Primary: paired CBB at L ∈ {10, 20, 40} frames.
  - Secondary rows: SB at the same mean lengths, plus the data-driven Politis–White length (Section 2).
  - Baseline row: the original i.i.d. frame bootstrap (L = 1), so the reader can see the CI inflation.

### Gaps
- I did not retrieve the original Politis & Romano (1992) circular-bootstrap paper ("A circular block-resampling procedure for stationary data", in *Exploring the Limits of Bootstrap*, Wiley, pp. 263–270). That citation is from background knowledge and should be checked before use.
- I did not retrieve Künsch (1989) directly; the citation comes via the `tsboot` reference list.
- The claim that SB is "less sensitive to block-length misspecification" is commonly stated, but I did not verify it against a primary source in this session.
- Lahiri (2003), *Resampling Methods for Dependent Data* (Springer), was not accessed. It is the canonical book-length reference for all of the above, but specific page numbers or theorems are not given here.

---

## 2. Block-length selection: Politis & White (2004) automatic selection with the Patton–Politis–White (2009) correction; n^{1/3} rules of thumb; reporting sensitivity

### Takeaway
The optimal block length grows like n^{1/3} for variance or bias estimation. For n = 182 that is about 5.7 frames (about 1.4 s) times a data-dependent constant that is larger when autocorrelation is stronger. The Politis–White estimator (corrected 2009; implemented as `arch.bootstrap.optimal_block_length`) estimates this constant from the sample autocorrelations. It needs a scalar per-frame series, so for AP you must feed it a per-frame proxy or linearisation (see Inferences). Report the CI at L = 10/20/40 frames and at the Politis–White estimate, and treat a plateau in CI width as L grows as evidence that the dependence has been captured.

### Cited Findings
- Hall, Horowitz & Jing (1995), "On blocking rules for the bootstrap with dependent data", *Biometrika* 82(3), 561–574. The optimal block size "depends significantly on context, being equal to n^{1/3}, n^{1/4} and n^{1/5} in the cases of variance or bias estimation, estimation of a one-sided distribution function, and estimation of a two-sided distribution function, respectively". The paper also gives an empirical (subsampling/cross-validation) rule for choosing the constant. — [R blocklength::hhj docs](https://search.r-project.org/CRAN/refmans/blocklength/html/hhj.html); [abstract mirror](https://academicnewsletter.sufe.edu.cn/info/449466)
- The R package `blocklength` implements the HHJ cross-validation method, the Politis–White plug-in, and the Lahiri–Furukawa–Lee plug-in. — [blocklength on CRAN](https://cran.r-project.org/web/packages/blocklength/refman/blocklength.html)
- Politis & White (2004), "Automatic Block-Length Selection for the Dependent Bootstrap", *Econometric Reviews* 23(1), 53–70, DOI 10.1081/ETC-120028836. — [IDEAS/RePEc](https://ideas.repec.org/a/taf/emetrv/v23y2004i1p53-70.html)
- The 2009 correction sets D_SB = 2ĝ²(0) and keeps the 2004 block-size formulas otherwise. On AR(1) data, the corrected estimators gave mean b̂/b_opt between about 0.86 and 1.14, i.e. "between 90% and 110% of the respective true optimal block sizes" on average. RMSE roughly halves when N goes from 200 to 800. — [Patton, Politis & White 2009](https://public.econ.duke.edu/~ap172/Patton_Politis_White_2009.pdf)
- Theoretical optimal block lengths from the same paper (AR(1) with φ = 0.7): b_opt,SB = 11.47 and b_opt,CB ≈ 13.12 at N = 200; 18.20 and 20.83 at N = 800. With φ = 0.1 at N = 200, they are 2.01 and 2.31. — [Patton, Politis & White 2009](https://public.econ.duke.edu/~ap172/Patton_Politis_White_2009.pdf)
- `arch.bootstrap.optimal_block_length(x)` returns a DataFrame with columns `b_sb` (stationary) and `b_cb` (circular). It cites both Politis & White (2004) and the Patton, Politis & White (2009) correction. Its implementation:
  - block length b = (2 g² n / d)^{1/3};
  - flat-top lag window h(x) = min(1, 2(1 − |x|));
  - constant c = 2 for SB and 4/3 for CB (so D_SB = 2ĝ(0)² and D_CB = (4/3)ĝ(0)²);
  - tuning lag m = the first lag after which k_n consecutive autocorrelations fall inside ±2√(log₁₀ n / n), with k_n = max(5, log₁₀ n) and maximum m = ⌈√n + k_n⌉;
  - autocovariances computed with the full sample length, so results can differ from Patton's MATLAB code.

  — [arch optimal_block_length docs](https://arch.readthedocs.io/en/latest/bootstrap/generated/arch.bootstrap.optimal_block_length.html)

### Inferences
- **Formulas to state in the paper (Politis–White, corrected):**
  - Autocovariances: R̂(k) = (1/n) Σ_{t=1}^{n−|k|} (X_t − X̄)(X_{t+|k|} − X̄).
  - Ĝ = Σ_{|k|≤M} λ(k/M) |k| R̂(k).
  - ĝ(0) = Σ_{|k|≤M} λ(k/M) R̂(k), which is the long-run variance / spectral density at 0 up to 2π.
  - λ is the flat-top window: λ(t) = 1 for |t| ≤ 1/2, 2(1 − |t|) for 1/2 < |t| ≤ 1, 0 otherwise.
  - M = 2m̂.
  - b̂_opt,SB = (2Ĝ² / D̂_SB)^{1/3} n^{1/3} with D̂_SB = 2ĝ(0)².
  - b̂_opt,CB = (2Ĝ² / D̂_CB)^{1/3} n^{1/3} with D̂_CB = (4/3)ĝ(0)².

  The general shape b = (2g²n/d)^{1/3} and the constants 2 and 4/3 are confirmed by the arch docs. The exact definitions of Ĝ and of M = 2m̂ are given from memory of Politis & White (2004) and should be checked against the paper.
- **Numbers for n = 182** (computed):
  - n^{1/3} = 5.67, n^{1/4} = 3.67, n^{1/5} = 2.83 frames.
  - The arch insignificance band is ±2√(log₁₀182/182) ≈ ±0.223. Autocorrelations below about 0.22 count as "insignificant" when choosing m̂, so the estimator can miss weak but long-range dependence.
  - Maximum m = ⌈√182 + 5⌉ = 19 lags (4.75 s).
  - The data-driven method therefore cannot "see" dependence beyond about 5 s. That is one more reason to report the fixed 10 s (40-frame) block as the conservative end of the sensitivity range.
- **Feeding a non-decomposable metric to Politis–White.** The estimator needs a scalar series x_t whose mean is the statistic. For AP, use one of these:
  - (a) **Jackknife pseudo-values** (linearisation). ψ_t = n·AP − (n − 1)·AP_(−t), where AP_(−t) is AP with frame t removed. Then AP ≈ mean(ψ_t), and the ψ_t series carries the per-frame influence. For a paired comparison, use ψ_t^A − ψ_t^B. This costs n = 182 extra AP evaluations, which is cheap with cached matches.
  - (b) A decomposable proxy such as the per-frame count of TPs minus FPs at IoU 0.5, or per-frame recall.
  - Report the b̂ obtained and round it to an integer number of frames.
- **Reporting sensitivity.** Make a table with rows L ∈ {1 (i.i.d.), b̂_PW, 10, 20, 40} frames. Columns: point estimate, 95% CI, CI half-width, the inflation factor (half-width ÷ i.i.d. half-width), and the paired-difference p-value.
  - Optionally plot CI half-width against L. Expect the half-width to rise and then plateau.
  - If it is still rising at L = 40, dependence extends past 10 s and the clip effectively contains only a few independent observations. Say so explicitly.
  - In the text, state which L is primary. The Politis–White estimate, or the reviewer's 5 s (L = 20), is a defensible primary; the other lengths are sensitivity analyses.

### Gaps
- I could not read the Politis & White (2004) paper itself, so the exact Ĝ / M definitions above are from background knowledge (see note).
- The arch docs paraphrase k_n as max(5, log₁₀ n). My recollection of the original is K_N = max(5, √(log₁₀ N)); for n = 182 both give 5, so the discrepancy does not matter here.
- I found no source on block-length selection specifically for ranking metrics like AP. The pseudo-value approach is my inference.

---

## 3. Bootstrapping a non-decomposable metric (COCO AP50:95): resample frames/blocks, recompute AP on the pooled resampled set; literature on bootstrap CIs for mAP

### Takeaway
The established practice, from PASCAL VOC and then ILSVRC, is to resample test **images** with replacement and recompute the full metric (AP) on each pooled replicate. Pairwise comparisons are made on the **difference** computed on the same replicate. For video, the only change is that the resampling unit becomes a temporal block rather than a single frame. AP is computed on the concatenated frames of the resampled blocks, with duplicated frames contributing their detections and ground truths once per copy.

### Cited Findings
- PASCAL VOC bootstrap note (Everingham, Eslami, Van Gool, Williams, Winn, Zisserman, 2013):
  - "the data points (images in our case) are sampled with replacement from the original n test points to produce B bootstrap replicates".
  - To compare methods A and B, "we first compute the difference in scores on each bootstrap sample. We then obtain a confidence interval by sorting the differences, and then returning the α/2 and 1 − α/2 quantiles", which is the percentile method.
  - The null that A is equivalent to B "is rejected if zero is not contained in the obtained confidence interval".
  - — [VOC bootstrap note](https://homepages.inf.ed.ac.uk/ckiw/postscript/bootstrap_note.pdf)
- The same note explains why pairing matters: the paired procedure "is more informative than the unpaired bootstrap confidence intervals… a variation in the hardness of the bootstrap replicates may give rise to overlapping score intervals, even if method A always beats method B". It also notes that AP's "computation depends on all of the images", so AP is recomputed per replicate. Rank CIs can be built from per-replicate ranks. — [VOC bootstrap note](https://homepages.inf.ed.ac.uk/ckiw/postscript/bootstrap_note.pdf)
- The method was published in the VOC retrospective: Everingham et al., "The Pascal Visual Object Classes Challenge: A Retrospective", *IJCV* 111, 98–136 (2015). — [IJCV preprint](https://homepages.inf.ed.ac.uk/ckiw/postscript/ijcv_voc14.pdf); [Oxford ORA record](https://ora.ox.ac.uk/objects/uuid:4be61639-601d-49f8-9a68-62ce1f52bb92)
- ILSVRC (Russakovsky et al. 2015) follows the VOC strategy. "For each method we obtain a confidence interval of its score using bootstrap sampling. During each bootstrap round, we sample N images with replacement", then discards the lower and upper α fraction (percentile interval). — [ILSVRC paper, arXiv:1409.0575](https://arxiv.org/abs/1409.0575)
- I found no detection-specific paper that applies a **block** bootstrap to video-frame AP. Searches for "bootstrap confidence intervals mAP object detection" returned only the VOC/ILSVRC image-level precedent and generic material. — [search result: ILSVRC](https://arxiv.org/abs/1409.0575); [Raschka, CIs for ML classifiers](https://sebastianraschka.com/blog/2022/confidence-intervals-for-ml.html)

### Inferences
- **Algorithm (paired circular block bootstrap for COCO AP), recommended implementation.**
  1. Run COCO matching **once per frame** for each model (A, B), at each IoU threshold τ ∈ {0.50, 0.55, …, 0.95}, per category, using COCOeval's per-image greedy matching and maxDets = 100.
     - Cache per frame t: the detection scores, the TP/FP flags per τ, the ignore flags, and the number of non-ignored GTs per category.
     - Matching is within-image, so these cached records are exactly what COCOeval's `accumulate` pools. Resampling frames does not require re-matching.
  2. For b = 1..B, draw a CBB index sequence I_b = (t_1, …, t_n) with block length L. Use the **same I_b for both models** (pairing).
  3. For each model, pool the cached records over the multiset I_b, so a duplicated frame contributes its detections and GTs twice. Then recompute precision/recall per category and τ, apply COCO's 101-point interpolated AP, and average over categories and τ. This gives AP*_A,b and AP*_B,b and the difference Δ*_b = AP*_A,b − AP*_B,b.
  4. CI for each AP: percentile [q_{α/2}, q_{1−α/2}] of {AP*_b}. CI for Δ: the percentile of {Δ*_b}. See Section 5 for basic/BCa alternatives.
  5. p-value: see Section 4.
- **Implementation pitfalls.**
  - pycocotools' `accumulate()` uses `params.imgIds` positions to index `evalImgs`, so passing a list with duplicate image ids does not reliably weight duplicates. Re-implement the pooling step over the cached arrays, which is about 30 lines of NumPy, or relabel duplicated frames with fresh image ids and duplicate their GT/detection entries.
  - Categories with zero GT in a replicate: COCO sets AP to −1 and excludes the category from the mean. Do the same, and report how often it happens. It is frequent with 5 blocks of 10 s if a class appears in only one segment.
  - B: use B ≥ 2,000 for 95% percentile CIs, and B = 9,999 or 10,000 if p-values near 10⁻³ will be reported (Section 4). Per-replicate cost is only a sort over pooled detections, so 10⁴ replicates for each of 4 block lengths is cheap.
- **Tracking metrics.** Ratio-of-sums metrics such as MOTA = 1 − Σ(FN + FP + IDSW)/ΣGT can be block-bootstrapped by caching **per-block sums** (FN_k, FP_k, IDSW_k, GT_k) from one full-sequence evaluation and resampling blocks. ID switches that fall exactly on block boundaries are assigned to the block containing the later frame.
  - IDF1 and HOTA depend on a global identity assignment over the whole sequence. Re-solving that assignment on a spliced, resampled sequence is ill-defined, because tracks would jump across splice points.
  - The practical approach is a *conditional* bootstrap: fix the global ID matching from the full-sequence evaluation, cache per-block IDTP/IDFP/IDFN (or HOTA's per-frame association quantities), and resample blocks. State in the paper that the CI is conditional on the global assignment.
  - For tracks, the natural resampling unit is a time segment, not a single track, because tracks overlap in time and share frames.
- **Concrete one-video protocol (summary for the writer):**
  - Units: frames are the observation unit; temporal blocks are the resampling unit.
  - Paired CBB with L ∈ {10, 20, 40} frames plus Politis–White b̂ plus i.i.d. (L = 1) for reference.
  - B = 10,000; percentile CIs for AP and ΔAP; p-value = (b + 1)/(B + 1) on the null-centred statistic (Section 4).
  - Report the CI inflation relative to i.i.d. and n_eff (Section 6).

### Gaps
- I found no peer-reviewed paper specifically validating the block bootstrap for mAP on video. The method is a direct combination of the VOC/ILSVRC image bootstrap and standard block-bootstrap theory. Formal validity for AP, a smooth-ish functional of the empirical distribution of (score, TP) pairs and GT counts, is plausible but not established in the sources I found.
- I found no reference for the effect of duplicated frames on COCO's maxDets or area-range handling. This is per-image, so duplication should be harmless, but I did not verify it.

---

## 4. Paired bootstrap tests for model comparison; p-values with the (b+1)/(B+1) correction (Davison & Hinkley 1997; Phipson & Smyth 2010); reporting p < 1/(B+1)

### Takeaway
With B = 1,000 resamples, the smallest valid Monte-Carlo p-value is 1/(B + 1) = 1/1001 ≈ 0.000999. A reported "p < 0.001" is therefore only defensible as "p ≤ 1/1001 (the minimum attainable with B = 1,000)". If it came from b/B = 0/1000, it was computed with a formula that understates p and can return an invalid zero.

Recompute with p = (1 + #{T*_b ≥ T_obs})/(B + 1), where T* is drawn from a **null-centred** bootstrap distribution, and use the same blocks for both models. Increase B to 9,999 if you want to resolve p near 10⁻³. Also note that bootstrap tests are only asymptotically valid, and with 5–19 blocks that asymptotic regime is weak.

### Cited Findings
- Phipson & Smyth (2010), "Permutation P-values Should Never Be Zero: Calculating Exact P-values When Permutations Are Randomly Drawn", *Stat. Appl. Genet. Mol. Biol.* 9(1), Art. 39. — [arXiv:1603.05766](https://arxiv.org/abs/1603.05766); [journal](https://www.degruyterbrill.com/document/doi/10.2202/1544-6115.1585)
  - "published p-values obtained from permutation tests are understated by about 1/m, where m is the number of permutations".
  - Software returning exactly zero is "of particular concern". Applying multiple-testing procedures to zero p-values can be "grossly misleading".
- Same paper, on why p̂ = B/m fails: under H₀, p̂ is discrete uniform on {0, 1/m, …, 1}, so P(p̂ ≤ α) = (⌊mα⌋ + 1)/(m + 1). "The type I error rate always exceeds α when α = i/m for some integer i, or when α < 1/(m + 1)", and P(p̂ ≤ α) "is never less than 1/(m + 1)". — [Phipson & Smyth, arXiv:1603.05766](https://arxiv.org/abs/1603.05766)
- Same paper, the exact Monte Carlo p-value: with b exceedances out of m simulated statistics, p_u = P(B ≤ b) = (b + 1)/(m + 1). The positive bias is "just enough to allow for the uncertainty of estimation and to produce a test with the correct size". This "is presented in a number of textbooks on randomization tests including Davison and Hinkley (1997), Manly (1997) and Edgington and Onghena (2007)". The same formula holds for permutations sampled without replacement. — [Phipson & Smyth, arXiv:1603.05766](https://arxiv.org/abs/1603.05766)
- Gordon Smyth (co-author), commenting on a zero result from n permutations: it should be read as the true p-value being below about 1/n, and 1/(n + 1) is the suggested estimate. — [Biostars thread](https://biostars.org/p/453567)
- Dror et al. (2018) describe the paired bootstrap test as resampling the test set with replacement, with "the p-value… calculated in a similar manner to the permutation test". They note it "is less effective for small test sets, as it assumes that the test set distribution does not deviate too much from the population distribution". — [Dror et al., ACL 2018, P18-1128](https://aclanthology.org/P18-1128/)
- The VOC procedure tests by CI inversion: reject equivalence of A and B at level α if 0 is outside the paired percentile CI of the difference. — [VOC bootstrap note](https://homepages.inf.ed.ac.uk/ckiw/postscript/bootstrap_note.pdf)

### Inferences
- **Why the "null-centred" step matters.** The bootstrap distribution of Δ* = AP*_A − AP*_B is centred near the observed Δ̂, not near 0. Counting #{Δ*_b ≤ 0} therefore gives a percentile/CI-inversion quantity (the achieved significance level of the percentile CI), not a classical tail probability under H₀. Two correct ways to get a p-value:
  - (a) **Shifted (null-centred) bootstrap test**, two-sided:
    p = (1 + #{b : |Δ*_b − Δ̂| ≥ |Δ̂|}) / (B + 1).
    The one-sided version for H₁: A > B is p = (1 + #{b : Δ*_b − Δ̂ ≥ Δ̂}) / (B + 1). This is equivalent to Efron–Tibshirani's recentring and to counting Δ*_b ≥ 2Δ̂.
  - (b) **CI inversion** (VOC style), two-sided:
    p = min(1, 2·min(1 + #{Δ*_b ≤ 0}, 1 + #{Δ*_b ≥ 0}) / (B + 1)).
    This is the smallest α at which the (1 − α) percentile CI excludes 0.
  - Both use the "+1" correction, so p ≥ 1/(B + 1) (two-sided (b): ≥ 2/(B + 1)).
- **Strictly exact alternative under dependence: paired block sign-flip / swap (approximate randomisation) test.**
  - Partition the 182 frames into K contiguous blocks (non-overlapping, L frames each).
  - Under H₀ ("the two models' outputs are exchangeable within each block"), randomly swap the A and B detections for all frames of each block, with probability 1/2 independently per block. Recompute ΔAP and repeat m times.
  - p = (1 + #{|Δ_perm| ≥ |Δ_obs|}) / (m + 1). Here the Phipson–Smyth exactness argument applies directly, because this is a genuine randomisation test.
  - Caveat: there are only 2^K distinct swap patterns (K = 5 for 10 s blocks gives 32 patterns). The smallest possible exact two-sided p is then about 2/2^K: 0.0625 for K = 5, 0.0039 for K = 9, 7.6·10⁻⁶ for K = 18 (computed). Enumerate all 2^K patterns when K ≤ about 20.
  - This makes the temporal-dependence penalty explicit and is a strong answer to the reviewer.
- **What to write about the original "p < 0.001".** Something like: "With B = 1,000 resamples, the smallest attainable p-value under the (b + 1)/(B + 1) convention [Davison & Hinkley 1997; Phipson & Smyth 2010] is 1/1001 ≈ 0.001. In the original submission, no resample produced a difference at least as extreme as observed, i.e. b = 0. We now report p = 1/(B + 1) and, in the revised analysis, use B = 9,999 paired circular-block resamples, so the minimum attainable p is 10⁻⁴."
  - Report "p ≤ 1/(B + 1)" or "p < 0.001 (b = 0 of B = 9,999)" rather than "p = 0".
  - Monte-Carlo SE of a p estimate: √(p(1 − p)/B). For p ≈ 0.01 and B = 1,000 that is about 0.003.
- **Multiple comparisons.** If several model pairs or several block lengths are tested, apply Holm or Bonferroni to the corrected p-values. Zero p-values break those procedures, per Phipson & Smyth. Block lengths are a sensitivity analysis, not independent tests, so designate one primary L in advance and do not multiplicity-correct across L.

### Gaps
- I did not access Davison & Hinkley (1997) directly. The attribution of (b + 1)/(B + 1) to that textbook comes via Phipson & Smyth (2010).
- The validity of (b + 1)/(B + 1) as an "exact" p-value is proven for Monte-Carlo and permutation tests. For bootstrap tests it is a conservative convention; bootstrap tests are only asymptotically valid. I found no source that quantifies this gap for block bootstraps with few blocks.
- I did not retrieve Berg-Kirkpatrick et al. (2012, EMNLP), which discusses the "Δ* ≥ 2Δ̂" form of the paired bootstrap in NLP. The recentring formulas above follow standard bootstrap-testing logic (Efron & Tibshirani 1993, §16) and were not checked against those texts in this session.

---

## 5. Percentile vs BCa vs basic intervals for the block bootstrap

### Takeaway
Use **percentile** intervals as the primary, following VOC, ILSVRC and Bouthillier et al. 2021, with **basic** intervals as a sensitivity check. Avoid standard BCa with a block bootstrap unless the acceleration is computed with a *block* jackknife. Common implementations, including arch, compute it with a leave-one-out jackknife over individual observations, which ignores temporal dependence. Studentized block-bootstrap intervals have better theoretical accuracy but need a block-consistent standard error and a nested bootstrap. That is overkill, and unstable, with 5–19 blocks.

### Cited Findings
- arch's interval definitions, with θ*_l and θ*_u the α/2 and 1 − α/2 bootstrap quantiles: — [arch confidence-interval docs](https://arch.readthedocs.io/en/latest/bootstrap/confidence-intervals.html)
  - basic = [θ̂ + (θ̂ − θ*_u), θ̂ + (θ̂ − θ*_l)];
  - percentile = [θ*_l, θ*_u];
  - norm = θ̂ ± z·σ̂*;
  - studentized = θ̂ + σ̂ Ĝ^{-1}(q), where the SE comes from `std_err_func` or a nested bootstrap;
  - bc = BCa with acceleration 0, bias-correction b̂ = #{θ*_b < θ̂}/B;
  - bca: the acceleration â is "a skewness-like estimator using a leave-one-out jackknife".
- The VOC bootstrap note uses the percentile interval method (citing Wasserman 2004, §8.3). — [VOC bootstrap note](https://homepages.inf.ed.ac.uk/ckiw/postscript/bootstrap_note.pdf)
- ILSVRC uses the percentile interval, discarding the lower and upper α fraction of bootstrap scores. — [ILSVRC, arXiv:1409.0575](https://arxiv.org/abs/1409.0575)
- Bouthillier et al. (2021) compute CIs for P(A > B) "with the non-parametric percentile bootstrap (Efron, 1982)". — [Bouthillier et al., MLSys 2021, arXiv:2103.03098](https://arxiv.org/abs/2103.03098)
- Lahiri (1999): block-bootstrap estimators share the same first-order bias, so all block methods inherit a block-length-dependent bias in the variance or distribution estimate. — [Lahiri 1999, Project Euclid](https://projecteuclid.org/euclid.aos/1018031117)

### Inferences
- **Why not plain BCa.** The acceleration â is estimated from leave-one-out jackknife influence values. With autocorrelated frames, deleting a single frame barely changes AP because its neighbours carry nearly the same information, so â is mis-estimated.
  - If BCa is wanted, compute â from a **delete-a-block jackknife** (Künsch 1989 introduced the blockwise jackknife): drop each non-overlapping block of length L in turn, compute θ̂_(−k), and use â = Σ(θ̄ − θ̂_(−k))³ / (6 [Σ(θ̄ − θ̂_(−k))²]^{3/2}).
  - With K = 5 to 19 blocks this is very noisy. I would not recommend it as the primary.
- **Percentile vs basic.**
  - Percentile is transformation-respecting and stays inside [0, 1] for AP.
  - Basic corrects for bias in θ̂ but can go outside [0, 1] when AP is skewed.
  - If the two differ materially at some L, report both. Material disagreement signals skewness/bias, which is likely with few blocks.
- **Block-bootstrap-specific issue.** CBB/MBB variance estimates are biased downward for L that is too short relative to the dependence range. That is the whole point of the sensitivity analysis.
- **Recommended table footnote:** "95% percentile intervals from B = 10,000 paired circular-block-bootstrap resamples; basic intervals agree to within X pp (Appendix)."

### Gaps
- I did not retrieve Götze & Künsch (1996, *Ann. Statist.*), which shows second-order correctness of the *studentized* block bootstrap with an appropriate block-variance estimator. The claim is from background knowledge.
- I found no simulation study of BCa versus percentile specifically for block bootstraps at very small numbers of blocks.

---

## 6. Effective sample size under autocorrelation; the "experimental unit" issue in ML evaluation (Bouthillier et al. 2021; Dror et al. 2018); clustered/hierarchical bootstrap across scenes

### Takeaway
182 autocorrelated frames carry far less information than 182 independent images. For a mean-type statistic, n_eff = n / (1 + 2Σ_{k≥1} (1 − k/n) ρ_k), which is about n(1 − ρ)/(1 + ρ) for AR(1). With lag-1 (0.25 s) autocorrelations of 0.8–0.95, typical for consecutive video frames, n_eff is only about 5–20.

The ML-methodology literature names non-independence of test items as a central open problem (Dror et al. 2018) and shows that correlated errors widen the sampling distribution (Bouthillier et al. 2021). The paper must state that **frames are the unit of observation**, **temporal blocks are the unit of resampling**, and the **video/scene is the unit of generalisation**. With one video, the CIs quantify within-video sampling variability only, and say nothing about variation across scenes.

### Cited Findings
- AR(1) effective number of independent observations: n·(1 − ρ̄)/(1 + ρ̄), where ρ̄ is the lag-1 autocorrelation (Chelton 1983, as cited). — [arXiv:2604.14414](https://arxiv.org/abs/2604.14414)
- Worked derivation: T_eff/T = (1 − φ)/(1 + φ); with φ = 0.5 and 1,000 observations, T_eff ≈ 330. — [QuantVault problem](https://quantvault.org/problem-1952-effective-sample-size-under-autocorrelation.html)
- An exact-form AR(1) ESS of {n(1 − ρ) + 2ρ}/(1 + ρ) is given under a Godambe-information definition. — [arXiv:2112.09840](https://arxiv.org/abs/2112.09840)
- Practical estimator ESS = n / (1 + 2Σρ̂_k), with the sum truncated at the first negative autocorrelation (as in MCMC practice). — [PINTS ESS notebook](https://nbviewer.jupyter.org/github/pints-team/pints/blob/v0.1.1/examples/sampling-effective-sample-size.ipynb)
- Bouthillier et al. (2021, MLSys):
  - For classification, if errors are i.i.d. the test accuracy is binomial. "If errors are correlated, not i.i.d., the degrees of freedom are smaller and the distribution is wider."
  - Their recommendations: randomise as many sources of variation as possible (initialisation, data sampling, augmentation, hyperparameter optimisation); account for variance rather than comparing averages; prefer resampling such as out-of-bootstrap over a single fixed held-out test set.
  - Decision rule: A is better than B if P(A > B) is significantly above 0.5 and "meaningful" (threshold γ ≈ 0.75 suggested).
  - — [Bouthillier et al., arXiv:2103.03098](https://arxiv.org/abs/2103.03098)
- Dror, Baumer, Shlomov & Reichart (2018, ACL), "The Hitchhiker's Guide to Testing Statistical Significance in NLP":
  - "while most significance tests assume the test set consists of independent observations, most NLP datasets consist of dependent data points". Examples: sentences from the same article, author or parliamentary session.
  - "This dependence between test examples violates the conditions under which the theoretical guarantees of the various tests were developed."
  - They list it as an open problem.
  - — [Dror et al., ACL 2018](https://aclanthology.org/P18-1128/)
- Hierarchical bootstrap (Saravanan, Berman & Sober 2020, *NBDT*): resampling sequentially over the levels of a hierarchy (e.g., subjects, then trials) "keeps the Type-I error rate within the intended bound and retains more statistical power than summarizing methods". On their real examples it performed better than linear mixed models. Code: soberlab/Hierarchical-Bootstrap-Paper. — [Saravanan et al. 2020, PMC7906290](https://pmc.ncbi.nlm.nih.gov/articles/PMC7906290); [NBDT article](https://nbdt.scholasticahq.com/article/13927-application-of-the-hierarchical-bootstrap-to-multi-level-data-in-neuroscience)
- Resampling schemes for multilevel data "have to take into account the hierarchical structure of data". — [Wild bootstrap for multilevel models, arXiv:1508.05713](https://ar5iv.arxiv.org/html/1508.05713)

### Inferences
- **n_eff for the 182-frame set (computed, AR(1) approximation):** ρ₁ = 0.5 gives about 61; 0.8 gives about 20; 0.9 gives about 9.6; 0.95 gives about 4.7.
  - Estimate ρ̂_k from a per-frame proxy series (the jackknife pseudo-values ψ_t of AP, or per-frame TP/FP counts) and report n̂_eff = n / (1 + 2Σ_{k=1}^{K}(1 − k/n)ρ̂_k), truncated at the first non-positive ρ̂_k.
  - Equivalent design-effect form using the bootstrap itself: deff(L) = (SE_block(L) / SE_iid)², and n_eff(L) = n / deff(L). This ties n_eff directly to the reported sensitivity table.
- **Experimental-unit definitions, suggested wording for the paper:**
  - *Frame* (observation unit): a single annotated image at time t. Detections are matched to GT within a frame. AP pools matched detections across frames. Frames are **not** independent: consecutive frames, 0.25 s apart, share objects, background, illumination and model failure modes.
  - *Temporal segment / block* (resampling unit): a contiguous run of L frames (L·0.25 s). Blocks are treated as approximately exchangeable. The CI is computed by resampling blocks, and sensitivity to L ∈ {10, 20, 40} frames (2.5/5/10 s) is reported.
  - *Track* (tracking unit): the time-ordered set of GT boxes sharing one identity. Tracking metrics (MOTA, IDF1, HOTA) are computed over tracks. For resampling, tracks are cut at block boundaries, and identity-level metrics are bootstrapped conditionally on the full-sequence ID assignment.
  - *Scene / video* (generalisation unit): one recording with fixed camera, site and conditions. **With a single test video, inference is conditional on that scene.** CIs describe uncertainty about performance on frames from this scene, not about performance on new scenes.
  - If a second held-out recording exists (e.g., a separate test-only clip), report it as a separate scene-level replicate, not pooled into the frame bootstrap.
- **Multi-scene extension (two-stage cluster bootstrap):**
  - Stage 1: resample S scenes with replacement.
  - Stage 2, optional: within each chosen scene, block-resample frames.
  - Recompute the pooled metric.
  - Resampling only at the top level is the conservative standard choice for clustered data. Two-stage resampling can overstate variance (cf. Field & Welsh 2007; see Gaps). With S < about 5–10 scenes, cluster bootstrap CIs are unreliable; report per-scene results instead.
- **Training-seed variance (Bouthillier).** Block-bootstrap CIs capture *test-sampling* variance only. If the reviewer or readers might read them as "model-level" uncertainty, add a sentence that training-seed variance is not included, or add seeds if compute allows.

### Gaps
- Field & Welsh (2007), "Bootstrapping clustered data", *JRSS-B*, the standard reference on which level(s) to resample in a cluster bootstrap, was not found in my searches. The statements about top-level resampling being conservative are from background knowledge.
- I found no ML-evaluation paper that defines experimental units specifically for video detection or tracking test sets. The definitions above are a proposal.
- I did not find a published estimate of typical frame-to-frame autocorrelation of detection correctness at 4 fps. The ρ values above are illustrative and should be estimated from the data.

---

## 7. McNemar and Wilcoxon tests with dependent data: caveats

### Takeaway
Both tests assume independent pairs. Applied to autocorrelated frames, they overstate the effective sample size and give anti-conservative p-values.

- **McNemar:** use a clustered McNemar (Obuchowski 1998; Durkalski et al. 2003; Eliasziw & Donner 1991) with temporal blocks as clusters (R `clust.bin.pair`).
- **Wilcoxon:** use the clustered signed-rank test of Rosner, Glynn & Lee (2006) (R `clusrank`), or aggregate to block level first.

With 5–19 blocks, power is low and exact p-values are bounded below by roughly 2/2^K. Neither test applies to AP directly, because AP is not a per-frame quantity.

### Cited Findings
- Durkalski, Palesch, Lipsitz & Rust (2003), "Analysis of clustered matched-pair data", *Statistics in Medicine* 22(15), 2417–2428, DOI 10.1002/sim.1438. — [clust.bin.pair README](https://cran.r-project.org/web/packages/clust.bin.pair/readme/README.html)
- R `clust.bin.pair` implements Eliasziw (1991), Obuchowski (1998), Durkalski (2003) and Yang (2010) adjustments, with McNemar (1947) for comparison. It notes that the standard McNemar test is unsuitable for clustered data. — [clust.bin.pair README](https://cran.r-project.org/web/packages/clust.bin.pair/readme/README.html); [reference manual](https://archive.linux.duke.edu/cran/web/packages/clust.bin.pair/refman/clust.bin.pair.html)
- Rosner, Glynn & Lee (2006), "The Wilcoxon Signed Rank Test for Paired Comparisons of Clustered Data", *Biometrics* 62, 185–192: randomisation at the cluster level, comparisons at the subunit level. The variance-adjusted version assumes a common intracluster correlation and does not handle informative cluster size. Datta & Satten (2008, *Biometrics* 64, 501–507) give an alternative. R `clusrank::clusWilcox.test(paired = TRUE, method = "rgl")`. — [clusrank paper, arXiv:1706.03409](https://arxiv.org/abs/1706.03409); [clusrank manual](https://cran.r-project.org/web/packages/clusrank/refman/clusrank.html)
- Dror et al. (2018) list McNemar and Wilcoxon signed-rank among the tests used in NLP, all of which assume i.i.d. samples. — [Dror et al., ACL 2018](https://aclanthology.org/P18-1128/)

### Inferences
- **Standard McNemar:** χ² = (n₀₁ − n₁₀)² / (n₀₁ + n₁₀) over discordant *objects or frames*. With autocorrelation, the discordances cluster in time: a model that misses an object usually misses it in several consecutive frames. Var(n₀₁ − n₁₀) is then much larger than n₀₁ + n₁₀, so χ² is inflated.
- **Cluster-robust form** (Durkalski-type; formula from background knowledge, verify against the paper). With blocks k = 1..K and discordant counts b_k (A right, B wrong) and c_k (A wrong, B right) per block:
  χ²_c = (Σ_k (b_k − c_k))² / Σ_k (b_k − c_k)², referred to χ²₁.
  This is a cluster-robust (sandwich) variance in place of the binomial one. With K = 5 the χ² approximation is poor, so use the exact block sign-flip test from Section 4 instead.
- **Wilcoxon on per-frame scores** (e.g., per-frame F1 or count error): the signed-rank variance n(n + 1)(2n + 1)/24 assumes independence. Use RGL (2006) with blocks as clusters, or average within blocks and run an exact signed-rank or sign test on the K block means.
  - Minimum exact two-sided p for a signed-rank or sign test on K block-level differences is 2/2^K: 0.125 (K = 4), 0.0625 (K = 5), 0.0039 (K = 9), 0.0020 (K = 10), 7.6·10⁻⁶ (K = 18) (computed).
  - With 10 s blocks, no rank test on this clip can reach p < 0.05 two-sided. That is a useful and honest statement for the rebuttal.
- **Applicability to AP.** McNemar and Wilcoxon need per-unit paired outcomes. AP is a pooled ranking metric, so for AP comparisons use the paired block bootstrap or the block randomisation test (Sections 3 and 4). Use McNemar/Wilcoxon only for decomposable secondary metrics such as per-object detection at a fixed operating threshold or per-frame counting error.
- **Choice of block length for these tests.** Use the same L as in the bootstrap sensitivity analysis, and report results for all three L.

### Gaps
- I could not verify the exact Durkalski (2003) and Obuchowski (1998) statistic formulas from primary text. The form above should be checked against the paper or the `clust.bin.pair` source.
- I found no literature applying clustered McNemar or Wilcoxon tests to temporal blocks of video frames specifically. Using contiguous time blocks as "clusters" is an inference. Adjacent blocks are still mildly dependent at their boundaries, which these tests ignore.
