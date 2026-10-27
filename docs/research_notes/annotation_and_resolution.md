# Human ground-truth annotation protocols for pedestrian detection/MOT evaluation, and resolution-controlled comparison of YOLO vs RF-DETR

Scope note: Primary sources were read in full text where possible (MOT16, MOT20, CityPersons, CrowdHuman, Caltech PAMI 2012, HOTA, RF-DETR ICLR 2026 paper, Extreme Clicking, Murrugarra-Llerena 2022, Krippendorff-alpha-for-CV 2019). Items marked "(snippet only)" were seen only through search-result abstracts or snippets and should be checked before being cited verbatim.

---

## A1. How MOTChallenge, CrowdHuman, CityPersons and Caltech define annotation rules (ignore regions, min size, visibility, full vs visible boxes, "reasonable" subsets)

### Takeaway
All four benchmarks annotate the **full-extent** box, estimating the occluded parts, plus a visibility measure (a visibility ratio in MOT, or a separate visible box in Caltech, CityPersons and CrowdHuman). All four also use **ignore** classes or regions. Detections that match an ignore box are neither rewarded nor penalized. The pedestrian-detection "Reasonable" setting is height ≥ 50 px with ≤ 35% occlusion, and the log-average miss rate (MR⁻²) is computed over 10⁻² to 10⁰ FPPI. MOT keeps one ID only while the path can be determined unambiguously. After a long ambiguous occlusion, or after leaving the field of view, the person gets a new ID.

### Cited Findings

**MOT16/MOT17 (Milan, Leal-Taixé, Reid, Roth, Schindler 2016)**
- Classes fall into three groups: (i) moving or standing upright pedestrians, which are the targets (cyclists and skaters are included in the target or "person on vehicle" handling); (ii) ambiguous classes: static people who are not upright (sitting, lying), and "distractors" such as mannequins, posters, statues, reflections and people behind glass; (iii) vehicles and occluders, which are annotated for training and occlusion computation only. "The idea is to use these annotations in the evaluation such that an algorithm is neither penalized nor rewarded for tracking, e.g., a sitting person or a reflection." — [MOT16, arXiv:1603.00831](https://arxiv.org/abs/1603.00831)
- Box rule: the box "should contain all pixels belonging to that object and at the same time be as tight as possible". For partially occluded persons, "the extent is estimated based on other available information such as expected size, shadows, reflections, previous and future frames". Persons cropped by the image border are boxed beyond the frame. — [MOT16](https://arxiv.org/abs/1603.00831)
- Track start and end: a track starts "as soon as the person's location and extent can be determined precisely. This is typically the case when ≈10% of the person becomes visible", and ends when this is no longer possible. — [MOT16](https://arxiv.org/abs/1603.00831)
- Identity: "Keep ID as long as the person is inside the field of view and its path can be determined unambiguously." "Should a person leave the field of view and appear at a later point, they will be assigned a new ID." If an occlusion is very long and the path cannot be determined by "simple reasoning (e.g. constant velocity assumption)", the person gets a new ID on reappearance. A target that is fully occluded and never reappears is terminated ("outside of view"). — [MOT16 Table 1, §2.3, §2.5](https://arxiv.org/abs/1603.00831)
- Minimal size: "annotations will contain all objects of all sizes as long as they are distinguishable by the annotator". Evaluation applies a minimum height. — [MOT16 §2.4](https://arxiv.org/abs/1603.00831)
- Occlusion level is not annotated by hand. It is "computed automatically using the ground plane assumption and the annotations". The GT file carries a visibility ratio in [0,1] that reflects both occlusion and border cropping. The class IDs are 7 static person, 8 distractor, 9 occluder, 10 occluder on ground, 11 occluder full, 12 reflection. — [MOT16 §2.5, Tables 5–6](https://arxiv.org/abs/1603.00831)
- QA: after annotation, a "sanity check" ran a pedestrian detector on all videos and "added all high-confidence detections that corresponded to either humans or distractors to the annotation list". This is a detector-assisted completeness pass. The paper also says annotations were "double-checked to ensure highest annotation accuracy". — [MOT16 §2.6, §1](https://arxiv.org/abs/1603.00831)
- Evaluation handling of ambiguous classes: (1) per frame, Hungarian-match all result boxes to GT; (2) remove result boxes that overlap > 50% with distractor, static person, reflection or person-on-vehicle boxes; (3) evaluate only on boxes annotated as pedestrians. The MOTP/MOTA match threshold is t_d = 50% IoU. MT means tracked for ≥ 80% of the lifespan and ML means < 20%. — [MOT16 §4.1](https://arxiv.org/abs/1603.00831)
- The MOT16 sequence table tags each sequence by camera motion, viewpoint (low/medium/high) and conditions (e.g., cloudy, night, indoor, sunny). This is a ready-made template for stratifying by viewpoint and illumination. — [MOT16 Table 2](https://arxiv.org/abs/1603.00831)

**MOT20 (Dendorfer et al. 2020)**
- MOT20 follows the MOT16 protocol. It has 8 very crowded sequences with density up to 246 pedestrians per frame. Training totals 8,931 frames, 2,215 tracks and 1,134,614 boxes, with a mean density of 127.04 per frame. Test totals 4,479 frames, 1,242 tracks, 517,426 boxes and 115.52 per frame. All annotations total 2,102,385, of which 1,652,040 are pedestrian boxes and 335,891 are static-person boxes. Static persons and non-motorized vehicles "are filter[ed] out during evaluation". Occluders and "crowd" regions are annotated. — [MOT20, arXiv:2003.09003](https://arxiv.org/abs/2003.09003)

**CityPersons (Zhang, Benenson, Schiele CVPR 2017)**
- For pedestrians and riders, the full body is annotated "by drawing a line from the top of the head to the middle of two feet, and the bounding box is generated using a fixed aspect ratio (0.41)". The visible box is "the tightest one fully covering the segment mask". The occlusion ratio uses area(BB-vis)/area(BB-full). Sitting and "other" persons get only the segment (visible) box. — [CityPersons, arXiv:1702.05693](https://arxiv.org/abs/1702.05693)
- Ignore regions: annotators search the whole image for "fake humans, for instance, people on posters, statue, mannequin, people's reflection in mirror or window" and mark them as ignore. Ignore regions also cover areas "where the annotator cannot tell if a person is present or absent, and person groups where individuals cannot be told apart". The dataset has 5,000 images, 35,016 persons and 13,172 ignore regions, about 7 persons per image. — [CityPersons](https://arxiv.org/abs/1702.05693)
- Evaluation follows Caltech. MR is the log-average miss rate on the "reasonable" setup: scale [50, ∞] px and occlusion ratio [0, 0.35]. "Cyclists/sitting persons/other persons/ignore regions are not considered, which means detections matching with those areas are not counted as mistakes." — [CityPersons §3.3](https://arxiv.org/abs/1702.05693)
- Other papers often give the "Heavy occlusion" subset as visibility 0.2–0.65, but definitions vary between papers. The CityPersons PDF text I read only says "Heavy occl. [50, ∞]". — (snippet only) [pedestrian-detection papers via search](https://arxiv.org/pdf/2001.09252)

**CrowdHuman (Shao et al. 2018)**
- Every person gets three boxes: a full body box (annotated exhaustively, "the annotator is required to complete the invisible part"), a visible-region box, and a head box. Unlike CityPersons, the full box is not forced to a 0.41 aspect ratio. Human-like objects such as statues get a specific label and "will be ignored during evaluation". "All the annotations are double-checked by at least one different annotator." — [CrowdHuman, arXiv:1805.00123](https://arxiv.org/abs/1805.00123)
- Training statistics: 15,000 images, 339,565 persons, 99,227 ignore regions, 22.64 persons per image. Train and val together hold about 470K human instances. Occlusion subsets are "bare" (≤ 30% occluded), "partial" (30–70%) and "heavy" (> 70%). Bare and partial make up 29.89% and 32.13% of CrowdHuman, against 46.79% and 24.19% of CityPersons. There are on average 9 person pairs per image with IoU > 0.3 (fewer than 1 in CityPersons) and 2.4 pairs with IoU > 0.5. — [CrowdHuman](https://arxiv.org/abs/1805.00123)
- Metric: mMR, the Caltech log-average miss rate over FPPI in [10⁻², 10⁰], plus AP and recall. Ignored regions are not counted. The authors note that full-body GT "might be suffered from high variance caused by different decision-makings by different annotators". — [CrowdHuman §4.2](https://arxiv.org/abs/1805.00123)

**Caltech Pedestrians (Dollár, Wojek, Schiele, Perona, PAMI 2012)**
- About 10 h of 30 Hz video were collected. Roughly 250k frames from about 137 minutes were annotated, giving about 350k boxes and about 2,300 unique pedestrians. Annotation used keyframes with **cubic interpolation** that is re-run whenever a box is edited. "Labeling the ∼2.3 hours of video, including verification, took ∼400 hours total (spread across multiple annotators)." — [Dollár et al. PAMI 2012](https://pdollar.github.io/files/papers/DollarPAMI12peds.pdf)
- Each pedestrian gets BB-full (estimating hidden parts) and BB-vis. Labels are "Person" (~1900), "People" for groups that are tedious or impossible to separate (~300), and "Person?" for ambiguous cases (~110). — [Caltech PAMI](https://pdollar.github.io/files/papers/DollarPAMI12peds.pdf)
- Scales are near (≥ 80 px), medium (30–80 px; about 69% of pedestrians) and far (≤ 30 px). Occlusion levels are partial (1–35% area occluded) and heavy (35–80%). Over 80% usually means full occlusion. The log-average aspect ratio is w ≈ 0.41h. — [Caltech PAMI §2.2](https://pdollar.github.io/files/papers/DollarPAMI12peds.pdf)
- Ignore semantics: ignored GT "need not be matched, however, matches are not considered mistakes either". A detection may match any subregion of an ignore box, using overlap area(dt∩ig)/area(dt). GT matches take priority over ignore matches. "Setting a BBgt to ignore is not the same as discarding it", because discarding would turn detections in those regions into FPs. Four types are always ignored: boxes under 20 px high, boxes truncated by the image boundary, 'Person?' and 'People'. Evaluation uses "expanded filtering" with r = 1.25 for scale ranges and standardizes all boxes to aspect ratio 0.41. The summary metric is the log-average miss rate over 9 FPPI points evenly spaced in log space in [10⁻², 10⁰]. — [Caltech PAMI §3](https://pdollar.github.io/files/papers/DollarPAMI12peds.pdf)

### Inferences
- A human-annotated eval subset for this project can take its rules directly from these sources:
  - Full-extent boxes plus a per-box visibility class (MOT-style ratio, or Caltech bins 0 / 1–35 / 35–80 / >80%).
  - Explicit "ignore" labels for groups, reflections, posters, static seated people and sub-20 px people, scored with Caltech/MOT semantics (detections there are neither TP nor FP).
  - A "reasonable" stratum of h ≥ 50 px and occlusion ≤ 35%.
  - Size strata using Caltech near/medium/far (≥ 80 / 30–80 / < 30 px) or COCO small/medium/large.
  - Density strata using persons per frame. MOT20 reaches 127–246 per frame; CrowdHuman reports 9 pairs per image at IoU > 0.3.
  - Illumination and viewpoint tags per sequence, as in the MOT16 sequence table.
- Whether boxes are full or visible matters for agreement with the pseudo-labels. If the teacher ensemble predicts visible boxes but humans draw full boxes, IoU-based agreement will be penalized systematically for occluded people. The protocol must state which box type is evaluated.

### Gaps
- The CityPersons "Heavy occlusion" and "Small" subset thresholds were not confirmed from the primary PDF text, and secondary papers give inconsistent values.
- None of the four dataset papers reports quantitative inter-annotator agreement numbers. They describe double-checking only.

---

## A2. Inter-annotator agreement (IAA) measures for boxes and tracks, and typical human agreement levels

### Takeaway
Human box agreement on clean, large objects is about **0.88 mean IoU** (PASCAL experts) to **0.93 mean IoU** (TAO track re-annotation). Agreement falls sharply for small and occluded objects. Krippendorff's α has been extended to boxes, with α ≥ 0.667 as the usual "acceptable" level and ≥ 0.8 as "good". For tracks, HOTA's symmetric Jaccard formulation makes it suitable for annotator-vs-annotator comparison, but I found no published HOTA/IDF1 inter-annotator figures for pedestrian MOT.

### Cited Findings
- **PASCAL expert agreement:** comparing the two independent expert box sets of VOC 2007 (GT Boxes vs boxes from the GT segmentations) gives **88% mIoU**, and **93%** of GT Boxes have IoU > 0.7 with their counterpart. Perturbing every coordinate of the GT boxes by **4 px** also gives 88% mIoU. Extreme-click boxes reach the same quality (88 mIoU, 92–94% at IoU > 0.7, 97–98% at IoU > 0.5) at 7.0 s per instance, against 34.5 s for conventional drawing. — [Papadopoulos et al., Extreme Clicking, ICCV 2017, arXiv:1708.02750](https://arxiv.org/abs/1708.02750)
- **Tracking re-annotation (TAO):** 177 re-annotated tracks had an average IoU of **0.93** with the originals, and **93%** of class labels matched. — (snippet only) [TAO ECCV 2020 supplement](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123500426-supp.pdf)
- **Krippendorff's α for CV annotations:** α = 1 − D_o/D_e, extended to boxes (computed at pixel level: each pixel is labeled or unlabeled per class channel). The authors chose it because it "handles multiple raters, missed bounding boxes, and box size variation". The benchmark is "good IAA using K-α is 0.67, with 0.8 and above being very good" (Hayes & Krippendorff 2007). Reported values: VIRAT α = 0.881, Stanford Drone α = 0.575. Per-class person α varied between 0.463 and 0.817 depending on annotator group. The paper also defines **Rater Vitality** V_i = K_α − K_α(without i) to flag outlier annotators. — [Nassar, Pavon-Harr, Bosch, McCulloh (Accenture Federal Services), "Assessing Data Quality of Annotations With Krippendorff's Alpha For Applications in Computer Vision", arXiv:1912.10107](https://arxiv.org/abs/1912.10107)
- **Tooling and newer work:** the `iaa-od` PyPI package implements Krippendorff's α for object detection ([PyPI iaa-od](https://pypi.org/project/iaa-od/)). KαLOS (Tschirschwitz & Rodehorst, arXiv:2603.27197, CVPR 2026 poster) proposes a meta-algorithm for IAA in localization-plus-classification tasks. — (snippet only) [arXiv:2603.27197](https://arxiv.org/abs/2603.27197)
- **Label convergence:** with contradictory test annotations, the achievable ceiling on LVIS is estimated at **62.63–67.52 mAP@[.5:.95]** (95% confidence). The authors recommend multi-annotated test data. — [Tschirschwitz & Rodehorst, Label Convergence, WACV 2025, arXiv:2409.09412](https://arxiv.org/abs/2409.09412)
- **HOTA as an agreement metric:**
  - Definition: HOTA_α = √(DetA_α · AssA_α), evaluated at 19 localization thresholds α ∈ {0.05, 0.10, …, 0.95} and averaged. DetA and AssA are Jaccard-type scores, and HOTA decomposes into DetRe/DetPr/AssRe/AssPr and LocA.
  - Symmetry: the paper discusses symmetry when predictions and GT are swapped and notes that the MODA/MOTA formulation is **not symmetric**.
  - Human alignment: HOTA "better align[s] with human visual evaluation of tracking performance" (user study).
  - [Luiten et al., HOTA, IJCV 2021, arXiv:2009.07736](https://arxiv.org/abs/2009.07736)
- Annotator variance for full-body boxes of occluded people is explicitly acknowledged by CrowdHuman ("high variance caused by different decision-makings by different annotators"). — [CrowdHuman](https://arxiv.org/abs/1805.00123)
- Murrugarra-Llerena et al. show that IoU between two plausible box sets for the same objects falls monotonically with object size (Spearman ρ = 0.79 between IoU and the smallest box dimension on VOC). — [CVPRW 2022](https://openaccess.thecvf.com/content/CVPR2022W/VDU/papers/Murrugarra-Llerena_Can_We_Trust_Bounding_Box_Annotations_for_Object_Detection_CVPRW_2022_paper.pdf)

### Inferences
- A practical IAA protocol for two annotators plus adjudication would report the following, stratified by size, occlusion, density, illumination and viewpoint:
  1. **Matched-F1 between annotators.** Hungarian-match A vs B at IoU ≥ 0.5, then F1 = 2·|matches|/(|A|+|B|). This is symmetric and equals the detection F1 of B scored against A.
  2. **Mean IoU of matched pairs.** Compare with the 0.88 PASCAL and 0.93 TAO reference points.
  3. **Krippendorff's α**, via iaa-od, with ≥ 0.667 acceptable and ≥ 0.8 good.
  4. **For tracks: HOTA, DetA, AssA and IDF1 of annotator B against annotator A**, plus the reverse. HOTA should be roughly symmetric; IDF1 is symmetric by construction as an F1 over ID-matched detections.
- These human-vs-human numbers set a "noise ceiling". Detector-vs-human differences smaller than the human-vs-human gap, especially at AP75 and above, or for small or heavily occluded strata, should not be over-interpreted.
- Because small objects lose IoU fastest, use IoU 0.5 as the primary threshold for small pedestrians (or report AP50 alongside AP50:95). Caltech and CityPersons both use 0.5.

### Gaps
- I found no paper reporting **HOTA or IDF1 between two human annotators** on pedestrian MOT. TAO reports IoU agreement only. This would be a novel contribution and should be framed as such.
- Note: arXiv:1912.10107 is by Nassar et al. (Accenture), not Tschirschwitz. The later Tschirschwitz work (Label Convergence, KαLOS) builds on the box-level K-α idea. Its K-α is computed at pixel level, not per matched box.
- No published "typical" Krippendorff α for crowded pedestrian video was found.

---

## A3. Sample-size guidance (frames, boxes, CIs) and stratified sampling

### Takeaway
No detection-specific minimum test-set size standard was found. Established practice is an **image-level, or for video sequence/block-level, bootstrap** for AP CIs (ILSVRC used up to 20,000 bootstrap rounds). Binomial (Wilson) CIs give a useful planning rule for recall, precision and F1 per stratum.

### Cited Findings
- ILSVRC computes detection-AP confidence intervals by bootstrapping: sample N test images with replacement, re-evaluate, discard the lower and upper α fraction, and run up to 20,000 bootstrap rounds (following the PASCAL VOC practice). — (snippet only) [Russakovsky et al., ILSVRC, arXiv:1409.0575](https://arxiv.org/abs/1409.0575)
- A medical detection study reports 95% BCa bootstrap CIs for mAP with 1,000 iterations. — (snippet only) [AI-Dentify, arXiv:2310.00354](https://arxiv.org/abs/2310.00354)
- Bootstrap-tool guidance: overlapping CIs do not rule out a significant difference, so use a **paired** bootstrap on the difference between two models. mAP is not a per-sample mean and needs whole-set re-evaluation per resample. — (snippet only) [bstrap on PyPI](https://pypi.org/project/bstrap/)
- MOT benchmarks are small in sequences but large in boxes. MOT20 train has 4 sequences, 8,931 frames and 1.13M boxes. Boxes inside a sequence are highly correlated, so the effective sample size is closer to the number of sequences or time blocks than the number of boxes. — [MOT20](https://arxiv.org/abs/2003.09003)

### Inferences
These are my own planning arithmetic and are not sourced.
- **Wilson/normal approximation** for a proportion p (recall or precision) estimated from n independent boxes: half-width ≈ 1.96·√(p(1−p)/n).
  - At p = 0.90: n = 100 gives ±5.9 pp, n = 400 gives ±2.9 pp, n = 1,000 gives ±1.9 pp.
  - At p = 0.70: n = 400 gives ±4.5 pp.
  - So **about 300–500 independent boxes per stratum** gives roughly ±3–4 pp. With 5 stratification factors and 2–3 levels each, you need several thousand boxes overall, unless the strata are reported marginally (one factor at a time) rather than as a full cross.
- Boxes in the same frame, and frames in the same second, are correlated. Inflate n by the design effect, or sample **sparse frames** (e.g., 1 frame per 2–5 s across many clips) to make boxes closer to independent. Use a **cluster bootstrap over frames, or a block bootstrap over clips**, for CIs.
- Stratified sampling: choose frames to fill strata quotas (small or occluded or dense or night or high-angle), then report per-stratum metrics with CIs. Report overall metrics using **post-stratification weights** that restore the natural distribution, or state clearly that the overall number is on a stratified (non-representative) sample.
- For tracking, CIs should be computed at the track or sequence level. A multi-minute subset of 3–5 clips gives very wide CIs on IDF1/HOTA. Report per-clip values and a range rather than a single pooled number.

### Gaps
- I found no authoritative paper prescribing a minimum number of frames or boxes for pedestrian detection or MOT evaluation, and no comparison of image-level versus box-level resampling for detection AP.

---

## A4. Annotation tooling, semi-automatic correction of pseudo-labels, anchoring bias, and keeping evaluation independent of teacher labels

### Takeaway
CVAT (track mode with keyframes, interpolation, and Outside/Occluded flags) and Label Studio (prediction import for pre-annotations, video rectangle sequences) both support **pre-annotation and correction**. That workflow is fast but vulnerable to **anchoring**: editors under-correct machine output, which inflates measured model performance. Even MOT16 used a detector pass, but only as a post-hoc completeness check.

### Cited Findings
- **CVAT track mode:** draw boxes on keyframes, and intermediate frames are interpolated automatically. The docs give the example "it isn't necessary to change the rectangle on each frame, simply update several keyframes". Mark **Outside** when the object leaves the view or becomes too small. For objects that vanish and return, create separate tracks and **Merge** them. Occluded is a separate per-frame attribute. Shortcut keys differ between doc mirrors (O/Q). — [CVAT docs, track mode](https://docs.cvat.ai/docs/annotation/manual-annotation/modes/track-mode-basics/); [LearnOpenCV CVAT guide](https://learnopencv.com/a-closer-look-at-cvat-perfecting-your-annotations/)
- **Label Studio:** pre-annotations are imported as `predictions` in Label Studio JSON and must match the labeling config. Video tracks use `VideoRectangle` with a per-keyframe `sequence` (frame, time, x, y, width, height). Interactive pre-annotation goes through an ML backend. — [Label Studio docs: import pre-annotations](https://docs.heartex.com/guide/predictions); (snippet only) [community thread on video pre-annotations](https://community.labelstud.io/t/importing-video-pre-annotations/251)
- **Interpolation precedent:** Caltech used cubic interpolation between user keyframes. The authors tried "more sophisticated interpolation schemes, including relying on tracking; however, cubic interpolation proved best". The effort was about 400 annotator-hours for about 2.3 h of video, including verification. — [Caltech PAMI 2012](https://pdollar.github.io/files/papers/DollarPAMI12peds.pdf)
- **Detector-assisted completeness:** MOT16 ran a pedestrian detector after manual annotation and added high-confidence detections of humans or distractors that had been missed. — [MOT16 §2.6](https://arxiv.org/abs/1603.00831)
- **Anchoring evidence (NLP):** in treebank creation by editing parser output, Berzak et al. find "a clear anchoring effect", leading to **overestimated parser performance** and lower quality than human-only annotation. — [Berzak et al., EMNLP 2016, arXiv:1605.04481](https://arxiv.org/abs/1605.04481)
- **Bias-in-the-loop evidence:** a study with 2,784 crowdworkers found that when corrections were required, annotators were **less likely to revise pre-annotations** (added effort reduced corrections). More automation-skeptical annotators were more accurate. Note: text extraction task, not boxes. — (snippet only) [arXiv:2509.08514](https://arxiv.org/abs/2509.08514)
- Iterative box annotation via detector proposals plus human correction reduces workload, because most proposals need no action. That paper does not measure bias. — (snippet only) [arXiv:2007.00961](https://arxiv.org/abs/2007.00961)

### Inferences
Ways to keep the human set independent of the teacher ensemble:
1. **From-scratch annotation of the eval subset.** No teacher boxes are shown, and annotators may use interpolation and tracking tools that are not seeded by the teacher.
2. If pre-annotation is unavoidable for cost reasons, run a **controlled audit**. Annotate a random 10–20% of frames from scratch by a third annotator and compare the "pre-annotated + corrected" labels against these. Report the agreement difference to quantify anchoring.
3. **Blind** annotators to which model produced any proposals. Seed proposals from a model that is **not** in the teacher ensemble (e.g., a different architecture).
4. Use an MOT16-style detector pass only **after** manual annotation, as a recall check, and have humans adjudicate every added box.
5. Two independent annotators, then adjudication by a third (or by consensus) on disagreements only, i.e., unmatched boxes and matched pairs with IoU < 0.5–0.7. Report IAA **before** adjudication.

### Gaps
- I found no image or box-specific controlled study quantifying anchoring bias of pre-annotated bounding boxes, such as the IoU shift toward model boxes. The evidence comes from NLP and text extraction.

---

## A5. How papers report pseudo-label/annotation vs human GT discrepancies and label-noise effects on mAP

### Takeaway
Small box discrepancies, often under 1 px for COCO and within ±16 px for VOC, change AP materially, especially AP75, AP50:95 and small objects. Recent re-annotation work finds missing objects are the dominant real error (+40% objects on COCO, +60% on KITTI). Absolute mAP is very sensitive to the GT, while **model rankings are largely stable**. This supports reporting both absolute metrics and rank agreement between pseudo-label GT and human GT.

### Cited Findings
- **Murrugarra-Llerena, Kirsten, Jung, "Can We Trust Bounding Box Annotations for Object Detection?" (CVPRW 2022):**
  - IoU is "strongly affected even by small annotation discrepancies", so AP is affected "particularly for small objects and tighter IoU thresholds".
  - On VOC 2012, 6,909 annotated-vs-segmentation-derived box pairs were matched (Hungarian, −IoU cost). About 95% of coordinate offsets fall within [−16, 16]² px, and some reach hundreds of px. IoU correlates with the smallest box dimension at Spearman 0.79.
  - EfficientDet-D0 and SSD300 trained on VOC 2007 lose about 2% AP50, 4.5% and 6.3% AP75, and **7.6% and 9.3% AP50:95** when evaluated on the alternative (mask-derived) GT instead of the annotated boxes. The authors compare this to the 1.6-point AP75 gain claimed by the DIoU loss.
  - An ideal-detector simulation (Table 2) gives a 4% AP50 reduction and over 22% COCO-AP degradation from annotation discrepancy alone.
  - On COCO, sub-pixel discrepancies changed the ranking of detectors for small objects: YOLOR-W6 was best on small-object AP50:95 with mask-derived boxes but third with annotated boxes.
  - [CVF open access PDF](https://openaccess.thecvf.com/content/CVPR2022W/VDU/papers/Murrugarra-Llerena_Can_We_Trust_Bounding_Box_Annotations_for_Object_Detection_CVPRW_2022_paper.pdf)
- **Penquitt et al., "Object Detection Benchmarks are Incomplete: The Role of Label Errors and Annotation Uncertainty" (arXiv:2609.21822, Sep 2026):** re-annotating COCO, VOC, Cityscapes and KITTI with soft labels from at least 11 annotators per object found up to **+60% more objects on KITTI and +40% on COCO**, mostly small, occluded or densely packed. Missing annotations were the main error type. "Benchmark performance is highly sensitive to annotation quality, yet model rankings remain largely stable." Label-error detectors tuned on synthetic noise struggle on real errors. — (abstract only) [arXiv:2609.21822](https://arxiv.org/abs/2609.21822)
- **Label convergence:** the LVIS ceiling is 62.63–67.52 mAP under contradictory annotations, and the authors argue SOTA models sit near the ceiling. — [arXiv:2409.09412](https://arxiv.org/abs/2409.09412)
- **Krippendorff-α paper:** RetinaNet trained on annotations from different annotator groups and evaluated against a custom GT showed precision, recall and F1 varying by several points depending on which annotator's labels were dropped (e.g., F1 0.59–0.65 across leave-one-annotator-out runs). — [arXiv:1912.10107](https://arxiv.org/abs/1912.10107)

### Inferences
Reporting template for the revision:
1. **Teacher pseudo-labels vs human GT** on the human subset: matched-F1, mean IoU, and per-stratum recall of the teacher, i.e., which pedestrians the teacher misses. Penquitt et al. suggest small, occluded and dense ones.
2. **Student metrics computed against both GTs**, side by side: AP50, AP50:95, MOTA, IDF1, HOTA.
3. **Rank agreement** (Kendall τ or Spearman) across all compared models or configs under the two GTs.
4. The **human-vs-human ceiling** from A2.
- If pseudo-label-based and human-based rankings agree, the paper's relative conclusions stand even if absolute numbers shift. This matches the Penquitt et al. finding.

### Gaps
- Full-text numbers from Penquitt et al. 2026 (mAP deltas, rank correlations) were not available, only the abstract.
- No paper was found that specifically evaluates **teacher-ensemble pseudo-labels as test GT** against human GT for pedestrian detection or MOT.

---

## A6. Counting-line and occupancy evaluation against annotated events

### Takeaway
No standard temporal-tolerance definition for matching predicted line-crossing events to annotated events was found. Established practice is either (a) **cumulative count errors at segment boundaries** (AI City Challenge nwRMSE), or (b) **per-interval count comparisons** against manual video counts. Practitioners warn that over-counts and under-counts can cancel in totals. Event-level precision/recall with a ±τ matching window has to be defined by the authors.

### Cited Findings
- **AI City Challenge (vehicle counting by movement), 2020:** each video is split into k segments, and predicted and true **cumulative** counts are compared at segment boundaries. The weighted RMSE is normalized by the true count per vehicle type and movement, weighted by vehicle counts, and set to 0 if it exceeds the true count. Later segments get more weight so early labeling discrepancies matter less. The 2021 edition added an efficiency term and edge-device online constraints. — (snippet only) [4th AI City Challenge, arXiv:2004.14619](https://arxiv.org/abs/2004.14619); [5th AI City Challenge, arXiv:2104.12233](https://arxiv.org/abs/2104.12233)
- **Robust crossing rule:** instead of two consecutive positions, a crossing is triggered when most position pairs **one second apart** on the same trajectory fall on opposite sides of the line. The naive two-point check is "highly sensitive to measurement noise". — (snippet only) [TU/e high-fidelity pedestrian tracking benchmark, arXiv:2108.11719](https://arxiv.org/abs/2108.11719)
- **Count cancellation warning:** "undercounting and overcounting could be occurring that cancel each other out, resulting in an automated count that matches the ground truth count" (NCHRP comparison against manual video ground truth at hourly resolution). — (snippet only) [NCHRP report, nap.edu](https://www.nap.edu/read/24732/chapter/5)
- Sárándi (RWTH Aachen thesis) proposes generalizations of precision and recall to continuous, instantaneous pedestrian-flow estimates. — (snippet only) [RWTH VCI publication 00168](https://www.vision.rwth-aachen.de/publication/00168/)
- Vendor practice: crossing-line accuracy = (GT count − (missed + over-counted)) / GT count. — (snippet only) [Swarm Analytics docs](https://docs.swarm-analytics.com/test-and-performance-measurements/how-many-streams-can-my-perception-box-compute/how-do-we-measure-performance)

### Inferences
Proposed protocol for the multi-minute manually annotated subset:
1. Annotators mark **each crossing event** (frame index, direction, and the track ID if available) from the human tracks, or by direct event annotation in a second pass.
2. Match predicted to GT events with the **same direction** whose times differ by ≤ τ, using one-to-one bipartite matching. Report precision, recall and F1 at τ ∈ {0.5 s, 1 s, 2 s}, so tolerance sensitivity is visible.
3. Also report **per-interval signed and absolute count error** (e.g., per 30 s or 60 s bins). This exposes the cancellation the NCHRP report warns about.
4. **Occupancy** (people in ROI): per-frame or per-second MAE, RMSE and bias against human counts, stratified by density.
5. Report re-entry and occlusion-specific identity errors separately: double counts caused by ID switches near the line, which IDF1 and AssA capture indirectly.

### Gaps
- No peer-reviewed standard for event-level temporal tolerance in pedestrian line counting was found. Full texts of the Sárándi thesis and arXiv:2108.11719 were not read.

---

## B1. RF-DETR supported resolutions, published AP/latency, YOLO AP vs imgsz, and how papers do resolution- or compute-matched comparisons

### Takeaway
RF-DETR-Nano's **native COCO resolution is 384 px**, not 640: patch 16, 2 windows, so valid resolutions are multiples of 32. It scores 48.4 AP, 31.9 GFLOPs and 2.3 ms on a T4. YOLO26n is 40.9 AP at 640 px with 5.5 GFLOPs and 1.7 ms. Because FLOPs scale roughly with pixel count, **YOLO26n at 1280 px (~22 GFLOPs, extrapolated) and RF-DETR-N at 640 px (~89 GFLOPs, extrapolated) are not compute-matched**. A fair sweep should report AP, GFLOPs and measured latency at a common set of input sizes, with all numbers from one harness. The real-time DETR papers (RT-DETR, D-FINE, DEIM) compare everything at 640×640 on T4 TensorRT FP16. RF-DETR instead treats resolution as a NAS knob and compares on **latency-matched** Pareto curves.

### Cited Findings

**RF-DETR configuration and resolution rules**
- Pareto-optimal COCO detection configs (resolution / patch / windows / decoder layers / backbone):
  - N: 384 / 16 / 2 / 2 / DINOv2-S
  - S: 512 / 16 / 2 / 3
  - M: 576 / 16 / 2 / 4
  - L: 704 / 16 / 2 / 4
  - XL: 700 / 20 / 1 / 5 / DINOv2-B
  - 2XL: 880 / 20 / 2 / 5
  - Max: 828 / 12 / 1 / 6
  - [Robinson et al., RF-DETR, ICLR 2026, arXiv:2511.09554, Table 7](https://arxiv.org/abs/2511.09554)
- NAS grid. Training resolutions were 320, 384, 448, …, 960 in steps of 64, with patch sizes 8–32 and windows 1/2/4. The same resolution list was used at inference. A total of 6,468 configurations were evaluated, at about 10,000 T4 GPU-hours. Positional embeddings are pre-allocated for "the largest image resolution divided by the smallest patch size" and interpolated for smaller ones. Multi-scale training uses 0.5–1.5× scale. "Higher resolution improves small object detection performance, while lower resolution improves runtime." — [RF-DETR paper §3, App.](https://arxiv.org/abs/2511.09554)
- Resolution rule (docs): "`resolution` must be a positive integer divisible by `patch_size × num_windows` for the selected variant". "Current detection checkpoints use a block size of 32". Non-divisible values raise `ValueError`. — [RF-DETR FAQ](https://rfdetr.roboflow.com/latest/faq/). A third-party HF demo says "divisible by 32 and 56", which matches older patch-14 configs. That is unverified. — [rf-detr-sahi demo](https://dmorawiec-rf-detr-sahi.hf.space)

**Published COCO numbers (T4, TensorRT FP16, batch size 1)**

RF-DETR models, from the GitHub README and benchmarks page:

| Model | Resolution | AP50:95 | AP50 | Latency (ms) | Params (M) |
|---|---|---|---|---|---|
| N | 384 | 48.4 | 67.6 | 2.3 | 30.5 |
| S | 512 | 53.0 | 72.1 | 3.5 | 32.1 |
| M | 576 | 54.7 | 73.6 | 4.4 | 33.7 |
| L | 704 | 56.5 | 75.1 | 6.8 | 33.9 |
| XL | 700 | 58.6 | 77.4 | 11.5 | 126.4 |
| 2XL | 880 | 60.1 | 78.5 | 17.2 | 126.9 |

The README also lists Atto, Femto and Pico models under the PML licence: 30.5, 37.8 and 41.6 AP at 380, 384 and 560 px. — [rf-detr GitHub](https://github.com/roboflow/rf-detr)

Paper Table 2 (COCO) adds GFLOPs and per-size AP:

| Model | Params | GFLOPs | Latency (ms) | AP | AP_S | AP_M | AP_L |
|---|---|---|---|---|---|---|---|
| RF-DETR-N | 30.5M | 31.9 | 2.3 | 48.0 | 25.2 | 53.5 | 70.0 |
| RF-DETR-S | — | 59.8 | 3.5 | 52.9 | 32.0 | — | — |
| RF-DETR-M | — | 78.8 | 4.4 | 54.7 | 36.1 | — | — |
| YOLOv11-N | 2.6M | 6.5 | 2.2 | 37.1 | 17.3 | — | — |
| YOLOv8-N | — | 8.7 | 2.1 | 35.2 | 15.8 | — | — |
| D-FINE-N | — | 7.3 | 2.1 | 42.7 | 22.9 | — | — |
| LW-DETR-T | — | 21.4 | 1.9 | 42.9 | 22.7 | — | — |

- In that table, YOLO latencies **include NMS**, and YOLO AP is re-measured in TensorRT FP16. That is why YOLOv11-N shows 37.1 rather than the Ultralytics 39.5. — [RF-DETR paper Table 2](https://arxiv.org/abs/2511.09554)
- **Resolution ablation inside RF-DETR (M-size path, Table 5):**
  - "Patch 14→16, Res 560→640": 78.5 GFLOPs, 4.7 ms, 54.4 AP, AP_S 35.9.
  - "Res 640→576": 64.2 GFLOPs, 4.0 ms, 53.6 AP, AP_S 34.8.
  - Dropping resolution 640→576 (−19% pixels) cut GFLOPs by 18% and latency by 0.7 ms, at a cost of −0.8 AP and −1.1 AP_S.
  - [RF-DETR paper Table 5](https://arxiv.org/abs/2511.09554)
- **Benchmarking protocol advocated by RF-DETR:**
  - Same model artifact for accuracy and latency.
  - FP16 TensorRT, with a 200 ms buffer between passes to avoid power throttling.
  - NMS included for YOLO.
  - YOLOv8/11 AP could not be reproduced in TensorRT (multi-class vs single-class NMS mismatch) and degrades further in FP16.
  - Naive D-FINE FP16 export gave 0.5 AP until fixed with ONNX opset 17.
  - Example: YOLOv11-M reported 51.5 AP / 4.7 ms, but buffered FP16 gives 48.3 AP / 5.2 ms.
  - [RF-DETR paper Table 1](https://arxiv.org/abs/2511.09554)
- **Roboflow in-house re-evaluation (vendor):** everything is measured with pycocotools on val2017, T4 / TRT 10.4 / FP16 / bs 1. YOLO26-N at 640 gives 40.3 AP50:95, 55.8 AP50, 1.7 ms. YOLO11-N at 640 gives 37.4 AP, 2.5 ms. D-FINE-N gives 42.7 AP, 2.1 ms. RF-DETR rows use native resolutions (384–880), while all others are 640. — [RF-DETR benchmarks page](https://rfdetr.roboflow.com/latest/learn/benchmarks/)

**YOLO26, YOLO11 and YOLOv5 official numbers**
- **YOLO26 (Ultralytics docs), all at 640 px**, with T4 TensorRT10 latency and NMS-free head:

| Model | mAP50-95 | mAP50-95 (e2e) | T4 (ms) | CPU ONNX (ms) | Params (M) | FLOPs (B) |
|---|---|---|---|---|---|---|
| YOLO26n | 40.9 | 40.1 | 1.7 | 38.9 | 2.4 | 5.5 |
| YOLO26s | 48.6 | 47.8 | 2.5 | — | 9.5 | 20.9 |
| YOLO26m | 53.1 | 52.5 | 4.7 | — | 20.4 | 68.4 |

- The YOLO26 docs give no imgsz sweep and no small-object AP. They state only that "STAL improves positive label coverage for small objects". — [Ultralytics YOLO26 docs](https://docs.ultralytics.com/models/yolo26/). The YOLO26 paper (Jocher et al., arXiv:2606.03748, Jun 2026) abstract reports "40.9–57.5 mAP on COCO at 1.7–11.8 ms T4". — [arXiv:2606.03748](https://arxiv.org/abs/2606.03748)
- **YOLO11 (Ultralytics docs), at 640:** YOLO11n 39.5 mAP, 1.5 ms T4, 2.6M params, 6.5 GFLOPs. YOLO11s 47.0, 2.5 ms, 21.6 GFLOPs. YOLO11m 51.5, 4.7 ms, 68.1 GFLOPs. — [Ultralytics YOLO11 docs](https://docs.ultralytics.com/models/yolo11/)
- **Only official Ultralytics nano-at-1280 datapoint (YOLOv5 P6):**
  - YOLOv5n at 640: 28.0 mAP50-95, 45.7 mAP50, 4.5 GFLOPs @640, V100 b1 6.3 ms, CPU 45 ms.
  - YOLOv5n6 at 1280: **36.0** mAP50-95, 54.4 mAP50, 3.2M params, 4.6 GFLOPs **reported @640**, V100 b1 8.1 ms, CPU 153 ms.
  - The +8.0 AP comes from both 4× pixels and an extra P6 output stride. Speeds exclude NMS (~1 ms).
  - [YOLOv5 README](https://github.com/ultralytics/yolov5); [PyTorch Hub YOLOv5](https://pytorch.org/hub/ultralytics_yolov5/)

**How other real-time DETR papers make comparisons**
- **RT-DETR:** RT-DETR and the YOLOs "share a common input size of (640, 640)", while other DETRs use 800×1333. FPS is on a T4 with TensorRT FP16. YOLO speed is measured including NMS via a TensorRT NMS plugin, in an "end-to-end speed benchmark". — (snippet only) [Zhao et al., DETRs Beat YOLOs, arXiv:2304.08069](https://arxiv.org/abs/2304.08069)
- **D-FINE / DEIM:** T4 TensorRT FP16 latency at 640×640, with YOLOs grouped as "non-end-to-end". The YOLO26 paper's literature table mixes primary-paper numbers, e.g., D-FINE-L 54.0 AP / 8.1 ms and DEIM-X 56.5 AP / 12.9 ms against YOLO26x 57.5 AP / 11.8 ms. — (snippet only) [D-FINE arXiv:2410.13842](https://arxiv.org/abs/2410.13842); [YOLO26 paper](https://arxiv.org/abs/2606.03748)

### Inferences
- **Compute arithmetic** (extrapolation, not measured). Conv-net FLOPs scale about with H×W, so 640→1280 is ×4.
  - YOLO26n: about 5.5 → **~22 GFLOPs** at 1280.
  - YOLOv5n6: 4.6 → ~18.4 GFLOPs at 1280.
  - RF-DETR-N (ViT, patch 16) at 384 has 24×24 = 576 tokens and 31.9 GFLOPs. At 640 it has 40×40 = 1,600 tokens (×2.8), so **≥ ~89 GFLOPs**, more if attention cost grows super-linearly. At 1280 it has 6,400 tokens, beyond the NAS training grid (max 960), so positional embeddings would be extrapolated.
  - So the current pairing (YOLO26n at 1280, RF-DETR-N at 640) is closer in pixel count than the defaults, but RF-DETR-N at 640 still uses about **4× the FLOPs** of YOLO26n at 1280. RF-DETR-N at its native 384 uses about 1.5× the FLOPs of YOLO26n at 1280.
- **Recommended resolution sweep:**
  - Common resolutions: {384, 512, 640, 960, 1280}, restricted to multiples of 32. All are valid for both YOLO (stride 32) and RF-DETR-N (block 32).
  - Train or fine-tune each model **at each evaluated resolution**, or at least at {640, 1280} for both. Also report the cross-resolution "train@A / test@B" cells, since the model was trained at one size.
  - For each point, report AP50, AP50:95, AP for small pedestrians (height < 32 px or Caltech far/medium bins), GFLOPs from the same profiler, and **measured latency on the actual edge device** (Jetson TensorRT FP16, NMS included for YOLO, same artifact for accuracy and latency, per RF-DETR's protocol).
  - Plot **AP vs latency** and **AP vs GFLOPs** Pareto curves, and compare models at **matched latency** (e.g., the best YOLO resolution whose latency ≤ RF-DETR-N@384 latency, and the reverse), as RF-DETR and LW-DETR do.
- Be careful with vendor-reported numbers. Roboflow's own table shows YOLO26-N at 40.3 rather than Ultralytics' 40.9 (e2e 40.1), so all numbers in the revision should come from one in-house harness.

### Gaps
- No official Ultralytics COCO table for YOLO26n or YOLO11n at imgsz 1280 was found. The YOLOv5n6 P6 model is the closest official nano-at-1280 datapoint, but it is a different architecture.
- RF-DETR does not publish a per-resolution AP table for a fixed Nano config (e.g., N at 384/512/640). The only resolution ablation is the M-size 640→576 step.
- The D-FINE/DEIM papers' exact NMS handling for YOLO rows and their exact latency numbers were seen only in snippets.

---

## B2. Small-object AP sensitivity to input resolution (SAHI tiling, etc.)

### Takeaway
Resolution is the dominant lever for small objects. RF-DETR's authors state it explicitly and lose 1.1 AP_S for a 640→576 reduction. Ultralytics' YOLOv5n gains 8 AP going from 640 to 1280 (P6), and SAHI slicing adds 5–7 AP at inference and 12.7–14.5 AP cumulatively with sliced fine-tuning on aerial small-object datasets. Annotation noise also hits small objects hardest, so small-object evaluation needs the human GT most.

### Cited Findings
- "Higher resolution improves small object detection performance, while lower resolution improves runtime." In RF-DETR-M, Res 640→576 cut AP_S from 35.9 to 34.8, overall AP from 54.4 to 53.6, and GFLOPs from 78.5 to 64.2. — [RF-DETR paper](https://arxiv.org/abs/2511.09554)
- At native resolutions, the small-object gap is large: RF-DETR-N (384) has AP_S 25.2, against YOLOv11-N (640) 17.3, YOLOv8-N 15.8 and D-FINE-N 22.9 (TensorRT FP16 re-evaluation). RF-DETR-N reaches higher AP_S despite fewer input pixels. — [RF-DETR paper Table 2](https://arxiv.org/abs/2511.09554)
- YOLOv5n at 640 gives 28.0 AP, and YOLOv5n6 at 1280 gives 36.0 AP (54.4 AP50), with CPU latency rising from 45 to 153 ms. — [YOLOv5 README](https://github.com/ultralytics/yolov5)
- **SAHI (Akyon, Altinuc, Temizel, ICIP 2022):** slicing-aided inference increases AP by **6.8% (FCOS), 5.1% (VFNet), 5.3% (TOOD)**. With slicing-aided fine-tuning the cumulative gains are **12.7%, 13.4% and 14.5%**, on VisDrone and xView. — [arXiv:2202.06934](https://arxiv.org/abs/2202.06934)
- Annotation discrepancy interacts with size: IoU between equivalent annotations correlates with the smallest box dimension (ρ = 0.79), and AP on small objects is most affected. — [Murrugarra-Llerena et al. 2022](https://openaccess.thecvf.com/content/CVPR2022W/VDU/papers/Murrugarra-Llerena_Can_We_Trust_Bounding_Box_Annotations_for_Object_Detection_CVPRW_2022_paper.pdf)
- In Caltech, about 69% of pedestrians are at medium scale (30–80 px). Detection "is disappointing at low resolutions", and below 30 px annotators also struggle (boxes under 20 px are always ignored). — [Caltech PAMI 2012](https://pdollar.github.io/files/papers/DollarPAMI12peds.pdf)

### Inferences
- The reviewer's concern is well founded. If the YOLO26n-at-1280 advantage over RF-DETR-N-at-640 is concentrated in small or far pedestrians, it may be a resolution effect rather than an architecture effect.
- The sweep should therefore report per-size-bin AP or recall (e.g., height < 32, 32–64, 64–128, > 128 px, or Caltech far/medium/near). Optionally add a SAHI-tiled RF-DETR-N@640 (or @384 tiles) as a third arm, to separate "more pixels" from "different model".
- Report small-object metrics at IoU 0.5 as primary, because human box agreement is weakest there (A2/A5).

### Gaps
- SAHI's gains are on aerial datasets (VisDrone, xView). I found no published SAHI numbers for surveillance-view pedestrian datasets.
- No published AP_S-vs-imgsz curve for YOLO26n or YOLO11n was found.
