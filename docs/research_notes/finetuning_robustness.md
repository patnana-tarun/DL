# Preventing scene-specific overfitting when fine-tuning COCO-pretrained detectors on single-camera pseudo-labels

Context being addressed: YOLO26n fine-tuned on WBF(RF-DETR-L + YOLO26x) pseudo-labels from one campus camera: in-domain AP50:95 0.760 -> 0.825; held-out camera 0.684 (pretrained) -> 0.412 (fine-tuned); small-person (32-64 px) recall 0.40 -> 0.05; new FPs on bins/benches/trees.

Source-verification note: items marked "(abstract-level)" were confirmed only from the abstract or search snippet, not from the paper's full tables. Items marked "(not fetched this session)" are well-known primary papers cited by URL whose numbers I am stating from the paper abstract as I know it. Re-check those before quoting numbers in the paper.

## 1. Weight-space ensembling (WiSE-FT, model soups, pretrained/fine-tuned interpolation) for detectors

### Takeaway
Yes, it has now been applied to object detectors. WiSE-OD (WACV 2026) interpolates COCO-pretrained and fine-tuned Faster R-CNN, FCOS, RetinaNet and YOLOv8 weights. It beats both endpoints on in-domain AND corrupted/OOD AP50 at λ≈0.5, at zero inference cost. It is the cheapest and best-supported first experiment for our case. The main pitfall for YOLO (BatchNorm) has a standard fix: re-estimate BN running stats after merging.

### Cited Findings
- WiSE-FT (Wortsman et al., CVPR 2022) fine-tunes a zero-shot model and then linearly interpolates its weights with the original zero-shot weights. The method and experiments are classification-only (CLIP), and the authors list detection fine-tuning as future work — [WiSE-FT CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Wortsman_Robust_Fine-Tuning_of_Zero-Shot_Models_CVPR_2022_paper.pdf); [arXiv 2109.01903](https://arxiv.org/html/2109.01903v3)
- WiSE-FT headline numbers (abstract-level). On ImageNet and five derived shifts it improves OOD accuracy by 4-6 pp over prior work while also raising ImageNet accuracy by 1.6 pp. On six further shifts the gains are 2-23 pp, and on seven transfer datasets it is 0.8-3.3 pp better than standard fine-tuning — [arXiv 2109.01903](https://arxiv.org/html/2109.01903v3)
- **WiSE-OD (Medeiros, Belal, Aminbeidokhti, Granger, Pedersoli; WACV 2026)** is the direct detection extension. It interpolates a COCO-RGB-pretrained detector ("zero-shot", ZS) with an IR-fine-tuned detector (WiSE-OD_ZS), or with a linear-probed detector whose cls/reg heads were trained on a frozen backbone (WiSE-OD_LP). It needs no extra training or inference cost. Detectors: Faster R-CNN, FCOS, RetinaNet, and YOLOv8n on M3FD — [arXiv 2507.18925](https://arxiv.org/abs/2507.18925); [HTML v2](https://arxiv.org/html/2507.18925v2)
- WiSE-OD numbers (Faster R-CNN, AP50) — [WiSE-OD v2](https://arxiv.org/html/2507.18925v2):
  - LLVIP in-domain: ZS 71.21, FT 93.68, LP 91.82, LP-FT 92.18, **WiSE-OD_ZS 96.06, WiSE-OD_LP 96.24**. The merged model beats the fine-tuned one in-domain too.
  - LLVIP-C mPC (corruptions): ZS 40.96, FT 56.40, LP 70.96, LP-FT 70.02, **WiSE-OD_ZS 75.08, WiSE-OD_LP 75.83**.
  - FLIR-C mPC: ZS 33.67, FT 44.80, LP 47.11, LP-FT 46.92, **WiSE-OD_ZS 50.52, WiSE-OD_LP 51.59**.
  - FLIR in-domain: ZS 65.52, FT 77.57, WiSE-OD_ZS 82.20 (λ=0.5).
  - Real OOD (M3FD), ZS/FT/WiSE-OD: rain 13.1/28.0/29.4, fog 87.4/93.0/97.2, **night 14.3/47.8/43.3 (WiSE worse than FT)**, indoor 5.8/59.8/59.2. The paper's text and Table 4 disagree on these gains.
- WiSE-OD λ choice. The default is λ=0.5 with no target tuning. Best LLVIP-C mPC is at λ=0.3 (76.98), and best FLIR-C at 0.5. A fixed λ is not uniformly optimal — [WiSE-OD v2](https://arxiv.org/html/2507.18925v2)
- WiSE-OD limitations as stated by the authors:
  - Gains are limited if either endpoint is weak.
  - Under extreme corruptions both models miss objects, and the merge propagates those errors.
  - BatchNorm handling during interpolation is not discussed.
  - Source: [WiSE-OD v2](https://arxiv.org/html/2507.18925v2)
- BatchNorm pitfall. Interpolating weights can cause "variance collapse" of activations, which is why BN generally breaks after interpolating independently trained nets. The standard fix is to reset BN running stats and recompute them by forwarding training data (RESET), or to use REPAIR (re-normalise pre-activations). REPAIR cut the ResNet18/CIFAR-10 interpolation barrier from about 16% to about 1.5% — [Git Re-Basin, arXiv 2209.04836](https://arxiv.org/pdf/2209.04836); [Non-local merging / variance collapse, arXiv 2410.12766](https://arxiv.org/pdf/2410.12766); [Rethinking re-basin, arXiv 2402.05966](https://arxiv.org/html/2402.05966v2)
- Related follow-ups (classification) build on WiSE interpolation:
  - DaWin: training-free per-sample dynamic interpolation coefficients — [arXiv 2410.03782](https://arxiv.org/pdf/2410.03782)
  - MonoSoup: reports Pareto fronts that dominate WiSE-FT and LiNeS — [arXiv 2602.09689](https://arxiv.org/html/2602.09689)
- Model soups (Wortsman et al., ICML 2022; not fetched this session) average the weights of several fine-tuned runs from the same pretrained init (greedy soup) and improve both ID and OOD without inference cost — [arXiv 2203.05482](https://arxiv.org/abs/2203.05482)

### Inferences
- **Our pitfalls (1): BatchNorm.** Our case is the favourable one for interpolation. YOLO26n-pretrained and YOLO26n-fine-tuned share an init and differ by a short fine-tune, so they should be linearly mode-connected; the BN variance-collapse problem mainly affects independently trained nets. Still, YOLO uses BN everywhere and running_mean/var are buffers, not weights. Safe procedure:
  - Interpolate all float tensors, including BN affine params and running stats.
  - Then (variant B) recalibrate BN by forwarding about 200-500 unlabeled images in train mode with no gradient.
  - Report both variants. Recalibrate on a mix of campus and generic images. Recalibrating on campus-only frames would re-specialise the BN stats.
- **Our pitfalls (2): detection head.** YOLO26's head has the same shape (nc=80 kept) in both endpoints, so plain interpolation is valid. If the fine-tune changed nc (person-only, nc=1), the cls-head tensors no longer match. Then either keep the 80-class head and train with person only, or interpolate only matching tensors and take the fine-tuned head (WiSE-OD_LP-style variants show that head choice matters).
- **The λ sweep is the experiment.** Sweep λ ∈ {0, 0.1, ..., 1.0} (11 evals, no training) and plot in-domain AP vs held-out-camera AP. This directly yields the ID/OOD trade-off curve described in Section 7. WiSE-OD's results suggest λ in 0.3-0.5 may recover most OOD loss while keeping much of the ID gain. Our gap (0.684 -> 0.412) is larger than WiSE-OD's settings, so this is a hypothesis to test.
- **Night-like caveat.** WiSE-OD lost to FT on M3FD-night. Interpolation helps most when the pretrained model is already decent on the OOD domain. Our pretrained model scores 0.684 on camera 2, so this condition holds.
- **Soup of runs.** A soup of 2-3 fine-tunes with different seeds or augmentations, then WiSE with pretrained, is a cheap extension. Each YOLO26n fine-tune is short on a 4060.

### Gaps
- I found no published WiSE-FT/WiSE-OD result on DETR-family detectors (RF-DETR, RT-DETR) or on YOLO26 specifically. WiSE-OD's YOLOv8n result is only on M3FD and I did not extract its per-number table.
- No paper found that examines small-object recall specifically under weight interpolation.
- WiSE-OD does not state its BN handling. The BN-recalibration advice comes from the model-merging literature (independently trained nets), not from detection fine-tuning.

## 2. Regularised fine-tuning: L2-SP, freezing, lower LR / fewer epochs, LP-FT, EMA

### Takeaway
There is detection-specific evidence that full fine-tuning distorts backbone features and can lower OOD AP (Saito et al., "Mind the Backbone"). Decoder-probe-then-fine-tune (the detection analogue of LP-FT) plus weight regularisation toward pretrained weights (L2-SP-style) helps. Freezing the backbone in YOLOv8n preserved COCO mAP to within 0.1%. On a single-camera dataset, freezing plus a short, low-LR schedule is the cheapest regulariser. LP-FT alone did not beat LP on OOD in the WiSE-OD detection benchmark.

### Cited Findings
- LP-FT (Kumar et al., ICLR 2022) argues that full fine-tuning with a randomly initialised head distorts pretrained features and underperforms OOD. Linear-probing the head first, then fine-tuning everything, fixes this. Abstract-level numbers: LP-FT gets the best ID (avg 85.7%) and OOD (avg 68.9%) accuracy across ten distribution-shift benchmarks — [arXiv 2202.10054](https://arxiv.org/pdf/2202.10054)
- **"Mind the Backbone" (Saito, Kim, Teterwak, Feris, Saenko; arXiv 2303.14744)** applies this to detection. DP = decoder-probing (train detector neck/head on a frozen backbone), FT = full fine-tune, DP-FT = DP then FT. Source: [ar5iv 2303.14744](https://ar5iv.labs.arxiv.org/html/2303.14744)
  - **ResNet50 (Instagram) Pascal -> Comic/Watercolor/Clipart.** OOD is DP 15.7/21.2/15.3 vs FT 7.5/19.4/11.4. **FT more than halves Comic AP**, even though ID rises from 44.6 to 50.4. DP-FT gives 9.1/21.0/12.9 with ID 52.6.
  - **ConvNeXt-B: FT improves OOD over DP**, with DP-FT best (Cityscapes -> Foggy 20.5, BDD 11.5, ID 37.1). Whether FT hurts OOD depends on the backbone.
  - **Relative Gradient Norm (RGN)** predicts distortion: ResNet50 1.34, ConvNeXt 0.13, EfficientNet-B2 0.06. Higher RGN means lower OOD gain from FT.
  - **Fixes.** Weight regularisation toward pretrained weights (WR, L2-SP-style) and SE blocks reduce distortion and are complementary. On COCO -> COCO-C (R50), relative performance under corruption goes from 55.3% (DP-FT) to 69.0% with WR+SE.
  - **Decoder strength.** A stronger decoder (NAS-FPN) increases distortion and lowers OOD.
  - **Longer DP.** Longer DP training reduces RGN and improves OOD.
- In WiSE-OD's detection benchmark (Faster R-CNN, LLVIP-C mPC), LP alone scores 70.96 vs FT 56.40, with ID 91.82 vs 93.68. LP-FT (70.02) did not beat LP OOD. Weight interpolation (75-76) beat all of them — [WiSE-OD v2](https://arxiv.org/html/2507.18925v2)
- **YOLOv8n freezing study (Gandhi & Gandhi, arXiv 2505.01016; preprint, not peer-reviewed).** Fine-tuning YOLOv8n on a fine-grained fruit dataset with freeze points at layers 22/15/10 lost under 0.1% absolute COCO mAP at every freeze level. Unfreezing down to layer 10 gave about +10% mAP50 on the target vs head-only training — [arXiv 2505.01016](https://arxiv.org/abs/2505.01016)
- DAFT critiques LP-FT. LP-FT's BatchNorm handling is a weakness, and its small FT learning rate can limit head optimisation. DAFT merges the two stages; its evaluation is classification only — [arXiv 2308.07728](https://arxiv.org/html/2308.07728v5)
- LP-FT for query-based (DETR-style) few-shot detectors is used to balance the ID-OOD trade-off (BIOT) — [HCC journal](https://journal.hep.com.cn/hcc/EN/1182989588496036466)
- L2-SP (Li, Grandvalet, Davoine, ICML 2018; not fetched this session) penalises ||w - w_pretrained||² instead of ||w||² during fine-tuning — [arXiv 1802.01483](https://arxiv.org/abs/1802.01483). A related result: decaying toward the pretrained weights is studied for robust fine-tuning of foundation models — [arXiv 2411.01713](https://arxiv.org/pdf/2411.01713)
- **Ultralytics knobs** (docs) — [Ultralytics train docs](https://docs.ultralytics.com/modes/train/):
  - `freeze` (int or list): freezes the first N layers or named modules, e.g. `23.cv2`.
  - `lr0`: default 0.01. **It is ignored under the default `optimizer='auto'`**, which picks AdamW for short runs (MuSGD for runs over about 10k iterations) and sets its own LR. Set `optimizer='SGD'` or `'AdamW'` explicitly for an LR ablation to take effect.
  - `lrf` 0.01, `epochs` 100, `warmup_epochs` 3.0, `close_mosaic` 10, `scale` 0.5, `mosaic` 1.0, `multi_scale` 0.0.

### Inferences
- **Our run may have been more aggressive than intended.** Check whether `optimizer='auto'` was used (if so, `lr0` settings were silently ignored) and how many epochs ran. A low-LR, short (5-15 epoch) fine-tune with `freeze=10` (backbone frozen in YOLOv8/11-style indexing; verify the YOLO26 layer indices in its yaml) is the cheapest regularised baseline.
- **Fit to Saito et al.** YOLO26n is a tiny CNN with BN. Its sensitivity is unknown, but our 0.684 -> 0.412 drop matches the "FT distorts features" ResNet50 pattern. Freezing directly limits backbone distortion.
- **L2-SP for Ultralytics** needs a small custom trainer (subclass DetectionTrainer, add λ·Σ||w−w0||² to the loss). That is more engineering than `freeze` or WiSE, so it is lower priority. WiSE interpolation is roughly a post-hoc analogue of pulling weights toward w0.
- **EMA** is on by default in Ultralytics training (ModelEMA); this comes from Ultralytics code/knowledge, not from the docs page fetched. It stabilises training but is EMA over the fine-tune trajectory, not toward pretrained weights. It does not by itself prevent drift from COCO.
- **LP-FT for YOLO** = freeze backbone+neck, train head for a few epochs, then unfreeze at low LR. Evidence (WiSE-OD) suggests LP/DP alone may be as robust as LP-FT. A "head-only" run (`freeze` = all but the Detect layer) is a cheap probe.

### Gaps
- No peer-reviewed study found of LP-FT/L2-SP/freezing specifically on YOLO-family nano models with an OOD evaluation. The YOLOv8n study is a non-peer-reviewed preprint, measures only COCO retention (not another camera), and does not report absolute COCO mAP values.
- No source found on fine-tuning RF-DETR/DINOv2-backbone detectors with frozen vs unfrozen backbone for OOD robustness.

## 3. Replay / rehearsal: mixing COCO (or other-scene) images into fine-tuning

### Takeaway
The closest prior work to our setting, Scenes100 scene adaptation (CVPR 2023), mixes **as many COCO train images as unlabeled scene frames (1:1)** during per-scene pseudo-label fine-tuning. Continual detection work shows replay buffers of 5-50% keep near-joint-training performance. A 1:1 (or at least 1:4) COCO-person replay ratio is the well-precedented default.

### Cited Findings
- **Scenes100 setup.** Zhang & Hoai (CVPR 2023) adapt a Faster R-CNN to each of 100 fixed-camera videos with self-generated pseudo-labels:
  - Pseudo-labels come from the detector plus a forward/backward DiMP-50 tracker; tracks are capped at 2 s to prevent drift.
  - Two base models (R-50, R-101) cross-teach.
  - **"During training, we include the same number of images from MSCOCO training set as the number of unlabeled frames from the videos."**
  - The same hyperparameters are used for all 100 videos.
  - Source: [Zhang & Hoai CVPR 2023 PDF](https://www3.cs.stonybrook.edu/~minhhoai/papers/Self-Supervised_Scene_Adaptation_CVPR23.pdf)
- Plain fine-tuning in class-incremental COCO detection (40-40, 70-10 splits) causes severe forgetting of old classes, and replay is a main remedy — [Augmented Box Replay, arXiv 2307.12427](https://arxiv.org/pdf/2307.12427); [Elastic Response Distillation, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Feng_Overcoming_Catastrophic_Forgetting_in_Incremental_Object_Detection_via_Elastic_Response_CVPR_2022_paper.pdf)
- Replay-based continual detection for indoor drones compared naive FT against ER/MIR/FAR at 5%-50% buffer budgets. Naive FT forgets; FAR and MIR stay close to joint training even with minimal memory (abstract-level) — [arXiv 2602.13440](https://arxiv.org/html/2602.13440v1)
- One-Shot Replay stores one augmented object crop per old class (<1 MB) and improves on VOC/COCO incremental detection (abstract-level) — [AAAI 2023](https://ojs.aaai.org/index.php/AAAI/article/view/25417/25189)
- Knowledge distillation (response-level) is the main non-replay alternative and narrows but does not close the gap to joint training — [ERD CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Feng_Overcoming_Catastrophic_Forgetting_in_Incremental_Object_Detection_via_Elastic_Response_CVPR_2022_paper.pdf)
- The Ultralytics docs page does not document multi-dataset or replay options in `data`. It only describes a YAML with train/val paths — [Ultralytics train docs](https://docs.ultralytics.com/modes/train/)

### Inferences
- **Our setting is not class-incremental, but similar.** It is a domain/scale-distribution shift within the person class. Replay matters for a specific reason here: COCO person images carry many small, distant, varied-background persons and many non-person "bench/bin/tree" negatives. That counters both observed failures (small-person recall collapse; FPs on street furniture).
- **Practical recipe:**
  - Build a COCO-person subset (images containing persons, plus about 10-20% person-free COCO images as hard negatives).
  - Use GT COCO labels remapped to the same class set.
  - List both image dirs under `train:` in the dataset YAML; Ultralytics generally accepts a list of train paths (an inference from common practice; verify against the installed version).
  - Ratio 1:1 per Scenes100. Ablate 0 / 0.25 / 1.0.
- **Alternative to raw COCO replay:** use pretrained-model distillation on campus frames (keep the pretrained YOLO26n's outputs as soft targets). This is more engineering.

### Gaps
- I found no detection paper that systematically sweeps the replay ratio for single-scene pseudo-label adaptation. The 1:1 ratio is one paper's design choice, not an ablated optimum.
- Scenes100's main paper reports only in-scene AP gains (+3.76 AP / +4.45 AP50 averaged), not performance on other scenes or COCO after adaptation. "Compound" multi-video models are in the supplement, which I did not read.

## 4. Scene-specific self-training domain adaptation for fixed-camera pedestrian detection (and its pitfalls)

### Takeaway
Scene-specific adaptation from a fixed camera is a long-standing line of work (2011 -> Scenes100 2023). Its typical in-scene gains are modest: +0.8 to +4.5 AP on Scenes100 with Faster R-CNN. Several popular DA methods (STAC, Adaptive Teacher, H2FA, TIA) actually **reduced** AP on Scenes100, and longer schedules caused "severe degradation" for LODS. Teacher bias toward the source domain and confirmation bias are the named failure modes. Our +6.5 AP50:95 in-scene gain is large, and paired with a -27 AP OOD loss it fits an over-specialised student rather than a generally better one.

### Cited Findings
- Scenes100 averaged AP gain per scene (AP / AP50, COCO-weighted) — [Zhang & Hoai CVPR 2023](https://www3.cs.stonybrook.edu/~minhhoai/papers/Self-Supervised_Scene_Adaptation_CVPR23.pdf):
  - Self-Train +0.80/+0.24
  - STAC −1.26/−5.12
  - Adaptive Teacher −0.75/−1.11
  - H2FA −3.10/−4.97
  - TIA −0.32/−0.37
  - LODS +0.45/+1.28
  - proposed (pseudo-labels + tracker + location-aware mixup + background mid-fusion) **+3.76/+4.45**
  - pseudo-labeling alone +0.95/+0.54
- In the same paper: "any training schedule longer than 1000 iterations would lead to severe performance degradation" for LODS. The base model on Scenes100 is only about 10 AP below its COCO AP, so gains above 5 points were not observed. Pseudo-box refinement "can neither remove false positives nor recover false negatives" — [Zhang & Hoai CVPR 2023](https://www3.cs.stonybrook.edu/~minhhoai/papers/Self-Supervised_Scene_Adaptation_CVPR23.pdf)
- Earlier scene-specific pedestrian detector adaptation (abstract-level):
  - iterative retraining of a generic detector on confident target positives/negatives selected with path models (2011) — [IEEE 5995698](https://ieeexplore.ieee.org/document/5995698/)
  - progressive latent models for self-learning scene-specific detectors (TITS 2019) — [IEEE 8701617](https://ieeexplore.ieee.org/document/8701617/); [arXiv 1611.07544](https://arxiv.org/pdf/1611.07544)
  - unsupervised DA scene-specific detection for static surveillance (Pattern Recognition 2021) — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0031320321002259)
  - synthetic scene-specific training data for a calibrated static camera (CVPR 2015) — [Hattori et al.](https://www.ri.cmu.edu/pub_files/2015/6/cvpr-2015.pdf)
- Unbiased Teacher (Liu et al., ICLR 2021) identifies a "pseudo-labeling bias" in semi-supervised detection. It uses a teacher-student scheme with a gradually updated teacher and a class-balance loss that down-weights over-confident pseudo-labels. Reported gain: about +6.8 mAP over prior SOTA at 1% COCO labels (abstract-level) — [arXiv 2102.09480](https://arxiv.org/abs/2102.09480)
- In cross-domain teacher-student detection, the teacher "generates many low-quality pseudo labels (e.g., false positives) due to its bias toward the source domain." Adaptive Unbiased Teacher and Adaptive Teacher (CVPR 2022; EMA teacher, weak/strong augmentation, adversarial feature alignment) counter this — [AUT arXiv 2111.13216](https://arxiv.org/abs/2111.13216v2); [Adaptive Teacher, CVPR 2022](https://mlanthology.org/cvpr/2022/li2022cvpr-crossdomain-a)
- Debiased Teacher (ICCV 2025) names three biases that self-training DAOD methods fail to address: distribution, training and **confirmation** bias — [ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/papers/Cui_Debiased_Teacher_for_Day-to-Night_Domain_Adaptive_Object_Detection_ICCV_2025_paper.pdf)

### Inferences
- **Our pipeline is offline single-round distillation** from a stronger teacher ensemble, not iterative self-training. Classic confirmation bias (the student re-labelling itself) does not apply directly. **Teacher bias does**: whatever the WBF ensemble misses (small/far persons below threshold) becomes a hard negative the student is trained to suppress.
- **Why the FPs appear on bins/benches/trees.** A single static background gives very low negative diversity. The student can overfit to scene-specific cues (e.g., "objects at these image locations or scales are persons"). Scenes100 mixes COCO 1:1 partly for this reason. Location-aware mixup (pasting pseudo-labeled persons at plausible positions) is their in-scene augmentation remedy.
- **EMA teacher frameworks** (Mean Teacher / Unbiased Teacher) are heavier to implement in Ultralytics. On a 4060 within a few hours they are lower priority than WiSE, freeze, and replay.

### Gaps
- I could not access Scenes100's supplement (per-scene failure cases, hyperparameters, compound models, any cross-scene evaluation).
- No fixed-camera pedestrian-adaptation paper found that reports OOD/other-camera degradation after adaptation. The literature mostly reports only in-scene gains, which is itself a gap our paper can fill.

## 5. Why pseudo-label fine-tuning hurts small objects

### Takeaway
Small objects get lower teacher confidence and noisier boxes. A single global confidence threshold therefore drops disproportionately many small or far persons. Unlabeled small persons then act as **background (negative) supervision**, which actively teaches the student to suppress them. Recall can collapse well beyond the teacher's own miss rate. The literature supports the mechanism (precision/recall trade-off of thresholds; small objects have noisier pseudo-labels) but has no size-stratified threshold study.

### Cited Findings
- SCMT (IJCAI 2022) reports its largest gains on small-object AP. It argues that weak features for small objects produce noisier pseudo-labels, and it re-weights student losses by teacher confidence instead of using one hard threshold — [SCMT IJCAI 2022](https://www.ijcai.org/proceedings/2022/0207.pdf)
- Using only the top-1 class score as the filter overlooks hard samples and lowers pseudo-label recall. Top-k cumulative scoring raises recall — [Double-Check Soft Teacher IJCAI 2022](https://ijcai.org/proceedings/2022/0199.pdf)
- Raising pseudo-label recall (0.299 -> 0.376 via two-stage mining) lowered pseudo-label precision. The authors conclude precision matters more than recall for final AP, so the trade-off is not one-directional — [ACRST arXiv 2107.05031](https://arxiv.org/pdf/2107.05031)
- One round of self-training greatly increases recall, but precision drops at low thresholds (event-camera detection) — [LEOD arXiv 2311.17286](https://arxiv.org/pdf/2311.17286)
- Threshold selection methods:
  - dynamic threshold by maximising F1 on teacher predictions — [S4OD arXiv 2204.04492](https://arxiv.org/pdf/2204.04492)
  - non-iterative threshold optimisation accounting for GT and pseudo-labels — [arXiv 2210.10221](https://arxiv.org/pdf/2210.10221)
  - in one cross-domain few-shot study, performance peaked at a 0.6 threshold — [GiPL arXiv 2605.29539](https://arxiv.org/pdf/2605.29539)
- Scenes100's own motivation for the tracker is that propagating detections to adjacent frames recovers false negatives the detector missed. Backward tracking is used because an object approaching the camera starts small, and the box seeded from its larger later appearance is more accurate — [Zhang & Hoai CVPR 2023](https://www3.cs.stonybrook.edu/~minhhoai/papers/Self-Supervised_Scene_Adaptation_CVPR23.pdf)

### Inferences
- **Three compounding mechanisms in our case.** These are hypotheses to check with our own data:
  - **(a) Threshold censoring.** The WBF confidence cut removes small persons first, so the campus pseudo-label scale distribution skews large.
  - **(b) Negative supervision.** Missed small persons become background and are explicitly penalised.
  - **(c) Scale-prior shift.** Campus camera-1 persons occupy a narrow size range, so the student's assigner/head stops firing at other scales. Camera 2 has more 32-64 px persons.
- **Diagnostic to add to the paper.** Histogram pseudo-label box sizes vs teacher raw detections (pre-threshold) vs camera-2 GT. Report pseudo-label recall by size bin on any hand-labeled camera-1 frames.
- **Cheap mitigations for small persons:**
  - Lower the threshold or use size-dependent thresholds for boxes under 64 px.
  - Use "ignore regions" for low-confidence teacher boxes (0.1-thr). These are neither positive nor negative; Ultralytics has no native ignore label, so approximate it by masking/blurring those regions or dropping images with many uncertain boxes.
  - Add `scale`/`multi_scale` augmentation (e.g., scale=0.7-0.9, multi_scale=0.25) to broaden the size distribution.
  - Use COCO replay (Section 3).

### Gaps
- No paper found that studies per-size-bin pseudo-label thresholds or quantifies small-object recall collapse after pseudo-label fine-tuning of a compact detector. This is likely a novel analysis for our paper.
- No source found on WBF-specific effects (box-score averaging across a DETR and a YOLO teacher) on small-object confidence.

## 6. Multi-scene / leave-one-scene-out evaluation protocols

### Takeaway
The closest established protocol is Scenes100's per-scene adaptation with per-video AP averaged across 100 scenes. It reports "AP gain" (adapted − base) and also trains compound all-scene models. For generalisation claims, the standard is held-out domains never seen in training: Pascal -> Comic/Watercolor/Clipart, Cityscapes -> Foggy/BDD, COCO -> COCO-C/COCO-O. With only 2 cameras, report both directions (adapt on A, test on A and B; adapt on B, test on B and A) where labels allow.

### Cited Findings
- Scenes100 has 100 fixed-camera videos (about 2 h total). The protocol:
  - AP and AP50 are evaluated per video and then averaged.
  - The headline metric is the averaged AP gain, APG = 1/100 Σ (AP_v,adapt − AP_v,base).
  - The same hyperparameters are used for all videos.
  - Compound models are trained on all videos jointly.
  - Source: [Zhang & Hoai CVPR 2023](https://www3.cs.stonybrook.edu/~minhhoai/papers/Self-Supervised_Scene_Adaptation_CVPR23.pdf); [code/data](https://github.com/cvlab-stonybrook/scenes100)
- Detection OOD protocols use ID test + several held-out domains:
  - Pascal -> Comic/Watercolor/Clipart and Cityscapes -> Foggy Cityscapes/BDD — [Saito et al. 2303.14744](https://ar5iv.labs.arxiv.org/html/2303.14744)
  - corruption suites with mPC at fixed severities (LLVIP-C, FLIR-C) plus real OOD (M3FD rain/fog/night/indoor) — [WiSE-OD](https://arxiv.org/html/2507.18925v2)
- COCO-O (ICCV 2023) is a natural-distribution-shift benchmark for COCO detectors with six domains. It is useful as a generic-robustness check after adaptation — [COCO-O ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/papers/Mao_COCO-O_A_Benchmark_for_Object_Detectors_under_Natural_Distribution_Shifts_ICCV_2023_paper.pdf); [arXiv 2307.12730](https://arxiv.org/pdf/2307.12730)

### Inferences
- **Recommended eval matrix for every variant:**
  - (i) camera-1 held-out frames (in-domain)
  - (ii) camera-2 (held-out scene, test-only)
  - (iii) a COCO val person subset, or COCO-O person images (generic retention, "forgetting")
  - (iv) size-stratified recall (<32, 32-64, 64-128, >128 px) on (i) and (ii)
  - (v) FP per image on person-free frames, which captures the bins/benches/trees failure
- Keep camera 2 strictly test-only for model and λ selection. Pick λ or freeze depth on (i)+(iii), **not** on (ii), to avoid selecting on the OOD test set. This matches the project's existing "test1.mp4 test-only" discipline.
- With more cameras (public fixed-camera sets such as Scenes100 videos containing persons, or MOT17/MOT20 static-camera sequences), a leave-one-scene-out table would be stronger. Even 3-5 extra scenes as OOD tests would give variance estimates.

### Gaps
- No standard "leave-one-camera-out" detection benchmark was found in this session beyond Scenes100. MOT-sequence suitability (static vs moving camera per sequence) was not verified here.

## 7. How to report "adaptation helps vs hurts": ID-vs-OOD trade-off curves and effective robustness

### Takeaway
Report (1) ID and OOD AP side by side for every method, (2) a 2-D ID-vs-OOD scatter or Pareto plot with the WiSE λ-path drawn as a curve, and (3) effective robustness: OOD performance relative to a baseline trend fitted across models. COCO-O fitted such a line for detectors (slope about 0.45, unverified). Effective robustness is rare and can vanish at convergence, so claims should be framed as Pareto improvements over the pretrained -> fine-tuned line.

### Cited Findings
- Effective robustness (Taori et al. 2020) is OOD accuracy above a baseline β(ID) fitted (logit/log-linear) across many models. "Accuracy on the line" shows a strong linear ID-OOD correlation across many shifts, though not all shifts follow linear trends — [Accuracy on the Line arXiv 2107.04649](https://arxiv.org/pdf/2107.04649); [Taori et al. arXiv 2007.00644](https://arxiv.org/abs/2007.00644) (not fetched this session)
- Effective robustness from pretraining shows up during fine-tuning but vanishes at convergence, and models above the line are rare — [Andreassen et al., arXiv 2106.15831](https://arxiv.org/pdf/2106.15831)
- COCO-O fits a linear trend of OOD mAP vs COCO mAP across 11 classic detectors, slope about 0.45. A secondary summary gives ER = mAP_ood − 0.45·mAP_id; I did not confirm the exact formula in the primary paper — [COCO-O arXiv 2307.12730](https://arxiv.org/pdf/2307.12730); [ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/papers/Mao_COCO-O_A_Benchmark_for_Object_Detectors_under_Natural_Distribution_Shifts_ICCV_2023_paper.pdf)
- WiSE-FT and WiSE-OD present results as ID/OOD frontiers traced by λ. Interpolated models lie above the straight line between endpoints, i.e. they Pareto-dominate both endpoints on some shifts — [WiSE-FT](https://openaccess.thecvf.com/content/CVPR2022/papers/Wortsman_Robust_Fine-Tuning_of_Zero-Shot_Models_CVPR_2022_paper.pdf); [WiSE-OD](https://arxiv.org/html/2507.18925v2)
- Saito et al. report OOD gain/loss per method relative to DP. This separates "FT helps ID" (+5.8 AP Pascal, R50) from "FT hurts OOD" (−8.2 AP Comic) — [ar5iv 2303.14744](https://ar5iv.labs.arxiv.org/html/2303.14744)

### Inferences
- **Paper figure.** X = camera-1 AP50:95, Y = camera-2 AP50:95. Plot:
  - pretrained (0.760, 0.684)
  - full FT (0.825, 0.412)
  - the WiSE λ-curve between them
  - points for freeze-k, replay-r, and low-LR/short variants
  - teacher models as reference
- **Reading the plot.** The "no-free-lunch" line is the straight segment pretrained -> FT. Methods above it have positive effective robustness relative to that segment. A method that reaches, say, ≥0.80 ID with ≥0.65 OOD would be a clear Pareto win.
- **Companion table.** Add a "forgetting" row: COCO-person AP before/after. Report ΔID, ΔOOD, and Δsmall-recall per method, with seed variance (≥2-3 seeds) for any difference under about 2 AP.
- **Honest framing for the paper.** Single-camera pseudo-label adaptation is a deployment specialisation. It is justified when the camera is fixed and the adapted model is used only on that camera. Report the OOD drop explicitly as the cost, and show that WiSE/replay/freeze recovers most of it.

### Gaps
- No detection-specific effective-robustness fit exists for our camera pair. With only two cameras, a fitted β line needs many models (e.g., the YOLO26 n/s/m/l/x and RF-DETR family pretrained checkpoints evaluated on both cameras). That is feasible and cheap (inference only) and would make an "effective robustness" claim well-founded.

## 8. Synthesis: which 2-4 cheap mitigations to run on an RTX 4060 8GB

### Takeaway
Priority order by (evidence strength × cost):

1. **WiSE interpolation λ-sweep.** No training, minutes; strongest detection-specific evidence (WiSE-OD).
2. **COCO-person replay at 1:1.** One extra fine-tune; Scenes100 precedent.
3. **Backbone freeze plus explicit low LR / short schedule.** One or two fine-tunes; Saito et al. and the YOLOv8n study.
4. **Small-object-aware pseudo-labels.** Lower or size-aware threshold, scale/multi-scale augmentation; one fine-tune; mechanism supported by SCMT/LEOD/ACRST.

Then apply WiSE on top of the best of 2-4. Weight-space ensembling is complementary to the others; WiSE-OD_LP uses an LP endpoint.

### Cited Findings
- WiSE-OD λ=0.5 improves both ID and corruption-OOD AP50 over FT for four detector families — [WiSE-OD](https://arxiv.org/html/2507.18925v2)
- Scenes100 mixes COCO 1:1 with scene frames during pseudo-label adaptation — [Zhang & Hoai CVPR 2023](https://www3.cs.stonybrook.edu/~minhhoai/papers/Self-Supervised_Scene_Adaptation_CVPR23.pdf)
- Full FT can halve OOD AP on some backbones; decoder-probe and weight regularisation toward pretrained weights mitigate it — [Saito et al.](https://ar5iv.labs.arxiv.org/html/2303.14744)
- YOLOv8n with frozen backbone layers retains COCO mAP within 0.1% — [arXiv 2505.01016](https://arxiv.org/abs/2505.01016)
- The Ultralytics `lr0` is ignored with `optimizer='auto'` — [Ultralytics docs](https://docs.ultralytics.com/modes/train/)

### Inferences
- **Concrete runs** (each YOLO26n fine-tune should fit in well under an hour on a 4060 at imgsz 640 for a few thousand frames; this is an estimate, not measured):
  - **E0.** Existing pretrained and FT endpoints (re-evaluate with the full metric matrix of Section 6).
  - **E1.** WiSE λ ∈ {0, .1, ..., 1}:
    - E1a: raw interpolation.
    - E1b: interpolation + BN recalibration on mixed campus+COCO images.
    - Select λ on camera-1 val + COCO-person val; report camera-2 as test.
  - **E2.** FT with COCO-person replay 1:1 (plus about 10-20% person-free COCO negatives), same epochs/LR as E0-FT.
  - **E3.** FT with `freeze=10` (verify the YOLO26 backbone layer index), `optimizer='SGD'` or `'AdamW'` explicitly, lr0 reduced 3-10×, epochs about 10-20.
  - **E4.** FT with pseudo-labels regenerated at a lower threshold for boxes <64 px (or with low-confidence boxes masked out), plus `scale=0.75`, `multi_scale=0.25`.
  - **E5 (combo).** Best of E2-E4 then WiSE-interpolated with the pretrained model.
- **Seeds.** Run ≥2 seeds for E2-E5 if time allows. Camera-2 differences of a few AP points from one seed are not reliable.
- **Expected direction** (hypothesis, not evidence-backed numbers): E1 recovers a large share of the OOD drop at a small ID cost; E2 mostly fixes the FPs and small-person recall; E3 reduces OOD drop but caps the ID gain.

### Gaps
- No direct published evidence for YOLO26 or RF-DETR on any of these mitigations. Every recommendation transfers from Faster R-CNN/FCOS/RetinaNet/YOLOv8 or classification results.
- Runtime per experiment on an RTX 4060 is estimated, not sourced.
