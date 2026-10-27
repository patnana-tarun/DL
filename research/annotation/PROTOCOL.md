# Human ground-truth protocol (review items 1 and 7)

Goal: a small, **independently human-annotated** evaluation set, so that detection, tracking and counting can be
scored against people rather than against the teacher ensemble, and so the teacher labels themselves can be
audited. The earlier audit used an AI assistant on crops (`results/test1/audit/`); this replaces it.

## Independence rules (the reason this set exists)

1. **Blind, from scratch.** Annotators never see teacher pseudo-labels, model detections or each other's work.
   `annotator.html` shows the raw frame only. (Editing machine boxes anchors annotators to the machine:
   Berzak et al., EMNLP 2016; MOT16 ran a detector only *after* manual labelling.)
2. **Two annotators per frame**, working separately (`?annotator=A` / `?annotator=B`). A third person adjudicates
   only the disagreements (`human_gt.py adjudicate`). Agreement is reported **before** adjudication.
3. Frames are chosen by `sample.py` with a fixed seed before anyone annotates. Strata use the pseudo-labels only to
   *choose* frames (density, size), never to draw boxes.
4. `test1.mp4` stays test-only: its human labels are used for evaluation, never for training or tuning.

## What to draw (MOT16 / Caltech / CityPersons conventions)

- **Every person**, including partly visible, sitting, cycling (the person, not the bike), and people inside
  vehicles only if clearly visible through glass (mark *ignore*).
- **Full-extent box**: the whole body as it would appear without the occluder; estimate the hidden part from body
  proportions and neighbouring frames (MOT16). Boxes stop at the image border (do not extrapolate outside).
- **Occlusion** (key `O`): 0 = ≤ 35 % hidden, 1 = 35–80 % hidden, 2 = > 80 % hidden.
- **Ignore** (key `I`): people shorter than 20 px, crowds where individuals cannot be separated, reflections,
  posters / mannequins / statues of people, people in vehicles. Detections overlapping ignore boxes are neither
  rewarded nor penalised (Caltech semantics).
- When unsure whether something is a person, zoom in (mouse wheel). If still unsure: *ignore*.
- Tracking clips (`track_*` tasks): give each person **one ID for as long as their path is unambiguous**; after
  leaving the view, or an occlusion after which you cannot be sure it is the same person, use a new ID (MOT16).
  `C` copies the previous frame's boxes and IDs; then move them.

## Tool

```
env -u PYTHONPATH .venv/bin/python research/annotation/serve.py          # http://127.0.0.1:8765
open  http://127.0.0.1:8765/?task=det_main&annotator=A
```
Mouse: drag on empty space = new box; drag inside a box = move; drag a corner = resize; wheel = zoom; right-drag =
pan. Keys: `N`/`→` next, `P`/`←` previous, `Del` delete, `I` ignore, `O` occlusion, `T` set track ID, `C` copy
previous frame, `F` fit view, `H` hide boxes. Work is saved on every frame change to
`research/annotation/labels/<annotator>/<task>.json`.

Budget: ~30–60 s per detection frame (5–15 people), ~2–4 h per annotator for the default 240-frame detection task;
a 30 s tracking clip at 5 fps (150 frames) takes roughly 2–3 h per annotator with copy-forward.

## Running it on an annotator's own laptop

The tool needs only Python 3 (standard library) and a browser, so each annotator can work on their own machine,
which also keeps them blind to each other.

1. Organiser: `bash research/annotation/pack.sh` -> `annotation_pack_A.zip`, `annotation_pack_B.zip` (tool, this
   protocol, the three tasks; no labels, no machine boxes). Give one to each annotator.
2. Annotator: unzip, open a terminal in the folder, run `python3 serve.py` (Windows: `python serve.py`), open the
   printed link with your letter, e.g. `http://127.0.0.1:8765/?task=det_main&annotator=A`. Do `det_main` first,
   then `track_aa1`, then `track_test1` (change `task=` in the address). Work saves itself on every frame change;
   you can close the browser and continue later, it reopens at the first unlabelled frame.
3. Annotator: when done, send back the folder `labels/<your letter>/` (one JSON per task). Do not discuss frames
   with the other annotator until both have sent their labels.
4. Organiser: copy each `labels/<letter>/` into `research/annotation/labels/` of the repository.

## Adjudication (third person, detection task only)

`human_gt.py adjudicate --task det_main` (step below) writes `tasks/det_main_adj/`: only the frames where A and B
disagree. The adjudicator (C, who did not annotate A or B) labels those frames blind and from scratch with the same
rules (`?task=det_main_adj&annotator=C`), then `human_gt.py adjudicate --task det_main --adjudicated C` keeps every
disputed box that C also drew and turns the rest into ignore regions. Tracking clips are not adjudicated; their A
vs B agreement is the result.

## After annotation

```
python research/annotation/human_gt.py agreement  --task det_main      # A vs B: matched F1, IoU, per stratum
python research/annotation/human_gt.py adjudicate --task det_main      # merged GT, disagreements, det_main_adj task
# ... C labels det_main_adj ...
python research/annotation/human_gt.py adjudicate --task det_main --adjudicated C
python research/annotation/human_gt.py eval       --task det_main      # every model + teacher vs human GT
python research/annotation/human_gt.py agreement  --task track_aa1     # tracking clip: HOTA / IDF1, A vs B
python research/annotation/human_gt.py agreement  --task track_test1
```
Reported in the paper (`\ifHaveHuman`): inter-annotator matched F1 / mean IoU (the human ceiling), teacher
precision / recall against humans by size, every student's AP against human GT, and the Kendall τ between the
model ranking under pseudo-labels and under human labels. For tracking clips: A-vs-B HOTA / IDF1 (a human ceiling
that, to our knowledge, has not been published for pedestrian MOT), and line-crossing precision / recall with
±0.5 / 1 / 2 s tolerance.
