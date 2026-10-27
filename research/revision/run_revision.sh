#!/usr/bin/env bash
# Revision experiments (scene adaptation, resolution, multi-scene), in dependency order. GPU jobs run one at a time.
#   research/revision/run_revision.sh            # everything not yet done (each step skips finished outputs)
# Needs: datasets/scene_v{1,2}/teachers.json (research/heldout/heldout_detect.py on videos/video_{1,2}.mp4),
#        datasets/coco2017_hf/val-*.parquet (COCO val2017, Hugging Face mirror detection-datasets/coco).
# Logs: results/logs/revision/<step>.log
set -euo pipefail
cd "$(dirname "$0")/../.."
PY="env -u PYTHONPATH .venv/bin/python"
LOG=results/logs/revision; mkdir -p "$LOG"
# a failed step is reported and skipped, so one bad experiment does not stop the overnight queue
step() { local name=$1; shift; echo "== $(date +%T) $name"; "$@" > "$LOG/$name.log" 2>&1 || echo "!! $name FAILED (see $LOG/$name.log)"; }

for v in 1 2; do
  [[ -f datasets/scene_v$v/teachers.json && -f results/scene_v$v/detections_full.npz ]] || \
    step detect_scene_v$v $PY research/heldout/heldout_detect.py --video videos/video_$v.mp4 --name scene_v$v
done
step scene_splits   $PY research/revision/make_scene_splits.py scene_v1 scene_v2
[[ -f research/annotation/tasks/det_main/manifest.json ]] || step annotation_sample $PY research/annotation/sample.py
[[ -f datasets/coco_person_val/stats.json ]] || step coco_person $PY research/revision/prepare_coco_person.py
[[ -f results/revision/label_threshold.json ]] || step aa1_ignore $PY research/revision/aa1_ignore.py

# GPU jobs below need the whole 8 GB: wait while a VLM evaluation holds the lock (see vlm_safety_eval.py)
while [[ -f results/logs/revision/vlm.lock ]]; do sleep 30; done
for p in $(pgrep -x llama-server); do kill "$p"; done; sleep 5          # free the GPU memory held by the VLM
step train_ft80     $PY research/revision/train_variants.py ft80
step wise_ft80      $PY research/revision/wise.py --ft runs/revision/ft80/weights/best.pt --tag ft80
step wise_aa1_1cls  $PY research/revision/wise.py --ft runs/yolo26n/weights/best.pt --tag aa1-1cls --bn interp
step eval_base      $PY research/revision/multiscene_eval.py --set base
step eval_wise      $PY research/revision/multiscene_eval.py --set wise
for v in replay multiscene freeze res640 ignore v1 v2; do
  step train_$v     $PY research/revision/train_variants.py $v
done
step eval_variants  $PY research/revision/multiscene_eval.py --set variants
step wise_replay    $PY research/revision/wise.py --ft runs/revision/replay/weights/best.pt --tag replay
step wise_multi     $PY research/revision/wise.py --ft runs/revision/multiscene/weights/best.pt --tag multiscene
step eval_wise2     $PY research/revision/multiscene_eval.py --set wise
step eval_ressweep  $PY research/revision/multiscene_eval.py --set ressweep
# longer multi-scene tracking (frozen aa1 parameters) on the two campus videos; counting line fixed in advance:
# vertical centre line (people walk across the view in both videos)
for v in 1 2; do
  [[ -f results/scene_v$v/tracking.json ]] || step track_scene_v$v $PY research/heldout/heldout_track.py \
      pretrained bytetrack trackers analytics --name scene_v$v --video videos/video_$v.mp4 --line 0.5,0.0,0.5,1.0
done
# desktop contention benchmark (detector on the GPU + VLM on the same GPU), 10 min per mode
(cd edge_deploy && nohup scripts/start_vlm.sh > ../$LOG/llama_bench.log 2>&1 &)
for i in $(seq 1 60); do curl -s localhost:8080/health | grep -q ok && break; sleep 5; done
step bench_desktop  bash -c "cd edge_deploy && env -u PYTHONPATH ../.venv/bin/python tools/bench.py --config configs/desktop.yaml \
  --video ../videos/aa1.mp4 --duration 600 --warmup 30 --set detector.backend=ultralytics \
  --set detector.weights=../runs/yolo26n/weights/best.pt --out ../results/revision/bench_desktop.json"
for p in $(pgrep -x llama-server); do kill "$p"; done
echo "== done $(date +%T)"
