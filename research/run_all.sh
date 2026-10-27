#!/usr/bin/env bash
# Reproduce the whole aa1 study + campus case study, in dependency order.
#   research/run_all.sh            # evaluation, tracking, analytics, figures (uses the trained weights)
#   research/run_all.sh --train    # also rebuild the dataset and retrain both students (hours)
#   research/run_all.sh --heldout  # also the test-only evaluation on videos/test1.mp4 (~1.5 h, research/heldout/)
# Logs go to results/logs/<step>.log. Run from anywhere; uses the project venv.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="env -u PYTHONPATH .venv/bin/python"     # keep ROS / system site-packages out of the venv
mkdir -p results/logs
step() { local name=$1; shift; echo "== $(date +%T) $name"; "$@" > "results/logs/$name.log" 2>&1; }

if [[ "${1:-}" == "--train" ]]; then
  step prepare_data   $PY research/prepare_data.py
  step train_yolo     $PY research/train_yolo.py
  step train_rfdetr   $PY research/train_rfdetr.py
fi
step evaluate        $PY research/evaluate.py          # detection metrics, CIs, significance, overfitting
step crosscheck_det  $PY research/crosscheck.py detection
step latency         $PY research/benchmark_latency.py
step teacher_bias    $PY research/teacher_bias.py
step track           $PY research/track.py detect tune render
step track_compare   $PY research/track_compare.py
step crosscheck_trk  $PY research/crosscheck.py tracking
step analytics_yolo  $PY research/analytics.py --model YOLO26n
step analytics_rf    $PY research/analytics.py --model "RF-DETR Nano"
step figures         $PY research/visualize.py
step select_model    $PY research/select_model.py      # picks the deploy model, exports + verifies ONNX
# campus case study (tracks + features are cached in results/campus/<video>/)
step campus_probe    $PY research/campus/campus_probe.py
step campus_report_1 $PY research/campus/campus_report.py --name video_1 --video videos/video_1.mp4
step campus_report_2 $PY research/campus/campus_report.py --name video_2 --video videos/video_2.mp4
if [[ " $* " == *" --heldout "* ]]; then
  research/heldout/run_heldout.sh videos/test1.mp4 test1      # logs in results/test1/logs/
fi
echo "== done"
