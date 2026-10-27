#!/usr/bin/env bash
# Evaluate the aa1-trained models on a held-out video (test only: no training, tuning or threshold search).
#   research/heldout/run_heldout.sh videos/test1.mp4 test1
# Outputs: datasets/<name>/ (frames + teacher labels), results/<name>/ (JSON, figures, logs).
# The audit verdicts (results/<name>/audit/verdicts.tsv) are filled by hand between `audit sheets` and `audit score`.
set -euo pipefail
cd "$(dirname "$0")/../.."
VIDEO=${1:?video}; NAME=${2:?name}
PY="env -u PYTHONPATH .venv/bin/python"
LOG=results/$NAME/logs; mkdir -p "$LOG"
step() { local name=$1; shift; echo "== $(date +%T) $name"; "$@" > "$LOG/$name.log" 2>&1; }

step heldout_detect   $PY research/heldout/heldout_detect.py --video "$VIDEO" --name "$NAME"   # ~1 h for 5 min of 1080p
step heldout_eval     $PY research/heldout/heldout_eval.py --name "$NAME"
step heldout_track    $PY research/heldout/heldout_track.py pretrained bytetrack trackers analytics --name "$NAME" --video "$VIDEO"
step heldout_edge     $PY research/heldout/heldout_edge.py --name "$NAME" --video "$VIDEO"
step heldout_resize   $PY research/heldout/heldout_resize.py --name "$NAME"
step heldout_audit    $PY research/heldout/heldout_audit.py sheets --name "$NAME"
if grep -qP '^\d+\t\S' "results/$NAME/audit/verdicts.tsv"; then
  step heldout_audit_score $PY research/heldout/heldout_audit.py score --name "$NAME"
fi
step heldout_figures  $PY research/heldout/heldout_figures.py --name "$NAME" --video "$VIDEO"
echo "== done"
