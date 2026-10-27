#!/usr/bin/env bash
# Build one self-contained annotation pack per annotator (tool + protocol + tasks, no labels). See PROTOCOL.md.
#   bash research/annotation/pack.sh [A B]      -> annotation_pack_<X>.zip in the repository root
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$(dirname "$(dirname "$HERE")")"
cd "$HERE"
for who in "${@:-A B}"; do
  for w in $who; do
    rm -f "$OUT/annotation_pack_$w.zip"
    zip -qr "$OUT/annotation_pack_$w.zip" annotator.html serve.py PROTOCOL.md tasks/det_main tasks/track_aa1 tasks/track_test1
    echo "$OUT/annotation_pack_$w.zip ($(du -h "$OUT/annotation_pack_$w.zip" | cut -f1)) -> annotator $w"
  done
done
