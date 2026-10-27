#!/usr/bin/env bash
# Rebuild the paper from the current results: numbers -> figures -> PDF.
#   bash paper/build.sh        (from anywhere)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
PY=(env -u PYTHONPATH "$ROOT/.venv/bin/python")
TECTONIC="${TECTONIC:-$HOME/.local/bin/tectonic}"

cd "$HERE"
echo "== numbers"; "${PY[@]}" make_numbers.py
echo "== figures"; "${PY[@]}" make_figures.py
echo "== LaTeX";   "$TECTONIC" -X compile main.tex --keep-logs 2>&1 | tee build.log | grep -E "^(error|warning)|Overfull|Underfull|undefined" || true
test -s main.pdf && echo "== done: $HERE/main.pdf ($(pdfinfo main.pdf 2>/dev/null | awk '/^Pages/{print $2}') pages)"
# supplement.tex is kept for reference; its content is now the appendix of main.tex
