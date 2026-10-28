#!/usr/bin/env bash
# Pack the files that git does not carry (videos, datasets, trained weights) into one archive for a teammate.
#
#   bash scripts/pack_data.sh              # writes MINOR_DL_data.tar (about 6.5 GB) in the repository root
#
# Share the archive privately (Google Drive, a USB disk). Do NOT upload it to GitHub or any public place:
# the videos show people and their licences do not allow redistribution.
#
# The teammate unpacks it in the root of their clone and checks it:
#   tar -xf MINOR_DL_data.tar && sha256sum -c --quiet --ignore-missing MANIFEST.sha256
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=${1:-MINOR_DL_data.tar}
PATHS=(videos datasets runs weights models/pretrained)
for p in "${PATHS[@]}"; do
  [[ -e "$p" ]] || { echo "missing: $p" >&2; exit 1; }
done

echo "packing: ${PATHS[*]}"
du -shc "${PATHS[@]}" | tail -1
# no compression: the videos are already compressed and the rest is mostly JPEG frames
tar -cf "$OUT" --exclude='__pycache__' "${PATHS[@]}"
sha256sum "$OUT" > "$OUT.sha256"
echo "wrote $OUT ($(du -h "$OUT" | cut -f1)) and $OUT.sha256"
