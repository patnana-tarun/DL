#!/usr/bin/env bash
# Step 3: download the vision-language model (Qwen3-VL GGUF for llama.cpp) and, on the Jetson, MediaMTX arm64.
#
#   scripts/03_download_models.sh               # Qwen3-VL-2B Q4_K_M + mmproj Q8_0 (~1.5 GB), all boards
#   VLM_SIZE=4B scripts/03_download_models.sh   # Qwen3-VL-4B Q4_K_M + mmproj Q8_0 (~2.9 GB), Orin NX 16 GB / AGX
#   MEDIAMTX=0 scripts/03_download_models.sh    # skip MediaMTX
#
# Files already present are skipped; interrupted downloads resume (curl -C -). The detector ONNX models
# (models/yolo26n_person_*.onnx) are part of the folder and are not downloaded.
set -euo pipefail
cd "$(dirname "$0")/.."

SIZE=${VLM_SIZE:-2B}
case "$SIZE" in 2B|4B) ;; *) echo "ERROR: VLM_SIZE must be 2B or 4B (got $SIZE)"; exit 1;; esac
REPO="https://huggingface.co/Qwen/Qwen3-VL-${SIZE}-Instruct-GGUF/resolve/main"
FILES=("Qwen3VL-${SIZE}-Instruct-Q4_K_M.gguf" "mmproj-Qwen3VL-${SIZE}-Instruct-Q8_0.gguf")
MTX_VER=v1.21.1
MTX_URL="https://github.com/bluenviron/mediamtx/releases/download/${MTX_VER}/mediamtx_${MTX_VER}_linux_arm64.tar.gz"

download() {   # url dest
  local url=$1 dest=$2
  if [[ -s "$dest" ]]; then
    echo "  present, skipping: $dest ($(du -h "$dest" | cut -f1))"
    return 0
  fi
  echo "  downloading $url"
  # Download to .part and rename at the end, so a half-finished file is never mistaken for a complete one.
  curl -L --fail --retry 5 --retry-delay 5 -C - -o "$dest.part" "$url"
  mv "$dest.part" "$dest"
  echo "  saved $dest ($(du -h "$dest" | cut -f1))"
}

# ------------------------------------------------------------------------------------------- VLM
echo "==> Qwen3-VL-$SIZE (GGUF) -> models/vlm/"
mkdir -p models/vlm
for f in "${FILES[@]}"; do
  download "$REPO/$f" "models/vlm/$f"
done
if [[ "$SIZE" == 4B ]]; then
  echo "  4B selected: start the server with  VLM_SIZE=4B scripts/start_vlm.sh  and set  vlm.model: qwen3-vl-4b"
fi

# ------------------------------------------------------------------------------------------- detector check
echo "==> detector models"
shopt -s nullglob
ONNX=(models/yolo26n_person_*.onnx)
shopt -u nullglob
if (( ${#ONNX[@]} )); then printf '  ok: %s\n' "${ONNX[@]}"
else echo "  WARNING: no models/yolo26n_person_*.onnx found: copy them from the PC (README step 3)"; fi

# ------------------------------------------------------------------------------------------- MediaMTX
if [[ "${MEDIAMTX:-1}" == 0 ]]; then
  echo "==> MediaMTX skipped (MEDIAMTX=0)"
elif [[ "$(uname -m)" != aarch64 ]]; then
  echo "==> MediaMTX: not an arm64 machine, skipping (the desktop already has tools/bin/mediamtx)"
else
  echo "==> MediaMTX $MTX_VER (arm64) -> tools/bin/mediamtx (local RTSP server for testing, README step 5)"
  mkdir -p tools/bin
  if [[ -x tools/bin/mediamtx ]] && tools/bin/mediamtx --version >/dev/null 2>&1; then
    echo "  present, skipping: tools/bin/mediamtx ($(tools/bin/mediamtx --version 2>&1 | head -n1))"
  else
    if [[ -e tools/bin/mediamtx ]]; then
      echo "  tools/bin/mediamtx does not run here (x86 binary copied from the PC?): renaming it to mediamtx.x86_64"
      mv tools/bin/mediamtx tools/bin/mediamtx.x86_64
    fi
    TMP=$(mktemp -d)
    trap 'rm -rf "$TMP"' EXIT
    curl -L --fail --retry 5 -o "$TMP/mediamtx.tar.gz" "$MTX_URL"
    tar -xzf "$TMP/mediamtx.tar.gz" -C "$TMP" mediamtx
    install -m 0755 "$TMP/mediamtx" tools/bin/mediamtx
    echo "  installed: $(tools/bin/mediamtx --version 2>&1 | head -n1)"
  fi
fi

echo
echo "OK. Next: scripts/04_warmup_detector.sh"
