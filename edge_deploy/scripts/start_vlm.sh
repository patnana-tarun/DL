#!/usr/bin/env bash
# Start the Qwen3-VL server (llama.cpp llama-server, OpenAI-compatible API on http://127.0.0.1:8080/v1).
#
#   scripts/start_vlm.sh                 # Qwen3-VL-2B Q4_K_M (default; fits a Jetson Orin Nano 8 GB)
#   VLM_SIZE=4B scripts/start_vlm.sh     # Qwen3-VL-4B (Orin NX 16 GB / AGX Orin)
#   CTX=4096 PORT=8081 scripts/start_vlm.sh
#
# Finds llama-server in: $LLAMA_SERVER, third_party/llama.cpp/build/bin (Jetson, built by
# scripts/02_build_llamacpp.sh), third_party/llama.cpp-vulkan-x64 (desktop prebuilt), or PATH.
set -euo pipefail
cd "$(dirname "$0")/.."
SIZE=${VLM_SIZE:-2B}
MODEL=${VLM_MODEL:-models/vlm/Qwen3VL-${SIZE}-Instruct-Q4_K_M.gguf}
MMPROJ=${VLM_MMPROJ:-models/vlm/mmproj-Qwen3VL-${SIZE}-Instruct-Q8_0.gguf}
CTX=${CTX:-8192}
PORT=${PORT:-8080}
HOST=${HOST:-127.0.0.1}

BIN=${LLAMA_SERVER:-}
for c in third_party/llama.cpp/build/bin/llama-server third_party/llama.cpp-vulkan-x64/llama-server "$(command -v llama-server || true)"; do
  [[ -z "$BIN" && -n "$c" && -x "$c" ]] && BIN=$c
done
[[ -n "$BIN" ]] || { echo "llama-server not found: run scripts/02_build_llamacpp.sh (Jetson) or see README"; exit 1; }
[[ -f "$MODEL" && -f "$MMPROJ" ]] || { echo "model files missing: run scripts/03_download_models.sh"; exit 1; }
export LD_LIBRARY_PATH="$(dirname "$BIN"):${LD_LIBRARY_PATH:-}"

EXTRA=()
if [[ "$BIN" == *vulkan* ]]; then          # desktop: pick the discrete NVIDIA GPU, not the iGPU
  DEV=$("$BIN" --list-devices 2>&1 | grep -m1 -i nvidia | sed -E 's/^ *([A-Za-z]+[0-9]+):.*/\1/' || true)
  [[ -n "$DEV" ]] && EXTRA+=(--device "$DEV")
fi

echo "llama-server: $BIN"
echo "model: $MODEL  mmproj: $MMPROJ  ctx: $CTX  -> http://$HOST:$PORT/v1"
exec "$BIN" -m "$MODEL" --mmproj "$MMPROJ" -ngl 99 -c "$CTX" --host "$HOST" --port "$PORT" \
  --jinja -a "qwen3-vl-$(echo "$SIZE" | tr 'A-Z' 'a-z')" --no-ui -fa auto "${EXTRA[@]}" "$@"
