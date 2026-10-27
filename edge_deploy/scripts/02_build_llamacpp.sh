#!/usr/bin/env bash
# Step 2: build llama.cpp (llama-server + llama-mtmd-cli) with CUDA for the Jetson Orin GPU.
#
#   scripts/02_build_llamacpp.sh
#
# Result: third_party/llama.cpp/build/bin/llama-server, which scripts/start_vlm.sh finds automatically.
# Takes roughly 20-60 minutes on an Orin Nano (expectation, not measured; the CUDA kernels dominate).
#
# Environment overrides:
#   LLAMA_REF=b1234        build a specific llama.cpp tag/branch instead of the newest master
#   UPDATE=1               git pull an existing checkout, then rebuild
#   REBUILD=1              rebuild even if llama-server already exists
#   JOBS=4                 parallel compile jobs (default: nproc, capped at 4 on boards with < 10 GB RAM)
#   CUDA_ARCH=87           GPU architecture; 87 = Jetson Orin (all Orin modules)
#
# Build flags (checked against llama.cpp master, docs/build.md and CMakeLists.txt):
#   -DGGML_CUDA=ON                 CUDA backend (the old LLAMA_CUBLAS / LLAMA_CUDA names are removed / deprecated)
#   -DCMAKE_CUDA_ARCHITECTURES=87  only compile kernels for Orin (much faster build)
#   -DLLAMA_OPENSSL=ON|OFF         HTTPS for model downloads (-hf). LLAMA_CURL is deprecated in current llama.cpp
#                                  (it only prints a warning), so it is not passed. We download models with curl.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="/usr/local/cuda/bin:$PATH"

SRC=third_party/llama.cpp
BUILD=$SRC/build
BIN=$BUILD/bin/llama-server
CUDA_ARCH=${CUDA_ARCH:-87}

if [[ -x "$BIN" && "${REBUILD:-0}" != 1 && "${UPDATE:-0}" != 1 ]]; then
  echo "llama-server already built: $BIN"
  "$BIN" --version 2>&1 | head -n 3 || true
  echo "(REBUILD=1 to rebuild, UPDATE=1 to pull the newest llama.cpp and rebuild)"
  exit 0
fi

command -v nvcc >/dev/null || {
  echo "ERROR: nvcc (CUDA compiler) not found in /usr/local/cuda/bin."
  echo "       Run scripts/01_install_deps.sh (installs nvidia-jetpack) or: sudo apt install nvidia-jetpack"
  exit 1
}
for t in git cmake g++; do command -v $t >/dev/null || { echo "ERROR: $t missing: run scripts/01_install_deps.sh"; exit 1; }; done
echo "CUDA: $(nvcc --version | sed -nE 's/.*release ([0-9.]+).*/\1/p'), arch sm_$CUDA_ARCH, cmake $(cmake --version | head -n1 | awk '{print $3}')"

# ------------------------------------------------------------------------------------------- source
mkdir -p third_party
if [[ -d $SRC/.git ]]; then
  echo "llama.cpp checkout exists: $SRC ($(git -C $SRC log -1 --format='%h %cs'))"
  if [[ "${UPDATE:-0}" == 1 ]]; then
    echo "updating..."
    git -C $SRC pull --ff-only
  fi
else
  echo "cloning llama.cpp into $SRC"
  if [[ -n "${LLAMA_REF:-}" ]]; then
    git clone --depth 1 --branch "$LLAMA_REF" https://github.com/ggml-org/llama.cpp "$SRC"
  else
    git clone --depth 1 https://github.com/ggml-org/llama.cpp "$SRC"
  fi
fi

# ------------------------------------------------------------------------------------------- configure
MEM_KB=$(awk '/MemTotal/ {print $2}' /proc/meminfo)
JOBS_DEFAULT=$(nproc)
(( MEM_KB < 10000000 && JOBS_DEFAULT > 4 )) && JOBS_DEFAULT=4      # nvcc needs ~1-1.5 GB per job
JOBS=${JOBS:-$JOBS_DEFAULT}
SSL=OFF
if pkg-config --exists openssl 2>/dev/null || [[ -f /usr/include/openssl/ssl.h ]]; then SSL=ON; fi

echo "configuring (jobs=$JOBS, openssl=$SSL)"
cmake -S "$SRC" -B "$BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
  -DLLAMA_OPENSSL=$SSL \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF

# ------------------------------------------------------------------------------------------- build
echo "building llama-server and llama-mtmd-cli (this is the slow part; watch memory with jtop)"
cmake --build "$BUILD" --config Release -j "$JOBS" --target llama-server llama-mtmd-cli

[[ -x "$BIN" ]] || { echo "ERROR: build finished but $BIN is missing"; exit 1; }
echo
"$BIN" --version 2>&1 | head -n 3 || true
echo
echo "OK: $BIN"
echo "Next: scripts/03_download_models.sh"
