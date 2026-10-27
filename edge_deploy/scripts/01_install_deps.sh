#!/usr/bin/env bash
# Step 1: install system packages and the Python environment (edge_deploy/.venv).
#
#   scripts/01_install_deps.sh
#
# Environment overrides:
#   JETSON_PIP_INDEX=https://pypi.jetson-ai-lab.io/jp6/cu126/+simple/   index for onnxruntime-gpu
#   ORT_VERSION=1.24.0          pin an onnxruntime-gpu version (default: newest on the index)
#   FORCE_ORT=1                 reinstall onnxruntime-gpu even if TensorRT is already available
#   SKIP_APT=1                  skip the apt-get part (no sudo needed)
#
# Safe to run again: apt skips installed packages, the venv is reused, pip only installs what is missing.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="/usr/local/cuda/bin:$PATH"

say() { printf '\n==> %s\n' "$*"; }
die() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }

ARCH=$(uname -m)
IS_JETSON=0
[[ -f /etc/nv_tegra_release ]] && IS_JETSON=1
if [[ $IS_JETSON == 0 ]]; then
  echo "NOTE: this is not a Jetson ($ARCH). Installing the desktop requirements instead (requirements-desktop.txt)."
fi

# ------------------------------------------------------------------------------------------- 1. apt
APT_PKGS=(
  ffmpeg                         # decoding of files / RTSP / USB cameras, and the optional re-stream
  python3-pip python3-venv python3-dev
  v4l-utils                      # v4l2-ctl: list USB cameras and their formats
  gstreamer1.0-tools             # gst-launch-1.0: used for CSI cameras (nvarguscamerasrc)
  gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-libav
  cmake build-essential git pkg-config   # building llama.cpp
  curl ca-certificates           # downloads, health checks in 05_run.sh
  libcurl4-openssl-dev libssl-dev        # llama.cpp HTTPS support (LLAMA_OPENSSL); harmless if unused
)
if [[ "${SKIP_APT:-0}" != 1 ]]; then
  say "Installing system packages with apt (asks for your sudo password)"
  sudo apt-get update
  sudo apt-get install -y --no-install-recommends "${APT_PKGS[@]}"
  if [[ $IS_JETSON == 1 ]]; then
    if ! dpkg-query -W -f='${Status}' nvidia-jetpack 2>/dev/null | grep -q "install ok installed"; then
      say "Installing nvidia-jetpack (CUDA, cuDNN, TensorRT; several GB, takes a while)"
      sudo apt-get install -y nvidia-jetpack
    else
      echo "nvidia-jetpack already installed: $(dpkg-query -W -f='${Version}' nvidia-jetpack)"
    fi
  fi
  if [[ -e /dev/video0 ]] && ! id -nG "$(id -un)" | grep -qw video; then
    say "Adding $(id -un) to the 'video' group (camera access). Log out and back in afterwards."
    sudo usermod -aG video "$(id -un)"
  fi
else
  echo "SKIP_APT=1: skipping apt-get"
fi

# ------------------------------------------------------------------------------------------- 2. venv
say "Python virtual environment in $(pwd)/.venv"
if [[ ! -x .venv/bin/python ]]; then
  # --system-site-packages: JetPack's own Python packages stay visible (e.g. the TensorRT bindings);
  # everything in requirements-*.txt is installed inside the venv and takes precedence.
  python3 -m venv --system-site-packages .venv
  echo "created .venv"
else
  echo ".venv already exists, reusing it"
fi
PY=(env -u PYTHONPATH .venv/bin/python)
PIP=(env -u PYTHONPATH .venv/bin/python -m pip)
"${PIP[@]}" install --upgrade pip wheel

if [[ $IS_JETSON == 1 ]]; then
  say "Installing requirements-jetson.txt"
  "${PIP[@]}" install -r requirements-jetson.txt
else
  say "Installing requirements-desktop.txt"
  "${PIP[@]}" install -r requirements-desktop.txt
fi

# ------------------------------------------------------------------------------------------- 3. onnxruntime-gpu
has_trt() {
  "${PY[@]}" -c "import onnxruntime as o, sys; sys.exit(0 if 'TensorrtExecutionProvider' in o.get_available_providers() else 1)" 2>/dev/null
}
if [[ $IS_JETSON == 1 ]]; then
  CUDA_VER=""
  if command -v nvcc >/dev/null; then CUDA_VER=$(nvcc --version | sed -nE 's/.*release ([0-9]+\.[0-9]+).*/\1/p'); fi
  DEFAULT_INDEX=https://pypi.jetson-ai-lab.io/jp6/cu126/+simple/
  INDEX=${JETSON_PIP_INDEX:-$DEFAULT_INDEX}
  if [[ -n "$CUDA_VER" && "$CUDA_VER" != 12.6 && -z "${JETSON_PIP_INDEX:-}" ]]; then
    echo "WARNING: CUDA $CUDA_VER found, but the default onnxruntime-gpu index is built for CUDA 12.6 (JetPack 6.1/6.2)."
    echo "         Upgrade to JetPack 6.2 (recommended), or set JETSON_PIP_INDEX to the matching index from"
    echo "         https://pypi.jetson-ai-lab.io/ (e.g. .../jp6/cu128/+simple/)."
  fi
  if has_trt && [[ "${FORCE_ORT:-0}" != 1 ]]; then
    say "onnxruntime-gpu with TensorRT is already installed (FORCE_ORT=1 to reinstall)"
  else
    say "Installing onnxruntime-gpu from $INDEX"
    # The CPU-only 'onnxruntime' package provides the same module and would shadow the GPU build.
    if "${PIP[@]}" show onnxruntime >/dev/null 2>&1; then
      echo "removing the CPU-only 'onnxruntime' package from the venv (it conflicts with onnxruntime-gpu)"
      "${PIP[@]}" uninstall -y onnxruntime
    fi
    # --index-url (not --extra-index-url): the Jetson AI Lab index mirrors PyPI for the dependencies, and this
    # guarantees pip cannot pick a non-Jetson onnxruntime-gpu build from PyPI.
    "${PIP[@]}" install --index-url "$INDEX" "onnxruntime-gpu${ORT_VERSION:+==$ORT_VERSION}"
  fi
fi

# ------------------------------------------------------------------------------------------- 4. verify
say "Verifying the installation"
"${PY[@]}" - <<'EOF'
import importlib, sys
ok = True
for m in ("numpy", "cv2", "yaml", "scipy", "supervision", "trackers", "onnxruntime"):
    try:
        mod = importlib.import_module(m)
        print(f"  {m:12s} {getattr(mod, '__version__', '?')}  ({getattr(mod, '__file__', '')})")
    except Exception as e:  # noqa: BLE001
        ok = False
        print(f"  {m:12s} FAILED: {type(e).__name__}: {e}")
import onnxruntime as ort
print("  onnxruntime providers:", ort.get_available_providers())
try:
    import edgevision.pipeline  # noqa: F401  (imports every runtime dependency)
    print("  edgevision   import ok")
except Exception as e:  # noqa: BLE001
    ok = False
    print(f"  edgevision   FAILED: {type(e).__name__}: {e}")
sys.exit(0 if ok else 1)
EOF

for t in ffmpeg ffprobe; do command -v "$t" >/dev/null || die "$t not found on PATH"; done

if [[ $IS_JETSON == 1 ]]; then
  if has_trt; then
    echo
    echo "OK: TensorrtExecutionProvider is available. Next: scripts/02_build_llamacpp.sh"
  else
    echo
    echo "WARNING: TensorrtExecutionProvider is NOT available. The detector will fall back to CUDA or CPU (slow)."
    echo "         Check that nvidia-jetpack (TensorRT 10, cuDNN 9) is installed and that JETSON_PIP_INDEX matches"
    echo "         your JetPack/CUDA version, then run again with FORCE_ORT=1. See README step 10."
    exit 1
  fi
else
  echo
  echo "OK (desktop). Next: scripts/start_vlm.sh, then python -m edgevision --config configs/desktop.yaml"
fi
