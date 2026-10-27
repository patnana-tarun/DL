#!/usr/bin/env bash
# Step 0: check the Jetson before installing anything. Read-only: changes nothing, safe to run any time.
#
#   scripts/00_check_system.sh
#
# Prints the board, JetPack / L4T version, CUDA, cuDNN, TensorRT, Python, memory, disk, power mode,
# cameras, tools and the state of this folder (venv, models, llama.cpp build), and warns about problems.
set -euo pipefail
cd "$(dirname "$0")/.."

WARN=0
ok()   { printf '  [ ok ] %s\n' "$*"; }
warn() { printf '  [WARN] %s\n' "$*"; WARN=$((WARN + 1)); }
info() { printf '  [info] %s\n' "$*"; }
hdr()  { printf '\n== %s\n' "$*"; }
has()  { command -v "$1" >/dev/null 2>&1; }
pkgver() { dpkg-query -W -f='${Version}' "$1" 2>/dev/null || true; }

# ------------------------------------------------------------------------------------------- board
hdr "Board"
ARCH=$(uname -m)
info "architecture: $ARCH, kernel $(uname -r)"
IS_JETSON=0
if [[ -r /proc/device-tree/model ]]; then
  MODEL=$(tr -d '\0' </proc/device-tree/model)
  info "model: $MODEL"
fi
if [[ -f /etc/nv_tegra_release ]]; then
  IS_JETSON=1
  info "L4T: $(head -n1 /etc/nv_tegra_release)"
  L4T_MAJOR=$(sed -nE 's/^# R([0-9]+).*/\1/p' /etc/nv_tegra_release | head -n1)
  if [[ "${L4T_MAJOR:-0}" == "36" ]]; then
    ok "L4T R36 = JetPack 6.x (supported)"
  elif [[ "${L4T_MAJOR:-0}" -lt 36 ]]; then
    warn "L4T R${L4T_MAJOR} is JetPack 5 or older: this guide needs JetPack 6.x (README step 1)"
  else
    warn "L4T R${L4T_MAJOR} is newer than JetPack 6: not tested (pip index / CUDA version in 01_install_deps.sh may differ)"
  fi
else
  warn "/etc/nv_tegra_release not found: this is not a Jetson (fine for a desktop test, but CUDA/TensorRT checks will fail)"
fi
JP=$(pkgver nvidia-jetpack)
if [[ -n "$JP" ]]; then ok "nvidia-jetpack package: $JP"
elif [[ $IS_JETSON == 1 ]]; then warn "nvidia-jetpack is not installed (CUDA/cuDNN/TensorRT may be missing): scripts/01_install_deps.sh installs it"
fi
L4T_CORE=$(pkgver nvidia-l4t-core)
[[ -n "$L4T_CORE" ]] && info "nvidia-l4t-core: $L4T_CORE"

# ------------------------------------------------------------------------------------------- CUDA / TensorRT
hdr "CUDA / cuDNN / TensorRT"
NVCC=""
for c in /usr/local/cuda/bin/nvcc "$(command -v nvcc || true)"; do
  [[ -z "$NVCC" && -n "$c" && -x "$c" ]] && NVCC=$c
done
if [[ -n "$NVCC" ]]; then
  ok "CUDA: $("$NVCC" --version | sed -nE 's/.*release ([0-9.]+).*/\1/p') ($NVCC)"
  case ":$PATH:" in *:/usr/local/cuda/bin:*) ;; *) info "/usr/local/cuda/bin is not on PATH (the scripts add it themselves)";; esac
elif [[ -f /usr/local/cuda/version.json ]]; then
  ok "CUDA runtime: $(grep -m1 -A2 '"cuda"' /usr/local/cuda/version.json | sed -nE 's/.*"version" *: *"([^"]+)".*/\1/p') (nvcc missing: install cuda-toolkit to build llama.cpp)"
else
  warn "CUDA not found in /usr/local/cuda (needed by onnxruntime-gpu and llama.cpp)"
fi
CUDNN=$(dpkg-query -W -f='${Package} ${Version}\n' 'libcudnn*' 2>/dev/null | grep -E '^libcudnn[0-9]+ ' | head -n1 || true)
if [[ -n "$CUDNN" ]]; then ok "cuDNN: $CUDNN"; elif [[ $IS_JETSON == 1 ]]; then warn "cuDNN not found (dpkg libcudnn*)"; fi
TRT=$(pkgver tensorrt)
[[ -z "$TRT" ]] && TRT=$(dpkg-query -W -f='${Version}\n' 'libnvinfer[0-9]*' 2>/dev/null | head -n1 || true)
if [[ -n "$TRT" ]]; then ok "TensorRT: $TRT"; elif [[ $IS_JETSON == 1 ]]; then warn "TensorRT not found (dpkg tensorrt / libnvinfer*)"; fi

# ------------------------------------------------------------------------------------------- Python
hdr "Python"
if has python3; then
  PYV=$(python3 -c 'import sys; print("%d.%d.%d" % sys.version_info[:3])')
  ok "python3 $PYV ($(command -v python3))"
  [[ $IS_JETSON == 1 && "$PYV" != 3.10.* ]] && warn "JetPack 6 ships Python 3.10; the onnxruntime-gpu wheels are built for cp310"
else
  warn "python3 not found"
fi
[[ -n "${PYTHONPATH:-}" ]] && info "PYTHONPATH is set (e.g. by ROS); the scripts unset it so the venv stays clean"
if [[ -x .venv/bin/python ]]; then
  ok "venv: .venv exists"
  env -u PYTHONPATH .venv/bin/python - <<'EOF' || true
try:
    import onnxruntime as ort
    p = ort.get_available_providers()
    print(f"  [{' ok ' if 'TensorrtExecutionProvider' in p else 'WARN'}] onnxruntime {ort.__version__}: {p}")
except Exception as e:  # noqa: BLE001
    print(f"  [WARN] onnxruntime not importable in .venv: {e}")
EOF
else
  info "venv .venv not created yet (scripts/01_install_deps.sh)"
fi

# ------------------------------------------------------------------------------------------- memory / disk / power
hdr "Memory, disk, power"
MEM_KB=$(awk '/MemTotal/ {print $2}' /proc/meminfo)
AVAIL_KB=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
SWAP_KB=$(awk '/SwapTotal/ {print $2}' /proc/meminfo)
info "RAM total $((MEM_KB / 1024)) MiB, available $((AVAIL_KB / 1024)) MiB, swap $((SWAP_KB / 1024)) MiB"
if (( MEM_KB < 7000000 )); then
  warn "less than 8 GB RAM: use the 2B VLM with CTX=2048..4096, or disable the VLM (vlm.enabled=false)"
fi
if (( SWAP_KB < 4000000 )); then
  warn "less than 4 GB swap/zram: add a swap file before building llama.cpp (README step 2)"
fi
if [[ -r /proc/swaps ]] && grep -q zram /proc/swaps; then info "zram swap is active"; fi
DISK_KB=$(df -Pk . | awk 'NR==2 {print $4}')
info "free disk in $(pwd): $((DISK_KB / 1024 / 1024)) GiB ($(df -P . | awk 'NR==2 {print $1}'))"
(( DISK_KB < 10000000 )) && warn "less than 10 GB free: llama.cpp build + models + TensorRT cache need ~6-8 GB (an NVMe SSD is recommended)"
if has nvpmodel; then
  PM=$(nvpmodel -q 2>/dev/null | grep -i 'power mode' | head -n1 || true)
  [[ -n "$PM" ]] && info "$PM (set the fastest mode with: sudo nvpmodel -m <id>, README step 2)"
fi
if has jtop; then ok "jtop installed (run: jtop)"; else info "jtop not installed (sudo pip3 install -U jetson-stats)"; fi

# ------------------------------------------------------------------------------------------- cameras
hdr "Cameras"
shopt -s nullglob
VIDS=(/dev/video*)
shopt -u nullglob
if (( ${#VIDS[@]} )); then
  info "video devices: ${VIDS[*]}"
  if has v4l2-ctl; then v4l2-ctl --list-devices 2>/dev/null | sed 's/^/    /' || true
  else info "v4l2-ctl not installed yet (v4l-utils, scripts/01_install_deps.sh)"; fi
  id -nG | grep -qw video || warn "user $(id -un) is not in the 'video' group: sudo usermod -aG video $(id -un), then log out/in"
else
  info "no /dev/video* devices (fine for files / RTSP; plug in a USB camera or connect a CSI camera)"
fi
if has systemctl && systemctl list-unit-files 2>/dev/null | grep -q '^nvargus-daemon'; then
  if systemctl is-active --quiet nvargus-daemon; then ok "nvargus-daemon running (needed for CSI cameras: source.uri csi://0)"
  else warn "nvargus-daemon not running (CSI cameras): sudo systemctl restart nvargus-daemon"; fi
fi

# ------------------------------------------------------------------------------------------- tools
hdr "Tools"
for t in ffmpeg ffprobe gst-launch-1.0 cmake git curl gcc; do
  if has "$t"; then ok "$t"; else warn "$t missing (scripts/01_install_deps.sh)"; fi
done

# ------------------------------------------------------------------------------------------- this folder
hdr "EdgeVision folder ($(pwd))"
shopt -s nullglob
ONNX=(models/yolo26n_person_*.onnx)
shopt -u nullglob
if (( ${#ONNX[@]} )); then ok "detector models: ${ONNX[*]}"
else warn "no models/yolo26n_person_*.onnx: copy the whole edge_deploy folder (README step 3)"; fi
if [[ -x third_party/llama.cpp/build/bin/llama-server ]]; then ok "llama-server built (third_party/llama.cpp/build/bin)"
else info "llama.cpp not built yet (scripts/02_build_llamacpp.sh)"; fi
for s in 2B 4B; do
  m=models/vlm/Qwen3VL-$s-Instruct-Q4_K_M.gguf; p=models/vlm/mmproj-Qwen3VL-$s-Instruct-Q8_0.gguf
  if [[ -f $m && -f $p ]]; then ok "VLM $s files present"; fi
done
[[ -f models/vlm/Qwen3VL-2B-Instruct-Q4_K_M.gguf ]] || info "VLM files not downloaded yet (scripts/03_download_models.sh)"
if [[ $ARCH == aarch64 && -d third_party/llama.cpp-vulkan-x64 ]]; then
  warn "third_party/llama.cpp-vulkan-x64 is an x86 desktop build and cannot run here (exclude it when copying)"
fi
if [[ -x tools/bin/mediamtx ]]; then
  if tools/bin/mediamtx --version >/dev/null 2>&1; then ok "mediamtx: $(tools/bin/mediamtx --version 2>&1 | head -n1)"
  else warn "tools/bin/mediamtx does not run on $ARCH (x86 binary?): scripts/03_download_models.sh fetches the arm64 one"; fi
fi
if has ss; then
  for port in 8000 8080 8554; do
    ss -ltnH "sport = :$port" 2>/dev/null | grep -q . && info "port $port is in use ($(ss -ltnpH "sport = :$port" 2>/dev/null | grep -oE 'users:\(\("[^"]+"' | head -n1 | cut -d'"' -f2 || true))"
  done
fi

hdr "Summary"
if (( WARN )); then echo "  $WARN warning(s) above. Fix them (see README step 10, troubleshooting) or continue if they do not apply."
else echo "  All checks passed. Next: scripts/01_install_deps.sh"; fi
exit 0
