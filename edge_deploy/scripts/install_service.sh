#!/usr/bin/env bash
# Step 8 (optional): start EdgeVision automatically at boot with systemd.
#
#   scripts/install_service.sh                      # install + enable + start both services
#   CONFIG=configs/jetson.yaml VLM_SIZE=2B CTX=4096 scripts/install_service.sh
#   NO_VLM=1 scripts/install_service.sh             # pipeline only (set vlm.enabled: false in the config too)
#   START=0 scripts/install_service.sh              # install + enable, start at next boot
#   scripts/install_service.sh --disable            # stop and disable both services (unit files are kept)
#
# Installs two units from scripts/systemd/ into /etc/systemd/system/:
#   edgevision-vlm.service   llama-server (scripts/start_vlm.sh)
#   edgevision.service       the pipeline (python -m edgevision --config $CONFIG); After/Wants the VLM unit
# Run as your normal user (not with sudo): the services run as that user; sudo is asked for when needed.
# Safe to run again: the unit files are rewritten and systemd is reloaded.
set -euo pipefail
cd "$(dirname "$0")/.."
DIR=$(pwd)
RUN_USER=${SUDO_USER:-$(id -un)}
CONFIG=${CONFIG:-configs/jetson.yaml}
VLM_SIZE=${VLM_SIZE:-2B}
CTX=${CTX:-8192}
UNITS=(edgevision-vlm edgevision)
[[ "${NO_VLM:-0}" == 1 ]] && UNITS=(edgevision)

if [[ "${1:-}" == "--disable" ]]; then
  echo "stopping and disabling: edgevision edgevision-vlm"
  sudo systemctl disable --now edgevision.service edgevision-vlm.service 2>/dev/null || true
  echo "done. The unit files stay in /etc/systemd/system/ (remove them by hand if you want)."
  exit 0
fi

[[ "$RUN_USER" != root ]] || { echo "ERROR: run this as your normal user, not as root"; exit 1; }
[[ -x .venv/bin/python ]] || { echo "ERROR: .venv missing: run scripts/01_install_deps.sh first"; exit 1; }
[[ -f "$CONFIG" ]] || { echo "ERROR: config $CONFIG not found"; exit 1; }
if [[ "${NO_VLM:-0}" != 1 ]]; then
  [[ -f models/vlm/Qwen3VL-${VLM_SIZE}-Instruct-Q4_K_M.gguf ]] || { echo "ERROR: VLM $VLM_SIZE not downloaded: VLM_SIZE=$VLM_SIZE scripts/03_download_models.sh"; exit 1; }
fi
CONFIG_ABS=$(realpath "$CONFIG")

echo "installing systemd units: user=$RUN_USER dir=$DIR config=$CONFIG_ABS vlm=$VLM_SIZE ctx=$CTX"
for u in "${UNITS[@]}"; do
  sed -e "s|@USER@|$RUN_USER|g" -e "s|@DIR@|$DIR|g" -e "s|@CONFIG@|$CONFIG_ABS|g" \
      -e "s|@VLM_SIZE@|$VLM_SIZE|g" -e "s|@CTX@|$CTX|g" "scripts/systemd/$u.service" \
    | sudo tee "/etc/systemd/system/$u.service" >/dev/null
  echo "  /etc/systemd/system/$u.service"
done
if [[ "${NO_VLM:-0}" == 1 ]]; then
  # The pipeline unit names the VLM unit in After/Wants; without that unit systemd just ignores it.
  echo "  (NO_VLM=1: edgevision-vlm.service not installed)"
fi

sudo systemctl daemon-reload
UNIT_FILES=()
for u in "${UNITS[@]}"; do UNIT_FILES+=("$u.service"); done
if [[ "${START:-1}" == 1 ]]; then
  # The TensorRT engine must already be cached (scripts/04_warmup_detector.sh), otherwise the first start
  # spends minutes building it.
  sudo systemctl enable --now "${UNIT_FILES[@]}"
  sudo systemctl restart "${UNIT_FILES[@]}"
else
  sudo systemctl enable "${UNIT_FILES[@]}"
fi

echo
systemctl --no-pager --lines=0 status "${UNIT_FILES[@]}" || true
cat <<EOF

Installed. Useful commands:
  journalctl -u edgevision -f            # pipeline log
  journalctl -u edgevision-vlm -f        # VLM server log
  sudo systemctl restart edgevision      # after editing $CONFIG
  sudo systemctl stop edgevision edgevision-vlm
  scripts/install_service.sh --disable   # no longer start at boot
UI: http://$(hostname -I 2>/dev/null | awk '{print $1}'):8000
EOF
