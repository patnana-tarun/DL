#!/usr/bin/env bash
# Step 5: run everything. Starts the VLM server (if the config uses one and it is not running yet), waits
# until it answers, then runs the pipeline in the foreground. Ctrl-C stops both.
#
#   scripts/05_run.sh
#   CONFIG=configs/desktop.yaml scripts/05_run.sh
#   scripts/05_run.sh --set source.uri=data/aa1.mp4 --set source.loop=true      # extra args go to edgevision
#   VLM_SIZE=4B CTX=8192 scripts/05_run.sh                                       # passed to start_vlm.sh
#   VLM_ARGS="-np 1" scripts/05_run.sh                                           # extra llama-server flags
#
# Environment: CONFIG (default configs/jetson.yaml), VLM_WAIT_S (default 600), VLM_SIZE, CTX, VLM_ARGS,
#              PYTHON (interpreter; default .venv/bin/python).
# The VLM log goes to data/llama_server.log. Open the UI at http://<this-machine-ip>:8000
set -euo pipefail
cd "$(dirname "$0")/.."
CONFIG=${CONFIG:-configs/jetson.yaml}
WAIT_S=${VLM_WAIT_S:-600}
PYBIN=${PYTHON:-.venv/bin/python}
[[ -x $PYBIN ]] || PYBIN=python3
PY=(env -u PYTHONPATH "$PYBIN")
mkdir -p data

[[ -f "$CONFIG" ]] || { echo "ERROR: config $CONFIG not found"; exit 1; }

# Read the VLM settings from the config (after defaults and --set overrides), so this script agrees with
# what the pipeline will use.
read -r VLM_ON VLM_EP < <("${PY[@]}" - "$CONFIG" "$@" <<'EOF'
import sys
from edgevision import config
args, ov = sys.argv[2:], []
for i, a in enumerate(args):
    if a == "--set" and i + 1 < len(args):
        ov.append(args[i + 1])
    elif a.startswith("--set="):
        ov.append(a[6:])
c = config.load(sys.argv[1], ov)
print(int(bool(c["vlm"]["enabled"])), c["vlm"]["endpoint"].rstrip("/"))
EOF
)
[[ -n "${VLM_ON:-}" ]] || { echo "ERROR: could not read $CONFIG with $PYBIN (run scripts/01_install_deps.sh first?)"; exit 1; }

VLM_PID=""
# shellcheck disable=SC2329  # invoked by the EXIT trap
cleanup() {
  if [[ -n "$VLM_PID" ]] && kill -0 "$VLM_PID" 2>/dev/null; then
    echo "stopping the VLM server (pid $VLM_PID)"
    kill "$VLM_PID" 2>/dev/null || true
    wait "$VLM_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

LOCAL_RE='^https?://(127\.0\.0\.1|localhost)(:([0-9]+))?(/|$)'
vlm_up() { curl -sf --max-time 3 "$VLM_EP/models" >/dev/null 2>&1; }

if [[ "$VLM_ON" == 1 ]]; then
  if vlm_up; then
    echo "VLM server already running at $VLM_EP"
  elif [[ "$VLM_EP" =~ $LOCAL_RE ]]; then
    PORT_FROM_EP=${BASH_REMATCH[3]:-80}
    read -ra VA <<<"${VLM_ARGS:-}"
    echo "starting the VLM server on port $PORT_FROM_EP (log: data/llama_server.log)"
    PORT=$PORT_FROM_EP scripts/start_vlm.sh "${VA[@]}" >data/llama_server.log 2>&1 &
    VLM_PID=$!
    printf 'waiting for %s/models (loading the model usually takes tens of seconds; giving up after %s s) ' "$VLM_EP" "$WAIT_S"
    for ((i = 0; i < WAIT_S; i += 2)); do
      vlm_up && break
      if ! kill -0 "$VLM_PID" 2>/dev/null; then
        echo; echo "WARNING: the VLM server exited. Last lines of data/llama_server.log:"
        tail -n 20 data/llama_server.log | sed 's/^/  | /'
        echo "Continuing WITHOUT the VLM (the /commands in the chat still work)."
        VLM_PID=""
        break
      fi
      printf '.'; sleep 2
    done
    echo
    if vlm_up; then echo "VLM ready at $VLM_EP"
    elif [[ -n "$VLM_PID" ]]; then echo "WARNING: VLM not ready after $WAIT_S s; the pipeline starts anyway and retries."; fi
  else
    echo "VLM endpoint $VLM_EP is remote and not reachable now; the pipeline starts anyway and retries."
  fi
else
  echo "vlm.enabled is false: running without the VLM"
fi

echo "starting EdgeVision: $PYBIN -m edgevision --config $CONFIG $*"
# Not exec: the EXIT trap must run afterwards to stop the VLM server we started.
set +e
"${PY[@]}" -m edgevision --config "$CONFIG" "$@"
RC=$?
set -e
exit $RC
