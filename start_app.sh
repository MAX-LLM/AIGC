#!/usr/bin/env bash
# Start the counselling service and Gradio page from any working directory.
set -euo pipefail

cd "$(dirname "$0")"
SCRIPT_PATH="$PWD/$(basename "$0")"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x .venv/bin/python ]]; then
    PYTHON_BIN="$PWD/.venv/bin/python"
  else
    PYTHON_BIN=python3
  fi
fi

# Read .env as data, using the same parser as the API. Never execute it as shell.
if [[ "${MND_LAUNCH_ENV_LOADED:-0}" != "1" ]]; then
  exec "$PYTHON_BIN" - "$SCRIPT_PATH" "$@" <<'PY'
import os
import sys
from model_config import load_environment

load_environment()
os.environ["MND_LAUNCH_ENV_LOADED"] = "1"
os.execvpe("bash", ["bash", sys.argv[1], *sys.argv[2:]], os.environ)
PY
fi

AGENT_PORT="${AGENT_PORT:-8000}"
UI_PORT="${UI_PORT:-8080}"
HOST="${HOST:-127.0.0.1}"
BACKEND_ENGINE="${BACKEND_ENGINE:-${MND_BACKEND:-api}}"
LLM="${MND_AGENT_MODEL:-${AGENT_MODEL:-${LLM:-}}}"
MND_MAX_NEW_TOKENS="${MND_AGENT_MAX_TOKENS:-${AGENT_MAX_TOKENS:-${MND_MAX_NEW_TOKENS:-512}}}"
if [[ "$BACKEND_ENGINE" == "vllm" ]]; then
  MND_MAX_CONCURRENT_GENERATIONS="${MND_MAX_CONCURRENT_GENERATIONS:-10}"
else
  MND_MAX_CONCURRENT_GENERATIONS="${MND_MAX_CONCURRENT_GENERATIONS:-1}"
fi
export MND_MAX_CONCURRENT_GENERATIONS MND_MAX_NEW_TOKENS
export MND_BACKEND="$BACKEND_ENGINE"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$PWD/.runtime/vllm}"
export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-10}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.95}"
API_BASE="${MND_API_BASE:-http://127.0.0.1:${AGENT_PORT}}"
START_BACKEND="${START_BACKEND:-1}"
GRADIO_SHARE="${GRADIO_SHARE:-0}"
STARTUP_TIMEOUT="${MND_STARTUP_TIMEOUT:-300}"
LOG_DIR="${MND_LOG_DIR:-logs}"
BACKEND_LOG="${LOG_DIR}/backend.log"
FRONTEND_LOG="${LOG_DIR}/frontend.log"

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  for child_pid in "${TAIL_PID:-}" "${FRONTEND_PID:-}" "${BACKEND_PID:-}"; do
    if [[ -n "$child_pid" ]]; then
      kill "$child_pid" 2>/dev/null || true
      wait "$child_pid" 2>/dev/null || true
    fi
  done
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

mkdir -p "$LOG_DIR"
touch "$BACKEND_LOG" "$FRONTEND_LOG"

port_is_busy() {
  "$PYTHON_BIN" - "$1" <<'PY'
import socket
import sys

with socket.socket() as sock:
    try:
        sock.bind(("0.0.0.0", int(sys.argv[1])))
    except OSError:
        sys.exit(0)
sys.exit(1)
PY
}

wait_for_service() {
  "$PYTHON_BIN" - "$1" "$2" "$STARTUP_TIMEOUT" <<'PY'
import os
import sys
import time
import urllib.error
import urllib.request

url, pid, timeout = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
deadline = time.monotonic() + timeout
while time.monotonic() < deadline:
    try:
        os.kill(pid, 0)
    except OSError:
        sys.exit(1)
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            if response.status == 200:
                sys.exit(0)
    except (OSError, urllib.error.URLError):
        pass
    time.sleep(1)
sys.exit(1)
PY
}

if [[ "$START_BACKEND" == "1" ]] && port_is_busy "$AGENT_PORT"; then
  echo "Backend port ${AGENT_PORT} is already in use. Choose AGENT_PORT=<free_port>."
  exit 1
fi
if port_is_busy "$UI_PORT"; then
  echo "Web port ${UI_PORT} is already in use. Choose UI_PORT=<free_port>."
  exit 1
fi

if [[ "$START_BACKEND" == "1" ]]; then
  backend_args=(--backend "$BACKEND_ENGINE" --max_new_tokens "$MND_MAX_NEW_TOKENS"
                --host 127.0.0.1 --port "$AGENT_PORT")
  if [[ -n "$LLM" ]]; then
    backend_args+=(--model "$LLM")
  fi
  echo "Starting ${BACKEND_ENGINE} counselling service on port ${AGENT_PORT}..."
  "$PYTHON_BIN" -u agent_api.py "${backend_args[@]}" > "$BACKEND_LOG" 2>&1 &
  BACKEND_PID=$!
  if ! wait_for_service "http://127.0.0.1:${AGENT_PORT}/health" "$BACKEND_PID"; then
    echo "Counselling service failed to become ready. Recent log:"
    tail -n 50 "$BACKEND_LOG"
    exit 1
  fi
else
  echo "Using the existing counselling service at ${API_BASE}."
fi

echo "Starting web interface on port ${UI_PORT}..."
export GRADIO_SERVER_NAME="$HOST"
export GRADIO_SERVER_PORT="$UI_PORT"
export GRADIO_SHARE
export MND_API_BASE="$API_BASE"
# This is the service's public alias, independent of the upstream provider model.
export MND_MODEL="${MND_MODEL:-mnd-local-agent}"
"$PYTHON_BIN" -u ui_gradio.py --api_base "$API_BASE" --model "$MND_MODEL" \
  --host "$HOST" --port "$UI_PORT" > "$FRONTEND_LOG" 2>&1 &
FRONTEND_PID=$!
if ! wait_for_service "http://127.0.0.1:${UI_PORT}/" "$FRONTEND_PID"; then
  echo "Web interface failed to become ready. Recent log:"
  tail -n 50 "$FRONTEND_LOG"
  exit 1
fi

echo "Open http://127.0.0.1:${UI_PORT}"
echo "Logs: ${BACKEND_LOG} and ${FRONTEND_LOG}"
echo "Press Ctrl+C to stop."
if [[ "$GRADIO_SHARE" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
  echo "The temporary public Gradio link appears in ${FRONTEND_LOG}."
fi
tail -n +1 -f "$BACKEND_LOG" "$FRONTEND_LOG" &
TAIL_PID=$!

# Monitor both children: a failed service must not leave a stranded web process.
while kill -0 "$FRONTEND_PID" 2>/dev/null; do
  if [[ -n "${BACKEND_PID:-}" ]] && ! kill -0 "$BACKEND_PID" 2>/dev/null; then
    wait "$BACKEND_PID"
    exit 1
  fi
  sleep 1
done
wait "$FRONTEND_PID"
