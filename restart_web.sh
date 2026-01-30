#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ENV_NAME="${VISIONAI_CONDA_ENV:-vision}"
APP_MODULE="${VISIONAI_APP:-webapp.main:app}"
HOST="${VISIONAI_HOST:-0.0.0.0}"
PORT="${VISIONAI_PORT:-8003}"

PID_FILE="${VISIONAI_PID_FILE:-$ROOT_DIR/.visionai_web.pid}"
LOG_FILE="${VISIONAI_LOG_FILE:-$ROOT_DIR/.visionai_web.log}"

say() { printf '%s\n' "$*"; }

conda_base() {
  if [[ -n "${CONDA_EXE:-}" ]]; then
    # .../bin/conda -> base
    local bin_dir
    bin_dir="$(cd "$(dirname "$CONDA_EXE")" && pwd)"
    (cd "$bin_dir/.." && pwd)
    return 0
  fi

  if command -v conda >/dev/null 2>&1; then
    conda info --base
    return 0
  fi

  # Common fallbacks
  if [[ -d "$HOME/miniconda3" ]]; then
    echo "$HOME/miniconda3"
    return 0
  fi
  if [[ -d "$HOME/anaconda3" ]]; then
    echo "$HOME/anaconda3"
    return 0
  fi
  if [[ -d "/opt/conda" ]]; then
    echo "/opt/conda"
    return 0
  fi

  return 1
}

activate_env() {
  local base
  if ! base="$(conda_base)"; then
    say "ERROR: conda not found. Make sure 'conda' is on PATH or set CONDA_EXE."
    exit 1
  fi

  # shellcheck disable=SC1090
  source "$base/etc/profile.d/conda.sh"
  conda activate "$ENV_NAME"
}

is_pid_running() {
  local pid="$1"
  [[ -n "$pid" ]] && kill -0 "$pid" >/dev/null 2>&1
}

read_pidfile() {
  if [[ -f "$PID_FILE" ]]; then
    tr -d ' \n\r\t' < "$PID_FILE" || true
  fi
}

find_pids() {
  local pids=()

  local pid
  pid="$(read_pidfile || true)"
  if [[ -n "$pid" ]] && is_pid_running "$pid"; then
    pids+=("$pid")
  fi

  # If no pidfile (or stale), try to find uvicorn running this app/port.
  if [[ "${#pids[@]}" -eq 0 ]]; then
    if command -v pgrep >/dev/null 2>&1; then
      while IFS= read -r line; do
        [[ -n "$line" ]] && pids+=("$line")
      done < <(pgrep -f "uvicorn.*${APP_MODULE}.*--port[ =]${PORT}" || true)
    fi
  fi

  # As a fallback, try to locate any process listening on the port, but only kill uvicorn.
  if [[ "${#pids[@]}" -eq 0 ]]; then
    local listening_pid=""
    if command -v lsof >/dev/null 2>&1; then
      listening_pid="$(lsof -tiTCP:"$PORT" -sTCP:LISTEN 2>/dev/null | head -n 1 || true)"
    elif command -v ss >/dev/null 2>&1; then
      listening_pid="$(ss -lptn "sport = :$PORT" 2>/dev/null | sed -n 's/.*pid=\\([0-9]\\+\\).*/\\1/p' | head -n 1 || true)"
    fi

    if [[ -n "$listening_pid" ]]; then
      if ps -p "$listening_pid" -o command= 2>/dev/null | grep -q "uvicorn"; then
        pids+=("$listening_pid")
      fi
    fi
  fi

  # De-duplicate
  if [[ "${#pids[@]}" -gt 0 ]]; then
    printf '%s\n' "${pids[@]}" | awk '!seen[$0]++'
  fi
}

status_cmd() {
  local pids
  pids="$(find_pids || true)"
  if [[ -z "${pids:-}" ]]; then
    say "status: stopped"
    return 0
  fi
  say "status: running"
  say "pids:"
  printf '%s\n' "$pids" | sed 's/^/  - /'
}

stop_cmd() {
  local pids
  pids="$(find_pids || true)"
  if [[ -z "${pids:-}" ]]; then
    say "stop: not running"
    rm -f "$PID_FILE" >/dev/null 2>&1 || true
    return 0
  fi

  say "stop: sending SIGTERM"
  while IFS= read -r pid; do
    [[ -z "$pid" ]] && continue
    kill "$pid" >/dev/null 2>&1 || true
  done <<< "$pids"

  # Wait up to ~15s
  for _ in {1..30}; do
    local still=""
    while IFS= read -r pid; do
      [[ -z "$pid" ]] && continue
      if is_pid_running "$pid"; then
        still="1"
      fi
    done <<< "$pids"
    [[ -z "$still" ]] && break
    sleep 0.5
  done

  # Force kill if still alive
  local forced=""
  while IFS= read -r pid; do
    [[ -z "$pid" ]] && continue
    if is_pid_running "$pid"; then
      forced="1"
      kill -9 "$pid" >/dev/null 2>&1 || true
    fi
  done <<< "$pids"

  rm -f "$PID_FILE" >/dev/null 2>&1 || true

  if [[ -n "$forced" ]]; then
    say "stop: SIGKILL applied (some processes did not exit in time)"
  else
    say "stop: done"
  fi
}

start_cmd() {
  local pid
  pid="$(read_pidfile || true)"
  if [[ -n "$pid" ]] && is_pid_running "$pid"; then
    say "start: already running (pid=$pid)"
    return 0
  fi

  activate_env
  cd "$ROOT_DIR"

  mkdir -p "$(dirname "$LOG_FILE")" >/dev/null 2>&1 || true

  say "start: uvicorn ${APP_MODULE} on ${HOST}:${PORT}"
  say "log: ${LOG_FILE}"
  nohup uvicorn "$APP_MODULE" --host "$HOST" --port "$PORT" >>"$LOG_FILE" 2>&1 &
  echo "$!" > "$PID_FILE"
  say "start: pid=$(cat "$PID_FILE")"

  # Best-effort readiness wait (helps when restart is scripted).
  local wait_seconds="${VISIONAI_WAIT_SECONDS:-8}"
  if [[ "$wait_seconds" =~ ^[0-9]+$ ]] && [[ "$wait_seconds" -gt 0 ]]; then
    say "start: waiting up to ${wait_seconds}s for 127.0.0.1:${PORT}"
    local tries=$((wait_seconds * 2))
    for _ in $(seq 1 "$tries"); do
      python - <<PY >/dev/null 2>&1 && break || true
import socket
sock = socket.create_connection(("127.0.0.1", int("${PORT}")), timeout=0.5)
sock.close()
PY
      sleep 0.5
    done
  fi
}

restart_cmd() {
  stop_cmd
  start_cmd
}

usage() {
  cat <<EOF
Usage:
  ./restart_web.sh [start|stop|restart|status]

Env vars (optional):
  VISIONAI_CONDA_ENV   (default: vision)
  VISIONAI_APP         (default: webapp.main:app)
  VISIONAI_HOST        (default: 0.0.0.0)
  VISIONAI_PORT        (default: 8003)
  VISIONAI_PID_FILE    (default: ./.visionai_web.pid)
  VISIONAI_LOG_FILE    (default: ./.visionai_web.log)
  VISIONAI_DEVICE      (used by the app to select torch device)
EOF
}

cmd="${1:-restart}"
case "$cmd" in
  start) start_cmd ;;
  stop) stop_cmd ;;
  restart) restart_cmd ;;
  status) status_cmd ;;
  -h|--help|help) usage ;;
  *)
    usage
    exit 2
    ;;
esac

