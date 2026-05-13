#!/usr/bin/env bash
# start-nova.sh — Launch or relaunch Nova (brain on Mac + services on Pi)
#
# Usage:
#   ./start-nova.sh          # start everything
#   ./start-nova.sh status   # check status only
#   ./start-nova.sh stop     # stop everything

set -euo pipefail

PI_HOST="pi@100.87.156.70"
BRAIN_PORT=8900
BRAIN_LOG="/tmp/nova-brain.log"
NOVA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="${NOVA_PYTHON:-/Users/chrismatthieu/miniconda3/envs/merlin311/bin/python3}"
MAIN_PY="$NOVA_DIR/main.py"
OLLAMA_URL="http://localhost:11434/api/tags"

# ── Helpers ────────────────────────────────────────────────────────────────────

log()  { echo "  $*"; }
ok()   { echo "  [OK]  $*"; }
fail() { echo "  [!!]  $*"; }
hr()   { echo ""; echo "=== $* ==="; }

brain_pid() { lsof -ti ":$BRAIN_PORT" 2>/dev/null | head -1 || true; }

brain_healthy() {
  curl -sf "http://localhost:$BRAIN_PORT/health" >/dev/null 2>&1
}

ollama_running() {
  curl -sf "$OLLAMA_URL" >/dev/null 2>&1
}

pi_reachable() {
  ssh -o ConnectTimeout=5 -o BatchMode=yes "$PI_HOST" true 2>/dev/null
}

# ── Status ─────────────────────────────────────────────────────────────────────

do_status() {
  hr "Nova Status"

  # Ollama
  if ollama_running; then
    ok "Ollama running"
  else
    fail "Ollama not running — start with: ollama serve"
  fi

  # Brain
  local pid
  pid=$(brain_pid)
  if [ -n "$pid" ] && brain_healthy; then
    ok "Brain running (PID $pid) — http://localhost:$BRAIN_PORT/health"
  elif [ -n "$pid" ]; then
    fail "Brain process alive (PID $pid) but /health not responding"
  else
    fail "Brain not running"
  fi

  # Pi
  if pi_reachable; then
    local tracker client
    tracker=$(ssh "$PI_HOST" "systemctl is-active merlin-tracker 2>/dev/null || echo inactive")
    client=$(ssh  "$PI_HOST" "systemctl is-active merlin-pi-client 2>/dev/null || echo inactive")
    if [ "$tracker" = "active" ] && [ "$client" = "active" ]; then
      ok "Pi services active (tracker: $tracker, client: $client)"
    else
      fail "Pi services degraded (tracker: $tracker, client: $client)"
    fi
  else
    fail "Pi not reachable ($PI_HOST)"
  fi

  echo ""
}

# ── Stop ───────────────────────────────────────────────────────────────────────

do_stop() {
  hr "Stopping Nova"

  local pid
  pid=$(brain_pid)
  if [ -n "$pid" ]; then
    kill "$pid" 2>/dev/null || true
    sleep 1
    pid=$(brain_pid)
    [ -n "$pid" ] && kill -9 "$pid" 2>/dev/null || true
    ok "Brain stopped"
  else
    log "Brain was not running"
  fi

  if pi_reachable; then
    ssh "$PI_HOST" "sudo systemctl stop merlin-tracker merlin-pi-client 2>/dev/null || true"
    ok "Pi services stopped"
  else
    fail "Pi not reachable — skipped"
  fi

  echo ""
}

# ── Start ──────────────────────────────────────────────────────────────────────

do_start() {
  hr "Starting Nova"

  # Check ollama
  if ! ollama_running; then
    fail "Ollama is not running. Start it first: ollama serve"
    exit 1
  fi
  ok "Ollama running"

  # Kill any existing brain
  local pid
  pid=$(brain_pid)
  if [ -n "$pid" ]; then
    log "Killing existing brain (PID $pid)..."
    kill "$pid" 2>/dev/null || true
    sleep 1
    pid=$(brain_pid)
    [ -n "$pid" ] && kill -9 "$pid" 2>/dev/null || true
    sleep 1
  fi

  # Validate venv
  if [ ! -x "$VENV_PYTHON" ]; then
    fail "Python not found at $VENV_PYTHON"
    echo "       Set NOVA_PYTHON env var to your Python executable and retry."
    exit 1
  fi

  # Start brain
  log "Starting brain (log: $BRAIN_LOG)..."
  echo "" >> "$BRAIN_LOG"
  echo "=== start-nova.sh — $(date) ===" >> "$BRAIN_LOG"
  cd "$NOVA_DIR"
  nohup "$VENV_PYTHON" -u "$MAIN_PY" >> "$BRAIN_LOG" 2>&1 &
  local brain_pid=$!

  # Wait for health
  log "Waiting for /health..."
  local healthy=0
  for _ in $(seq 1 60); do
    if ! kill -0 "$brain_pid" 2>/dev/null; then
      fail "Brain exited before becoming healthy — check $BRAIN_LOG"
      tail -10 "$BRAIN_LOG" 2>/dev/null || true
      exit 1
    fi
    if brain_healthy; then
      healthy=1
      break
    fi
    sleep 0.5
  done

  if [ "$healthy" = "1" ]; then
    ok "Brain healthy (PID $brain_pid)"
  else
    fail "Brain did not become healthy in 30s — check $BRAIN_LOG"
    tail -10 "$BRAIN_LOG" 2>/dev/null || true
    exit 1
  fi

  # Pi services
  if pi_reachable; then
    log "Restarting Pi services..."
    ssh "$PI_HOST" "sudo systemctl restart merlin-tracker merlin-pi-client"
    sleep 2
    local tracker client
    tracker=$(ssh "$PI_HOST" "systemctl is-active merlin-tracker 2>/dev/null || echo inactive")
    client=$(ssh  "$PI_HOST" "systemctl is-active merlin-pi-client 2>/dev/null || echo inactive")
    if [ "$tracker" = "active" ] && [ "$client" = "active" ]; then
      ok "Pi services running"
    else
      fail "Pi services may not have started (tracker: $tracker, client: $client)"
    fi
  else
    fail "Pi not reachable ($PI_HOST) — skipping Pi services"
  fi

  hr "Nova is up — say 'Hey Nova' to wake"
  echo "  Brain log:  tail -f $BRAIN_LOG"
  echo "  Pi tracker: ssh $PI_HOST 'journalctl -u merlin-tracker -f --no-pager'"
  echo "  Pi client:  ssh $PI_HOST 'journalctl -u merlin-pi-client -f --no-pager'"
  echo ""
}

# ── Dispatch ───────────────────────────────────────────────────────────────────

case "${1:-start}" in
  start)   do_start ;;
  stop)    do_stop  ;;
  restart) do_stop; do_start ;;
  status)  do_status ;;
  *)
    echo "Usage: $0 [start|stop|restart|status]"
    exit 1
    ;;
esac
