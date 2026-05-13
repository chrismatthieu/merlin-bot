#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/Users/chrismatthieu/miniconda3/envs/merlin311/bin/python}"

MERLIN_LLM_URL="${MERLIN_LLM_URL:-http://localhost:11434/v1/chat/completions}"
MERLIN_MODEL="${MERLIN_MODEL:-qwen3-vl:2b}"
MERLIN_VISION_MODEL="${MERLIN_VISION_MODEL:-qwen3-vl:2b}"
MERLIN_AUDIO_SOURCE="${MERLIN_AUDIO_SOURCE:-usb}"
MERLIN_CAMERA_INDEX="${MERLIN_CAMERA_INDEX:-0}"
# Set MERLIN_START_TRACKER=0 to skip tracker_usb.py (orchestrator only).
MERLIN_START_TRACKER="${MERLIN_START_TRACKER:-1}"

MAIN_PID=""
TRACKER_PID=""
_CLEANED_UP=0

cleanup() {
  [ "$_CLEANED_UP" = "1" ] && return
  _CLEANED_UP=1
  if [ -n "${TRACKER_PID:-}" ] && kill -0 "$TRACKER_PID" 2>/dev/null; then
    kill -TERM "$TRACKER_PID" 2>/dev/null || true
    wait "$TRACKER_PID" 2>/dev/null || true
  fi
  if [ -n "${MAIN_PID:-}" ] && kill -0 "$MAIN_PID" 2>/dev/null; then
    kill -TERM "$MAIN_PID" 2>/dev/null || true
    wait "$MAIN_PID" 2>/dev/null || true
  fi
}

trap 'cleanup; exit 130' INT
trap 'cleanup; exit 143' TERM HUP
trap cleanup EXIT

if ! command -v ollama >/dev/null 2>&1; then
  echo "Error: ollama is not installed or not on PATH."
  exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "Error: ffmpeg is required for STT. Install with: brew install ffmpeg"
  exit 1
fi

if [ ! -x "$PYTHON_BIN" ]; then
  echo "Error: Python not found at: $PYTHON_BIN"
  echo "Set PYTHON_BIN to your merlin Python executable and retry."
  exit 1
fi

if ! curl -sSf "http://localhost:11434/api/tags" >/dev/null; then
  echo "Error: Ollama server is not reachable on localhost:11434."
  echo "Start it with: ollama serve"
  exit 1
fi

if [ "$MERLIN_AUDIO_SOURCE" = "usb" ]; then
  DEVICE_LIST="$(ffmpeg -f avfoundation -list_devices true -i "" 2>&1 || true)"
  # Only the AVFoundation *video* list — avoids matching EMEET USB audio on the same composite device.
  VIDEO_BLOCK="$(printf "%s\n" "$DEVICE_LIST" | awk '/AVFoundation video devices/,/AVFoundation audio devices/')"
  PIXY_LINES="$(printf "%s\n" "$VIDEO_BLOCK" | grep -Ei "pixy" || true)"
  if [ -z "$PIXY_LINES" ]; then
    PIXY_LINES="$(printf "%s\n" "$VIDEO_BLOCK" | grep -Ei "emeet" || true)"
  fi
  if [ -z "$PIXY_LINES" ]; then
    # No section headers (older ffmpeg): fall back to whole device list
    PIXY_LINES="$(printf "%s\n" "$DEVICE_LIST" | grep -Ei "emeet|pixy" || true)"
  fi
  if [ -z "$PIXY_LINES" ]; then
    echo "Error: EMEET PIXY is not detected by macOS."
    echo "Detected cameras:"
    printf "%s\n" "$DEVICE_LIST" | grep -E "AVFoundation video devices|\\[[0-9]+\\]"
    echo
    echo "Fixes to try:"
    echo "  - Replug PIXY directly into Mac (avoid passive hubs)."
    echo "  - Use a USB data cable (not charge-only)."
    echo "  - Quit apps that may own camera (Zoom/Teams/Photo Booth)."
    echo "  - Check camera permission for your terminal/Cursor in macOS Privacy settings."
    exit 1
  fi

  DETECTED_INDEX="$(printf "%s\n" "$PIXY_LINES" | sed -E 's/.*\[([0-9]+)\].*/\1/' | head -n 1)"
  if [ -n "$DETECTED_INDEX" ]; then
    MERLIN_CAMERA_INDEX="$DETECTED_INDEX"
  fi
fi

echo "Starting Merlin with Ollama..."
echo "  LLM URL:      $MERLIN_LLM_URL"
echo "  Model:        $MERLIN_MODEL"
echo "  Vision model: $MERLIN_VISION_MODEL"
echo "  Audio source: $MERLIN_AUDIO_SOURCE"
echo "  Camera index: $MERLIN_CAMERA_INDEX"
if [ "$MERLIN_AUDIO_SOURCE" = "usb" ] && [ "$MERLIN_START_TRACKER" != "0" ]; then
  echo "  Face tracker: tracker_usb.py (after /health)"
else
  echo "  Face tracker: (skipped)"
fi
echo

cd "$ROOT_DIR"

# Avoid false "healthy" from a stale Merlin while a new main.py hits EADDRINUSE on :8900.
_free_merlin_listener() {
  local port="${MERLIN_HTTP_PORT:-8900}"
  for pid in $(lsof -nP -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null); do
    echo "Freeing port $port (stopping PID $pid)"
    kill -9 "$pid" 2>/dev/null || true
  done
  pkill -9 -f "tracker_usb.py" 2>/dev/null || true
  sleep 0.5
}

_run_main_env() {
  MERLIN_LLM_URL="$MERLIN_LLM_URL" \
  MERLIN_MODEL="$MERLIN_MODEL" \
  MERLIN_VISION_MODEL="$MERLIN_VISION_MODEL" \
  MERLIN_AUDIO_SOURCE="$MERLIN_AUDIO_SOURCE" \
  MERLIN_CAMERA_INDEX="$MERLIN_CAMERA_INDEX" \
    "$@"
}

if [ "$MERLIN_AUDIO_SOURCE" = "usb" ] && [ "$MERLIN_START_TRACKER" != "0" ]; then
  _free_merlin_listener
  _run_main_env "$PYTHON_BIN" -u main.py &
  MAIN_PID=$!

  echo "Waiting for http://localhost:8900/health ..."
  HEALTH_OK=0
  for _ in $(seq 1 80); do
    if curl -sf "http://localhost:8900/health" >/dev/null; then
      HEALTH_OK=1
      break
    fi
    if ! kill -0 "$MAIN_PID" 2>/dev/null; then
      echo "Error: main.py exited before becoming healthy."
      wait "$MAIN_PID" || true
      exit 1
    fi
    sleep 0.25
  done
  if [ "$HEALTH_OK" != "1" ]; then
    echo "Error: Timed out waiting for /health (is port 8900 blocked?)."
    exit 1
  fi

  echo "Starting USB face tracker..."
  MERLIN_CAMERA_INDEX="$MERLIN_CAMERA_INDEX" \
  MERLIN_BRAIN_URL="${MERLIN_BRAIN_URL:-http://localhost:8900/event}" \
    "$PYTHON_BIN" -u tracker_usb.py &
  TRACKER_PID=$!
  echo "  Tracker PID: $TRACKER_PID"
  echo

  wait "$MAIN_PID"
  RET=$?
  exit "$RET"
fi

_run_main_env "$PYTHON_BIN" -u main.py
