#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/Users/chrismatthieu/miniconda3/envs/merlin311/bin/python}"

if [ ! -x "$PYTHON_BIN" ]; then
  echo "Error: Python not found at: $PYTHON_BIN"
  echo "Set PYTHON_BIN to your merlin Python executable and retry."
  exit 1
fi

DEVICE_LIST="$(ffmpeg -f avfoundation -list_devices true -i "" 2>&1 || true)"
PIXY_LINE="$(printf "%s\n" "$DEVICE_LIST" | grep -Ei "emeet|pixy" || true)"

if [ -z "$PIXY_LINE" ]; then
  echo "Error: EMEET PIXY is not detected by macOS."
  echo "Detected cameras:"
  printf "%s\n" "$DEVICE_LIST" | grep -E "AVFoundation video devices|\\[[0-9]+\\]"
  exit 1
fi

PIXY_INDEX="$(printf "%s\n" "$PIXY_LINE" | sed -E 's/.*\[([0-9]+)\].*/\1/' | head -n 1)"
if [ -z "$PIXY_INDEX" ]; then
  echo "Error: Could not parse PIXY camera index from AVFoundation output."
  exit 1
fi

echo "Starting tracker on PIXY index: $PIXY_INDEX"
cd "$ROOT_DIR"
MERLIN_CAMERA_INDEX="$PIXY_INDEX" "$PYTHON_BIN" -u tracker_usb.py
