#!/usr/bin/env sh
set -eu

MARKER_FILE="${DOWNLOADER_ONCE_MARKER:-/app/_container_data/.download_completed}"

mkdir -p "$(dirname "$MARKER_FILE")"

if [ -f "$MARKER_FILE" ]; then
  echo "Downloader already executed once. Skipping data download."
  exec tail -f /dev/null
fi

echo "Starting one-time downloader run..."
if python main.py; then
  touch "$MARKER_FILE"
  echo "Downloader run completed. Marker created at $MARKER_FILE"
else
  status=$?
  echo "Downloader run FAILED (exit code $status). Marker NOT created." >&2
  exit "$status"
fi

exec tail -f /dev/null
