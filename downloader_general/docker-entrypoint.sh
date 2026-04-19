#!/usr/bin/env sh
set -eu

MARKER_FILE="${DOWNLOADER_ONCE_MARKER:-/app/_container_data/.download_completed}"

mkdir -p "$(dirname "$MARKER_FILE")"

if [ -f "$MARKER_FILE" ]; then
  echo "Downloader already executed once. Skipping data download."
  exec tail -f /dev/null
fi

echo "Starting one-time downloader run..."
python main.py

touch "$MARKER_FILE"
echo "Downloader run completed. Marker created at $MARKER_FILE"

exec tail -f /dev/null
