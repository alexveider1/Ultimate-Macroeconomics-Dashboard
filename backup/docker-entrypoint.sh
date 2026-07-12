#!/usr/bin/env sh
set -eu

# Long-running scheduler. All logic (enabled check, interval loop, per-run
# error handling, SIGTERM shutdown) lives in Python.
exec python main.py
