#!/usr/bin/env bash
# Claude Code PostToolUse hook: auto-format the just-edited Python file with the
# repo's single-source-of-truth ruff config (root ruff.toml). Non-blocking —
# always exits 0 so a formatting hiccup never aborts an edit.
#
# Uses `uvx ruff` so it works without any service .venv activated. ruff walks up
# from the file to discover the root ruff.toml automatically.
set -uo pipefail

payload="$(cat)"
file="$(printf '%s' "$payload" \
  | python3 -c 'import json,sys;
d=json.load(sys.stdin);
print(d.get("tool_input", {}).get("file_path", ""))' 2>/dev/null || true)"

# Only touch Python files that still exist on disk.
case "$file" in
  *.py) ;;
  *) exit 0 ;;
esac
[ -f "$file" ] || exit 0

uvx --quiet ruff format "$file" >/dev/null 2>&1 || true
uvx --quiet ruff check --fix --exit-zero "$file" >/dev/null 2>&1 || true
exit 0
