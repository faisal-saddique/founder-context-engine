#!/usr/bin/env bash
# Convenience wrapper — starts the LightRAG admin server.
# Run from anywhere: ./admin/start.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

exec uv run --project "$PROJECT_ROOT" python "$SCRIPT_DIR/start.py" "$@"
