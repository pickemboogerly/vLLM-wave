#!/usr/bin/env bash
# Launch the vLLM-MLX Textual harness from this repo (optional local venv).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
if [[ -x "$ROOT/.venv/bin/python" ]]; then
  exec "$ROOT/.venv/bin/python" -m vllm_mlx_tui "$@"
fi
exec python3 -m vllm_mlx_tui "$@"
