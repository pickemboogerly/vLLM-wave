#!/bin/bash
#
# start_ai_cached_models.sh — start vllm-mlx with a model chosen from the Hugging Face Hub cache.
#
# Where vllm-mlx / mlx-lm store downloaded weights:
#   - Hub repo ids are fetched via huggingface_hub (e.g. snapshot_download). Files go under the
#     Hub cache directory (env HF_HUB_CACHE, or HF_HOME/hub, default ~/.cache/huggingface/hub).
#   - On disk: models--<org--name>/snapshots/<revision_hash>/  (symlinks into blobs/)
#   - Passing that snapshot directory (or any local folder with config.json) to vllm-mlx avoids
#     ambiguity and uses the copy you already downloaded.
#
# Separate from weights: vllm-mlx may persist prefix/KV helpers under ~/.cache/vllm-mlx/ — that
# is not scanned here.
#
# Requirements: bash 3.2+, python3 with huggingface_hub (same env as vllm-mlx is ideal).
# Optional: curl for API readiness checks.

set -euo pipefail

# Fallback when you press Enter at the model prompt and no default is set (Hub id or absolute path).
#DEFAULT_MODEL="mlx-community/Qwen3.5-27B-heretic-8bit"

PORT="${VLLM_PORT:-8001}"
VLLM_BIN="${VLLM_MLX_BIN:-vllm-mlx}"
# Max seconds to wait for /v1/models after starting the server
API_READY_TIMEOUT="${API_READY_TIMEOUT:-120}"

# Hub cache root (see https://huggingface.co/docs/huggingface_hub/guides/manage-cache )
if [ -n "${HF_HUB_CACHE:-}" ]; then
  HUB_CACHE="${HF_HUB_CACHE}"
elif [ -n "${HF_HOME:-}" ]; then
  HUB_CACHE="${HF_HOME}/hub"
else
  HUB_CACHE="${HOME}/.cache/huggingface/hub"
fi
HUB_CACHE="${HUB_CACHE%/}"

# Prints one line per model: MENU_LABEL<TAB>SNAPSHOT_PATH
discover_cached_models_tsv() {
  local hub="${1:-}"
  python3 - "$hub" <<'PY'
import os
import subprocess
import sys
from collections import Counter

cache = os.path.expanduser(sys.argv[1])


def decode_models_folder(folder_name: str) -> str:
    if folder_name.startswith("models--"):
        return folder_name[len("models--") :].replace("--", "/")
    return folder_name


rows: list[dict] = []

if os.path.isdir(cache):
    try:
        from huggingface_hub import scan_cache_dir

        info = scan_cache_dir(cache)
        for repo in info.repos:
            if repo.repo_type != "model":
                continue
            for rev in repo.revisions:
                p = str(rev.snapshot_path)
                if os.path.isdir(p):
                    rows.append(
                        {
                            "repo_id": repo.repo_id,
                            "commit_hash": rev.commit_hash or "",
                            "path": p,
                            "last_modified_str": getattr(
                                rev, "last_modified_str", ""
                            )
                            or "",
                        }
                    )
    except Exception:
        pass

if not rows:
    try:
        proc = subprocess.run(
            [
                "find",
                cache,
                "-type",
                "f",
                "-path",
                "*/snapshots/*/config.json",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        seen_paths: set[str] = set()
        for line in proc.stdout.splitlines():
            cfg = line.strip()
            if not cfg:
                continue
            snap = os.path.dirname(cfg)
            if snap in seen_paths or not os.path.isdir(snap):
                continue
            seen_paths.add(snap)
            repo_folder = os.path.basename(
                os.path.dirname(os.path.dirname(snap))
            )
            rid = (
                decode_models_folder(repo_folder)
                if repo_folder.startswith("models--")
                else repo_folder
            )
            rows.append(
                {
                    "repo_id": rid,
                    "commit_hash": os.path.basename(snap),
                    "path": snap,
                    "last_modified_str": "",
                }
            )
    except (subprocess.TimeoutExpired, OSError):
        pass

# Deduplicate by path (stable order)
seen: set[str] = set()
uniq: list[dict] = []
for r in rows:
    p = r["path"]
    if p in seen:
        continue
    seen.add(p)
    uniq.append(r)
rows = uniq

rows.sort(
    key=lambda r: (
        r.get("last_modified_str") or "",
        r.get("commit_hash") or "",
    ),
    reverse=True,
)

counts = Counter(r["repo_id"] for r in rows)
for r in rows:
    rid = r["repo_id"]
    h = r.get("commit_hash") or ""
    short = h[:8] if h else ""
    if counts[rid] > 1 and short:
        label = f"{rid} ({short})"
    else:
        label = rid
    # Tabs/newlines in repo_id are extremely rare; strip for safe TSV
    label = label.replace("\t", " ").replace("\n", " ")
    path = r["path"]
    sys.stdout.write(f"{label}\t{path}\n")
PY
}

echo "🔍 Hub cache directory: $HUB_CACHE"
MODEL="${DEFAULT_MODEL:-}"

if ! command -v "$VLLM_BIN" >/dev/null 2>&1; then
  echo "Error: '$VLLM_BIN' not found on PATH (set VLLM_MLX_BIN to override)." >&2
  exit 1
fi

declare -a LOCAL_MODELS=()
declare -a LOCAL_LABELS=()

while IFS= read -r line || [ -n "${line:-}" ]; do
  [ -z "$line" ] && continue
  IFS=$'\t' read -r label path <<<"$line"
  [ -n "${path:-}" ] && [ -d "$path" ] || continue
  LOCAL_LABELS+=("$label")
  LOCAL_MODELS+=("$path")
done < <(discover_cached_models_tsv "$HUB_CACHE")

if [ "${#LOCAL_MODELS[@]}" -gt 0 ]; then
  echo "Cached models (name only; revision shown if the same repo has multiple snapshots):"
  for i in "${!LOCAL_MODELS[@]}"; do
    idx=$((i + 1))
    echo "  $idx) ${LOCAL_LABELS[$i]}"
  done
  echo ""
  defmsg="${DEFAULT_MODEL:-<none — pick a number>}"
  read -r -p "Select model by number, or press Enter for default ($defmsg): " MODEL_CHOICE
  if [[ "$MODEL_CHOICE" =~ ^[0-9]+$ ]]; then
    MODEL_INDEX=$((MODEL_CHOICE - 1))
    if [ "$MODEL_INDEX" -ge 0 ] && [ "$MODEL_INDEX" -lt "${#LOCAL_MODELS[@]}" ]; then
      MODEL="${LOCAL_MODELS[$MODEL_INDEX]}"
      echo "Selected: ${LOCAL_LABELS[$MODEL_INDEX]}"
    else
      echo "Invalid selection."
      if [ -n "${DEFAULT_MODEL:-}" ]; then
        MODEL="$DEFAULT_MODEL"
        echo "Using default: $MODEL"
      else
        echo "No default set (DEFAULT_MODEL). Exiting." >&2
        exit 1
      fi
    fi
  else
    if [ -n "${DEFAULT_MODEL:-}" ]; then
      MODEL="$DEFAULT_MODEL"
      echo "Using default model: $MODEL"
    else
      echo "No default set (DEFAULT_MODEL). Exiting." >&2
      exit 1
    fi
  fi
else
  echo "No cached model snapshots found under $HUB_CACHE."
  if [ -n "${DEFAULT_MODEL:-}" ]; then
    MODEL="$DEFAULT_MODEL"
    echo "Using DEFAULT_MODEL: $MODEL"
  else
    read -r -p "Enter Hugging Face model id or local path: " MODEL
    if [ -z "$MODEL" ]; then
      echo "No model specified. Exiting." >&2
      exit 1
    fi
  fi
fi

echo ""

DEFAULT_CACHE_PCT="0.30"
read -r -p "KV cache memory fraction (0-1) [${DEFAULT_CACHE_PCT}]: " CACHE_PCT
if [ -z "$CACHE_PCT" ]; then
  CACHE_PCT="$DEFAULT_CACHE_PCT"
fi
if ! awk -v x="$CACHE_PCT" 'BEGIN { exit !(x+0 >= 0 && x+0 <= 1) }' </dev/null 2>/dev/null; then
  echo "⚠️  Invalid cache fraction '$CACHE_PCT' (expected 0–1). Using ${DEFAULT_CACHE_PCT}."
  CACHE_PCT="$DEFAULT_CACHE_PCT"
fi

read -r -p "Load as multimodal MLLM (--mllm) for vision-capable checkpoints? [y/N]: " USE_MLLM
USE_MLLM_ENABLED=false
if [ -n "$USE_MLLM" ]; then
  USE_MLLM_LOWER=$(echo "$USE_MLLM" | tr '[:upper:]' '[:lower:]')
  if [ "$USE_MLLM_LOWER" = "y" ] || [ "$USE_MLLM_LOWER" = "yes" ]; then
    USE_MLLM_ENABLED=true
  fi
fi

read -r -p "Enable ngrok tunnel for Cursor access? [y/N]: " USE_NGROK
USE_NGROK_LOWER=$(echo "$USE_NGROK" | tr '[:upper:]' '[:lower:]')
USE_NGROK_ENABLED=false

if [ "$USE_NGROK_LOWER" = "y" ] || [ "$USE_NGROK_LOWER" = "yes" ]; then
  if command -v ngrok >/dev/null 2>&1; then
    USE_NGROK_ENABLED=true
  else
    echo "⚠️  ngrok not found on PATH. Continuing without public tunnel."
  fi
fi

if command -v lsof >/dev/null 2>&1; then
  if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "Error: something is already listening on port $PORT (set VLLM_PORT to use another)." >&2
    exit 1
  fi
fi

MLLM_TAG="off"
if [ "$USE_MLLM_ENABLED" = true ]; then
  MLLM_TAG="on (--mllm)"
fi
echo "🚀 Starting vLLM-MLX engine with model: $MODEL (cache-memory-percent=$CACHE_PCT, mllm=$MLLM_TAG)"

VLLM_EXTRA=()
if [ "$USE_MLLM_ENABLED" = true ]; then
  VLLM_EXTRA+=(--mllm)
fi
# "${arr[@]}" with set -u errors on empty array (bash); use conditional expansion.
"$VLLM_BIN" serve "$MODEL" \
  "${VLLM_EXTRA[@]+"${VLLM_EXTRA[@]}"}" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --cache-memory-percent "$CACHE_PCT" &
VLLM_PID=$!

echo "⏳ Waiting for the API (up to ${API_READY_TIMEOUT}s)…"
READY=0
for ((i = 1; i <= API_READY_TIMEOUT; i++)); do
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "❌ vllm-mlx exited before the API became ready (see messages above)." >&2
    exit 1
  fi
  if command -v curl >/dev/null 2>&1; then
    if curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
      READY=1
      break
    fi
  else
    # No curl: short fixed delay only
    if [ "$i" -ge 15 ]; then
      READY=1
      break
    fi
  fi
  sleep 1
done

NGROK_PID=""
NGROK_URL=""

if [ "$USE_NGROK_ENABLED" = true ]; then
  echo "🌐 Opening secure ngrok tunnel..."
  ngrok http "$PORT" --log=stdout > /dev/null &
  NGROK_PID=$!

  echo "⏳ Fetching your public URL..."
  sleep 5

  NGROK_URL=$(curl -s http://localhost:4040/api/tunnels 2>/dev/null | grep -o '"public_url":"[^"]*' | grep -o 'https://[^"]*' || true)
fi

echo ""
echo "========================================================"
if [ "$READY" -eq 1 ]; then
  echo "✅ AI WORKSTATION ONLINE"
else
  echo "⚠️  SERVER STARTED (API not confirmed — install curl for readiness checks, or wait longer)"
fi
echo "========================================================"
echo "🏠 Local (VS Code): http://localhost:$PORT/v1"
if [ "$USE_NGROK_ENABLED" = true ]; then
  if [ -n "$NGROK_URL" ]; then
    echo "🌍 Public (Cursor): $NGROK_URL/v1"
    echo "   -> Paste this URL into Cursor's 'Override Base URL' field."
  else
    echo "⚠️ Could not auto-fetch ngrok URL. Open http://localhost:4040 in your browser to see it."
  fi
else
  echo "🌍 Public (Cursor): Disabled (ngrok tunnel not started)."
fi
echo "========================================================"
echo "Press [Ctrl+C] to gracefully shut down the server${USE_NGROK_ENABLED:+ and ngrok}."

cleanup() {
  echo -e "\n🛑 Shutting down vLLM and ngrok..."
  kill "$VLLM_PID" 2>/dev/null || true
  if [ -n "${NGROK_PID:-}" ]; then
    kill "$NGROK_PID" 2>/dev/null || true
  fi
  exit 0
}
trap cleanup SIGINT SIGTERM

wait "$VLLM_PID" || true
