#!/bin/bash

# Configuration
#DEFAULT_MODEL="mlx-community/Qwen2.5-Coder-14B-Instruct-4bit"
#DEFAULT_MODEL="mlx-community/Qwen3.5-27B-8bit"
#DEFAULT_MODEL="mlx-community/Qwen3.5-27B-heretic-8bit"
#DEFAULT_MODEL="mlx-community/Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-6bit"
#DEFAULT_MODEL="Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled"
#DEFAULT_MODEL="mlx-community/GLM-4.6V-Flash-4bit"
#txgsync/gpt-oss-20b-Derestricted-mxfp4-mlx
#mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit
#DEFAULT_MODEL="HauhauCS/Qwen3.5-27B-Uncensored-HauhauCS-Aggressive"
#DEFAULT_MODEL="Qwen3.5-27B-Uncensored-HauhauCS-Aggressive-Q6_K.gguf"


MODEL_DIR="$HOME/.cache/huggingface/hub/"
PORT=8001

echo "🔍 Checking for local models under subdirectories of '$MODEL_DIR'..."
MODEL="$DEFAULT_MODEL"

if [ -d "$MODEL_DIR" ]; then
  MD="${MODEL_DIR%/}"
  # Prefer real model roots (HF/MLX: directory containing config.json), only under subdirs of MODEL_DIR.
  mapfile -t LOCAL_MODELS < <(
    find "$MD" -mindepth 2 -type f -name config.json 2>/dev/null \
      | while IFS= read -r f; do dirname "$f"; done \
      | sort -u
  )
  if [ "${#LOCAL_MODELS[@]}" -eq 0 ]; then
    mapfile -t LOCAL_MODELS < <(find "$MD" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort -u)
  fi

  if [ "${#LOCAL_MODELS[@]}" -gt 0 ]; then
    echo "Available local models:"
    for i in "${!LOCAL_MODELS[@]}"; do
      idx=$((i + 1))
      rel="${LOCAL_MODELS[$i]#$MD/}"
      echo "  $idx) $rel"
    done

    read -r -p "Select a model by number, or press Enter to use default ($DEFAULT_MODEL): " MODEL_CHOICE

    if [[ "$MODEL_CHOICE" =~ ^[0-9]+$ ]]; then
      MODEL_INDEX=$((MODEL_CHOICE - 1))
      if [ "$MODEL_INDEX" -ge 0 ] && [ "$MODEL_INDEX" -lt "${#LOCAL_MODELS[@]}" ]; then
        MODEL="${LOCAL_MODELS[$MODEL_INDEX]}"
      else
        echo "Invalid selection. Using default model: $DEFAULT_MODEL"
        MODEL="$DEFAULT_MODEL"
      fi
    else
      echo "Using default model: $DEFAULT_MODEL"
      MODEL="$DEFAULT_MODEL"
    fi
  else
    echo "No models found under '$MODEL_DIR'. Using default model: $DEFAULT_MODEL"
  fi
else
  echo "Model directory '$MODEL_DIR' not found. Using default model: $DEFAULT_MODEL"
fi

echo ""

# Configure KV cache memory usage (fraction of total RAM)
DEFAULT_CACHE_PCT="0.30"
read -r -p "KV cache memory fraction (0-1) [${DEFAULT_CACHE_PCT}]: " CACHE_PCT
if [ -z "$CACHE_PCT" ]; then
  CACHE_PCT="$DEFAULT_CACHE_PCT"
fi

# vLLM-MLX only auto-detects VL models from the id (e.g. -VL-). Checkpoints that still
# include vision weights (language_model.vision_tower.*) need --mllm or loading fails.
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

MLLM_TAG="off"
if [ "$USE_MLLM_ENABLED" = true ]; then
  MLLM_TAG="on (--mllm)"
fi
echo "🚀 Starting vLLM-MLX engine with model: $MODEL (cache-memory-percent=$CACHE_PCT, mllm=$MLLM_TAG)"
# Start vLLM in the background
VLLM_EXTRA=()
if [ "$USE_MLLM_ENABLED" = true ]; then
  VLLM_EXTRA+=(--mllm)
fi
# "${arr[@]}" with set -u errors on empty array (bash); use conditional expansion.
vllm-mlx serve "$MODEL" \
  "${VLLM_EXTRA[@]+"${VLLM_EXTRA[@]}"}" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --cache-memory-percent "$CACHE_PCT" &
VLLM_PID=$!

echo "⏳ Waiting for the local API to spin up (10 seconds)..."
sleep 10

NGROK_PID=""
NGROK_URL=""

if [ "$USE_NGROK_ENABLED" = true ]; then
  echo "🌐 Opening secure ngrok tunnel..."
  ngrok http "$PORT" --log=stdout > /dev/null &
  NGROK_PID=$!

  echo "⏳ Fetching your public URL..."
  sleep 5 # Give ngrok a few seconds to establish the connection

  # Query ngrok's local API to grab the generated URL
  NGROK_URL=$(curl -s http://localhost:4040/api/tunnels | grep -o '"public_url":"[^"]*' | grep -o 'https://[^"]*')
fi

echo ""
echo "========================================================"
echo "✅ AI WORKSTATION ONLINE"
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

# Trap Ctrl+C (SIGINT) to kill background processes cleanly
trap 'echo -e "\n🛑 Shutting down vLLM and ngrok..."; kill "$VLLM_PID" ${NGROK_PID:+$NGROK_PID}; exit' SIGINT

# Keep the script running
wait
