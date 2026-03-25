# local-ai

Helpers for running **vLLM-MLX** on Apple Silicon with models from your local Hugging Face Hub cache.

## TUI harness (`vllm_mlx_tui`)

A [Textual](https://textual.textualize.io/) wizard lists cached snapshot directories, lets you set KV memory fraction and multimodal (`--mllm`) / ngrok options, starts `vllm-mlx serve` in a **subprocess** (weights never load in the UI process), waits for `GET /v1/models`, then **hands off** to a small streaming chat client (`httpx` only) so memory stays lower during long chats.

### Requirements

- Python **3.10+**
- **`vllm-mlx`** on your `PATH` (override with `VLLM_MLX_BIN`)
- **`huggingface_hub`** is pulled in by this package for cache scanning (same env as vLLM is fine)

### Install

```bash
cd /path/to/local-ai
python3 -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -e .
```

### Run

**Interactive TUI** (recommended):

```bash
python -m vllm_mlx_tui
# or, after install:
vllm-mlx-tui
# convenience script (uses .venv if present in this repo):
./start_ai_tui.sh
```

**Non-interactive** (scripting / CI):

```bash
python -m vllm_mlx_tui --no-tui --model /path/to/hf/hub/snapshots/<hash>
```

Use `--model` or set `DEFAULT_MODEL` to a Hub id or absolute snapshot path.

### CLI options

| Flag | Meaning |
|------|--------|
| `--no-tui` | Skip Textual; require `--model` or `DEFAULT_MODEL` |
| `--model <path\|id>` | Model path or Hub id |
| `--cache-pct <0–1>` | KV cache memory fraction (default `0.30`) |
| `--mllm` | Pass `--mllm` to the server |
| `--ngrok` | Start an `ngrok http` tunnel if `ngrok` is on `PATH` |
| `--skip-chat` | Keep server running after ready; no REPL (stop with Ctrl+C) |

### Environment variables

| Variable | Role |
|----------|------|
| `HF_HUB_CACHE` / `HF_HOME` | Hub cache root (see [HF cache docs](https://huggingface.co/docs/huggingface_hub/guides/manage-cache)) |
| `VLLM_PORT` | API port (default `8001`) |
| `VLLM_HOST` | Bind host for the server (default `127.0.0.1`) |
| `VLLM_MLX_BIN` | Server binary name or path (default `vllm-mlx`) |
| `API_READY_TIMEOUT` | Seconds to wait for `/v1/models` (default `120`) |
| `DEFAULT_MODEL` | Prefill manual model field in the TUI, or default for `--no-tui` |

### Shell scripts (legacy)

- `start_ai_cached_models.sh` — bash flow with HF cache discovery and `curl` readiness
- `start_ai.sh` — simpler bash menu

The Python TUI mirrors that behavior with a nicer terminal UI and a deliberate post-load chat handoff.

## License

Use and modify freely for local development; vLLM-MLX and model weights are subject to their own licenses.
