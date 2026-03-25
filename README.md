# vLLM-wave

Helpers for running **vLLM-wave** on Apple Silicon with models from your local Hugging Face Hub cache (MLX / `*.safetensors` layouts; GGUF is not supported by `vllm-mlx`).

## TUI harness (vLLM-wave)

A [Textual](https://textual.textualize.io/) wizard lists cached Hugging Face snapshot directories, lets you set KV memory fraction, optional tool-call parser, multimodal (`--mllm`) / ngrok options, starts `vllm-mlx serve` in a **subprocess** (weights never load in the UI process), waits for `GET /v1/models` on the correct client URL for your `VLLM_HOST` bind, then **hands off** to a small streaming chat client (`httpx` only) so memory stays lower during long chats.

### Requirements

- Python **3.10+**
- **`vllm-mlx`** on your `PATH` (override with `VLLM_MLX_BIN`)
- **`huggingface_hub`** is pulled in by this package for cache scanning (same env as vLLM is fine)

### Install

```bash
cd /path/to/vllm-wave
python3 -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -e .
```

### Run

**Interactive TUI** (recommended):

```bash
python -m vllm_wave
# or, after install:
vllm-wave
# convenience script (uses .venv if present in this repo):
./start_ai_tui.sh
```

**Non-interactive** (scripting / CI):

```bash
python -m vllm_wave --no-tui --model /path/to/hf/hub/snapshots/<hash>
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
| `--tool-parser <name>` | Pass `--enable-auto-tool-choice` and `--tool-call-parser` (overrides `VLLM_TOOL_CALL_PARSER`) |
| `--no-tool-parser` | Do not pass tool-call flags (ignores `VLLM_TOOL_CALL_PARSER`) |

### Environment variables

| Variable | Role |
|----------|------|
| `HF_HUB_CACHE` / `HF_HOME` | Hub cache root (see [HF cache docs](https://huggingface.co/docs/huggingface_hub/guides/manage-cache)) |
| `VLLM_PORT` | API port (default `8001`) |
| `VLLM_HOST` | Bind host for the server (default `127.0.0.1`). Readiness checks and the chat client use a matching local URL (e.g. bind `0.0.0.0` → connect via `127.0.0.1`) |
| `VLLM_TOOL_CALL_PARSER` | If set (non-empty), enables auto tool choice with this parser name (e.g. `qwen3_coder`). If unset, those flags are omitted so other models are not forced into one parser |
| `VLLM_MLX_BIN` | Server binary name or path (default `vllm-mlx`) |
| `API_READY_TIMEOUT` | Seconds to wait for `/v1/models` (default `120`) |
| `DEFAULT_MODEL` | Prefill manual model field in the TUI, or default for `--no-tui` |

### Shell scripts

- `start_ai_tui.sh` — convenience wrapper to run the `vllm_wave` Textual launcher

The Python TUI provides a richer terminal UI and a deliberate post-load chat handoff.

## License

This project (the code) is licensed under the MIT License. Model weights and any third-party model assets are subject to their own licenses.
