# Product Requirements Document: vLLM-wave

**Version:** 0.1.0 (as of codebase)  
**Purpose:** Single reference for current behavior, constraints, and intentional gaps—useful for refactors and roadmap work without re-deriving behavior from code.

---

## 1. Product overview

**vLLM-wave** is a Python terminal application that:

1. Helps users pick a locally cached Hugging Face model (MLX / safetensors layout), configure a few runtime options, and start **`vllm-mlx serve`** as a **subprocess** (weights do not load in the UI process).
2. Waits until the OpenAI-compatible **`GET /v1/models`** endpoint responds on the correct client URL for the configured bind host.
3. Hands off to a **Textual chat UI** that talks to the server over HTTP with **`httpx`** only (streaming chat completions), keeping the launcher process lighter during long sessions.

**Out of scope (today):** Serving GGUF-only weights with `vllm-mlx` (explicitly unsupported upstream); embedding the inference engine inside the TUI process.

---

## 2. Goals

| Goal | Notes |
|------|--------|
| Discover models from local HF Hub cache | Primary path: `huggingface_hub.scan_cache_dir`; fallback: `find` for `*/snapshots/*/config.json`. |
| Safe local defaults | Default bind `127.0.0.1`; client URL maps `0.0.0.0` / `::` to loopback for checks and chat. |
| Observable startup | vLLM stderr streamed into a boot log in the launcher TUI; copy boot log (`c`) and file fallback under `~/.cache/vllm-wave/boot_log.txt`. |
| Usable chat UX | Multi-session sidebar, streaming replies, optional system prompt per chat, markdown rendering, copy selection, switch model without restarting the whole CLI. |
| Scriptable path | `--no-tui` pipeline for CI/automation with the same boot semantics. |

---

## 3. Non-goals

- Replacing or wrapping `vllm-mlx` with a different backend.
- Full IDE-style editing or collaborative chat.
- Guaranteed clipboard support in all terminals (OSC52 + file fallback for boot log only; chat copy uses screen selection + app clipboard APIs).

---

## 4. User-facing surfaces

### 4.1 Launcher wizard (Textual)

**Entry:** `python -m vllm_wave`, `vllm-wave`, or `./start_ai_tui.sh`.

**Capabilities:**

- **Model table:** Rows from `discover_cached_models()` — columns **Model** (human-readable label) and **Path** (snapshot directory). Labels use Hub `repo_id` and optional short revision hash when multiple revisions exist for the same repo.
- **Manual model field:** Overrides table when non-empty; prefilled from `DEFAULT_MODEL` if set.
- **Options:** KV cache fraction (`--cache-memory-percent`), **Multimodal** (`--mllm`), **Ngrok** tunnel (if `ngrok` on `PATH`), **Disable tool-call flags**, **Parser override** (or env-driven default).
- **Boot:** Validates `vllm-mlx` on PATH, port free, model path/hub id (GGUF / invalid dir rejected early), clears boot log, disables Start/Quit during boot, worker thread starts server and polls readiness.
- **Exit:** Success → `WizardResult` (handles, base URL, chat model id, **display name**, optional ngrok hint). Quit → `None`.

**Bindings:** `q` quit (when not booting), `c` copy boot log.

### 4.2 Server process

**Binary:** `VLLM_MLX_BIN` (default `vllm-mlx`).

**Invocation (conceptual):** `serve <model> --host … --port … --cache-memory-percent …` plus optional `--mllm`, optional `--enable-auto-tool-choice` / `--tool-call-parser` when resolved.

**Readiness:** Poll `GET {base}/v1/models` until 200 or timeout (`API_READY_TIMEOUT`, default 120s). On failure, terminate subprocess (with kill fallback).

**Ngrok:** Optional child process; public URL may be discovered via local ngrok API; hint surfaced in UI/stderr.

**Lifecycle:** `ServerHandles` holds `vllm`, optional `ngrok`, stderr deque; `terminate_all()` stops tunnel then engine.

### 4.3 Chat TUI (Textual)

**Entry:** After successful boot when `--skip-chat` is not set.

**Protocol:** `POST {base}/v1/chat/completions` with `stream: true`; parses SSE `data:` lines and deltas.

**Features:**

- **Sessions:** New chat, list selection, first user message can rename session title (truncated).
- **Per-session system prompt** via optional input (applied to API payload for that chat).
- **Assistant streaming** with Stop button / cancel flag; Send disabled while streaming.
- **Transcript:** Built as markdown (`### System` / `### You` / `### Assistant`, horizontal rules); rendered with Textual `Markdown` inside `VerticalScroll`. Throttled redraw during streaming (~80ms) to limit cost.
- **Model display:** UI shows **human-readable** name when provided (e.g. `models--org/model` from HF cache path); API `model` id remains the value returned by `/v1/models` (or basename fallback).
- **Copy:** Select text in chat; **Ctrl+C** / **Cmd+C** (screen copy) or **Copy** button copies selection via `get_selected_text()` + `copy_to_clipboard`.
- **Switch model:** **Ctrl+R** or **Switch Model** exits chat with reason `switch_model`; outer CLI loop terminates server and reopens launcher wizard.

**Bindings:** Ctrl+N new chat, Ctrl+L clear, Ctrl+R switch model, Ctrl+Q quit.

### 4.4 Non-interactive CLI (`--no-tui`)

**Requires:** `--model` or `DEFAULT_MODEL`.

**Same boot path** as TUI (port, validation, readiness, optional ngrok). Then chat unless `--skip-chat` (server runs until Ctrl+C). Cleanup on exit and signals.

---

## 5. Configuration

### 5.1 CLI flags (summary)

| Flag | Role |
|------|------|
| `--no-tui` | Non-interactive mode |
| `--model` | Hub id or path |
| `--cache-pct` | KV cache memory fraction 0–1 |
| `--mllm` | Pass `--mllm` to server |
| `--ngrok` | Start ngrok http tunnel |
| `--skip-chat` | Do not launch chat; keep server running |
| `--tool-parser` | Tool parser name (mutually exclusive with `--no-tool-parser`) |
| `--no-tool-parser` | Omit tool-call flags |

### 5.2 Environment variables

| Variable | Role |
|----------|------|
| `HF_HUB_CACHE` / `HF_HOME` | Hub cache root for discovery |
| `VLLM_PORT` | Listen port (default `8001`) |
| `VLLM_HOST` | Bind address (default `127.0.0.1`) |
| `VLLM_MLX_BIN` | Server executable name/path |
| `VLLM_TOOL_CALL_PARSER` | If set, enables tool parser with this name (unless overridden/disabled by CLI) |
| `API_READY_TIMEOUT` | Seconds to wait for `/v1/models` |
| `DEFAULT_MODEL` | Prefill manual model / `--no-tui` default |

---

## 6. Technical constraints

- **Python:** 3.10+.
- **Dependencies:** `textual`, `httpx`, `huggingface_hub` (see `pyproject.toml`).
- **Platform focus:** Apple Silicon + MLX-oriented stack; README positions the project for local HF cache + `vllm-mlx`.
- **Model validation:** Local directories must be usable by `mlx_lm` / vLLM path rules (e.g. `config.json`, safetensors); GGUF-only layouts rejected with a clear message.

---

## 7. Human-readable model naming

- **Display strings** for HF cache paths under `…/models--org--name/snapshots/<hash>` map to **`models--org/name`** for UI labels.
- **Chat API** continues to use the model **id** from `/v1/models` when available.

---

## 8. Known limitations (acceptable for v0.1)

- Full markdown rebuild on each transcript refresh (throttled during streaming); very long histories may be heavy.
- User/assistant content in markdown can theoretically interact with markdown syntax (e.g. headings in user text); acceptable tradeoff for rich rendering.
- `find` fallback for cache discovery is best-effort and environment-dependent.
- Single port / single server instance per process flow; no multi-server orchestration.

---

## 9. Future work / refactor hooks (not committed)

Use this section for backlog items; verify against code before implementing.

- **Split packages:** `vllm_wave.server` (spawn + readiness), `vllm_wave.cache`, `vllm_wave.tui.launcher`, `vllm_wave.tui.chat`, `vllm_wave.cli`.
- **Tests:** Expand integration tests (mocked subprocess), cache label edge cases, chat stream parser.
- **Chat transcript:** Incremental markdown updates or block-based UI to avoid full-document reparses.
- **Accessibility:** Footer hints and binding discoverability for copy/switch model.
- **Configuration file:** Optional TOML/YAML for default host, port, tool parser—today env + CLI only.

---

## 10. Document maintenance

- Update this PRD when user-visible behavior, CLI, or env contract changes.
- **Source of truth:** `vllm_wave/*.py`, `README.md`, and `tests/`—if the PRD disagrees with code, treat code as authoritative until the doc is fixed.
