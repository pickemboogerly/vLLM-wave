"""Spawn the vLLM-wave engine, poll /v1/models, optional ngrok."""

from __future__ import annotations

import json
import logging
import os
import shutil
import socket
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

import httpx

logger = logging.getLogger(__name__)


@dataclass
class ServerHandles:
    vllm: subprocess.Popen[str]
    ngrok: subprocess.Popen[str] | None = None
    ngrok_public_url: str | None = None
    stderr_lines: deque[str] = field(default_factory=lambda: deque(maxlen=200))

    def terminate_all(self) -> None:
        for proc in (self.ngrok, self.vllm):
            if proc is None:
                continue
            if proc.poll() is None:
                proc.terminate()
        for proc in (self.ngrok, self.vllm):
            if proc is None:
                continue
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()


def default_port() -> int:
    raw = os.environ.get("VLLM_PORT", "8001")
    try:
        port = int(raw)
    except ValueError:
        return 8001
    if 1 <= port <= 65535:
        return port
    return 8001


def default_host() -> str:
    # Localhost by default to avoid accidental LAN exposure.
    host = os.environ.get("VLLM_HOST", "127.0.0.1").strip()
    return host or "127.0.0.1"


def vllm_bin() -> str:
    return os.environ.get("VLLM_MLX_BIN", "vllm-mlx")


def api_ready_timeout() -> int:
    raw = os.environ.get("API_READY_TIMEOUT", "120")
    try:
        timeout = int(raw)
    except ValueError:
        return 120
    return max(1, timeout)


def client_connect_host(bind_host: str) -> str:
    """
    Hostname to use for local HTTP clients given the server's --host bind address.

    Binds like 0.0.0.0 or :: mean "all interfaces"; we probe and chat via loopback.
    """
    h = (bind_host or "").strip()
    if not h:
        return "127.0.0.1"
    low = h.lower()
    if low == "0.0.0.0":
        return "127.0.0.1"
    if low in ("::", "[::]"):
        return "::1"
    return h


def api_base_url(bind_host: str, port: int) -> str:
    """Base URL (no trailing slash) for OpenAI-compatible routes on this machine."""
    host = client_connect_host(bind_host)
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    return f"http://{host}:{port}"


def port_in_use(port: int, bind_host: str | None = None) -> bool:
    connect_host = client_connect_host(
        bind_host if bind_host is not None else default_host()
    )
    try:
        sock = socket.create_connection((connect_host, port), timeout=0.5)
    except OSError:
        return False
    try:
        sock.close()
    except OSError:
        pass
    return True


def tool_call_parser_for_run(
    *, force_off: bool, explicit_override: str | None
) -> str | None:
    """
    Resolve which tool-call parser to pass to vllm-mlx, if any.

    When force_off is True, never pass tool flags.
    When explicit_override is non-empty, it wins over VLLM_TOOL_CALL_PARSER.
    Otherwise use VLLM_TOOL_CALL_PARSER when set; omit flags when unset (default).
    """
    if force_off:
        return None
    if explicit_override is not None and explicit_override.strip():
        return explicit_override.strip()
    env = os.environ.get("VLLM_TOOL_CALL_PARSER", "").strip()
    return env or None


def _stderr_reader(proc: subprocess.Popen[str], lines: deque[str]) -> None:
    if proc.stderr is None:
        return
    try:
        for line in proc.stderr:
            lines.append(line.rstrip("\n\r"))
    except Exception:
        logger.debug("stderr reader exited with error", exc_info=True)


def build_serve_argv(
    model: str,
    host: str,
    port: int,
    cache_memory_percent: float,
    mllm: bool,
    tool_call_parser: str | None = None,
) -> list[str]:
    cmd = [
        vllm_bin(),
        "serve",
        model,
        "--host",
        host,
        "--port",
        str(port),
        "--cache-memory-percent",
        str(cache_memory_percent),
    ]
    if tool_call_parser:
        cmd.extend(
            ["--enable-auto-tool-choice", "--tool-call-parser", tool_call_parser]
        )
    if mllm:
        cmd.append("--mllm")
    return cmd


def start_vllm(
    model: str,
    host: str,
    port: int,
    cache_memory_percent: float,
    mllm: bool,
    tool_call_parser: str | None = None,
    stderr_lines: deque[str] | None = None,
) -> tuple[subprocess.Popen[str], deque[str]]:
    lines = stderr_lines if stderr_lines is not None else deque(maxlen=200)
    proc = subprocess.Popen(
        build_serve_argv(
            model, host, port, cache_memory_percent, mllm, tool_call_parser
        ),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    t = threading.Thread(
        target=_stderr_reader, args=(proc, lines), daemon=True
    )
    t.start()
    return proc, lines


def wait_for_models_endpoint(
    base_url: str,
    timeout_sec: int,
    proc: subprocess.Popen[str],
    on_tick: Callable[[int], None] | None = None,
) -> bool:
    url = base_url.rstrip("/") + "/v1/models"
    deadline = time.monotonic() + timeout_sec
    i = 0
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return False
        i += 1
        if on_tick:
            on_tick(i)
        try:
            r = httpx.get(url, timeout=2.0)
            if r.status_code == 200:
                return True
        except httpx.HTTPError:
            pass
        time.sleep(1)
    return False


def start_ngrok(port: int) -> subprocess.Popen[str] | None:
    ngrok = shutil.which("ngrok")
    if not ngrok:
        return None
    return subprocess.Popen(
        [ngrok, "http", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def fetch_ngrok_public_url() -> str | None:
    try:
        r = httpx.get("http://127.0.0.1:4040/api/tunnels", timeout=5.0)
        r.raise_for_status()
        data = r.json()
        tunnels = data.get("tunnels") or []
        for t in tunnels:
            u = t.get("public_url") or ""
            if u.startswith("https://"):
                return u
        for t in tunnels:
            u = t.get("public_url") or ""
            if u.startswith("http://"):
                return u
    except httpx.HTTPError:
        pass
    return None


def wait_for_ngrok_url(sleep_sec: float = 5.0, retries: int = 6) -> str | None:
    time.sleep(sleep_sec)
    for _ in range(retries):
        url = fetch_ngrok_public_url()
        if url:
            return url
        time.sleep(1)
    return None


def first_model_id_from_api(base_url: str) -> str | None:
    try:
        r = httpx.get(base_url.rstrip("/") + "/v1/models", timeout=5.0)
        r.raise_for_status()
        body = r.json()
        data = body.get("data") or []
        if data and isinstance(data[0], dict):
            mid = data[0].get("id")
            if isinstance(mid, str):
                return mid
    except (httpx.HTTPError, json.JSONDecodeError, KeyError):
        pass
    return None


def ensure_vllm_on_path() -> str | None:
    """Return error message if binary missing, else None."""
    b = vllm_bin()
    if shutil.which(b):
        return None
    return f"'{b}' not found on PATH (set VLLM_MLX_BIN)."


_GGUF_UNSUPPORTED_MSG = (
    "vllm-mlx loads weights via mlx_lm, which expects `*.safetensors` (HF/MLX layout). "
    "GGUF weights are not supported. Use an MLX or safetensors model directory / Hub id, "
    "or run GGUF models with a GGUF-native runtime (e.g. llama.cpp, LM Studio)."
)


def _dir_has_extension(d: str, suffix: str) -> bool:
    suffix = suffix.lower()
    try:
        for name in os.listdir(d):
            if name.lower().endswith(suffix) and os.path.isfile(os.path.join(d, name)):
                return True
    except OSError:
        pass
    return False


def resolve_model_arg_for_vllm_serve(model: str) -> tuple[str, str | None]:
    """
    Validate local paths before spawning vllm-mlx.

    mlx_lm requires a model directory with `config.json` and `*.safetensors` weights.
    GGUF-only paths are rejected up front.
    """
    m = (model or "").strip()
    if not m:
        return m, None

    if os.path.isfile(m) and m.lower().endswith(".gguf"):
        return m, _GGUF_UNSUPPORTED_MSG

    if os.path.isdir(m):
        cfg = os.path.join(m, "config.json")
        if not os.path.isfile(cfg):
            return (
                m,
                "Selected model directory is missing `config.json`, which `vllm-mlx` requires. "
                "Use an MLX/Hugging Face model directory that includes config/tokenizer files.",
            )
        has_st = _dir_has_extension(m, ".safetensors")
        has_gguf = _dir_has_extension(m, ".gguf")
        if has_gguf and not has_st:
            return m, _GGUF_UNSUPPORTED_MSG
        return m, None

    return m, None
