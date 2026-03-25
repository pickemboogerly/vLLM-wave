"""Spawn the vLLM-wave engine, poll /v1/models, optional ngrok."""

from __future__ import annotations

import json
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


def port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def _stderr_reader(proc: subprocess.Popen[str], lines: deque[str]) -> None:
    if proc.stderr is None:
        return
    try:
        for line in proc.stderr:
            lines.append(line.rstrip("\n\r"))
    except Exception:
        pass


def build_serve_argv(
    model: str,
    host: str,
    port: int,
    cache_memory_percent: float,
    mllm: bool,
) -> list[str]:
    cmd = [
        vllm_bin(),
        "serve",
        model,
        "--host",
        host,
        "--port",
        str(port),
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "qwen3_coder",
        "--cache-memory-percent",
        str(cache_memory_percent),
    ]
    if mllm:
        cmd.append("--mllm")
    return cmd


def start_vllm(
    model: str,
    host: str,
    port: int,
    cache_memory_percent: float,
    mllm: bool,
    stderr_lines: deque[str] | None = None,
) -> tuple[subprocess.Popen[str], deque[str]]:
    lines = stderr_lines if stderr_lines is not None else deque(maxlen=200)
    proc = subprocess.Popen(
        build_serve_argv(model, host, port, cache_memory_percent, mllm),
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
    port: int,
    timeout_sec: int,
    proc: subprocess.Popen[str],
    on_tick: Callable[[int], None] | None = None,
) -> bool:
    url = f"http://127.0.0.1:{port}/v1/models"
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


def first_model_id_from_api(port: int) -> str | None:
    try:
        r = httpx.get(f"http://127.0.0.1:{port}/v1/models", timeout=5.0)
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
