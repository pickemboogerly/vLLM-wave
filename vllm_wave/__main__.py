"""CLI for vLLM-wave: launcher wizard or --no-tui pipeline, then chat."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
from collections import deque
from typing import Any

from vllm_wave.app import VllmHarnessApp, WizardResult
from vllm_wave.cache import discover_cached_models
from vllm_wave.chat import run_interactive_chat
from vllm_wave.server import (
    ServerHandles,
    api_base_url,
    api_ready_timeout,
    default_host,
    default_port,
    ensure_vllm_on_path,
    first_model_id_from_api,
    port_in_use,
    start_ngrok,
    start_vllm,
    resolve_model_arg_for_vllm_serve,
    tool_call_parser_for_run,
    wait_for_models_endpoint,
    wait_for_ngrok_url,
)


def _install_cleanup(handles_holder: dict[str, Any]) -> None:
    def cleanup(_sig: int | None = None, _frame: object | None = None) -> None:
        h = handles_holder.get("handles")
        if isinstance(h, ServerHandles):
            h.terminate_all()
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)


def _boot_cli(
    model: str,
    host: str,
    port: int,
    cache_pct: float,
    mllm: bool,
    want_ngrok: bool,
    tool_call_parser: str | None,
) -> tuple[ServerHandles, str, str, str | None]:
    err = ensure_vllm_on_path()
    if err:
        print(err, file=sys.stderr)
        sys.exit(1)
    if port_in_use(port, host):
        print(f"Port {port} in use (set VLLM_PORT).", file=sys.stderr)
        sys.exit(1)

    model_for_serve, resolution_err = resolve_model_arg_for_vllm_serve(model)
    if resolution_err:
        print(resolution_err, file=sys.stderr)
        sys.exit(2)

    stderr_buf: deque[str] = deque(maxlen=200)
    proc, lines = start_vllm(
        model_for_serve,
        host,
        port,
        cache_pct,
        mllm,
        tool_call_parser=tool_call_parser,
        stderr_lines=stderr_buf,
    )
    timeout = api_ready_timeout()
    base = api_base_url(host, port)
    ok = wait_for_models_endpoint(base, timeout, proc, on_tick=lambda _i: None)
    if not ok:
        proc.terminate()
        try:
            proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            proc.kill()
        tail = "\n".join(list(lines)[-40:])
        print(
            f"vLLM-wave engine did not become ready in {timeout}s.\n{tail}",
            file=sys.stderr,
        )
        sys.exit(1)
    ngrok_proc = None
    ngrok_url = None
    ngrok_hint = None
    if want_ngrok:
        ngrok_proc = start_ngrok(port)
        if ngrok_proc is None:
            ngrok_hint = "ngrok not on PATH; skipped."
        else:
            ngrok_url = wait_for_ngrok_url()
            if ngrok_url:
                ngrok_hint = f"Public: {ngrok_url}/v1"
            else:
                ngrok_hint = "Ngrok running; see http://127.0.0.1:4040"
    chat_id = first_model_id_from_api(base) or os.path.basename(
        model_for_serve.rstrip("/")
    )
    handles = ServerHandles(
        vllm=proc,
        ngrok=ngrok_proc,
        ngrok_public_url=ngrok_url,
        stderr_lines=lines,
    )
    return handles, base, chat_id, ngrok_hint


def main() -> None:
    p = argparse.ArgumentParser(description="vLLM-wave TUI harness")
    p.add_argument(
        "--no-tui",
        action="store_true",
        help="Non-interactive: requires --model (or DEFAULT_MODEL env)",
    )
    p.add_argument(
        "--model",
        default="",
        help="Hub id or local snapshot path (non-interactive / default pick)",
    )
    p.add_argument(
        "--cache-pct",
        type=float,
        default=0.30,
        help="KV cache memory fraction 0–1 (default 0.30)",
    )
    p.add_argument("--mllm", action="store_true", help="Pass --mllm to vllm-mlx")
    p.add_argument("--ngrok", action="store_true", help="Start ngrok http tunnel")
    p.add_argument(
        "--skip-chat",
        action="store_true",
        help="After server is ready, do not start interactive chat",
    )
    p.add_argument(
        "--tool-parser",
        metavar="NAME",
        default=None,
        help=(
            "Enable auto tool choice with this vLLM parser name "
            "(overrides VLLM_TOOL_CALL_PARSER; e.g. qwen3_coder)"
        ),
    )
    p.add_argument(
        "--no-tool-parser",
        action="store_true",
        help="Do not pass tool-call flags (ignores VLLM_TOOL_CALL_PARSER)",
    )
    args = p.parse_args()

    if args.no_tool_parser and args.tool_parser is not None:
        print(
            "Use only one of --no-tool-parser and --tool-parser.",
            file=sys.stderr,
        )
        sys.exit(2)

    handles_holder: dict[str, Any] = {"handles": None}

    if args.no_tui:
        model = args.model.strip() or os.environ.get("DEFAULT_MODEL", "").strip()
        if not model:
            print(
                "With --no-tui, pass --model or set DEFAULT_MODEL.",
                file=sys.stderr,
            )
            sys.exit(2)
        pct = args.cache_pct
        if not 0.0 <= pct <= 1.0:
            print("--cache-pct must be between 0 and 1.", file=sys.stderr)
            sys.exit(2)
        port = default_port()
        host = default_host()
        tparser = tool_call_parser_for_run(
            force_off=args.no_tool_parser,
            explicit_override=args.tool_parser,
        )
        handles, base, chat_id, hint = _boot_cli(
            model, host, port, pct, args.mllm, args.ngrok, tparser
        )
        handles_holder["handles"] = handles
        _install_cleanup(handles_holder)
        print(f"Local API: {base}/v1", file=sys.stderr)
        if hint:
            print(hint, file=sys.stderr)
        if args.skip_chat:
            print("Server running. Ctrl+C to stop.", file=sys.stderr)
            handles.vllm.wait()
        else:
            run_interactive_chat(base, chat_id)
        handles.terminate_all()
        return

    _install_cleanup(handles_holder)
    while True:
        models = discover_cached_models()
        app = VllmHarnessApp(
            models,
            cli_tool_parser=args.tool_parser,
            cli_no_tool_parser=args.no_tool_parser,
        )
        result: WizardResult | None = app.run()

        if result is None:
            sys.exit(0)

        handles_holder["handles"] = result.handles
        print(f"\nLocal API: {result.base_url}/v1", file=sys.stderr)
        if result.ngrok_hint:
            print(result.ngrok_hint, file=sys.stderr)

        if args.skip_chat:
            print("Server running. Ctrl+C to stop.", file=sys.stderr)
            result.handles.vllm.wait()
            result.handles.terminate_all()
            break

        chat_exit_reason = run_interactive_chat(result.base_url, result.chat_model_id)
        result.handles.terminate_all()
        handles_holder["handles"] = None
        if chat_exit_reason != "switch_model":
            break


if __name__ == "__main__":
    main()
