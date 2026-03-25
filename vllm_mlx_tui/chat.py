"""Minimal OpenAI-compatible streaming chat client (post–TUI handoff)."""

from __future__ import annotations

import json
import sys
from typing import Iterator

import httpx


def stream_chat_chunks(
    base_url: str,
    model: str,
    messages: list[dict],
    timeout: float = 300.0,
) -> Iterator[str]:
    url = base_url.rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
    }
    with httpx.Client(timeout=timeout) as client:
        with client.stream("POST", url, json=payload) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                if line.startswith("data: "):
                    data = line[6:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        obj = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    choices = obj.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    content = delta.get("content")
                    if content:
                        yield content


def run_interactive_chat(base_url: str, model: str) -> None:
    print(
        f"Chatting with model {model!r} at {base_url}\n"
        "Commands: /quit or Ctrl+D to exit; /system <text> sets system message.\n",
        file=sys.stderr,
    )
    messages: list[dict] = []
    while True:
        try:
            user = input("You: ").strip()
        except EOFError:
            print(file=sys.stderr)
            break
        if not user:
            continue
        if user.lower() in ("/quit", "/q", "exit"):
            break
        if user.lower().startswith("/system "):
            text = user[len("/system ") :].strip()
            rest = [m for m in messages if m.get("role") != "system"]
            messages = [{"role": "system", "content": text}] + rest
            print("System message updated.", file=sys.stderr)
            continue
        messages.append({"role": "user", "content": user})
        print("Assistant: ", end="", flush=True)
        buf: list[str] = []
        try:
            for piece in stream_chat_chunks(base_url, model, messages):
                print(piece, end="", flush=True)
                buf.append(piece)
            print()
            messages.append({"role": "assistant", "content": "".join(buf)})
        except httpx.HTTPStatusError as e:
            print(f"\n[HTTP {e.response.status_code}] {e.response.text[:500]}", file=sys.stderr)
            messages.pop()
            continue
        except httpx.HTTPError as e:
            print(f"\n[Error] {e}", file=sys.stderr)
            messages.pop()
            continue
