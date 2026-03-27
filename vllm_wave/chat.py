"""Textual chat UI for OpenAI-compatible streaming completions."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterator

import httpx
from textual import on, work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import (
    Button,
    Footer,
    Header,
    Input,
    Label,
    ListItem,
    ListView,
    Markdown,
    Static,
)

# Full chat redraw cost scales with history; throttle during streaming token delivery.
_STREAM_UI_MIN_INTERVAL_SEC = 0.08


@dataclass
class ChatMessage:
    role: str
    content: str


@dataclass
class ChatSession:
    title: str
    messages: list[ChatMessage] = field(default_factory=list)
    system_prompt: str = ""


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


class AiChatApp(App[str | None]):
    TITLE = "vLLM-wave"
    CSS = """
    Screen { background: #0f111a; }
    #root { height: 100%; }
    #sidebar {
        width: 30;
        min-width: 24;
        border: tall #30354a;
        background: #131722;
        padding: 1;
    }
    #brand {
        height: auto;
        color: #9cc3ff;
        text-style: bold;
        margin-bottom: 1;
    }
    #btn_new { margin-bottom: 1; }
    #sessions { height: 1fr; margin-top: 1; }
    .meta { height: auto; color: #8992aa; margin-top: 1; }
    #main { width: 1fr; padding: 1 1 1 2; }
    #topbar {
        height: auto;
        border: round #2f3550;
        background: #141a2b;
        color: #c9d4ff;
        padding: 0 1;
        margin-bottom: 1;
    }
    #chat_title { color: #dbe5ff; text-style: bold; }
    #chat_log {
        height: 1fr;
        border: round #2f3550;
        background: #0f1424;
        padding: 0 1;
    }
    #chat_log Markdown {
        background: transparent;
        padding: 0 1;
    }
    #composer {
        height: auto;
        border: round #2f3550;
        background: #141a2b;
        padding: 1;
        margin-top: 1;
    }
    #msg_input { margin-top: 1; margin-bottom: 1; }
    #composer_actions { height: auto; }
    .hint { color: #8a93aa; }
    """

    BINDINGS = [
        ("ctrl+n", "new_chat", "New chat"),
        ("ctrl+l", "clear_chat", "Clear chat"),
        ("ctrl+r", "switch_model", "Switch model"),
        ("ctrl+q", "quit", "Quit"),
    ]

    def __init__(self, base_url: str, model: str, model_display: str | None = None) -> None:
        super().__init__()
        self.base_url = base_url
        self.model = model
        shown = (model_display or model).strip()
        self.model_display = shown or "(unknown)"
        self.sessions: list[ChatSession] = []
        self.active_idx = 0
        self._streaming = False
        self._cancel_stream = False
        self._active_response_parts: list[str] = []
        self._last_stream_render_mono: float = 0.0

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="root"):
            with Vertical(id="sidebar"):
                yield Static("vLLM-wave", id="brand")
                yield Button("New Chat", id="btn_new", variant="primary")
                yield Button("Switch Model", id="btn_switch_model", variant="warning")
                yield ListView(id="sessions")
                yield Label(f"Model: {self.model_display}", classes="meta")
                yield Label("Ctrl+N new  Ctrl+L clear  Ctrl+R switch", classes="meta")
            with Vertical(id="main"):
                yield Static("", id="topbar")
                yield Static("", id="chat_title")
                yield VerticalScroll(
                    Markdown("", id="chat_markdown"),
                    id="chat_log",
                )
                with Vertical(id="composer"):
                    yield Input(
                        id="system_input",
                        placeholder="System prompt (optional, applies to this chat)",
                    )
                    yield Input(
                        id="msg_input",
                        placeholder="Send a message... (Enter to send)",
                    )
                    with Horizontal(id="composer_actions"):
                        yield Button("Send", id="btn_send", variant="success")
                        yield Button("Stop", id="btn_stop", variant="warning", disabled=True)
                        yield Button("Clear", id="btn_clear")
                        yield Button("Copy", id="btn_copy")
                        yield Static("Ctrl+C copy  Ctrl+Q quit", classes="hint")
        yield Footer()

    def on_mount(self) -> None:
        self._new_session(select=True)
        self._update_header()

    def _current(self) -> ChatSession:
        return self.sessions[self.active_idx]

    def _new_session(self, select: bool = True) -> None:
        stamp = datetime.now().strftime("%H:%M")
        session = ChatSession(title=f"New Chat {stamp}")
        self.sessions.append(session)
        item = ListItem(Label(session.title))
        sessions = self.query_one("#sessions", ListView)
        sessions.append(item)
        if select:
            self.active_idx = len(self.sessions) - 1
            sessions.index = self.active_idx
            self._render_active_chat()

    def _update_header(self) -> None:
        self.query_one("#topbar", Static).update(
            f"Endpoint: {self.base_url}/v1   |   Model: {self.model_display}"
        )

    def _render_active_chat(self) -> None:
        session = self._current()
        self.query_one("#chat_title", Static).update(session.title)
        parts: list[str] = []
        if session.system_prompt:
            parts.append(
                "### System\n\n"
                + session.system_prompt.strip()
                + "\n\n---\n"
            )
        for msg in session.messages:
            if msg.role == "user":
                parts.append("### You\n\n" + msg.content.strip() + "\n\n---\n")
            elif msg.role == "assistant":
                body = (msg.content or "...").strip() or "..."
                parts.append("### Assistant\n\n" + body + "\n\n---\n")
            else:
                parts.append(
                    f"### {msg.role.title()}\n\n"
                    + msg.content.strip()
                    + "\n\n---\n"
                )
        doc = "\n".join(parts).strip()
        self.call_later(self._refresh_chat_markdown, doc)

    async def _refresh_chat_markdown(self, markdown: str) -> None:
        await self.query_one("#chat_markdown", Markdown).update(markdown)

    def _payload_messages(self, session: ChatSession) -> list[dict]:
        payload: list[dict] = []
        if session.system_prompt:
            payload.append({"role": "system", "content": session.system_prompt})
        for msg in session.messages:
            payload.append({"role": msg.role, "content": msg.content})
        return payload

    @on(Button.Pressed, "#btn_new")
    def on_new_click(self) -> None:
        if self._streaming:
            return
        self._new_session(select=True)

    @on(Button.Pressed, "#btn_clear")
    def on_clear_click(self) -> None:
        if self._streaming:
            return
        self.action_clear_chat()

    @on(Button.Pressed, "#btn_switch_model")
    def on_switch_model_click(self) -> None:
        self.action_switch_model()

    @on(Button.Pressed, "#btn_send")
    def on_send_click(self) -> None:
        self._send_message()

    @on(Button.Pressed, "#btn_stop")
    def on_stop_click(self) -> None:
        if self._streaming:
            self._cancel_stream = True

    @on(Button.Pressed, "#btn_copy")
    def on_copy_click(self) -> None:
        self.action_copy_selection()

    @on(Input.Submitted, "#msg_input")
    def on_msg_submit(self) -> None:
        self._send_message()

    @on(ListView.Selected, "#sessions")
    def on_session_selected(self, event: ListView.Selected) -> None:
        if self._streaming:
            return
        if event.list_view.index is None:
            return
        idx = int(event.list_view.index)
        if 0 <= idx < len(self.sessions):
            self.active_idx = idx
            self._render_active_chat()

    def action_new_chat(self) -> None:
        if self._streaming:
            return
        self._new_session(select=True)

    def action_clear_chat(self) -> None:
        if self._streaming:
            return
        session = self._current()
        session.messages.clear()
        self._render_active_chat()

    def action_switch_model(self) -> None:
        if self._streaming:
            return
        self.exit("switch_model")

    def action_copy_selection(self) -> None:
        text = self.screen.get_selected_text()
        if text:
            self.app.copy_to_clipboard(text)

    def _send_message(self) -> None:
        if self._streaming:
            return
        msg_input = self.query_one("#msg_input", Input)
        text = msg_input.value.strip()
        if not text:
            return
        sys_input = self.query_one("#system_input", Input)
        session = self._current()
        session.system_prompt = sys_input.value.strip()
        session.messages.append(ChatMessage(role="user", content=text))
        if len(session.messages) == 1:
            session.title = (text[:24] + "...") if len(text) > 24 else text
            sessions = self.query_one("#sessions", ListView)
            item = sessions.children[self.active_idx]
            if isinstance(item, ListItem):
                item.query_one(Label).update(session.title)
        session.messages.append(ChatMessage(role="assistant", content=""))
        msg_input.value = ""
        self._streaming = True
        self._cancel_stream = False
        self._active_response_parts = []
        self._last_stream_render_mono = 0.0
        self.query_one("#btn_send", Button).disabled = True
        self.query_one("#btn_stop", Button).disabled = False
        self._render_active_chat()
        self._stream_worker(self.active_idx, self._payload_messages(session))

    @work(thread=True, exclusive=False, exit_on_error=False)
    def _stream_worker(self, session_idx: int, payload: list[dict]) -> None:
        try:
            for piece in stream_chat_chunks(self.base_url, self.model, payload):
                if self._cancel_stream:
                    break
                self.call_from_thread(self._append_stream_piece, session_idx, piece)
        except httpx.HTTPStatusError as exc:
            self.call_from_thread(
                self._finish_stream_with_error,
                session_idx,
                f"HTTP {exc.response.status_code}: {exc.response.text[:280]}",
            )
            return
        except httpx.HTTPError as exc:
            self.call_from_thread(self._finish_stream_with_error, session_idx, f"{exc}")
            return
        self.call_from_thread(self._finish_stream_ok, session_idx)

    def _append_stream_piece(self, session_idx: int, piece: str) -> None:
        if not (0 <= session_idx < len(self.sessions)):
            return
        session = self.sessions[session_idx]
        if not session.messages or session.messages[-1].role != "assistant":
            return
        self._active_response_parts.append(piece)
        session.messages[-1].content = "".join(self._active_response_parts)
        if session_idx == self.active_idx:
            now = time.monotonic()
            n_parts = len(self._active_response_parts)
            first_chunk = n_parts == 1
            if (
                first_chunk
                or now - self._last_stream_render_mono >= _STREAM_UI_MIN_INTERVAL_SEC
            ):
                self._last_stream_render_mono = now
                self._render_active_chat()

    def _finish_stream_with_error(self, session_idx: int, err: str) -> None:
        if 0 <= session_idx < len(self.sessions):
            session = self.sessions[session_idx]
            if session.messages and session.messages[-1].role == "assistant":
                if not session.messages[-1].content:
                    session.messages.pop()
            session.messages.append(ChatMessage(role="assistant", content=f"[Error] {err}"))
        self._streaming = False
        self.query_one("#btn_send", Button).disabled = False
        self.query_one("#btn_stop", Button).disabled = True
        if session_idx == self.active_idx:
            self._render_active_chat()

    def _finish_stream_ok(self, session_idx: int) -> None:
        self._streaming = False
        self.query_one("#btn_send", Button).disabled = False
        self.query_one("#btn_stop", Button).disabled = True
        if 0 <= session_idx < len(self.sessions):
            session = self.sessions[session_idx]
            if session.messages and session.messages[-1].role == "assistant":
                if not session.messages[-1].content:
                    session.messages[-1].content = "(no response)"
        if session_idx == self.active_idx:
            self._render_active_chat()


def run_interactive_chat(
    base_url: str, model: str, model_display: str | None = None
) -> str | None:
    """Launch the richer Textual chat interface."""
    return AiChatApp(base_url, model, model_display=model_display).run()
