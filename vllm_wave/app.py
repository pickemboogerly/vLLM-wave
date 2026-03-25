"""vLLM-wave launcher: pick cached model, options, start engine, then hand off to chat."""

from __future__ import annotations

import os
import subprocess
from collections import deque
from dataclasses import dataclass

from textual import on, work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import (
    Button,
    Checkbox,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    RichLog,
    Static,
)
from textual.worker import Worker, WorkerState

from vllm_wave.cache import CachedModel
from vllm_wave.server import (
    ServerHandles,
    api_ready_timeout,
    default_host,
    default_port,
    ensure_vllm_on_path,
    first_model_id_from_api,
    port_in_use,
    start_ngrok,
    start_vllm,
    wait_for_models_endpoint,
    wait_for_ngrok_url,
)


@dataclass
class WizardResult:
    handles: ServerHandles
    base_url: str
    chat_model_id: str
    ngrok_hint: str | None = None


class VllmHarnessApp(App[WizardResult | None]):
    TITLE = "vLLM-wave"
    CSS = """
    Screen { align: center middle; }
    #main { width: 100%; height: 100%; padding: 1 2; }
    DataTable { height: 1fr; min-height: 8; }
    #options { height: auto; margin-top: 1; }
    RichLog { height: 8; border: solid $primary; margin-top: 1; }
    #buttons { margin-top: 1; height: auto; }
    .row { height: auto; margin-top: 1; }
    """

    BINDINGS = [
        ("q", "quit_app", "Quit"),
    ]

    def __init__(self, models: list[CachedModel]) -> None:
        super().__init__()
        self.models = models
        self._boot_in_progress = False
        self._boot_stderr: deque[str] = deque(maxlen=200)

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical(id="main"):
            yield Static("Select a cached snapshot (arrow keys) or enter a path / Hub id below.")
            yield DataTable(cursor_type="row", zebra_stripes=True, id="models_table")
            yield Label("Manual model (path or Hub id; overrides table when non-empty):")
            yield Input(
                id="manual_model",
                placeholder="Optional — uses selected row if empty",
                value=os.environ.get("DEFAULT_MODEL", ""),
            )
            with Vertical(id="options"):
                with Horizontal(classes="row"):
                    yield Label("KV cache fraction [0–1]:  ")
                    yield Input(id="cache_pct", value="0.30", classes="cache-input")
                with Horizontal(classes="row"):
                    yield Checkbox("Multimodal (--mllm)", id="chk_mllm")
                    yield Checkbox("Ngrok tunnel (Cursor / public URL)", id="chk_ngrok")
            yield RichLog(id="boot_log", highlight=True, max_lines=80)
            with Horizontal(id="buttons"):
                yield Button("Start server", variant="primary", id="btn_start")
                yield Button("Quit", variant="error", id="btn_quit")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#models_table", DataTable)
        table.add_columns("Model", "Path")
        for i, m in enumerate(self.models):
            table.add_row(m.label, m.path, key=str(i))
        if self.models:
            table.move_cursor(row=0)
        self.set_interval(0.35, self._drain_stderr_log)

    def _drain_stderr_log(self) -> None:
        if not self._boot_in_progress:
            return
        log = self.query_one("#boot_log", RichLog)
        while self._boot_stderr:
            line = self._boot_stderr.popleft()
            log.write(line)

    def action_quit_app(self) -> None:
        if self._boot_in_progress:
            return
        self.exit(None)

    @on(Button.Pressed, "#btn_quit")
    def quit_pressed(self) -> None:
        if self._boot_in_progress:
            return
        self.exit(None)

    @on(Button.Pressed, "#btn_start")
    def start_pressed(self) -> None:
        if self._boot_in_progress:
            return
        err = ensure_vllm_on_path()
        if err:
            self.query_one("#boot_log", RichLog).write(f"[red]{err}[/red]")
            return
        port = default_port()
        host = default_host()
        if port_in_use(port):
            self.query_one("#boot_log", RichLog).write(
                f"[red]Port {port} is in use (set VLLM_PORT).[/red]"
            )
            return
        manual = self.query_one("#manual_model", Input).value.strip()
        model_path = manual
        if not model_path:
            if not self.models:
                self.query_one("#boot_log", RichLog).write(
                    "[red]No cached models and manual path empty. Enter a model path or Hub id.[/red]"
                )
                return
            table = self.query_one("#models_table", DataTable)
            row_idx = table.cursor_row
            if row_idx is None or row_idx < 0 or row_idx >= len(self.models):
                self.query_one("#boot_log", RichLog).write(
                    "[red]Select a row in the table.[/red]"
                )
                return
            model_path = self.models[row_idx].path
        cache_raw = self.query_one("#cache_pct", Input).value.strip() or "0.30"
        try:
            cache_pct = float(cache_raw)
        except ValueError:
            self.query_one("#boot_log", RichLog).write(
                f"[red]Invalid cache fraction: {cache_raw!r}[/red]"
            )
            return
        if not 0.0 <= cache_pct <= 1.0:
            self.query_one("#boot_log", RichLog).write(
                "[red]Cache fraction must be between 0 and 1.[/red]"
            )
            return
        mllm = self.query_one("#chk_mllm", Checkbox).value
        ngrok = self.query_one("#chk_ngrok", Checkbox).value
        self.query_one("#boot_log", RichLog).clear()
        self._boot_in_progress = True
        self._boot_stderr.clear()
        self.query_one("#btn_start", Button).disabled = True
        self.query_one("#btn_quit", Button).disabled = True
        self._boot_worker(model_path, host, port, cache_pct, mllm, ngrok)

    @work(thread=True, exclusive=True, name="boot", exit_on_error=False)
    def _boot_worker(
        self,
        model_path: str,
        host: str,
        port: int,
        cache_pct: float,
        mllm: bool,
        want_ngrok: bool,
    ) -> WizardResult | str:
        proc, lines = start_vllm(
            model_path,
            host,
            port,
            cache_pct,
            mllm,
            stderr_lines=self._boot_stderr,
        )
        timeout = api_ready_timeout()
        ok = wait_for_models_endpoint(port, timeout, proc, on_tick=lambda _i: None)
        if not ok:
            proc.terminate()
            try:
                proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                proc.kill()
            tail = "\n".join(list(lines)[-30:])
            return f"vLLM-wave engine did not become ready in {timeout}s.\n{tail}"

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
                    ngrok_hint = f"Public base URL: {ngrok_url}/v1"
                else:
                    ngrok_hint = (
                        "Ngrok started; open http://127.0.0.1:4040 for the public URL."
                    )

        chat_id = first_model_id_from_api(port) or os.path.basename(
            model_path.rstrip("/")
        )
        handles = ServerHandles(
            vllm=proc,
            ngrok=ngrok_proc,
            ngrok_public_url=ngrok_url,
            stderr_lines=lines,
        )
        base = f"http://127.0.0.1:{port}"
        return WizardResult(
            handles=handles,
            base_url=base,
            chat_model_id=chat_id,
            ngrok_hint=ngrok_hint,
        )

    @on(Worker.StateChanged)
    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        w = event.worker
        if w.name != "boot":
            return
        if event.state not in (
            WorkerState.SUCCESS,
            WorkerState.ERROR,
            WorkerState.CANCELLED,
        ):
            return
        self._boot_in_progress = False
        self.query_one("#btn_start", Button).disabled = False
        self.query_one("#btn_quit", Button).disabled = False
        log = self.query_one("#boot_log", RichLog)
        if event.state == WorkerState.SUCCESS:
            result = w.result
            if isinstance(result, str):
                log.write(f"[red]{result}[/red]")
                return
            if isinstance(result, WizardResult):
                if result.ngrok_hint:
                    log.write(result.ngrok_hint)
                log.write(f"[green]Ready[/green] — {result.base_url}/v1")
                self.exit(result)
        elif event.state == WorkerState.ERROR:
            err = w.error
            log.write(f"[red]Worker error: {err!r}[/red]")
        elif event.state == WorkerState.CANCELLED:
            log.write("[yellow]Start cancelled.[/yellow]")
