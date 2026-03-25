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
    api_base_url,
    api_ready_timeout,
    default_host,
    default_port,
    ensure_vllm_on_path,
    resolve_model_arg_for_vllm_serve,
    first_model_id_from_api,
    port_in_use,
    start_ngrok,
    start_vllm,
    tool_call_parser_for_run,
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
        ("c", "copy_boot_log", "Copy boot log"),
    ]

    def __init__(
        self,
        models: list[CachedModel],
        *,
        cli_tool_parser: str | None = None,
        cli_no_tool_parser: bool = False,
    ) -> None:
        super().__init__()
        self.models = models
        self._cli_tool_parser = cli_tool_parser
        self._cli_no_tool_parser = cli_no_tool_parser
        self._boot_in_progress = False
        self._boot_stderr: deque[str] = deque(maxlen=200)
        # Plain-text archive of everything shown in `#boot_log`, so users can copy it.
        self._boot_log_archive: deque[str] = deque(maxlen=800)

    def _boot_log_write(self, rich_text: str, *, plain_text: str | None = None) -> None:
        log = self.query_one("#boot_log", RichLog)
        log.write(rich_text)
        if plain_text is None:
            # Best-effort: strip common rich tags produced by this app.
            # (We avoid depending on Rich internals here.)
            plain_text = rich_text
            plain_text = plain_text.replace("[red]", "").replace("[/red]", "")
            plain_text = plain_text.replace("[green]", "").replace("[/green]", "")
            plain_text = plain_text.replace("[yellow]", "").replace("[/yellow]", "")
            plain_text = plain_text.replace("[blue]", "").replace("[/blue]", "")
            plain_text = plain_text.replace("[magenta]", "").replace("[/magenta]", "")
            plain_text = plain_text.replace("[cyan]", "").replace("[/cyan]", "")
        self._boot_log_archive.append(plain_text)

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical(id="main"):
            yield Static(
                "Select a cached snapshot (arrow keys) or enter a path / Hub id below. "
                "Press 'c' to copy the boot log. (GGUF weights are not supported by vllm-mlx.)"
            )
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
                with Horizontal(classes="row"):
                    yield Checkbox(
                        "Disable tool-call flags",
                        id="chk_no_tool",
                        value=self._cli_no_tool_parser,
                    )
                    yield Label("Parser override:")
                    yield Input(
                        id="tool_parser",
                        placeholder="Empty → VLLM_TOOL_CALL_PARSER",
                        value=self._cli_tool_parser or "",
                        classes="tool-parser-input",
                    )
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
            self._boot_log_archive.append(line)

    def action_copy_boot_log(self) -> None:
        text = "\n".join(self._boot_log_archive).strip()
        if not text:
            return
        # Uses Textual's clipboard implementation (OSC52), supported by many terminals.
        self.copy_to_clipboard(text)
        # Fallback for terminals without OSC52 clipboard support.
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "vllm-wave")
        try:
            os.makedirs(cache_dir, exist_ok=True)
            file_path = os.path.join(cache_dir, "boot_log.txt")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(text + "\n")
        except OSError:
            file_path = None
        if file_path:
            self._boot_log_write(
                "[green]Copied boot log to clipboard[/green] (also saved to:"
                f" {file_path})",
                plain_text=f"Copied boot log to clipboard (saved to: {file_path})",
            )
        else:
            self._boot_log_write(
                "[green]Copied boot log to clipboard[/green]",
                plain_text="Copied boot log to clipboard",
            )

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
            self._boot_log_write(f"[red]{err}[/red]", plain_text=err)
            return
        port = default_port()
        host = default_host()
        if port_in_use(port, host):
            self._boot_log_write(
                f"[red]Port {port} is in use (set VLLM_PORT).[/red]",
                plain_text=f"Port {port} is in use (set VLLM_PORT).",
            )
            return
        manual = self.query_one("#manual_model", Input).value.strip()
        model_path = manual
        if not model_path:
            if not self.models:
                self._boot_log_write(
                    "[red]No cached models and manual path empty. Enter a model path or Hub id.[/red]",
                    plain_text="No cached models and manual path empty. Enter a model path or Hub id.",
                )
                return
            table = self.query_one("#models_table", DataTable)
            row_idx = table.cursor_row
            if row_idx is None or row_idx < 0 or row_idx >= len(self.models):
                self._boot_log_write(
                    "[red]Select a row in the table.[/red]",
                    plain_text="Select a row in the table.",
                )
                return
            model_path = self.models[row_idx].path

        model_for_serve, resolution_err = resolve_model_arg_for_vllm_serve(model_path)
        if resolution_err:
            self._boot_log_write(f"[red]{resolution_err}[/red]", plain_text=resolution_err)
            return
        cache_raw = self.query_one("#cache_pct", Input).value.strip() or "0.30"
        try:
            cache_pct = float(cache_raw)
        except ValueError:
            self._boot_log_write(
                f"[red]Invalid cache fraction: {cache_raw!r}[/red]",
                plain_text=f"Invalid cache fraction: {cache_raw!r}",
            )
            return
        if not 0.0 <= cache_pct <= 1.0:
            self._boot_log_write(
                "[red]Cache fraction must be between 0 and 1.[/red]",
                plain_text="Cache fraction must be between 0 and 1.",
            )
            return
        mllm = self.query_one("#chk_mllm", Checkbox).value
        ngrok = self.query_one("#chk_ngrok", Checkbox).value
        no_tool = self.query_one("#chk_no_tool", Checkbox).value
        tp_raw = self.query_one("#tool_parser", Input).value.strip()
        tp_explicit: str | None = tp_raw if tp_raw else None
        tparser = tool_call_parser_for_run(
            force_off=no_tool, explicit_override=tp_explicit
        )
        self.query_one("#boot_log", RichLog).clear()
        self._boot_log_archive.clear()
        self._boot_in_progress = True
        self._boot_stderr.clear()
        self.query_one("#btn_start", Button).disabled = True
        self.query_one("#btn_quit", Button).disabled = True
        self._boot_worker(
            model_for_serve, host, port, cache_pct, mllm, ngrok, tparser
        )

    @work(thread=True, exclusive=True, name="boot", exit_on_error=False)
    def _boot_worker(
        self,
        model_path: str,
        host: str,
        port: int,
        cache_pct: float,
        mllm: bool,
        want_ngrok: bool,
        tool_call_parser: str | None,
    ) -> WizardResult | str:
        proc, lines = start_vllm(
            model_path,
            host,
            port,
            cache_pct,
            mllm,
            tool_call_parser=tool_call_parser,
            stderr_lines=self._boot_stderr,
        )
        timeout = api_ready_timeout()
        base = api_base_url(host, port)
        ok = wait_for_models_endpoint(
            base, timeout, proc, on_tick=lambda _i: None
        )
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

        chat_id = first_model_id_from_api(base) or os.path.basename(
            model_path.rstrip("/")
        )
        handles = ServerHandles(
            vllm=proc,
            ngrok=ngrok_proc,
            ngrok_public_url=ngrok_url,
            stderr_lines=lines,
        )
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
                self._boot_log_write(f"[red]{result}[/red]", plain_text=result)
                return
            if isinstance(result, WizardResult):
                if result.ngrok_hint:
                    self._boot_log_write(result.ngrok_hint)
                self._boot_log_write(
                    f"[green]Ready[/green] — {result.base_url}/v1",
                    plain_text=f"Ready — {result.base_url}/v1",
                )
                self.exit(result)
        elif event.state == WorkerState.ERROR:
            err = w.error
            self._boot_log_write(
                f"[red]Worker error: {err!r}[/red]",
                plain_text=f"Worker error: {err!r}",
            )
        elif event.state == WorkerState.CANCELLED:
            self._boot_log_write(
                "[yellow]Start cancelled.[/yellow]",
                plain_text="Start cancelled.",
            )
