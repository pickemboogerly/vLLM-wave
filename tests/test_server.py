"""Unit tests for vllm_wave.server helpers."""

from __future__ import annotations

import os
from unittest import mock

import pytest

from vllm_wave import server as s


def test_client_connect_host_all_interfaces() -> None:
    assert s.client_connect_host("0.0.0.0") == "127.0.0.1"
    assert s.client_connect_host("::") == "::1"
    assert s.client_connect_host("[::]") == "::1"


def test_client_connect_host_passthrough() -> None:
    assert s.client_connect_host("192.168.1.10") == "192.168.1.10"
    assert s.client_connect_host("127.0.0.1") == "127.0.0.1"


def test_api_base_url_ipv6_brackets() -> None:
    assert s.api_base_url("::", 8001) == "http://[::1]:8001"
    assert s.api_base_url("127.0.0.1", 55) == "http://127.0.0.1:55"


def test_tool_call_parser_for_run() -> None:
    with mock.patch.dict(os.environ, {"VLLM_TOOL_CALL_PARSER": ""}, clear=False):
        assert s.tool_call_parser_for_run(force_off=False, explicit_override=None) is None
        assert (
            s.tool_call_parser_for_run(force_off=False, explicit_override="x") == "x"
        )
        assert s.tool_call_parser_for_run(force_off=True, explicit_override="x") is None
    with mock.patch.dict(
        os.environ, {"VLLM_TOOL_CALL_PARSER": "qwen3_coder"}, clear=False
    ):
        assert (
            s.tool_call_parser_for_run(force_off=False, explicit_override=None)
            == "qwen3_coder"
        )
        assert (
            s.tool_call_parser_for_run(force_off=False, explicit_override="other")
            == "other"
        )


def test_build_serve_argv_tool_flags_optional() -> None:
    base = s.build_serve_argv(
        "m",
        "127.0.0.1",
        9,
        0.2,
        False,
        tool_call_parser=None,
    )
    assert "--enable-auto-tool-choice" not in base
    assert "--tool-call-parser" not in base
    assert "--cache-memory-percent" in base

    with_tools = s.build_serve_argv(
        "m",
        "127.0.0.1",
        9,
        0.2,
        False,
        tool_call_parser="hermes",
    )
    i = with_tools.index("--tool-call-parser")
    assert with_tools[i + 1] == "hermes"
    assert "--enable-auto-tool-choice" in with_tools


def test_resolve_non_local_paths_passthrough() -> None:
    r0, e0 = s.resolve_model_arg_for_vllm_serve("")
    assert r0 == "" and e0 is None
    r1, e1 = s.resolve_model_arg_for_vllm_serve("org/model-id")
    assert r1 == "org/model-id" and e1 is None
    r2, e2 = s.resolve_model_arg_for_vllm_serve("/no/such/dir/hopefully")
    assert r2 == "/no/such/dir/hopefully" and e2 is None


def test_resolve_gguf_file_rejected(tmp_path) -> None:
    p = tmp_path / "w.gguf"
    p.write_text("x")
    _, err = s.resolve_model_arg_for_vllm_serve(str(p))
    assert err is not None
    assert "GGUF" in err or "safetensors" in err


def test_resolve_dir_missing_config(tmp_path) -> None:
    d = tmp_path / "m"
    d.mkdir()
    _, err = s.resolve_model_arg_for_vllm_serve(str(d))
    assert err is not None
    assert "config.json" in err


def test_first_model_id_from_api_uses_base_url() -> None:
    r = mock.MagicMock()
    r.raise_for_status.return_value = None
    r.json.return_value = {"data": [{"id": "my-model"}]}
    with mock.patch("vllm_wave.server.httpx.get", return_value=r) as p:
        assert s.first_model_id_from_api("http://10.0.0.5:9999") == "my-model"
    assert p.call_args is not None
    assert "10.0.0.5" in str(p.call_args[0][0])


def test_wait_for_models_endpoint_success() -> None:
    proc = mock.Mock()
    proc.poll.return_value = None

    def get(url: str, timeout: float) -> object:
        assert url.endswith("/v1/models")
        return mock.Mock(status_code=200)

    with mock.patch("vllm_wave.server.httpx.get", side_effect=get):
        assert s.wait_for_models_endpoint("http://127.0.0.1:1", 5, proc) is True
