"""Tests for OpenRouter usage extraction and the local usage ledger."""

import hashlib
import json
import os
import stat
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from openai import APIStatusError
from openai.types.chat import ChatCompletion
from openai.types.responses import Response

import utils.usage_ledger as usage_ledger
from providers.openrouter import OpenRouterProvider
from utils.usage_ledger import (
    LEDGER_VERSION,
    UsageLedger,
    classify_client,
    extract_usage_fields,
    key_id_for,
    reset_call_context,
    set_call_context,
)

SECRET_KEY = "sk-or-v1-SENTINEL-do-not-leak-0123456789abcdef"
SECRET_PROMPT = "SENTINEL-PROMPT-must-never-be-logged"
SECRET_REPLY = "SENTINEL-REPLY-must-never-be-logged"

EXPECTED_FIELDS = {
    "v",
    "ts_start",
    "ts_end",
    "generation_id",
    "key_id",
    "model",
    "served_model",
    "tool",
    "client",
    "ao_session_id",
    "ao_project_id",
    "prompt_tokens",
    "cached_tokens",
    "completion_tokens",
    "reasoning_tokens",
    "cost_usd",
    "is_byok",
    "status",
}


def openrouter_chat_payload(*, with_cost=True, with_cached=True, content=SECRET_REPLY):
    """A real-shaped OpenRouter /chat/completions response body."""

    usage = {
        "prompt_tokens": 1200,
        "completion_tokens": 300,
        "total_tokens": 1500,
        "completion_tokens_details": {"reasoning_tokens": 120},
    }
    if with_cached:
        usage["prompt_tokens_details"] = {"cached_tokens": 800, "audio_tokens": 0}
    if with_cost:
        usage["cost"] = 0.0123
        usage["is_byok"] = False
        usage["cost_details"] = {"upstream_inference_cost": None}
    return {
        "id": "gen-1764000000-AbCdEf",
        "object": "chat.completion",
        "created": 1764000000,
        "model": "openai/gpt-5-2025-08-07",
        "provider": "OpenAI",
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content}}],
        "usage": usage,
    }


def read_lines(base: Path):
    files = sorted(base.rglob("*.jsonl"))
    lines = []
    for f in files:
        for raw in f.read_text().splitlines():
            lines.append(json.loads(raw))
    return files, lines


@pytest.fixture
def ledger_dir(tmp_path, monkeypatch):
    base = tmp_path / "usage"
    monkeypatch.setenv("PAL_USAGE_LEDGER_DIR", str(base))
    monkeypatch.delenv("PAL_USAGE_LEDGER", raising=False)
    monkeypatch.setattr(usage_ledger, "_ledger", None)
    return base


@pytest.fixture
def provider(monkeypatch):
    monkeypatch.delenv("OPENROUTER_ALLOWED_MODELS", raising=False)
    p = OpenRouterProvider(SECRET_KEY)
    client = MagicMock()
    monkeypatch.setattr(OpenRouterProvider, "client", property(lambda self: client))
    return p, client


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


class TestExtraction:
    def test_openrouter_chat_with_cost_and_cached(self):
        resp = ChatCompletion.model_validate(openrouter_chat_payload())
        fields = extract_usage_fields(resp.usage)
        assert fields == {
            "prompt_tokens": 1200,
            "completion_tokens": 300,
            "total_tokens": 1500,
            "cached_tokens": 800,
            "reasoning_tokens": 120,
            "cost_usd": 0.0123,
            "is_byok": False,
        }

    def test_openrouter_chat_without_cost_or_cached(self):
        resp = ChatCompletion.model_validate(openrouter_chat_payload(with_cost=False, with_cached=False))
        fields = extract_usage_fields(resp.usage)
        assert fields["prompt_tokens"] == 1200
        assert fields["cached_tokens"] is None
        assert fields["cost_usd"] is None
        assert fields["is_byok"] is None

    def test_responses_api_shape(self):
        resp = Response.model_construct(
            id="gen-resp-1",
            model="openai/o3-pro",
            usage={
                "input_tokens": 50,
                "output_tokens": 20,
                "total_tokens": 70,
                "input_tokens_details": {"cached_tokens": 10},
                "output_tokens_details": {"reasoning_tokens": 15},
                "cost": 0.5,
            },
        )
        fields = extract_usage_fields(resp.usage)
        assert fields["prompt_tokens"] == 50
        assert fields["completion_tokens"] == 20
        assert fields["cached_tokens"] == 10
        assert fields["reasoning_tokens"] == 15
        assert fields["cost_usd"] == 0.5

    def test_garbage_values_ignored(self):
        fields = extract_usage_fields(
            {"prompt_tokens": "12", "completion_tokens": True, "cost": float("nan"), "is_byok": "no"}
        )
        assert fields["prompt_tokens"] is None
        assert fields["completion_tokens"] is None
        assert fields["cost_usd"] is None
        assert fields["is_byok"] is None
        assert extract_usage_fields({"cost": -1})["cost_usd"] is None
        assert extract_usage_fields({"cost": "0.002"})["cost_usd"] == 0.002
        assert extract_usage_fields(MagicMock())["cost_usd"] is None

    def test_provider_extract_usage_keeps_cost_and_cached(self):
        p = OpenRouterProvider(SECRET_KEY)
        resp = ChatCompletion.model_validate(openrouter_chat_payload())
        usage = p._extract_usage(resp)
        assert usage == {
            "input_tokens": 1200,
            "output_tokens": 300,
            "total_tokens": 1500,
            "cached_tokens": 800,
            "reasoning_tokens": 120,
            "cost_usd": 0.0123,
            "is_byok": False,
        }

    def test_provider_extract_usage_without_extras_is_unchanged(self):
        p = OpenRouterProvider(SECRET_KEY)
        resp = ChatCompletion.model_validate(openrouter_chat_payload(with_cost=False, with_cached=False))
        usage = p._extract_usage(resp)
        assert usage == {"input_tokens": 1200, "output_tokens": 300, "total_tokens": 1500, "reasoning_tokens": 120}


# ---------------------------------------------------------------------------
# Ledger via the provider
# ---------------------------------------------------------------------------


class TestLedgerFromProvider:
    def test_line_schema_and_no_leakage(self, ledger_dir, provider, monkeypatch):
        p, client = provider
        client.chat.completions.create.return_value = ChatCompletion.model_validate(openrouter_chat_payload())
        monkeypatch.setenv("AO_SESSION_ID", "pal-mcp-server-7")
        monkeypatch.setenv("AO_PROJECT_ID", "pal-mcp-server")

        tokens = set_call_context("chat", "claude")
        try:
            result = p.generate_content(SECRET_PROMPT, "openai/gpt-5", system_prompt="SYSTEM-" + SECRET_PROMPT)
        finally:
            reset_call_context(tokens)

        assert result.content == SECRET_REPLY  # model behaviour unchanged
        assert result.usage["cost_usd"] == 0.0123

        files, lines = read_lines(ledger_dir)
        assert len(files) == 1 and len(lines) == 1
        line = lines[0]
        assert set(line) == EXPECTED_FIELDS
        assert line["v"] == LEDGER_VERSION
        assert line["generation_id"] == "gen-1764000000-AbCdEf"
        assert line["key_id"] == hashlib.sha256(SECRET_KEY.encode()).hexdigest()[:12]
        assert line["model"] == "openai/gpt-5"
        assert line["served_model"] == "openai/gpt-5-2025-08-07"
        assert line["tool"] == "chat"
        assert line["client"] == "claude"
        assert line["ao_session_id"] == "pal-mcp-server-7"
        assert line["ao_project_id"] == "pal-mcp-server"
        assert (line["prompt_tokens"], line["cached_tokens"]) == (1200, 800)
        assert (line["completion_tokens"], line["reasoning_tokens"]) == (300, 120)
        assert line["cost_usd"] == 0.0123
        assert line["is_byok"] is False
        assert line["status"] == "ok"
        assert line["ts_start"].endswith("Z") and line["ts_end"] >= line["ts_start"]

        # File layout: <dir>/YYYY-MM/<host>-<pid>-<start>.jsonl, private permissions
        f = files[0]
        assert f.parent.name == datetime.now(timezone.utc).strftime("%Y-%m")
        assert f.name.split("-")[-2] == str(os.getpid())
        assert stat.S_IMODE(f.stat().st_mode) == 0o600
        assert stat.S_IMODE(f.parent.stat().st_mode) == 0o700
        assert stat.S_IMODE(ledger_dir.stat().st_mode) == 0o700

        raw = f.read_text()
        for secret in (SECRET_KEY, SECRET_PROMPT, SECRET_REPLY, "sk-or-v1"):
            assert secret not in raw

    def test_missing_cost_and_no_ao_env(self, ledger_dir, provider, monkeypatch):
        p, client = provider
        monkeypatch.delenv("AO_SESSION_ID", raising=False)
        monkeypatch.delenv("AO_PROJECT_ID", raising=False)
        client.chat.completions.create.return_value = ChatCompletion.model_validate(
            openrouter_chat_payload(with_cost=False, with_cached=False)
        )
        p.generate_content("hi", "openai/gpt-5")
        _, lines = read_lines(ledger_dir)
        assert lines[0]["cost_usd"] is None
        assert lines[0]["cached_tokens"] is None
        assert lines[0]["ao_session_id"] is None
        assert lines[0]["tool"] is None
        assert lines[0]["client"] == "unknown"

    def test_billed_error_response_is_logged(self, ledger_dir, provider):
        p, client = provider
        request = MagicMock()
        response = MagicMock(status_code=400, headers={}, request=request)
        body = {"id": "gen-err-1", "error": {"message": SECRET_PROMPT}, "usage": {"prompt_tokens": 10, "cost": 0.001}}
        client.chat.completions.create.side_effect = APIStatusError("bad", response=response, body=body)
        with pytest.raises(RuntimeError):
            p.generate_content("hi", "openai/gpt-5")
        _, lines = read_lines(ledger_dir)
        assert len(lines) == 1  # 400 is not retried
        assert lines[0]["status"] == "error"
        assert lines[0]["cost_usd"] == 0.001
        assert lines[0]["generation_id"] == "gen-err-1"
        assert SECRET_PROMPT not in json.dumps(lines)

    def test_unbilled_error_is_not_logged(self, ledger_dir, provider):
        p, client = provider
        client.chat.completions.create.side_effect = ValueError("boom")
        with pytest.raises(RuntimeError):
            p.generate_content("hi", "openai/gpt-5")
        assert not list(ledger_dir.rglob("*.jsonl"))

    def test_disable_switch(self, ledger_dir, provider, monkeypatch):
        p, client = provider
        monkeypatch.setenv("PAL_USAGE_LEDGER", "0")
        client.chat.completions.create.return_value = ChatCompletion.model_validate(openrouter_chat_payload())
        p.generate_content("hi", "openai/gpt-5")
        assert not ledger_dir.exists()

    def test_write_failure_never_breaks_call_and_warns_once(self, tmp_path, provider, monkeypatch, caplog):
        p, client = provider
        blocker = tmp_path / "not-a-dir"
        blocker.write_text("x")
        monkeypatch.setenv("PAL_USAGE_LEDGER_DIR", str(blocker / "usage"))
        monkeypatch.setattr(usage_ledger, "_ledger", None)
        client.chat.completions.create.return_value = ChatCompletion.model_validate(openrouter_chat_payload())
        with caplog.at_level("WARNING", logger="utils.usage_ledger"):
            for _ in range(3):
                assert p.generate_content("hi", "openai/gpt-5").content == SECRET_REPLY
        warnings = [r for r in caplog.records if "usage ledger" in r.getMessage()]
        assert len(warnings) == 1
        assert str(blocker) not in warnings[0].getMessage()

    def test_non_openrouter_providers_do_not_write(self, ledger_dir):
        from providers.openai_compatible import OpenAICompatibleProvider

        assert OpenAICompatibleProvider.USAGE_LEDGER_ENABLED is False
        assert OpenRouterProvider.USAGE_LEDGER_ENABLED is True


# ---------------------------------------------------------------------------
# Ledger writer: concurrency, retention, helpers
# ---------------------------------------------------------------------------

_WRITER_SCRIPT = """
import sys, time
sys.path.insert(0, sys.argv[1])
from utils.usage_ledger import get_ledger
led = get_ledger()
for i in range(int(sys.argv[2])):
    led.record_call(api_key="k", model="m", ts_start=time.time(), ts_end=time.time(),
                    usage={"prompt_tokens": i, "cost": 0.001}, generation_id=f"gen-{sys.argv[3]}-{i}")
"""


class TestLedgerWriter:
    def test_two_processes_write_separate_files(self, ledger_dir):
        repo = str(Path(__file__).resolve().parent.parent)
        env = {**os.environ, "PAL_USAGE_LEDGER_DIR": str(ledger_dir)}
        env.pop("PAL_USAGE_LEDGER", None)
        procs = [
            subprocess.Popen([sys.executable, "-c", _WRITER_SCRIPT, repo, "200", tag], env=env) for tag in ("a", "b")
        ]
        assert [proc.wait(timeout=60) for proc in procs] == [0, 0]
        files, lines = read_lines(ledger_dir)
        assert len(files) == 2
        ids = {line["generation_id"] for line in lines}
        assert len(lines) == 400 and len(ids) == 400

    def test_threads_in_one_process_never_interleave(self, ledger_dir):
        led = UsageLedger()

        def worker(tag):
            for i in range(100):
                led.record_call(
                    api_key="k", model="m" * 150, ts_start=0, ts_end=1e9, usage={"cost": 1}, generation_id=f"{tag}-{i}"
                )

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        files, lines = read_lines(ledger_dir)  # json.loads would fail on a torn line
        assert len(files) == 1 and len(lines) == 800

    def test_month_dir_follows_ts_end(self, ledger_dir):
        led = UsageLedger()
        ts = datetime(2026, 3, 31, 23, 59, 59, tzinfo=timezone.utc).timestamp()
        path = led.record_call(api_key="k", model="m", ts_start=ts - 5, ts_end=ts, usage={})
        assert path.parent.name == "2026-03"
        path2 = led.record_call(api_key="k", model="m", ts_start=ts, ts_end=ts + 2, usage={})
        assert path2.parent.name == "2026-04"
        assert path.name == path2.name

    def test_retention_prunes_only_old_owned_files(self, ledger_dir):
        ledger_dir.mkdir(parents=True)
        for month in ("2025-09", "2025-10", "2026-10"):
            (ledger_dir / month).mkdir()
            (ledger_dir / month / "h-1-x.jsonl").write_text("{}\n")
        (ledger_dir / "2024-01").mkdir()
        (ledger_dir / "2024-01" / "keep.txt").write_text("foreign")
        (ledger_dir / "2024-01" / "old.jsonl").write_text("{}\n")
        (ledger_dir / "notes").mkdir()

        led = UsageLedger()
        now = datetime(2026, 11, 24, 12, 0, tzinfo=timezone.utc)
        led.record_call(api_key="k", model="m", ts_start=now.timestamp(), ts_end=now.timestamp(), usage={})

        remaining = sorted(p.name for p in ledger_dir.iterdir())
        # 13 months before 2026-11 is 2025-10: kept; 2025-09 pruned.
        assert remaining == ["2024-01", "2025-10", "2026-10", "2026-11", "notes"]
        assert [p.name for p in (ledger_dir / "2024-01").iterdir()] == ["keep.txt"]

    def test_prune_runs_once_per_month(self, ledger_dir):
        led = UsageLedger()
        now = datetime(2026, 11, 24, tzinfo=timezone.utc).timestamp()
        with patch.object(UsageLedger, "prune") as prune:
            for _ in range(5):
                led.record_call(api_key="k", model="m", ts_start=now, ts_end=now, usage={})
        assert prune.call_count == 1

    def test_helpers(self):
        assert key_id_for(SECRET_KEY) == hashlib.sha256(SECRET_KEY.encode()).hexdigest()[:12]
        assert key_id_for("") is None and key_id_for(None) is None
        assert classify_client("claude-code") == "claude"
        assert classify_client("codex-mcp-client") == "codex"
        assert classify_client("cursor") == "unknown"
        assert classify_client(None) == "unknown"

    @pytest.mark.parametrize("value", ["0", "false", "OFF", "no"])
    def test_disable_values(self, ledger_dir, monkeypatch, value):
        monkeypatch.setenv("PAL_USAGE_LEDGER", value)
        assert UsageLedger().record_call(api_key="k", model="m", ts_start=0, ts_end=0, usage={}) is None
        assert not ledger_dir.exists()

    def test_default_dir(self, monkeypatch):
        monkeypatch.delenv("PAL_USAGE_LEDGER_DIR", raising=False)
        assert usage_ledger.ledger_dir() == Path("~/.pal/usage").expanduser()


@pytest.mark.asyncio
async def test_call_tool_sets_and_resets_context(monkeypatch):
    import server

    seen = {}

    async def fake_impl(name, arguments):
        seen["tool"] = usage_ledger._current_tool.get()
        seen["client"] = usage_ledger._current_client.get()
        return []

    monkeypatch.setattr(server, "_handle_call_tool", fake_impl)
    monkeypatch.setattr(server, "_usage_ledger_client", lambda: "codex")
    await server.handle_call_tool("thinkdeep", {})
    assert seen == {"tool": "thinkdeep", "client": "codex"}
    assert usage_ledger._current_tool.get() is None
