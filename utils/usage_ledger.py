"""Local, append-only usage ledger for billed provider calls (OpenRouter).

Each PAL process appends one JSON line per provider response to its *own*
file::

    <ledger_dir>/YYYY-MM/<host>-<pid>-<process-start>.jsonl

One writer per file means concurrent PAL processes never interleave or need
locking; readers (e.g. Agent Orchestrator) scan the month directories
statelessly and never write. Readers should treat any ``*.jsonl`` in a month
dir as a ledger file: after a torn write the process continues in
``...-r<N>.jsonl``.

One line is written per billed HTTP attempt. A retried call that OpenRouter
billed more than once yields several lines with distinct ``generation_id``s;
readers de-duplicate by ``(key_id, generation_id)``.

Privacy and safety rules:

* Only metadata is written: timestamps, ids, model, tool name, client kind,
  token counts, provider-reported cost. Never prompts, responses or secrets.
* The API key is represented by ``key_id`` = first 12 hex chars of
  SHA-256(key); the key itself never leaves memory.
* Nothing is sent anywhere; the ledger is a local file only.
* Write failures never propagate: they are logged once and swallowed.

Settings (process environment):

* ``PAL_USAGE_LEDGER=0`` (or ``false``/``no``/``off``) disables the ledger.
* ``PAL_USAGE_LEDGER_DIR`` overrides the default ``~/.pal/usage``.

Retention: month directories older than ``RETENTION_MONTHS`` months before the
current month are pruned (only ``*.jsonl`` files inside ``YYYY-MM`` dirs are
deleted). Pruning runs at most once per process per calendar month.
"""

import contextvars
import hashlib
import json
import logging
import os
import re
import socket
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

LEDGER_VERSION = 1
RETENTION_MONTHS = 13
DEFAULT_LEDGER_DIR = Path("~/.pal/usage")
_MAX_STR_LEN = 200
_MONTH_DIR_RE = re.compile(r"^(\d{4})-(\d{2})$")
_DISABLED_VALUES = {"0", "false", "no", "off"}

# Per-request context (set by the MCP call_tool handler). ContextVars are
# task-local, so concurrent tool calls never see each other's values.
_current_tool: contextvars.ContextVar[str | None] = contextvars.ContextVar("pal_usage_tool", default=None)
_current_client: contextvars.ContextVar[str | None] = contextvars.ContextVar("pal_usage_client", default=None)


def set_call_context(tool: str | None, client: str | None) -> tuple[contextvars.Token, contextvars.Token]:
    """Record the tool name and client kind for provider calls in this context."""

    return _current_tool.set(tool), _current_client.set(client)


def reset_call_context(tokens: tuple[contextvars.Token, contextvars.Token]) -> None:
    tool_token, client_token = tokens
    _current_tool.reset(tool_token)
    _current_client.reset(client_token)


def classify_client(client_name: str | None) -> str:
    """Map a raw MCP clientInfo name to ``claude`` / ``codex`` / ``unknown``."""

    if not client_name or not isinstance(client_name, str):
        return "unknown"
    lowered = client_name.lower()
    if "codex" in lowered:
        return "codex"
    if "claude" in lowered or "anthropic" in lowered:
        return "claude"
    return "unknown"


def key_id_for(api_key: str | None) -> str | None:
    """Return a non-reversible short identifier for an API key."""

    if not api_key or not isinstance(api_key, str):
        return None
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Usage extraction (works for SDK objects and plain dicts, e.g. error bodies)
# ---------------------------------------------------------------------------


def _field(obj: Any, name: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    value = getattr(obj, name, None)
    if value is None:
        # Pydantic models keep unknown provider fields (OpenRouter's cost,
        # is_byok, ...) in model_extra; getattr normally finds them, but be
        # explicit in case attribute access is restricted.
        extra = getattr(obj, "model_extra", None)
        if isinstance(extra, dict):
            value = extra.get(name)
    return value


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, float) and value.is_integer() and value >= 0:
        return int(value)
    return None


def _as_cost(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        cost = float(value)
        return cost if cost >= 0 and cost == cost and cost != float("inf") else None
    if isinstance(value, str):
        try:
            cost = float(value)
        except ValueError:
            return None
        return cost if cost >= 0 and cost == cost and cost != float("inf") else None
    return None


def extract_usage_fields(usage: Any) -> dict[str, Any]:
    """Extract token counts and provider-reported cost from a usage object.

    Handles Chat Completions (``prompt_tokens``/``completion_tokens``) and the
    Responses API (``input_tokens``/``output_tokens``), plus OpenRouter's
    usage-accounting extras (``cost``, ``is_byok``, cached/reasoning tokens).
    Values of unexpected types are ignored rather than trusted.
    """

    if usage is None:
        return {}

    prompt = _as_int(_field(usage, "prompt_tokens"))
    if prompt is None:
        prompt = _as_int(_field(usage, "input_tokens"))
    completion = _as_int(_field(usage, "completion_tokens"))
    if completion is None:
        completion = _as_int(_field(usage, "output_tokens"))
    total = _as_int(_field(usage, "total_tokens"))

    prompt_details = _field(usage, "prompt_tokens_details")
    if prompt_details is None:
        prompt_details = _field(usage, "input_tokens_details")
    completion_details = _field(usage, "completion_tokens_details")
    if completion_details is None:
        completion_details = _field(usage, "output_tokens_details")

    is_byok = _field(usage, "is_byok")

    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
        "cached_tokens": _as_int(_field(prompt_details, "cached_tokens")),
        "reasoning_tokens": _as_int(_field(completion_details, "reasoning_tokens")),
        "cost_usd": _as_cost(_field(usage, "cost")),
        "is_byok": is_byok if isinstance(is_byok, bool) else None,
    }


# ---------------------------------------------------------------------------
# Ledger writer
# ---------------------------------------------------------------------------


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _clip(value: Any) -> str | None:
    if value is None:
        return None
    text = value if isinstance(value, str) else str(value)
    return text[:_MAX_STR_LEN]


def _safe_host() -> str:
    try:
        host = socket.gethostname() or "host"
    except Exception:
        host = "host"
    host = re.sub(r"[^A-Za-z0-9._-]", "_", host).lstrip(".")[:64]
    return host or "host"


def ledger_enabled() -> bool:
    return (os.environ.get("PAL_USAGE_LEDGER", "") or "").strip().lower() not in _DISABLED_VALUES


def ledger_dir() -> Path:
    override = (os.environ.get("PAL_USAGE_LEDGER_DIR", "") or "").strip()
    return Path(override).expanduser() if override else DEFAULT_LEDGER_DIR.expanduser()


class UsageLedger:
    """Per-process JSONL ledger writer. All public methods are fail-soft."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._process_start = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self._host = _safe_host()
        self._pruned: set[tuple[str, str]] = set()  # (base dir, month) already pruned
        self._warned = False
        self._file_seq = 0

    def _file_name(self) -> str:
        # pid is read at write time so a forked child never shares its parent's file.
        # A non-zero sequence means an earlier file was retired after a torn write.
        suffix = f"-r{self._file_seq}" if self._file_seq else ""
        return f"{self._host}-{os.getpid()}-{self._process_start}{suffix}.jsonl"

    def _write_all(self, fd: int, line: bytes) -> None:
        """Write one complete line (single writer per file, caller holds the lock).

        If the line cannot be completed after some bytes were written, the file
        is retired so later records never get glued onto the torn line; readers
        skip the torn line as malformed/partial.
        """
        written = 0
        try:
            while written < len(line):
                count = os.write(fd, line[written:])
                if count <= 0:
                    raise OSError("usage ledger short write")
                written += count
        except BaseException:
            if written:
                self._file_seq += 1
            raise

    def _warn_once(self, exc: BaseException) -> None:
        if not self._warned:
            self._warned = True
            logger.warning(
                "PAL usage ledger write failed (%s); further ledger errors are suppressed. "
                "Tool calls are unaffected. Set PAL_USAGE_LEDGER=0 to disable.",
                type(exc).__name__,
            )

    def prune(self, base: Path, now: datetime) -> None:
        """Delete ledger files in month dirs older than RETENTION_MONTHS."""

        current_index = now.year * 12 + (now.month - 1)
        cutoff = current_index - RETENTION_MONTHS
        try:
            entries = list(base.iterdir())
        except FileNotFoundError:
            return
        for entry in entries:
            match = _MONTH_DIR_RE.match(entry.name)
            if not match or not entry.is_dir() or entry.is_symlink():
                continue
            year, month = int(match.group(1)), int(match.group(2))
            if not 1 <= month <= 12 or year * 12 + (month - 1) >= cutoff:
                continue
            for item in entry.iterdir():
                if item.suffix == ".jsonl" and (item.is_file() or item.is_symlink()):
                    item.unlink(missing_ok=True)
            try:
                entry.rmdir()
            except OSError:
                pass  # foreign files left in place; never delete what we don't own

    def append(self, record: dict[str, Any], now: datetime | None = None) -> Path | None:
        """Append one record. Returns the file path, or None if skipped/failed."""

        try:
            if not ledger_enabled():
                return None
            now = now or datetime.now(timezone.utc)
            base = ledger_dir()
            month = now.strftime("%Y-%m")
            line = (json.dumps(record, ensure_ascii=True, separators=(",", ":"), allow_nan=False) + "\n").encode(
                "utf-8"
            )
            with self._lock:
                month_dir = base / month
                base.mkdir(mode=0o700, parents=True, exist_ok=True)
                if month_dir.is_symlink():
                    raise OSError("usage ledger month dir must not be a symlink")
                month_dir.mkdir(mode=0o700, exist_ok=True)
                # mkdir's mode is ignored for existing dirs (and masked by umask):
                # enforce private permissions on the dirs we write into.
                os.chmod(month_dir, 0o700)
                prune_key = (str(base), month)
                if prune_key not in self._pruned:
                    self._pruned.add(prune_key)
                    os.chmod(base, 0o700)
                    try:
                        self.prune(base, now)
                    except Exception as exc:  # pruning must never block writing
                        logger.debug("PAL usage ledger prune skipped: %s", type(exc).__name__)
                path = month_dir / self._file_name()
                flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
                fd = os.open(path, flags, 0o600)
                try:
                    os.fchmod(fd, 0o600)  # also tighten a pre-existing file
                    self._write_all(fd, line)
                finally:
                    os.close(fd)
            return path
        except Exception as exc:
            self._warn_once(exc)
            return None

    def record_call(
        self,
        *,
        api_key: str | None,
        model: str | None,
        ts_start: float,
        ts_end: float,
        usage: Any,
        generation_id: str | None = None,
        served_model: str | None = None,
        status: str = "ok",
    ) -> Path | None:
        """Build and append the ledger line for one provider call (fail-soft)."""

        try:
            fields = extract_usage_fields(usage)
            record = {
                "v": LEDGER_VERSION,
                "ts_start": _iso(ts_start),
                "ts_end": _iso(ts_end),
                "generation_id": _clip(generation_id) or None,
                "key_id": key_id_for(api_key),
                "model": _clip(model),
                "served_model": served if (served := _clip(served_model)) and served != _clip(model) else None,
                "tool": _clip(_current_tool.get()),
                "client": _current_client.get() or "unknown",
                "ao_session_id": _clip(os.environ.get("AO_SESSION_ID") or None),
                "ao_project_id": _clip(os.environ.get("AO_PROJECT_ID") or None),
                "prompt_tokens": fields.get("prompt_tokens"),
                "cached_tokens": fields.get("cached_tokens"),
                "completion_tokens": fields.get("completion_tokens"),
                "reasoning_tokens": fields.get("reasoning_tokens"),
                "cost_usd": fields.get("cost_usd"),
                "is_byok": fields.get("is_byok"),
                "status": "error" if status == "error" else "ok",
            }
            return self.append(record, now=datetime.fromtimestamp(ts_end, tz=timezone.utc))
        except Exception as exc:
            self._warn_once(exc)
            return None


_ledger: UsageLedger | None = None
_ledger_lock = threading.Lock()


def get_ledger() -> UsageLedger:
    global _ledger
    if _ledger is None:
        with _ledger_lock:
            if _ledger is None:
                _ledger = UsageLedger()
    return _ledger
