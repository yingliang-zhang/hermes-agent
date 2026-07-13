"""Hindsight memory plugin — MemoryProvider interface.

Long-term memory with knowledge graph, entity resolution, and multi-strategy
retrieval. Supports cloud (API key) and local modes.

Configurable request timeout via HINDSIGHT_TIMEOUT env var or config.json.
Configurable embedded daemon idle timeout via HINDSIGHT_IDLE_TIMEOUT env var
or config.json idle_timeout.

Original PR #1811 by benfrank241, adapted to MemoryProvider ABC.

Config via environment variables:
  HINDSIGHT_API_KEY                — API key for Hindsight Cloud
  HINDSIGHT_BANK_ID                — memory bank identifier (default: hermes)
  HINDSIGHT_BUDGET                 — recall budget: low/mid/high (default: mid)
  HINDSIGHT_API_URL                — API endpoint
  HINDSIGHT_MODE                   — cloud or local (default: cloud)
  HINDSIGHT_TIMEOUT                — API request timeout in seconds (default: 120)
  HINDSIGHT_IDLE_TIMEOUT           — embedded daemon idle timeout seconds; 0 disables shutdown (default: 300)
  HINDSIGHT_EMBED_PORT_HEALTH_GRACE_TIMEOUT — seconds to wait for a slow embedded daemon /health before treating it as stale (default: 30; set via config.json port_health_grace_timeout)
  HINDSIGHT_RETAIN_TAGS            — comma-separated tags attached to retained memories
  HINDSIGHT_RETAIN_OBSERVATION_SCOPES — observation scoping for retained memories: per_tag/combined/all_combinations, or a JSON list of tag-lists for custom scopes
  HINDSIGHT_RETAIN_SOURCE          — metadata source value attached to retained memories
  HINDSIGHT_RETAIN_USER_PREFIX     — label used before user turns in retained transcripts
  HINDSIGHT_RETAIN_ASSISTANT_PREFIX — label used before assistant turns in retained transcripts

Or via $HERMES_HOME/hindsight/config.json (profile-scoped), falling back to
~/.hindsight/config.json (legacy, shared) for backward compatibility.
"""

from __future__ import annotations

import asyncio
import atexit
import hashlib
import inspect
import importlib
import json
import logging
import math
import os
import queue
import sys
import threading
import time

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List
from uuid import uuid4

from agent.memory_provider import MemoryProvider
from hermes_constants import get_hermes_home
from tools.registry import tool_error
from hermes_cli.config import cfg_get

logger = logging.getLogger(__name__)

_DEFAULT_API_URL = "https://api.hindsight.vectorize.io"
_DEFAULT_LOCAL_URL = "http://localhost:8888"
# Keep in sync with tools/lazy_deps.py ("memory.hindsight") and plugin.yaml.
_MIN_CLIENT_VERSION = "0.8.4"
_MAX_CLIENT_VERSION = "0.10"
_CLIENT_REQUIREMENT = (
    f"hindsight-client>={_MIN_CLIENT_VERSION},<{_MAX_CLIENT_VERSION}"
)
_DEFAULT_TIMEOUT = 120  # seconds — cloud API can take 30-40s per request
_DEFAULT_IDLE_TIMEOUT = 300  # seconds — Hindsight embedded daemon default
_VALID_BUDGETS = {"low", "mid", "high"}
_VALID_SCOPE_DIMENSIONS = {"profile", "workspace", "session"}
_VALID_MIN_SCORE_KEYS = {"semantic", "keyword", "reranker", "final"}
_MIN_SCORES_CLIENT_VERSION = _MIN_CLIENT_VERSION
_PREFETCH_WAIT_SECONDS = 3.0
_MAX_PREFETCH_WORKERS = 2
_PROVIDER_DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-haiku-4-5",
    "gemini": "gemini-2.5-flash",
    "groq": "openai/gpt-oss-120b",
    "openrouter": "qwen/qwen3.5-9b",
    "minimax": "MiniMax-M2.7",
    "ollama": "gemma3:12b",
    "lmstudio": "local-model",
    "openai_compatible": "your-model-name",
}


def _parse_int_setting(value: Any, default: int) -> int:
    """Parse an integer config/env value, falling back on invalid input."""
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        logger.warning("Invalid integer Hindsight setting %r; using default %s", value, default)
        return default


# Env var the embedded daemon manager reads (at import time, as a module-level
# constant) to size the grace window it waits for a slow /health before
# declaring a daemon stale and killing it. Default upstream is 30s; on
# resource-contended hosts a busy daemon can exceed a single 2s health check
# and get needlessly killed + restarted (issue #13125 comment thread). We
# surface it as plugin config so users can raise it without hand-setting an
# env var, consistent with "config.json, not raw env vars".
_PORT_HEALTH_GRACE_ENV = "HINDSIGHT_EMBED_PORT_HEALTH_GRACE_TIMEOUT"


def _export_port_health_grace_timeout(config: dict[str, Any]) -> None:
    """Export the embedded-daemon health grace timeout to the process env.

    Must run BEFORE ``hindsight_embed.daemon_embed_manager`` is imported,
    because the package reads the env var into a module-level constant at
    import time. We only set it when the user configured a value AND the
    env var isn't already set, so an explicit env override always wins.
    """
    raw = config.get("port_health_grace_timeout")
    if raw is None or raw == "":
        return
    try:
        seconds = float(raw)
    except (TypeError, ValueError):
        logger.warning(
            "Invalid Hindsight port_health_grace_timeout %r; ignoring.", raw
        )
        return
    if seconds < 0:
        logger.warning(
            "Negative Hindsight port_health_grace_timeout %r; ignoring.", raw
        )
        return
    # setdefault: an explicit env var the operator set wins over config.
    os.environ.setdefault(_PORT_HEALTH_GRACE_ENV, repr(seconds))


def _check_local_runtime() -> tuple[bool, str | None]:
    """Return whether local embedded Hindsight imports cleanly.

    On older CPUs, importing the local Hindsight stack can raise a runtime
    error from NumPy before the daemon starts. Treat that as "unavailable"
    so Hermes can degrade gracefully instead of repeatedly trying to start
    a broken local memory backend.
    """
    try:
        importlib.import_module("hindsight")
        importlib.import_module("hindsight_embed.daemon_embed_manager")
        return True, None
    except Exception as exc:
        return False, str(exc)


def _ensure_cloud_client_dependency() -> None:
    """Install the Hindsight cloud client lazily before importing it."""
    try:
        from tools.lazy_deps import ensure as _lazy_ensure
        _lazy_ensure("memory.hindsight", prompt=False)
    except ImportError:
        pass
    except Exception as exc:
        raise ImportError(str(exc)) from exc




# ---------------------------------------------------------------------------
# Dedicated event loop for Hindsight async calls (one per process, reused).
# Avoids creating ephemeral loops that leak aiohttp sessions.
# ---------------------------------------------------------------------------

_loop: asyncio.AbstractEventLoop | None = None
_loop_thread: threading.Thread | None = None
_loop_lock = threading.Lock()

# Sentinel pushed to the per-provider retain queue to wake the writer for a
# clean exit. A unique object so it can never collide with a real job.
_WRITER_SENTINEL = object()


@dataclass(slots=True)
class _AutomaticRetainBatch:
    start_turn: int
    end_turn: int
    document_id: str
    item: Dict[str, Any]


@dataclass(slots=True)
class _RetainSessionState:
    session_id: str
    parent_session_id: str
    document_id: str
    nonce: str = field(default_factory=lambda: uuid4().hex)
    turns: list[str] = field(default_factory=list)
    committed_turn_count: int = 0
    failed_batch: _AutomaticRetainBatch | None = None


def _get_loop() -> asyncio.AbstractEventLoop:
    """Return a long-lived event loop running on a background thread."""
    global _loop, _loop_thread
    with _loop_lock:
        if _loop is not None and _loop.is_running():
            return _loop
        _loop = asyncio.new_event_loop()

        def _run():
            asyncio.set_event_loop(_loop)
            _loop.run_forever()

        _loop_thread = threading.Thread(target=_run, daemon=True, name="hindsight-loop")
        _loop_thread.start()
        return _loop


def _run_sync(coro, timeout: float = _DEFAULT_TIMEOUT):
    """Schedule *coro* on the shared loop and block until done."""
    from agent.async_utils import safe_schedule_threadsafe
    loop = _get_loop()
    future = safe_schedule_threadsafe(coro, loop)
    if future is None:
        raise RuntimeError("Hindsight loop unavailable")
    return future.result(timeout=timeout)


# ---------------------------------------------------------------------------
# Backward-compatible alias — instances use self._run_sync() instead.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

RETAIN_SCHEMA = {
    "name": "hindsight_retain",
    "description": (
        "Store information to long-term memory. Hindsight automatically "
        "extracts structured facts, resolves entities, and indexes for retrieval."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "The information to store."},
            "context": {"type": "string", "description": "Short label (e.g. 'user preference', 'project decision')."},
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional per-call tags to merge with configured default retain tags.",
            },
        },
        "required": ["content"],
    },
}

RECALL_SCHEMA = {
    "name": "hindsight_recall",
    "description": (
        "Search long-term memory. Returns memories ranked by relevance using "
        "semantic search, keyword matching, entity graph traversal, and reranking."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for."},
        },
        "required": ["query"],
    },
}

REFLECT_SCHEMA = {
    "name": "hindsight_reflect",
    "description": (
        "Synthesize a reasoned answer from long-term memories. Unlike recall, "
        "this reasons across all stored memories to produce a coherent response. "
        "Configured recall_min_scores apply only to recall, not reflection."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The question to reflect on."},
        },
        "required": ["query"],
    },
}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    """Load config from profile-scoped path, legacy path, or env vars.

    Resolution order:
      1. $HERMES_HOME/hindsight/config.json  (profile-scoped)
      2. ~/.hindsight/config.json             (legacy, shared)
      3. Environment variables
    """
    from pathlib import Path

    # Profile-scoped path (preferred)
    profile_path = get_hermes_home() / "hindsight" / "config.json"
    if profile_path.exists():
        try:
            return json.loads(profile_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    # Legacy shared path (backward compat)
    legacy_path = Path.home() / ".hindsight" / "config.json"
    if legacy_path.exists():
        try:
            return json.loads(legacy_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    return {
        "mode": os.environ.get("HINDSIGHT_MODE", "cloud"),
        "apiKey": os.environ.get("HINDSIGHT_API_KEY", ""),
        "timeout": _parse_int_setting(os.environ.get("HINDSIGHT_TIMEOUT"), _DEFAULT_TIMEOUT),
        "idle_timeout": _parse_int_setting(os.environ.get("HINDSIGHT_IDLE_TIMEOUT"), _DEFAULT_IDLE_TIMEOUT),
        "retain_tags": os.environ.get("HINDSIGHT_RETAIN_TAGS", ""),
        "observation_scopes": os.environ.get("HINDSIGHT_RETAIN_OBSERVATION_SCOPES", ""),
        "retain_source": os.environ.get("HINDSIGHT_RETAIN_SOURCE", ""),
        "retain_user_prefix": os.environ.get("HINDSIGHT_RETAIN_USER_PREFIX", "User"),
        "retain_assistant_prefix": os.environ.get("HINDSIGHT_RETAIN_ASSISTANT_PREFIX", "Assistant"),
        "banks": {
            "hermes": {
                "bankId": os.environ.get("HINDSIGHT_BANK_ID", "hermes"),
                "budget": os.environ.get("HINDSIGHT_BUDGET", "mid"),
                "enabled": True,
            }
        },
    }


def _normalize_retain_tags(value: Any) -> List[str]:
    """Normalize tag config/tool values to a deduplicated list of strings."""
    if value is None:
        return []

    raw_items: list[Any]
    if isinstance(value, list):
        raw_items = value
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("["):
            try:
                parsed = json.loads(text)
            except Exception:
                parsed = None
            if isinstance(parsed, list):
                raw_items = parsed
            else:
                raw_items = text.split(",")
        else:
            raw_items = text.split(",")
    else:
        raw_items = [value]

    normalized = []
    seen = set()
    for item in raw_items:
        tag = str(item).strip()
        if not tag or tag in seen:
            continue
        seen.add(tag)
        normalized.append(tag)
    return normalized


def _normalize_scope_dimensions(value: Any) -> list[str]:
    """Normalize opt-in dynamic scope dimensions, warning on invalid values."""
    if value is None or value == "":
        return []
    if isinstance(value, str):
        raw_items: list[Any] = value.split(",")
    elif isinstance(value, list):
        raw_items = value
    else:
        logger.warning(
            "Hindsight scope_tags must be a list or comma-separated string; "
            "ignoring %r.",
            value,
        )
        return []

    dimensions: list[str] = []
    seen: set[str] = set()
    for raw in raw_items:
        if not isinstance(raw, str):
            logger.warning(
                "Ignoring invalid Hindsight scope_tags value %r; expected one of %s.",
                raw,
                sorted(_VALID_SCOPE_DIMENSIONS),
            )
            continue
        dimension = raw.strip()
        if not dimension:
            continue
        if dimension not in _VALID_SCOPE_DIMENSIONS:
            logger.warning(
                "Ignoring invalid Hindsight scope_tags value %r; expected one of %s.",
                dimension,
                sorted(_VALID_SCOPE_DIMENSIONS),
            )
            continue
        if dimension not in seen:
            seen.add(dimension)
            dimensions.append(dimension)
    return dimensions


def _normalize_recall_min_scores(value: Any) -> dict[str, float]:
    """Validate inclusive Hindsight recall score floors."""
    if value is None or value == {}:
        return {}
    if not isinstance(value, dict):
        logger.warning(
            "Hindsight recall_min_scores must be an object using semantic, "
            "keyword, reranker, or final keys; filtering is disabled."
        )
        return {}

    normalized: dict[str, float] = {}
    for key, raw_score in value.items():
        if key not in _VALID_MIN_SCORE_KEYS:
            logger.warning(
                "Ignoring unknown Hindsight recall_min_scores key %r; "
                "expected one of %s.",
                key,
                sorted(_VALID_MIN_SCORE_KEYS),
            )
            continue
        if isinstance(raw_score, bool) or not isinstance(raw_score, (int, float)):
            logger.warning(
                "Invalid Hindsight recall_min_scores entry %r=%r; "
                "expected a finite number >= 0.",
                key,
                raw_score,
            )
            continue
        score = float(raw_score)
        if not math.isfinite(score) or score < 0:
            logger.warning(
                "Invalid Hindsight recall_min_scores entry %r=%r; "
                "expected a finite number >= 0.",
                key,
                raw_score,
            )
            continue
        normalized[key] = score
    return normalized


def _dynamic_scope_tag(dimension: str, value: str) -> str:
    """Hash a canonical scope value without exposing it in the tag."""
    canonical = str(value or "").strip()
    if not canonical:
        return ""
    if dimension == "workspace":
        canonical = os.path.realpath(
            os.path.abspath(os.path.expanduser(canonical))
        )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    return f"scope:{dimension}:{digest}"


_OBSERVATION_SCOPE_KEYWORDS = {"per_tag", "combined", "all_combinations"}


def _normalize_observation_scopes(value: Any) -> Any:
    """Normalize an observation_scopes config value to a Hindsight-accepted form.

    Returns one of:
      * ``None`` — nothing configured; Hindsight applies its ``combined`` default.
      * a keyword string — ``"per_tag"`` / ``"combined"`` / ``"all_combinations"``.
      * ``list[list[str]]`` — custom scopes, one inner list per consolidation pass.

    Accepts a keyword string, a JSON-encoded list, a flat list of tags (treated as
    a single scope), or a list of tag-lists. Anything unrecognized yields ``None``
    so we never send an invalid payload.
    """
    if value is None:
        return None

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text in _OBSERVATION_SCOPE_KEYWORDS:
            return text
        if text.startswith("["):
            try:
                parsed = json.loads(text)
            except Exception:
                return None
            return _normalize_observation_scopes(parsed)
        return None

    if isinstance(value, (list, tuple)):
        # A flat list of tag strings is one scope; a list of lists is many.
        if all(isinstance(entry, str) for entry in value):
            inner = [entry.strip() for entry in value if entry.strip()]
            return [inner] if inner else None
        scopes: list[list[str]] = []
        for entry in value:
            if isinstance(entry, (list, tuple)):
                inner = [str(tag).strip() for tag in entry if str(tag).strip()]
                if inner:
                    scopes.append(inner)
            elif isinstance(entry, str) and entry.strip():
                scopes.append([entry.strip()])
        return scopes or None

    return None


def _utc_timestamp() -> str:
    """Return current UTC timestamp in ISO-8601 with milliseconds and Z suffix."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _embedded_profile_name(config: dict[str, Any]) -> str:
    """Return the Hindsight embedded profile name for this Hermes config."""
    profile = config.get("profile", "hermes")
    return str(profile or "hermes")


def _load_simple_env(path) -> dict[str, str]:
    """Parse a simple KEY=VALUE env file, ignoring comments and blank lines."""
    if not path.exists():
        return {}

    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def _build_embedded_profile_env(config: dict[str, Any], *, llm_api_key: str | None = None) -> dict[str, str]:
    """Build the profile-scoped env file that standalone hindsight-embed consumes."""
    current_key = llm_api_key
    if current_key is None:
        current_key = (
            config.get("llmApiKey")
            or config.get("llm_api_key")
            or os.environ.get("HINDSIGHT_LLM_API_KEY", "")
        )

    current_provider = config.get("llm_provider", "")
    current_model = config.get("llm_model", "")
    current_base_url = config.get("llm_base_url") or os.environ.get("HINDSIGHT_API_LLM_BASE_URL", "")

    # The embedded daemon expects OpenAI wire format for these providers.
    daemon_provider = "openai" if current_provider in {"openai_compatible", "openrouter"} else current_provider

    env_values = {
        "HINDSIGHT_API_LLM_PROVIDER": str(daemon_provider),
        "HINDSIGHT_API_LLM_API_KEY": str(current_key or ""),
        "HINDSIGHT_API_LLM_MODEL": str(current_model),
        "HINDSIGHT_API_LOG_LEVEL": "info",
    }
    if current_base_url:
        env_values["HINDSIGHT_API_LLM_BASE_URL"] = str(current_base_url)

    idle_timeout = (
        config.get("idle_timeout")
        if config.get("idle_timeout") is not None
        else os.environ.get("HINDSIGHT_IDLE_TIMEOUT")
    )
    if idle_timeout is not None and idle_timeout != "":
        env_values["HINDSIGHT_EMBED_DAEMON_IDLE_TIMEOUT"] = str(
            _parse_int_setting(idle_timeout, _DEFAULT_IDLE_TIMEOUT)
        )
    return env_values


def _embedded_profile_env_path(config: dict[str, Any]):
    from pathlib import Path

    return Path.home() / ".hindsight" / "profiles" / f"{_embedded_profile_name(config)}.env"


def _materialize_embedded_profile_env(config: dict[str, Any], *, llm_api_key: str | None = None):
    """Write the profile-scoped env file that standalone hindsight-embed uses."""
    profile_env = _embedded_profile_env_path(config)
    profile_env.parent.mkdir(parents=True, exist_ok=True)
    env_values = _build_embedded_profile_env(config, llm_api_key=llm_api_key)
    profile_env.write_text(
        "".join(f"{key}={value}\n" for key, value in env_values.items()),
        encoding="utf-8",
    )
    return profile_env

def _sanitize_bank_segment(value: str) -> str:
    """Sanitize a bank_id_template placeholder value.

    Bank IDs should be safe for URL paths and filesystem use. Replaces any
    character that isn't alphanumeric, dash, or underscore with a dash, and
    collapses runs of dashes.
    """
    if not value:
        return ""
    out = []
    prev_dash = False
    for ch in str(value):
        if ch.isalnum() or ch == "-" or ch == "_":
            out.append(ch)
            prev_dash = False
        else:
            if not prev_dash:
                out.append("-")
                prev_dash = True
    return "".join(out).strip("-_")


def _resolve_bank_id_template(template: str, fallback: str, **placeholders: str) -> str:
    """Resolve a bank_id template string with the given placeholders.

    Supported placeholders (each is sanitized before substitution):
      {profile}   — active Hermes profile name (from agent_identity)
      {workspace} — Hermes workspace name (from agent_workspace)
      {platform}  — "cli", "telegram", "discord", etc.
      {user}      — platform user id (gateway sessions)
      {session}   — current session id

    Missing/empty placeholders are rendered as the empty string and then
    collapsed — e.g. ``hermes-{user}`` with no user becomes ``hermes``.

    If the template is empty, resolution falls back to *fallback*.
    Returns the sanitized bank id.
    """
    if not template:
        return fallback
    sanitized = {k: _sanitize_bank_segment(v) for k, v in placeholders.items()}
    try:
        rendered = template.format(**sanitized)
    except (KeyError, IndexError) as exc:
        logger.warning("Invalid bank_id_template %r: %s — using fallback %r",
                       template, exc, fallback)
        return fallback
    while "--" in rendered:
        rendered = rendered.replace("--", "-")
    while "__" in rendered:
        rendered = rendered.replace("__", "_")
    rendered = rendered.strip("-_")
    return rendered or fallback


# ---------------------------------------------------------------------------
# MemoryProvider implementation
# ---------------------------------------------------------------------------

class HindsightMemoryProvider(MemoryProvider):
    """Hindsight long-term memory with knowledge graph and multi-strategy retrieval."""

    def backup_paths(self) -> List[str]:
        """Hindsight's legacy shared config and embedded-mode profile env
        files live under ~/.hindsight (see _load_config / line ~509)."""
        try:
            from pathlib import Path
            legacy_dir = Path.home() / ".hindsight"
            return [str(legacy_dir)]
        except Exception:
            return []

    def __init__(self):
        self._config = None
        self._api_key = None
        self._api_url = _DEFAULT_API_URL
        self._bank_id = "hermes"
        self._budget = "mid"
        self._mode = "cloud"
        self._llm_base_url = ""
        self._memory_mode = "hybrid"  # "context", "tools", or "hybrid"
        self._prefetch_method = "recall"  # "recall" or "reflect"
        self._retain_tags: List[str] = []
        self._retain_source = ""
        self._retain_user_prefix = "User"
        self._retain_assistant_prefix = "Assistant"
        self._platform = ""
        self._user_id = ""
        self._user_name = ""
        self._chat_id = ""
        self._chat_name = ""
        self._chat_type = ""
        self._thread_id = ""
        self._agent_identity = ""
        self._agent_workspace = ""
        self._scope_workspace = ""
        self._turn_index = 0
        self._client = None
        self._timeout = _DEFAULT_TIMEOUT
        self._idle_timeout = _DEFAULT_IDLE_TIMEOUT
        self._prefetch_result = ""
        self._prefetch_result_query = ""
        self._prefetch_result_generation = 0
        self._prefetch_result_session_id = ""
        self._prefetch_completed_query = ""
        self._prefetch_completed_generation = 0
        self._prefetch_completed_session_id = ""
        self._prefetch_requested_query = ""
        self._prefetch_requested_generation = 0
        self._prefetch_requested_session_id = ""
        self._prefetch_inflight_query = ""
        self._prefetch_inflight_generation = 0
        self._prefetch_inflight_session_id = ""
        self._prefetch_generation = 0
        self._prefetch_post_turn_skip_query = ""
        self._prefetch_post_turn_skip_session_id = ""
        self._prefetch_lock = threading.Lock()
        self._prefetch_condition = threading.Condition(self._prefetch_lock)
        self._prefetch_thread: threading.Thread | None = None
        self._prefetch_threads: set[threading.Thread] = set()
        self._prefetch_inflight: dict[int, tuple[str, str]] = {}
        # Single-writer model for retain. sync_turn() enqueues; the writer
        # thread drains sequentially. Avoids spawning ad-hoc threads that
        # can race the interpreter shutdown and emit "cannot schedule new
        # futures after interpreter shutdown" / "Unclosed client session".
        self._retain_queue: queue.Queue = queue.Queue()
        self._writer_thread: threading.Thread | None = None
        self._shutting_down = threading.Event()
        self._retain_lifecycle_lock = threading.RLock()
        self._atexit_registered = False
        # Legacy alias — older tests/callers reference _sync_thread directly.
        # Points at _writer_thread once the writer is running.
        self._sync_thread = None
        self._session_id = ""
        self._parent_session_id = ""
        self._document_id = ""

        # Tags
        self._tags: list[str] | None = None
        self._recall_tags: list[str] | None = None
        self._recall_tags_match = "any"
        self._scope_dimensions: list[str] = []

        # Retain controls
        self._auto_retain = True
        self._retain_every_n_turns = 1
        self._retain_async = True
        self._retain_context = "conversation between Hermes Agent and the User"
        self._turn_counter = 0
        self._session_turns: list[str] = []
        self._committed_turn_count = 0
        self._retain_state: _RetainSessionState | None = None

        # Recall controls
        self._auto_recall = True
        self._recall_max_tokens = 4096
        # Default to observation-only recall. Observations are Hindsight's
        # consolidated knowledge layer — deduplicated, evidence-grounded
        # beliefs built from many raw facts, with proof counts and
        # freshness signals (see hindsight.vectorize.io/developer/observations).
        # Including raw world/experience facts re-ships the supporting
        # evidence that observations already summarize, burning the
        # `recall_max_tokens` budget. Users can restore the broader
        # recall via the `recall_types` config key.
        self._recall_types: list[str] = ["observation"]
        self._recall_prompt_preamble = ""
        self._recall_max_input_chars = 800
        self._recall_min_scores: dict[str, float] = {}
        self._min_scores_support_warning_emitted = False
        self._reflect_min_scores_warning_emitted = False

        # Bank
        self._bank_mission = ""
        self._bank_retain_mission: str | None = None
        self._bank_id_template = ""

    @property
    def name(self) -> str:
        return "hindsight"

    def is_available(self) -> bool:
        try:
            cfg = _load_config()
            mode = cfg.get("mode", "cloud")
            if mode in {"local", "local_embedded"}:
                available, _ = _check_local_runtime()
                return available
            if mode == "local_external":
                return True
            has_key = bool(
                cfg.get("apiKey")
                or cfg.get("api_key")
                or os.environ.get("HINDSIGHT_API_KEY", "")
            )
            has_url = bool(cfg.get("api_url") or os.environ.get("HINDSIGHT_API_URL", ""))
            return has_key or has_url
        except Exception:
            return False

    def save_config(self, values, hermes_home):
        """Write config to $HERMES_HOME/hindsight/config.json."""
        import json
        from pathlib import Path
        config_dir = Path(hermes_home) / "hindsight"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_path = config_dir / "config.json"
        existing = {}
        if config_path.exists():
            try:
                existing = json.loads(config_path.read_text())
            except Exception:
                pass
        existing.update(values)
        from utils import atomic_json_write
        atomic_json_write(config_path, existing, mode=0o600)

    def post_setup(self, hermes_home: str, config: dict) -> None:
        """Custom setup wizard — installs only the deps needed for the selected mode."""
        import subprocess
        import shutil
        import sys
        from pathlib import Path

        from hermes_cli.config import save_config
        from hermes_cli.secret_prompt import masked_secret_prompt

        from hermes_cli.memory_setup import _CANCELLED, _curses_select, _print_cancelled_setup

        print("\n  Configuring Hindsight memory:\n")

        existing_config = self._config if isinstance(self._config, dict) else _load_config()
        if not isinstance(existing_config, dict):
            existing_config = {}

        # Step 1: Mode selection
        mode_values = ["cloud", "local_embedded", "local_external"]
        mode_items = [
            ("Cloud", "Hindsight Cloud API (lightweight, just needs an API key)"),
            ("Local Embedded", "Run Hindsight locally (downloads ~200MB, needs LLM key)"),
            ("Local External", "Connect to an existing Hindsight instance"),
        ]
        existing_mode = existing_config.get("mode")
        mode_default_idx = mode_values.index(existing_mode) if existing_mode in mode_values else 0
        mode_idx = _curses_select("  Select mode", mode_items, default=mode_default_idx, cancel_returns=_CANCELLED)
        if mode_idx == _CANCELLED:
            _print_cancelled_setup()
            return
        mode = mode_values[mode_idx]

        provider_config: dict = dict(existing_config)
        provider_config["mode"] = mode
        env_writes: dict = {}

        # Step 2: Install/upgrade deps for selected mode
        cloud_dep = _CLIENT_REQUIREMENT
        local_dep = "hindsight-all"
        if mode == "local_embedded":
            deps_to_install = [local_dep]
        elif mode == "local_external":
            deps_to_install = [cloud_dep]
        else:
            deps_to_install = [cloud_dep]

        llm_provider = ""
        if mode == "local_embedded":
            providers_list = list(_PROVIDER_DEFAULT_MODELS.keys())
            llm_items = [
                (p, f"default model: {_PROVIDER_DEFAULT_MODELS[p]}")
                for p in providers_list
            ]
            existing_llm_provider = provider_config.get("llm_provider")
            llm_default_idx = providers_list.index(existing_llm_provider) if existing_llm_provider in providers_list else 0
            llm_idx = _curses_select(
                "  Select LLM provider",
                llm_items,
                default=llm_default_idx,
                cancel_returns=_CANCELLED,
            )
            if llm_idx == _CANCELLED:
                _print_cancelled_setup()
                return
            llm_provider = providers_list[llm_idx]
            provider_config["llm_provider"] = llm_provider

        print("\n  Checking dependencies...")
        uv_path = shutil.which("uv")
        if not uv_path:
            print("  ⚠ uv not found — install it: curl -LsSf https://astral.sh/uv/install.sh | sh")
            print(f"  Then run manually: uv pip install --python {sys.executable} {' '.join(deps_to_install)}")
        else:
            try:
                subprocess.run(
                    [uv_path, "pip", "install", "--python", sys.executable, "--quiet", "--upgrade"] + deps_to_install,
                    check=True, timeout=120, capture_output=True,
                    stdin=subprocess.DEVNULL,
                )
                print("  ✓ Dependencies up to date")
            except Exception as e:
                print(f"  ⚠ Install failed: {e}")
                print(f"  Run manually: uv pip install --python {sys.executable} {' '.join(deps_to_install)}")

        # Step 3: Mode-specific config
        if mode == "cloud":
            print("\n  Get your API key at https://ui.hindsight.vectorize.io\n")
            existing_key = os.environ.get("HINDSIGHT_API_KEY", "")
            if existing_key:
                masked = f"...{existing_key[-4:]}" if len(existing_key) > 4 else "set"
                sys.stdout.write(f"  API key (current: {masked}, blank to keep): ")
                sys.stdout.flush()
                api_key = masked_secret_prompt("") if sys.stdin.isatty() else sys.stdin.readline().strip()
            else:
                sys.stdout.write("  API key: ")
                sys.stdout.flush()
                api_key = masked_secret_prompt("") if sys.stdin.isatty() else sys.stdin.readline().strip()
            if api_key:
                env_writes["HINDSIGHT_API_KEY"] = api_key

            val = input(f"  API URL [{_DEFAULT_API_URL}]: ").strip()
            if val:
                provider_config["api_url"] = val

        elif mode == "local_external":
            val = input(f"  Hindsight API URL [{_DEFAULT_LOCAL_URL}]: ").strip()
            provider_config["api_url"] = val or _DEFAULT_LOCAL_URL

            sys.stdout.write("  API key (optional, blank to skip): ")
            sys.stdout.flush()
            api_key = masked_secret_prompt("") if sys.stdin.isatty() else sys.stdin.readline().strip()
            if api_key:
                env_writes["HINDSIGHT_API_KEY"] = api_key

        else:  # local_embedded
            if llm_provider == "openai_compatible":
                existing_base_url = provider_config.get("llm_base_url", "")
                prompt = "  LLM endpoint URL (e.g. http://192.168.1.10:8080/v1)"
                if existing_base_url:
                    prompt += f" [{existing_base_url}]"
                prompt += ": "
                val = input(prompt).strip()
                if val:
                    provider_config["llm_base_url"] = val
            elif llm_provider == "openrouter":
                provider_config["llm_base_url"] = "https://openrouter.ai/api/v1"

            provider_default_model = _PROVIDER_DEFAULT_MODELS.get(llm_provider, "gpt-4o-mini")
            current_model = provider_config.get("llm_model") or provider_default_model
            val = input(f"  LLM model [{current_model}]: ").strip()
            provider_config["llm_model"] = val or current_model

            sys.stdout.write("  LLM API key: ")
            sys.stdout.flush()
            llm_key = masked_secret_prompt("") if sys.stdin.isatty() else sys.stdin.readline().strip()
            if llm_key:
                env_writes["HINDSIGHT_LLM_API_KEY"] = llm_key
            else:
                env_path = Path(hermes_home) / ".env"
                existing_llm_key = ""
                if env_path.exists():
                    for line in env_path.read_text().splitlines():
                        if line.startswith("HINDSIGHT_LLM_API_KEY="):
                            existing_llm_key = line.split("=", 1)[1]
                            break
                env_writes["HINDSIGHT_LLM_API_KEY"] = existing_llm_key

        # Step 4: Save everything
        provider_config.setdefault("bank_id", "hermes")
        provider_config.setdefault("recall_budget", "mid")
        # Read existing timeout from config if present, otherwise use default.
        # Preserve explicit 0 values instead of treating them as blank.
        existing_timeout = provider_config.get("timeout")
        timeout_val = existing_timeout if existing_timeout is not None else _DEFAULT_TIMEOUT
        provider_config["timeout"] = timeout_val
        env_writes["HINDSIGHT_TIMEOUT"] = str(timeout_val)
        if mode == "local_embedded":
            existing_idle_timeout = provider_config.get("idle_timeout")
            idle_timeout_val = existing_idle_timeout if existing_idle_timeout is not None else _DEFAULT_IDLE_TIMEOUT
            provider_config["idle_timeout"] = idle_timeout_val
            env_writes["HINDSIGHT_IDLE_TIMEOUT"] = str(idle_timeout_val)
        config["memory"]["provider"] = "hindsight"
        save_config(config)

        self.save_config(provider_config, hermes_home)

        if env_writes:
            env_path = Path(hermes_home) / ".env"
            env_path.parent.mkdir(parents=True, exist_ok=True)
            existing_lines = []
            if env_path.exists():
                existing_lines = env_path.read_text().splitlines()
            updated_keys = set()
            new_lines = []
            for line in existing_lines:
                key_match = line.split("=", 1)[0].strip() if "=" in line and not line.startswith("#") else None
                if key_match and key_match in env_writes:
                    new_lines.append(f"{key_match}={env_writes[key_match]}")
                    updated_keys.add(key_match)
                else:
                    new_lines.append(line)
            for k, v in env_writes.items():
                if k not in updated_keys:
                    new_lines.append(f"{k}={v}")
            env_path.write_text("\n".join(new_lines) + "\n")

        if mode == "local_embedded":
            materialized_config = dict(provider_config)
            config_path = Path(hermes_home) / "hindsight" / "config.json"
            try:
                materialized_config = json.loads(config_path.read_text(encoding="utf-8"))
            except Exception:
                pass

            llm_api_key = env_writes.get("HINDSIGHT_LLM_API_KEY", "")
            if not llm_api_key:
                llm_api_key = _load_simple_env(Path(hermes_home) / ".env").get("HINDSIGHT_LLM_API_KEY", "")
            if not llm_api_key:
                llm_api_key = _load_simple_env(_embedded_profile_env_path(materialized_config)).get(
                    "HINDSIGHT_API_LLM_API_KEY",
                    "",
                )

            _materialize_embedded_profile_env(
                materialized_config,
                llm_api_key=llm_api_key or None,
            )

        print(f"\n  ✓ Hindsight memory configured ({mode} mode)")
        if env_writes:
            print("  API keys saved to .env")
        print("\n  Start a new session to activate.\n")

    def get_config_schema(self):
        return [
            {"key": "mode", "description": "Connection mode", "default": "cloud", "choices": ["cloud", "local_embedded", "local_external"]},
            # Cloud mode
            {"key": "api_url", "description": "Hindsight Cloud API URL", "default": _DEFAULT_API_URL, "when": {"mode": "cloud"}},
            {"key": "api_key", "description": "Hindsight Cloud API key", "secret": True, "env_var": "HINDSIGHT_API_KEY", "url": "https://ui.hindsight.vectorize.io", "when": {"mode": "cloud"}},
            # Local external mode
            {"key": "api_url", "description": "Hindsight API URL", "default": _DEFAULT_LOCAL_URL, "when": {"mode": "local_external"}},
            {"key": "api_key", "description": "API key (optional)", "secret": True, "env_var": "HINDSIGHT_API_KEY", "when": {"mode": "local_external"}},
            # Local embedded mode
            {"key": "llm_provider", "description": "LLM provider", "default": "openai", "choices": ["openai", "anthropic", "gemini", "groq", "openrouter", "minimax", "ollama", "lmstudio", "openai_compatible"], "when": {"mode": "local_embedded"}},
            {"key": "llm_base_url", "description": "Endpoint URL (e.g. http://192.168.1.10:8080/v1)", "default": "", "when": {"mode": "local_embedded", "llm_provider": "openai_compatible"}},
            {"key": "llm_api_key", "description": "LLM API key (optional for openai_compatible)", "secret": True, "env_var": "HINDSIGHT_LLM_API_KEY", "when": {"mode": "local_embedded"}},
            {"key": "llm_model", "description": "LLM model", "default": "gpt-4o-mini", "default_from": {"field": "llm_provider", "map": _PROVIDER_DEFAULT_MODELS}, "when": {"mode": "local_embedded"}},
            {"key": "bank_id", "description": "Memory bank name (static fallback when bank_id_template is unset)", "default": "hermes"},
            {"key": "bank_id_template", "description": "Optional template to derive bank_id dynamically. Placeholders: {profile}, {workspace}, {platform}, {user}, {session}. Example: hermes-{profile}", "default": ""},
            {"key": "bank_mission", "description": "Mission/purpose description for the memory bank"},
            {"key": "bank_retain_mission", "description": "Custom extraction prompt for memory retention"},
            {"key": "recall_budget", "description": "Recall thoroughness", "default": "mid", "choices": ["low", "mid", "high"]},
            {"key": "memory_mode", "description": "Memory integration mode", "default": "hybrid", "choices": ["hybrid", "context", "tools"]},
            {"key": "recall_prefetch_method", "description": "Auto-recall method", "default": "recall", "choices": ["recall", "reflect"]},
            {"key": "retain_tags", "description": "Default tags applied to retained memories (comma-separated)", "default": ""},
            {"key": "observation_scopes", "description": "How observations are scoped during consolidation: 'combined' (default — one pass over all tags), 'per_tag' (one isolated observation per tag), 'all_combinations' (every tag subset — expensive), or a JSON list of tag-lists for explicit custom scopes. Empty uses Hindsight's 'combined' default.", "default": ""},
            {"key": "retain_source", "description": "Metadata source value attached to retained memories", "default": ""},
            {"key": "retain_user_prefix", "description": "Label used before user turns in retained transcripts", "default": "User"},
            {"key": "retain_assistant_prefix", "description": "Label used before assistant turns in retained transcripts", "default": "Assistant"},
            {"key": "recall_tags", "description": "Tags to filter when searching memories (comma-separated)", "default": ""},
            {"key": "recall_tags_match", "description": "Tag matching mode for recall", "default": "any", "choices": ["any", "all", "any_strict", "all_strict"]},
            {"key": "scope_tags", "description": "Optional dynamic isolation dimensions applied as hashed tags. Accepts a list or comma-separated profile, workspace, and session values.", "default": []},
            {"key": "recall_min_scores", "description": "Inclusive recall score floors using semantic, keyword, reranker, and/or final keys. Applies to recall only, not reflection.", "default": {}},
            {"key": "recall_types", "description": "Fact types to surface on recall — applies to both auto-recall and the hindsight_recall tool (comma-separated or list). Defaults to observation-only — observations are Hindsight's consolidated, deduplicated, evidence-grounded knowledge layer; raw world/experience facts are the supporting evidence observations already summarize. Set to e.g. 'observation,world,experience' to also include raw facts.", "default": "observation"},
            {"key": "auto_recall", "description": "Automatically recall memories before each turn", "default": True},
            {"key": "auto_retain", "description": "Automatically retain conversation turns", "default": True},
            {"key": "retain_every_n_turns", "description": "Retain every N turns (1 = every turn)", "default": 1},
            {"key": "retain_async","description": "Process retain asynchronously on the Hindsight server", "default": True},
            {"key": "retain_context", "description": "Context label for retained memories", "default": "conversation between Hermes Agent and the User"},
            {"key": "recall_max_tokens", "description": "Maximum tokens for recall results", "default": 4096},
            {"key": "recall_max_input_chars", "description": "Maximum input query length for auto-recall", "default": 800},
            {"key": "recall_prompt_preamble", "description": "Custom preamble for recalled memories in context"},
            {"key": "timeout", "description": "API request timeout in seconds", "default": _DEFAULT_TIMEOUT},
            {"key": "idle_timeout", "description": "Embedded daemon idle timeout in seconds (0 disables auto-shutdown)", "default": _DEFAULT_IDLE_TIMEOUT, "when": {"mode": "local_embedded"}},
            {"key": "port_health_grace_timeout", "description": "Seconds to wait for a slow daemon /health before treating it as stale (raise on busy/low-resource hosts; blank uses the 30s default)", "default": "", "when": {"mode": "local_embedded"}},
        ]

    def _get_client(self):
        """Return the cached Hindsight client (created once, reused)."""
        if self._client is None:
            if self._mode == "local_embedded":
                available, reason = _check_local_runtime()
                if not available:
                    raise RuntimeError(
                        "Hindsight local runtime is unavailable"
                        + (f": {reason}" if reason else "")
                    )
                try:
                    from tools.lazy_deps import ensure as _lazy_ensure
                    _lazy_ensure("memory.hindsight", prompt=False)
                except ImportError:
                    pass
                except Exception as _e:
                    raise ImportError(str(_e))
                from hindsight import HindsightEmbedded
                HindsightEmbedded.__del__ = lambda self: None
                llm_provider = self._config.get("llm_provider", "")
                if llm_provider in {"openai_compatible", "openrouter"}:
                    llm_provider = "openai"
                logger.debug("Creating HindsightEmbedded client (profile=%s, provider=%s)",
                             self._config.get("profile", "hermes"), llm_provider)
                kwargs = dict(
                    profile=self._config.get("profile", "hermes"),
                    llm_provider=llm_provider,
                    llm_api_key=self._config.get("llmApiKey") or self._config.get("llm_api_key") or os.environ.get("HINDSIGHT_LLM_API_KEY", ""),
                    llm_model=self._config.get("llm_model", ""),
                )
                if self._llm_base_url:
                    kwargs["llm_base_url"] = self._llm_base_url
                idle_timeout = _parse_int_setting(
                    self._config.get("idle_timeout")
                    if self._config.get("idle_timeout") is not None
                    else os.environ.get("HINDSIGHT_IDLE_TIMEOUT", self._idle_timeout),
                    _DEFAULT_IDLE_TIMEOUT,
                )
                self._idle_timeout = idle_timeout
                kwargs["idle_timeout"] = idle_timeout
                self._client = HindsightEmbedded(**kwargs)
            else:
                _ensure_cloud_client_dependency()
                from hindsight_client import Hindsight
                timeout = self._timeout or _DEFAULT_TIMEOUT
                kwargs = {"base_url": self._api_url, "timeout": float(timeout)}
                if self._api_key:
                    kwargs["api_key"] = self._api_key
                logger.debug("Creating Hindsight cloud client (url=%s, has_key=%s, timeout=%s)",
                             self._api_url, bool(self._api_key), kwargs["timeout"])
                self._client = Hindsight(**kwargs)
        return self._client

    def _run_sync(self, coro):
        """Schedule *coro* on the shared loop using the configured timeout."""
        return _run_sync(coro, timeout=self._timeout)

    def _is_retriable_embedded_connection_error(self, exc: Exception) -> bool:
        """Return True for stale embedded-daemon connection failures."""
        if self._mode != "local_embedded":
            return False
        text = f"{type(exc).__name__}: {exc}".lower()
        return any(
            marker in text
            for marker in (
                "cannot connect to host",
                "connection refused",
                "connect call failed",
                "clientconnectorerror",
            )
        )

    def _ensure_writer(self) -> None:
        """Lazy-start the single retain-writer thread.

        We don't start the writer in initialize() so providers that never
        retain (e.g. tools-only mode) don't pay for an idle thread.
        """
        thread = self._writer_thread
        if thread is not None and thread.is_alive():
            return
        if self._shutting_down.is_set():
            return
        thread = threading.Thread(
            target=self._writer_loop,
            daemon=True,
            name="hindsight-writer",
        )
        self._writer_thread = thread
        # Keep the legacy _sync_thread alias pointing at the writer so any
        # external code that joins _sync_thread keeps working.
        self._sync_thread = thread
        thread.start()

    def _writer_loop(self) -> None:
        """Drain the retain queue serially. Exits on sentinel.

        Each job() is wrapped so a single failure can't kill the writer.
        task_done() always fires so queue.join() works in tests.
        """
        while True:
            try:
                job = self._retain_queue.get(timeout=1.0)
            except queue.Empty:
                if self._shutting_down.is_set():
                    return
                continue
            try:
                if job is _WRITER_SENTINEL:
                    return
                try:
                    job()
                except Exception as exc:
                    logger.warning("Hindsight retain failed: %s", exc, exc_info=True)
            finally:
                self._retain_queue.task_done()

    def _register_atexit(self) -> None:
        """Register an idempotent atexit hook to drain the writer.

        Without this, a CLI exit that doesn't go through MemoryManager.
        shutdown_all() would leave in-flight retain jobs racing interpreter
        teardown, producing "cannot schedule new futures" warnings and
        unclosed aiohttp sessions.
        """
        if self._atexit_registered:
            return
        self._atexit_registered = True
        atexit.register(self._atexit_shutdown)

    def _atexit_shutdown(self) -> None:
        if self._shutting_down.is_set():
            return
        try:
            self.shutdown()
        except Exception as exc:
            logger.debug("Hindsight atexit shutdown failed: %s", exc)

    def _run_hindsight_operation(self, operation):
        """Run an async Hindsight client operation, retrying once after idle shutdown."""
        client = self._get_client()
        try:
            return self._run_sync(operation(client))
        except Exception as exc:
            if not self._is_retriable_embedded_connection_error(exc):
                raise
            logger.info(
                "Hindsight embedded daemon appears unreachable; recreating client and retrying once: %s",
                exc,
            )
            self._client = None
            client = self._get_client()
            self._client = client
            return self._run_sync(operation(client))

    def _start_retain_session(
        self, session_id: str, parent_session_id: str
    ) -> _RetainSessionState:
        """Rotate automatic-retain state to a fresh session nonce."""
        self._session_id = session_id
        self._parent_session_id = parent_session_id
        start_ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self._document_id = f"{session_id}-{start_ts}"
        state = _RetainSessionState(
            session_id=session_id,
            parent_session_id=parent_session_id,
            document_id=self._document_id,
        )
        self._retain_state = state
        self._session_turns = state.turns
        self._turn_counter = 0
        self._turn_index = 0
        self._committed_turn_count = 0
        return state

    def _build_automatic_retain_batch(
        self,
        state: _RetainSessionState,
        start_turn: int,
        end_turn: int,
    ) -> _AutomaticRetainBatch:
        """Build one immutable automatic-retain batch for a stable turn range."""
        turns = state.turns[start_turn - 1:end_turn]
        content = "[" + ",".join(turns) + "]"
        metadata = self._build_metadata(
            message_count=len(turns) * 2,
            turn_index=end_turn,
        )
        if state.session_id:
            metadata["session_id"] = state.session_id
        else:
            metadata.pop("session_id", None)

        lineage_tags: list[str] = []
        if state.session_id:
            lineage_tags.append(f"session:{state.session_id}")
        if state.parent_session_id:
            lineage_tags.append(f"parent:{state.parent_session_id}")

        item = self._build_retain_kwargs(
            content,
            context=self._retain_context,
            metadata=metadata,
            tags=lineage_tags or None,
            scope_session_id=state.session_id,
        )
        item.pop("bank_id", None)
        item.pop("retain_async", None)
        return _AutomaticRetainBatch(
            start_turn=start_turn,
            end_turn=end_turn,
            document_id=f"{state.document_id}-{state.nonce}-t{start_turn}-{end_turn}",
            item=item,
        )

    def _retain_automatic_through(
        self, state: _RetainSessionState, target_turn_count: int
    ) -> None:
        """Retain one queued target in FIFO order without advancing on failure."""
        target_turn_count = min(target_turn_count, len(state.turns))
        while state.committed_turn_count < target_turn_count:
            batch = state.failed_batch
            if batch is None:
                batch = self._build_automatic_retain_batch(
                    state,
                    state.committed_turn_count + 1,
                    target_turn_count,
                )

            logger.debug(
                "Hindsight retain: bank=%s, doc=%s, async=%s, turns=%d-%d",
                self._bank_id,
                batch.document_id,
                self._retain_async,
                batch.start_turn,
                batch.end_turn,
            )
            try:
                self._run_hindsight_operation(
                    lambda client: client.aretain_batch(
                        bank_id=self._bank_id,
                        items=[batch.item],
                        document_id=batch.document_id,
                        retain_async=self._retain_async,
                    )
                )
            except Exception:
                state.failed_batch = batch
                raise

            state.failed_batch = None
            state.committed_turn_count = batch.end_turn
            if self._retain_state is state:
                self._committed_turn_count = state.committed_turn_count
            logger.debug("Hindsight retain succeeded: doc=%s", batch.document_id)

    def _enqueue_automatic_retain(
        self, state: _RetainSessionState, target_turn_count: int
    ) -> None:
        """Queue one automatic retain target on the provider's single writer."""
        def _do_retain() -> None:
            self._retain_automatic_through(state, target_turn_count)

        self._ensure_writer()
        self._register_atexit()
        self._retain_queue.put(_do_retain)

    def initialize(self, session_id: str, **kwargs) -> None:
        session_id = str(session_id or "").strip()
        parent_session_id = str(kwargs.get("parent_session_id", "") or "").strip()
        self._start_retain_session(session_id, parent_session_id)

        # Check client version and auto-upgrade if needed
        try:
            from importlib.metadata import version as pkg_version
            from packaging.version import Version
            installed = pkg_version("hindsight-client")
            if Version(installed) < Version(_MIN_CLIENT_VERSION):
                logger.warning(
                    "hindsight-client %s is outdated (need %s), attempting upgrade...",
                    installed,
                    _CLIENT_REQUIREMENT,
                )
                import shutil
                import subprocess
                import sys
                uv_path = shutil.which("uv")
                if uv_path:
                    try:
                        subprocess.run(
                            [uv_path, "pip", "install", "--python", sys.executable,
                             "--quiet", "--upgrade", _CLIENT_REQUIREMENT],
                            check=True, timeout=120, capture_output=True,
                            stdin=subprocess.DEVNULL,
                        )
                        logger.info("hindsight-client upgraded to %s", _CLIENT_REQUIREMENT)
                    except Exception as e:
                        logger.warning(
                            "Auto-upgrade failed: %s. Run: uv pip install '%s'",
                            e,
                            _CLIENT_REQUIREMENT,
                        )
                else:
                    logger.warning("uv not found. Run: pip install '%s'", _CLIENT_REQUIREMENT)
        except Exception:
            pass  # packaging not available or other issue — proceed anyway

        self._config = _load_config()
        self._platform = str(kwargs.get("platform") or "").strip()
        self._user_id = str(kwargs.get("user_id") or "").strip()
        self._user_name = str(kwargs.get("user_name") or "").strip()
        self._chat_id = str(kwargs.get("chat_id") or "").strip()
        self._chat_name = str(kwargs.get("chat_name") or "").strip()
        self._chat_type = str(kwargs.get("chat_type") or "").strip()
        self._thread_id = str(kwargs.get("thread_id") or "").strip()
        self._agent_identity = str(kwargs.get("agent_identity") or "").strip()
        self._agent_workspace = str(kwargs.get("agent_workspace") or "").strip()
        self._scope_workspace = str(kwargs.get("scope_workspace") or "").strip()
        self._mode = self._config.get("mode", "cloud")
        # Read timeout from config or env var, fall back to default
        self._timeout = _parse_int_setting(
            self._config.get("timeout") if self._config.get("timeout") is not None else os.environ.get("HINDSIGHT_TIMEOUT"),
            _DEFAULT_TIMEOUT,
        )
        self._idle_timeout = _parse_int_setting(
            self._config.get("idle_timeout") if self._config.get("idle_timeout") is not None else os.environ.get("HINDSIGHT_IDLE_TIMEOUT"),
            _DEFAULT_IDLE_TIMEOUT,
        )
        # "local" is a legacy alias for "local_embedded"
        if self._mode == "local":
            self._mode = "local_embedded"
        if self._mode == "local_embedded":
            # Export the daemon health grace timeout BEFORE importing
            # daemon_embed_manager (which reads it at import time).
            _export_port_health_grace_timeout(self._config)
            available, reason = _check_local_runtime()
            if not available:
                logger.warning(
                    "Hindsight local mode disabled because its runtime could not be imported: %s",
                    reason,
                )
                self._mode = "disabled"
                return
        self._api_key = self._config.get("apiKey") or self._config.get("api_key") or os.environ.get("HINDSIGHT_API_KEY", "")
        default_url = _DEFAULT_LOCAL_URL if self._mode in {"local_embedded", "local_external"} else _DEFAULT_API_URL
        self._api_url = self._config.get("api_url") or os.environ.get("HINDSIGHT_API_URL", default_url)
        self._llm_base_url = self._config.get("llm_base_url", "")

        banks = cfg_get(self._config, "banks", "hermes", default={})
        static_bank_id = self._config.get("bank_id") or banks.get("bankId", "hermes")
        self._bank_id_template = self._config.get("bank_id_template", "") or ""
        self._bank_id = _resolve_bank_id_template(
            self._bank_id_template,
            fallback=static_bank_id,
            profile=self._agent_identity,
            workspace=self._agent_workspace,
            platform=self._platform,
            user=self._user_id,
            session=self._session_id,
        )
        budget = self._config.get("recall_budget") or self._config.get("budget") or banks.get("budget", "mid")
        self._budget = budget if budget in _VALID_BUDGETS else "mid"

        memory_mode = self._config.get("memory_mode", "hybrid")
        self._memory_mode = memory_mode if memory_mode in {"context", "tools", "hybrid"} else "hybrid"

        prefetch_method = self._config.get("recall_prefetch_method") or self._config.get("prefetch_method", "recall")
        self._prefetch_method = prefetch_method if prefetch_method in {"recall", "reflect"} else "recall"

        # Bank options
        self._bank_mission = self._config.get("bank_mission", "")
        self._bank_retain_mission = self._config.get("bank_retain_mission") or None

        # Tags
        self._retain_tags = _normalize_retain_tags(
            self._config.get("retain_tags")
            or os.environ.get("HINDSIGHT_RETAIN_TAGS", "")
        )
        self._tags = self._retain_tags or None
        self._observation_scopes = _normalize_observation_scopes(
            self._config.get("observation_scopes")
            or os.environ.get("HINDSIGHT_RETAIN_OBSERVATION_SCOPES", "")
        )
        self._recall_tags = self._config.get("recall_tags") or None
        self._recall_tags_match = self._config.get("recall_tags_match", "any")
        self._scope_dimensions = _normalize_scope_dimensions(
            self._config.get("scope_tags", [])
        )
        self._retain_source = str(
            self._config.get("retain_source") or os.environ.get("HINDSIGHT_RETAIN_SOURCE", "")
        ).strip()
        self._retain_user_prefix = str(
            self._config.get("retain_user_prefix") or os.environ.get("HINDSIGHT_RETAIN_USER_PREFIX", "User")
        ).strip() or "User"
        self._retain_assistant_prefix = str(
            self._config.get("retain_assistant_prefix") or os.environ.get("HINDSIGHT_RETAIN_ASSISTANT_PREFIX", "Assistant")
        ).strip() or "Assistant"

        # Retain controls
        self._auto_retain = self._config.get("auto_retain", True)
        self._retain_every_n_turns = max(1, int(self._config.get("retain_every_n_turns", 1)))
        self._retain_context = self._config.get("retain_context", "conversation between Hermes Agent and the User")

        # Recall controls
        self._auto_recall = self._config.get("auto_recall", True)
        self._recall_max_tokens = int(self._config.get("recall_max_tokens", 4096))
        self._recall_min_scores = _normalize_recall_min_scores(
            self._config.get("recall_min_scores", {})
        )
        # Default narrows recall to observation-only; pass an explicit
        # `recall_types` list in config.json to broaden (e.g. include
        # "world" / "experience") or to disable the filter entirely.
        configured_types = self._config.get("recall_types")
        if configured_types is None:
            self._recall_types = ["observation"]
        elif isinstance(configured_types, str):
            # Allow comma-separated strings for parity with recall_tags.
            self._recall_types = [t.strip() for t in configured_types.split(",") if t.strip()]
        else:
            self._recall_types = list(configured_types) or ["observation"]
        self._recall_prompt_preamble = self._config.get("recall_prompt_preamble", "")
        self._recall_max_input_chars = int(self._config.get("recall_max_input_chars", 800))
        self._retain_async = self._config.get("retain_async", True)

        _client_version = "unknown"
        try:
            from importlib.metadata import version as pkg_version
            _client_version = pkg_version("hindsight-client")
        except Exception:
            pass
        logger.info("Hindsight initialized: mode=%s, api_url=%s, bank=%s, budget=%s, memory_mode=%s, prefetch_method=%s, client=%s",
                     self._mode, self._api_url, self._bank_id, self._budget, self._memory_mode, self._prefetch_method, _client_version)
        if self._bank_id_template:
            logger.debug(
                "Hindsight bank resolved from template %r: profile=%s "
                "workspace_set=%s platform=%s user=%s -> bank=%s",
                self._bank_id_template,
                self._agent_identity,
                bool(self._agent_workspace),
                self._platform,
                self._user_id,
                self._bank_id,
            )
        logger.debug(
            "Hindsight config: auto_retain=%s, auto_recall=%s, "
            "retain_every_n=%d, retain_async=%s, retain_context=%s, "
            "recall_max_tokens=%d, recall_max_input_chars=%d, tags=%s, "
            "recall_tags=%s, scope_dimensions=%s, recall_min_scores=%s",
            self._auto_retain,
            self._auto_recall,
            self._retain_every_n_turns,
            self._retain_async,
            self._retain_context,
            self._recall_max_tokens,
            self._recall_max_input_chars,
            self._tags,
            self._recall_tags,
            self._scope_dimensions,
            self._recall_min_scores,
        )

        # For local mode, start the embedded daemon in the background so it
        # doesn't block the chat. Redirect stdout/stderr to a log file to
        # prevent rich startup output from spamming the terminal.
        if self._mode == "local_embedded":
            # PostgreSQL's initdb refuses to run as root by design, so the
            # embedded daemon can never initialize its data directory under
            # root. Without this guard the daemon-start thread would fail,
            # retry, and loop forever — each cycle reloading embedding models
            # (~958MB RAM, ~33% CPU) with no user-visible error. Detect root
            # up front and skip daemon startup with a clear message instead.
            if hasattr(os, "geteuid") and os.geteuid() == 0:
                msg = (
                    "Hindsight local_embedded mode cannot run as root "
                    "(PostgreSQL initdb refuses root). Skipping the embedded "
                    "memory daemon. Run Hermes as a non-root user, or switch "
                    "to cloud / local_external mode via 'hermes memory setup'."
                )
                logger.warning(msg)
                # Surface to the terminal too — a daemon that never starts
                # would otherwise fail silently and the user would only see
                # Hermes get sluggish. (issue #13125)
                try:
                    print(f"  ⚠ {msg}", file=sys.stderr, flush=True)
                except Exception:
                    pass
                self._mode = "disabled"
                return

            def _start_daemon():
                import traceback
                log_dir = get_hermes_home() / "logs"
                log_dir.mkdir(parents=True, exist_ok=True)
                log_path = log_dir / "hindsight-embed.log"
                try:
                    # Redirect the daemon manager's Rich console to our log file
                    # instead of stderr. This avoids global fd redirects that
                    # would capture output from other threads.
                    import hindsight_embed.daemon_embed_manager as dem
                    from rich.console import Console
                    dem.console = Console(file=open(log_path, "a", encoding="utf-8"), force_terminal=False)

                    client = self._get_client()
                    profile = self._config.get("profile", "hermes")

                    # Update the profile .env to match our current config so
                    # the daemon always starts with the right settings.
                    # If the config changed and the daemon is running, stop it.
                    profile_env = _embedded_profile_env_path(self._config)
                    expected_env = _build_embedded_profile_env(self._config)
                    saved = _load_simple_env(profile_env)
                    config_changed = saved != expected_env

                    if config_changed:
                        profile_env = _materialize_embedded_profile_env(self._config)
                        if client._manager.is_running(profile):
                            with open(log_path, "a", encoding="utf-8") as f:
                                f.write("\n=== Config changed, restarting daemon ===\n")
                            client._manager.stop(profile)

                    client._ensure_started()
                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write("\n=== Daemon started successfully ===\n")
                except Exception as e:
                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write(f"\n=== Daemon startup failed: {e} ===\n")
                        traceback.print_exc(file=f)

            t = threading.Thread(target=_start_daemon, daemon=True, name="hindsight-daemon-start")
            t.start()

    def _scope_tags_for_session(self, session_id: str) -> list[str]:
        values = {
            "profile": self._agent_identity,
            "workspace": self._scope_workspace,
            "session": str(session_id or "").strip(),
        }
        missing_dimensions = [
            dimension
            for dimension in self._scope_dimensions
            if not str(values[dimension] or "").strip()
        ]
        if missing_dimensions:
            dimensions = ", ".join(missing_dimensions)
            raise ValueError(
                "Missing required Hindsight scope context for configured "
                f"dimension(s): {dimensions}"
            )
        return [
            _dynamic_scope_tag(dimension, values[dimension])
            for dimension in self._scope_dimensions
        ]

    def _automatic_scope_context_available(
        self, session_id: str, operation: str
    ) -> bool:
        try:
            self._scope_tags_for_session(session_id)
        except ValueError as exc:
            logger.warning("Hindsight automatic %s skipped: %s", operation, exc)
            return False
        return True

    def _query_tag_kwargs(
        self,
        session_id: str,
        *,
        include_legacy_without_scope: bool,
    ) -> dict[str, Any]:
        dynamic_tags = self._scope_tags_for_session(session_id)
        if not dynamic_tags:
            if include_legacy_without_scope and self._recall_tags:
                return {
                    "tags": self._recall_tags,
                    "tags_match": self._recall_tags_match,
                }
            return {}

        tags = _normalize_retain_tags(self._recall_tags)
        for tag in dynamic_tags:
            if tag not in tags:
                tags.append(tag)
        return {"tags": tags, "tags_match": "all_strict"}

    @staticmethod
    def _client_supports_recall_min_scores(client: Any) -> bool:
        try:
            parameters = inspect.signature(client.arecall).parameters.values()
        except (AttributeError, TypeError, ValueError):
            return False
        return any(
            parameter.name == "min_scores"
            or parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters
        )

    def _recall_min_scores_support_error(self, client: Any) -> str:
        if not self._recall_min_scores:
            return ""
        if self._client_supports_recall_min_scores(client):
            return ""
        return (
            "Configured Hindsight recall_min_scores cannot be enforced because "
            "this client does not support arecall(min_scores=...). Upgrade to "
            f"{_CLIENT_REQUIREMENT}; unfiltered recall was not run."
        )

    def _warn_unsupported_min_scores_once(self, message: str) -> None:
        with self._prefetch_condition:
            if self._min_scores_support_warning_emitted:
                return
            self._min_scores_support_warning_emitted = True
        logger.warning(
            "%s Automatic Hindsight memory injection is disabled until the "
            "client is upgraded.",
            message,
        )

    def _warn_reflect_min_scores_once(self) -> None:
        with self._prefetch_condition:
            if self._reflect_min_scores_warning_emitted:
                return
            self._reflect_min_scores_warning_emitted = True
        logger.warning(
            "Hindsight recall_prefetch_method=reflect cannot apply "
            "recall_min_scores because reflection has no score-filter API. "
            "Automatic memory injection is disabled; use recall prefetch or "
            "invoke hindsight_reflect explicitly."
        )

    def _normalize_prefetch_query(self, query: str) -> str:
        normalized = str(query or "").strip()
        if (
            self._recall_max_input_chars
            and len(normalized) > self._recall_max_input_chars
        ):
            normalized = normalized[: self._recall_max_input_chars]
        return normalized

    def _prefetch_enabled(self) -> bool:
        if self._memory_mode == "tools":
            logger.debug("Prefetch: skipped (tools-only mode)")
            return False
        if not self._auto_recall:
            logger.debug("Prefetch: skipped (auto_recall disabled)")
            return False
        if self._shutting_down.is_set():
            logger.debug("Prefetch: skipped (shutting down)")
            return False
        return True

    def _build_recall_kwargs(self, query: str, session_id: str) -> dict[str, Any]:
        recall_kwargs: dict[str, Any] = {
            "bank_id": self._bank_id,
            "query": query,
            "budget": self._budget,
            "max_tokens": self._recall_max_tokens,
        }
        recall_kwargs.update(
            self._query_tag_kwargs(
                session_id,
                include_legacy_without_scope=True,
            )
        )
        if self._recall_types:
            recall_kwargs["types"] = self._recall_types
        if self._recall_min_scores:
            recall_kwargs["min_scores"] = self._recall_min_scores
        return recall_kwargs

    def _execute_prefetch_query(self, query: str, session_id: str) -> str:
        if self._prefetch_method == "reflect":
            if self._recall_min_scores:
                self._warn_reflect_min_scores_once()
                return ""
            reflect_kwargs: dict[str, Any] = {
                "bank_id": self._bank_id,
                "query": query,
                "budget": self._budget,
            }
            reflect_kwargs.update(
                self._query_tag_kwargs(
                    session_id,
                    include_legacy_without_scope=False,
                )
            )
            logger.debug(
                "Prefetch: calling reflect (bank=%s, query_len=%d)",
                self._bank_id,
                len(query),
            )
            response = self._run_hindsight_operation(
                lambda client: client.areflect(**reflect_kwargs)
            )
            return response.text or ""

        client = self._client if self._client is not None else self._get_client()
        support_error = self._recall_min_scores_support_error(client)
        if support_error:
            self._warn_unsupported_min_scores_once(support_error)
            return ""
        recall_kwargs = self._build_recall_kwargs(query, session_id)
        logger.debug(
            "Prefetch: calling recall (bank=%s, query_len=%d, budget=%s)",
            self._bank_id,
            len(query),
            self._budget,
        )
        response = self._run_hindsight_operation(
            lambda operation_client: operation_client.arecall(**recall_kwargs)
        )
        results = response.results or []
        logger.debug("Prefetch: recall returned %d results", len(results))
        return "\n".join(f"- {result.text}" for result in results if result.text)

    def _start_prefetch_worker_locked(self) -> None:
        thread = threading.Thread(
            target=self._prefetch_worker,
            daemon=True,
            name="hindsight-prefetch",
        )
        self._prefetch_threads.add(thread)
        self._prefetch_thread = thread
        thread.start()

    def _request_prefetch(
        self,
        query: str,
        session_id: str,
        *,
        post_turn: bool,
    ) -> int | None:
        identity = (query, session_id)
        with self._prefetch_condition:
            if post_turn and self._prefetch_post_turn_skip_query:
                skip_identity = (
                    self._prefetch_post_turn_skip_query,
                    self._prefetch_post_turn_skip_session_id,
                )
                self._prefetch_post_turn_skip_query = ""
                self._prefetch_post_turn_skip_session_id = ""
                if skip_identity == identity:
                    logger.debug(
                        "Prefetch: skipped duplicate post-turn query (query_len=%d)",
                        len(query),
                    )
                    return None

            if (
                self._prefetch_result_query == query
                and self._prefetch_result_session_id == session_id
            ):
                return self._prefetch_result_generation
            if (
                self._prefetch_requested_query == query
                and self._prefetch_requested_session_id == session_id
                and self._prefetch_requested_generation
                > self._prefetch_completed_generation
            ):
                return self._prefetch_requested_generation

            self._prefetch_generation += 1
            generation = self._prefetch_generation
            self._prefetch_requested_query = query
            self._prefetch_requested_generation = generation
            self._prefetch_requested_session_id = session_id
            self._prefetch_result = ""
            self._prefetch_result_query = ""
            self._prefetch_result_generation = 0
            self._prefetch_result_session_id = ""
            if len(self._prefetch_threads) < _MAX_PREFETCH_WORKERS:
                self._start_prefetch_worker_locked()
            self._prefetch_condition.notify_all()
            return generation

    def _prefetch_worker(self) -> None:
        current_thread = threading.current_thread()
        try:
            while True:
                with self._prefetch_condition:
                    generation = self._prefetch_requested_generation
                    query = self._prefetch_requested_query
                    session_id = self._prefetch_requested_session_id
                    if (
                        not query
                        or generation <= self._prefetch_completed_generation
                        or generation in self._prefetch_inflight
                    ):
                        return
                    self._prefetch_inflight[generation] = (query, session_id)
                    if generation >= self._prefetch_inflight_generation:
                        self._prefetch_inflight_query = query
                        self._prefetch_inflight_session_id = session_id
                        self._prefetch_inflight_generation = generation

                text = ""
                try:
                    text = self._execute_prefetch_query(query, session_id)
                except Exception as exc:
                    logger.debug(
                        "Hindsight prefetch failed: %s",
                        exc,
                        exc_info=True,
                    )

                with self._prefetch_condition:
                    self._prefetch_inflight.pop(generation, None)
                    if self._prefetch_inflight_generation == generation:
                        self._prefetch_inflight_query = ""
                        self._prefetch_inflight_session_id = ""
                        self._prefetch_inflight_generation = 0
                    if (
                        generation == self._prefetch_generation
                        and generation == self._prefetch_requested_generation
                        and query == self._prefetch_requested_query
                        and session_id == self._prefetch_requested_session_id
                    ):
                        self._prefetch_result = text
                        self._prefetch_result_query = query
                        self._prefetch_result_generation = generation
                        self._prefetch_result_session_id = session_id
                        self._prefetch_completed_query = query
                        self._prefetch_completed_generation = generation
                        self._prefetch_completed_session_id = session_id
                    self._prefetch_condition.notify_all()
                    latest_generation = self._prefetch_requested_generation
                    if (
                        latest_generation <= self._prefetch_completed_generation
                        or latest_generation in self._prefetch_inflight
                        or not self._prefetch_requested_query
                    ):
                        return
        finally:
            with self._prefetch_condition:
                self._prefetch_threads.discard(current_thread)
                if (
                    self._prefetch_thread is current_thread
                    and self._prefetch_threads
                ):
                    self._prefetch_thread = next(iter(self._prefetch_threads))
                pending_generation = self._prefetch_requested_generation
                if (
                    not self._shutting_down.is_set()
                    and self._prefetch_requested_query
                    and pending_generation > self._prefetch_completed_generation
                    and pending_generation not in self._prefetch_inflight
                    and len(self._prefetch_threads) < _MAX_PREFETCH_WORKERS
                ):
                    self._start_prefetch_worker_locked()
                self._prefetch_condition.notify_all()

    def _join_prefetch_workers(self, timeout: float) -> None:
        with self._prefetch_condition:
            threads = set(self._prefetch_threads)
            if self._prefetch_thread is not None:
                threads.add(self._prefetch_thread)
        deadline = time.monotonic() + timeout
        for thread in threads:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            if thread.is_alive():
                thread.join(timeout=remaining)

    def _invalidate_prefetch(self, timeout: float) -> None:
        with self._prefetch_condition:
            self._prefetch_generation += 1
            self._prefetch_requested_query = ""
            self._prefetch_requested_generation = self._prefetch_generation
            self._prefetch_requested_session_id = ""
            self._prefetch_result = ""
            self._prefetch_result_query = ""
            self._prefetch_result_generation = 0
            self._prefetch_result_session_id = ""
            self._prefetch_completed_query = ""
            self._prefetch_completed_generation = 0
            self._prefetch_completed_session_id = ""
            self._prefetch_post_turn_skip_query = ""
            self._prefetch_post_turn_skip_session_id = ""
            self._prefetch_inflight_query = ""
            self._prefetch_inflight_session_id = ""
            self._prefetch_inflight_generation = 0
            self._prefetch_condition.notify_all()
        self._join_prefetch_workers(timeout)
        with self._prefetch_condition:
            self._prefetch_result = ""
            self._prefetch_result_query = ""
            self._prefetch_result_generation = 0
            self._prefetch_result_session_id = ""
            self._prefetch_condition.notify_all()

    def system_prompt_block(self) -> str:
        if self._memory_mode == "context":
            return (
                f"# Hindsight Memory\n"
                f"Active (context mode). Bank: {self._bank_id}, budget: {self._budget}.\n"
                f"Relevant memories are automatically injected into context."
            )
        if self._memory_mode == "tools":
            return (
                f"# Hindsight Memory\n"
                f"Active (tools mode). Bank: {self._bank_id}, budget: {self._budget}.\n"
                f"Use hindsight_recall to search, hindsight_reflect for synthesis, "
                f"hindsight_retain to store facts."
            )
        return (
            f"# Hindsight Memory\n"
            f"Active. Bank: {self._bank_id}, budget: {self._budget}.\n"
            f"Relevant memories are automatically injected into context. "
            f"Use hindsight_recall to search, hindsight_reflect for synthesis, "
            f"hindsight_retain to store facts."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if not self._prefetch_enabled():
            return ""
        normalized_query = self._normalize_prefetch_query(query)
        if not normalized_query:
            return ""
        effective_session_id = str(session_id or self._session_id or "").strip()
        if not self._automatic_scope_context_available(
            effective_session_id, "prefetch"
        ):
            return ""
        generation = self._request_prefetch(
            normalized_query,
            effective_session_id,
            post_turn=False,
        )
        if generation is None:
            return ""

        deadline = time.monotonic() + _PREFETCH_WAIT_SECONDS
        with self._prefetch_condition:
            while True:
                if (
                    self._prefetch_completed_generation == generation
                    and self._prefetch_completed_query == normalized_query
                    and self._prefetch_completed_session_id == effective_session_id
                ):
                    result = ""
                    if (
                        self._prefetch_result_generation == generation
                        and self._prefetch_result_query == normalized_query
                        and self._prefetch_result_session_id == effective_session_id
                    ):
                        result = self._prefetch_result
                        self._prefetch_result = ""
                        self._prefetch_result_query = ""
                        self._prefetch_result_generation = 0
                        self._prefetch_result_session_id = ""
                    self._prefetch_post_turn_skip_query = normalized_query
                    self._prefetch_post_turn_skip_session_id = effective_session_id
                    break
                if self._prefetch_generation != generation:
                    logger.debug(
                        "Prefetch: query superseded before completion"
                    )
                    return ""
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    logger.debug(
                        "Prefetch: current query timed out after %.1fs",
                        _PREFETCH_WAIT_SECONDS,
                    )
                    return ""
                self._prefetch_condition.wait(timeout=remaining)

        if not result:
            logger.debug("Prefetch: no results available for current query")
            return ""
        logger.debug("Prefetch: returning %d chars of context", len(result))
        header = self._recall_prompt_preamble or (
            "# Hindsight Memory (persistent cross-session context)\n"
            "Use this to answer questions about the user and prior sessions. "
            "Do not call tools to look up information that is already present here."
        )
        return f"{header}\n\n{result}"

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        if not self._prefetch_enabled():
            return
        normalized_query = self._normalize_prefetch_query(query)
        if not normalized_query:
            return
        effective_session_id = str(session_id or self._session_id or "").strip()
        if not self._automatic_scope_context_available(
            effective_session_id, "prefetch"
        ):
            return
        self._request_prefetch(
            normalized_query,
            effective_session_id,
            post_turn=True,
        )

    def _build_turn_messages(self, user_content: str, assistant_content: str) -> List[Dict[str, str]]:
        now = datetime.now(timezone.utc).isoformat()
        return [
            {
                "role": "user",
                "content": f"{self._retain_user_prefix}: {user_content}",
                "timestamp": now,
            },
            {
                "role": "assistant",
                "content": f"{self._retain_assistant_prefix}: {assistant_content}",
                "timestamp": now,
            },
        ]

    def _build_metadata(self, *, message_count: int, turn_index: int) -> Dict[str, str]:
        metadata: Dict[str, str] = {
            "retained_at": _utc_timestamp(),
            "message_count": str(message_count),
            "turn_index": str(turn_index),
        }
        if self._retain_source:
            metadata["source"] = self._retain_source
        if self._session_id:
            metadata["session_id"] = self._session_id
        if self._platform:
            metadata["platform"] = self._platform
        if self._user_id:
            metadata["user_id"] = self._user_id
        if self._user_name:
            metadata["user_name"] = self._user_name
        if self._chat_id:
            metadata["chat_id"] = self._chat_id
        if self._chat_name:
            metadata["chat_name"] = self._chat_name
        if self._chat_type:
            metadata["chat_type"] = self._chat_type
        if self._thread_id:
            metadata["thread_id"] = self._thread_id
        if self._agent_identity:
            metadata["agent_identity"] = self._agent_identity
        return metadata

    def _build_retain_kwargs(
        self,
        content: str,
        *,
        context: str | None = None,
        document_id: str | None = None,
        metadata: Dict[str, str] | None = None,
        tags: List[str] | None = None,
        retain_async: bool | None = None,
        scope_session_id: str | None = None,
    ) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "bank_id": self._bank_id,
            "content": content,
            "metadata": metadata or self._build_metadata(message_count=1, turn_index=self._turn_index),
        }
        if context is not None:
            kwargs["context"] = context
        if document_id:
            kwargs["document_id"] = document_id
        if retain_async is not None:
            kwargs["retain_async"] = retain_async
        merged_tags = _normalize_retain_tags(self._retain_tags)
        for tag in _normalize_retain_tags(tags):
            if tag not in merged_tags:
                merged_tags.append(tag)
        effective_session_id = (
            self._session_id
            if scope_session_id is None
            else str(scope_session_id or "").strip()
        )
        for tag in self._scope_tags_for_session(effective_session_id):
            if tag not in merged_tags:
                merged_tags.append(tag)
        if merged_tags:
            kwargs["tags"] = merged_tags
        if self._observation_scopes:
            kwargs["observation_scopes"] = self._observation_scopes
        return kwargs

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        """Enqueue the current turn for immutable, delta-only retention."""
        with self._retain_lifecycle_lock:
            if not self._auto_retain:
                logger.debug("sync_turn: skipped (auto_retain disabled)")
                return
            if self._shutting_down.is_set():
                logger.debug("sync_turn: skipped (shutting down)")
                return

            requested_session_id = str(session_id or "").strip()
            if requested_session_id and requested_session_id != self._session_id:
                self.on_session_switch(requested_session_id)

            state = self._retain_state
            if state is None:
                logger.debug("sync_turn: skipped (provider not initialized)")
                return
            if not self._automatic_scope_context_available(
                state.session_id, "retain"
            ):
                return

            turn = json.dumps(
                self._build_turn_messages(user_content, assistant_content),
                ensure_ascii=False,
            )
            state.turns.append(turn)
            self._turn_counter = len(state.turns)
            self._turn_index = self._turn_counter

            if self._turn_counter % self._retain_every_n_turns != 0:
                logger.debug(
                    "sync_turn: buffered turn %d (will retain at turn %d)",
                    self._turn_counter,
                    self._turn_counter
                    + (self._retain_every_n_turns - self._turn_counter % self._retain_every_n_turns),
                )
                return

            target_turn_count = self._turn_counter
            logger.debug(
                "sync_turn: queued immutable retain through turn %d (%d uncommitted)",
                target_turn_count,
                target_turn_count - state.committed_turn_count,
            )
            self._enqueue_automatic_retain(state, target_turn_count)

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        if self._memory_mode == "context":
            return []
        return [RETAIN_SCHEMA, RECALL_SCHEMA, REFLECT_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: dict, **kwargs) -> str:
        if tool_name == "hindsight_retain":
            content = args.get("content", "")
            if not content:
                return tool_error("Missing required parameter: content")
            context = args.get("context")
            try:
                item = self._build_retain_kwargs(
                    content,
                    context=context,
                    tags=args.get("tags"),
                )
                # aretain_batch takes bank_id/retain_async as call args, not item keys.
                item.pop("bank_id", None)
                item.pop("retain_async", None)
                logger.debug("Tool hindsight_retain: bank=%s, content_len=%d, context=%s",
                             self._bank_id, len(content), context)
                self._run_hindsight_operation(
                    lambda client: client.aretain_batch(
                        bank_id=self._bank_id, items=[item], retain_async=self._retain_async,
                    )
                )
                logger.debug("Tool hindsight_retain: success")
                return json.dumps({"result": "Memory stored successfully."})
            except Exception as e:
                logger.warning("hindsight_retain failed: %s", e, exc_info=True)
                return tool_error(f"Failed to store memory: {e}")

        elif tool_name == "hindsight_recall":
            query = args.get("query", "")
            if not query:
                return tool_error("Missing required parameter: query")
            try:
                recall_kwargs = self._build_recall_kwargs(
                    query,
                    self._session_id,
                )
                client = self._client if self._client is not None else self._get_client()
                support_error = self._recall_min_scores_support_error(client)
                if support_error:
                    return tool_error(support_error)
                logger.debug(
                    "Tool hindsight_recall: bank=%s, query_len=%d, budget=%s",
                    self._bank_id,
                    len(query),
                    self._budget,
                )
                resp = self._run_hindsight_operation(
                    lambda operation_client: operation_client.arecall(
                        **recall_kwargs
                    )
                )
                num_results = len(resp.results) if resp.results else 0
                logger.debug("Tool hindsight_recall: %d results", num_results)
                if not resp.results:
                    return json.dumps({"result": "No relevant memories found."})
                lines = [f"{i}. {r.text}" for i, r in enumerate(resp.results, 1)]
                return json.dumps({"result": "\n".join(lines)})
            except Exception as e:
                logger.warning("hindsight_recall failed: %s", e, exc_info=True)
                return tool_error(f"Failed to search memory: {e}")

        elif tool_name == "hindsight_reflect":
            query = args.get("query", "")
            if not query:
                return tool_error("Missing required parameter: query")
            try:
                reflect_kwargs: dict[str, Any] = {
                    "bank_id": self._bank_id,
                    "query": query,
                    "budget": self._budget,
                }
                reflect_kwargs.update(
                    self._query_tag_kwargs(
                        self._session_id,
                        include_legacy_without_scope=False,
                    )
                )
                if self._recall_min_scores:
                    logger.debug(
                        "Tool hindsight_reflect: recall_min_scores applies to "
                        "recall only; reflection remains available unfiltered"
                    )
                logger.debug(
                    "Tool hindsight_reflect: bank=%s, query_len=%d, budget=%s",
                    self._bank_id,
                    len(query),
                    self._budget,
                )
                resp = self._run_hindsight_operation(
                    lambda client: client.areflect(**reflect_kwargs)
                )
                logger.debug("Tool hindsight_reflect: response_len=%d", len(resp.text or ""))
                return json.dumps({"result": resp.text or "No relevant memories found."})
            except Exception as e:
                logger.warning("hindsight_reflect failed: %s", e, exc_info=True)
                return tool_error(f"Failed to reflect: {e}")

        return tool_error(f"Unknown tool: {tool_name}")

    def on_session_switch(
        self,
        new_session_id: str,
        *,
        parent_session_id: str = "",
        reset: bool = False,
        **kwargs,
    ) -> None:
        """Flush the old uncommitted tail, then rotate session state.

        The flush uses the same single-writer queue as regular retains. Its
        closure owns the old session state, so queued work cannot mutate the
        freshly rotated session's watermark or identifiers.
        """
        new_id = str(new_session_id or "").strip()
        if not new_id:
            return

        with self._retain_lifecycle_lock:
            old_state = self._retain_state
            if old_state is not None:
                old_target = len(old_state.turns)
                if (
                    old_target > old_state.committed_turn_count
                    or old_state.failed_batch is not None
                ) and not self._shutting_down.is_set():
                    self._enqueue_automatic_retain(old_state, old_target)

            self._invalidate_prefetch(_PREFETCH_WAIT_SECONDS)

            next_parent_session_id = self._parent_session_id
            if parent_session_id:
                next_parent_session_id = str(parent_session_id).strip()
            self._start_retain_session(new_id, next_parent_session_id)
            logger.debug(
                "Hindsight on_session_switch: new_session=%s parent=%s reset=%s doc=%s",
                self._session_id,
                self._parent_session_id,
                reset,
                self._document_id,
            )

    def shutdown(self) -> None:
        logger.debug("Hindsight shutdown: stopping writer + waiting for background threads")
        with self._retain_lifecycle_lock:
            if self._shutting_down.is_set():
                return

            state = self._retain_state
            if state is not None:
                target_turn_count = len(state.turns)
                if (
                    target_turn_count > state.committed_turn_count
                    or state.failed_batch is not None
                ):
                    try:
                        self._enqueue_automatic_retain(state, target_turn_count)
                    except Exception as exc:
                        logger.warning(
                            "Hindsight shutdown could not queue the final retain: %s",
                            exc,
                        )

            # The tail is now queued behind all prior retains. Only after that
            # immutable snapshot is accepted do new sync_turn calls get dropped.
            self._shutting_down.set()
            writer = self._writer_thread

        # Drain the writer: it will finish in-flight work, then exit on
        # the FIFO sentinel. Bounded join keeps shutdown predictable even if
        # the daemon is wedged.
        if writer is not None and writer.is_alive():
            try:
                self._retain_queue.put(_WRITER_SENTINEL)
            except Exception:
                pass
            writer.join(timeout=10.0)
            if writer.is_alive():
                logger.warning(
                    "Hindsight writer did not stop within 10s; "
                    "abandoning %d pending retain(s)",
                    self._retain_queue.qsize(),
                )
        self._invalidate_prefetch(5.0)
        if self._client is not None:
            try:
                if self._mode == "local_embedded":
                    # HindsightEmbedded.close() delegates to its sync client.close().
                    # When Hermes created/used that client on the shared async loop,
                    # closing it from this thread can raise "attached to a different
                    # loop" before aiohttp releases the session. Close the embedded
                    # inner async client on the shared loop first, then let the
                    # wrapper clean up daemon/UI bookkeeping.
                    inner_client = getattr(self._client, "_client", None)
                    if inner_client is not None and hasattr(inner_client, "aclose"):
                        _run_sync(inner_client.aclose())
                        try:
                            self._client._client = None
                        except Exception:
                            pass
                    try:
                        self._client.close()
                    except RuntimeError:
                        pass
                else:
                    self._run_sync(self._client.aclose())
            except Exception:
                pass
            self._client = None
        # The module-global background event loop (_loop / _loop_thread)
        # is intentionally NOT stopped here. It is shared across every
        # HindsightMemoryProvider instance in the process — the plugin
        # loader creates a new provider per AIAgent, and the gateway
        # creates one AIAgent per concurrent chat session. Stopping the
        # loop from one provider's shutdown() strands the aiohttp
        # ClientSession + TCPConnector owned by every sibling provider
        # on a dead loop, which surfaces as the "Unclosed client session"
        # / "Unclosed connector" warnings reported in #11923. The loop
        # runs on a daemon thread and is reclaimed on process exit;
        # per-session cleanup happens via self._client.aclose() above.


def register(ctx) -> None:
    """Register Hindsight as a memory provider plugin."""
    ctx.register_memory_provider(HindsightMemoryProvider())
