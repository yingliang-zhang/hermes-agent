"""Slice 2B contracts for durable local Desktop session rollover commits."""

from __future__ import annotations

import contextlib
import queue
import json
import sqlite3
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from agent.context_compressor import SUMMARY_PREFIX
from hermes_cli.active_sessions import ActiveSessionLease
from hermes_state import SessionDB
from tui_gateway import server

_REAL_ROLLOVER_NOTIFICATION_BLOCKERS = server._rollover_notification_blockers


class _Agent:
    def __init__(
        self,
        session_id: str,
        db: SessionDB,
        *,
        compression_count: int,
    ) -> None:
        self.session_id = session_id
        self._session_db = db
        self.model = "openai/gpt-5.6"
        self.provider = "openai-codex"
        self.base_url = "https://runtime.invalid/v1"
        self.api_key = "runtime-secret"
        self.api_mode = "codex_responses"
        self.acp_command = ""
        self.acp_args = []
        self._credential_pool = object()
        self.reasoning_config = {"enabled": True, "effort": "high"}
        self.service_tier = "priority"
        self.request_overrides = {"extra_body": {"route": "pinned"}}
        self.providers_allowed = ["provider-a"]
        self.providers_ignored = ["provider-b"]
        self.providers_order = ["provider-a"]
        self.provider_sort = "throughput"
        self.provider_require_parameters = True
        self.provider_data_collection = "deny"
        self.enabled_toolsets = ["terminal", "file"]
        self.disabled_toolsets = ["browser"]
        self.platform = "tui"
        self.context_compressor = types.SimpleNamespace(
            compression_count=compression_count
        )
        self._pending_steer = None
        self._pending_steer_lock = threading.Lock()
        self.background_review_callback = lambda _message: None
        self.memory_notifications = "verbose"
        self._end_session_on_close = True
        self.closed = False
        self.memory_shutdown_calls = []
        self.lifecycle = []

    def shutdown_memory_provider(self, messages=None) -> None:
        self.memory_shutdown_calls.append(messages)
        self.lifecycle.append("memory_shutdown")

    def close(self) -> None:
        self.lifecycle.append("close")
        self.closed = True


class _Worker:
    def __init__(self, session_key: str, model: str, profile_home=None) -> None:
        self.session_key = session_key
        self.model = model
        self.profile_home = profile_home
        self.closed = False

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def db(tmp_path: Path) -> SessionDB:
    value = SessionDB(tmp_path / "state.db")
    yield value
    value.close()


def _seed_parent(db: SessionDB) -> int:
    db.create_session(
        "predecessor",
        source="desktop",
        model="openai/gpt-5.6",
        model_config={
            "model": "openai/gpt-5.6",
            "provider": "openai-codex",
            "base_url": "https://runtime.invalid/v1",
            "api_mode": "codex_responses",
            "reasoning_config": {"enabled": True, "effort": "high"},
            "service_tier": "priority",
        },
        system_prompt="stable predecessor prompt",
        cwd="/workspace/project",
        profile_name="coding",
    )
    db.update_session_cwd(
        "predecessor",
        "/workspace/project",
        git_branch="feature/rollover",
        git_repo_root="/workspace/project",
    )
    db.set_session_title("predecessor", "Durable work")
    db.append_message(
        "predecessor",
        role="user",
        content=(
            f"{SUMMARY_PREFIX}\n"
            "## Historical Task Snapshot\n"
            "Keep the accepted storage and offer contracts."
        ),
    )
    db.append_message(
        "predecessor", role="assistant", content="Compaction acknowledged."
    )
    db.append_message(
        "predecessor", role="user", content="Implement the safe runtime swap."
    )
    return db.append_message(
        "predecessor",
        role="assistant",
        content="The exact final assistant response.",
        finish_reason="stop",
    )


def _history() -> list[dict]:
    return [
        {
            "role": "user",
            "content": (
                f"{SUMMARY_PREFIX}\n"
                "## Historical Task Snapshot\n"
                "Keep the accepted storage and offer contracts."
            ),
        },
        {"role": "assistant", "content": "Compaction acknowledged."},
        {"role": "user", "content": "Implement the safe runtime swap."},
        {"role": "assistant", "content": "The exact final assistant response."},
    ]


def _session(db: SessionDB) -> dict:
    old_agent = _Agent("predecessor", db, compression_count=2)
    old_worker = _Worker("predecessor", old_agent.model)
    lease = ActiveSessionLease(
        lease_id="lease-1",
        session_id="predecessor",
        surface="desktop",
        enabled=False,
    )
    return {
        "agent": old_agent,
        "agent_error": None,
        "agent_ready": threading.Event(),
        "active_session_lease": lease,
        "attached_images": [{"name": "draft.png"}],
        "close_on_disconnect": False,
        "cols": 100,
        "created_at": 1.0,
        "cwd": "/workspace/project",
        "edit_snapshots": {"draft": {"content": "not migrated"}},
        "explicit_cwd": True,
        "history": _history(),
        "history_lock": threading.Lock(),
        "history_version": 7,
        "image_counter": 4,
        "inflight_turn": None,
        "last_active": 2.0,
        "model_override": {"model": old_agent.model, "provider": old_agent.provider},
        "create_reasoning_override": dict(old_agent.reasoning_config),
        "create_service_tier_override": "priority",
        "parent_session_id": None,
        "pending_title": None,
        "profile_home": "/profiles/coding",
        "queued_prompt": None,
        "queued_prompts": [],
        "running": False,
        "session_key": "predecessor",
        "show_reasoning": True,
        "source": "desktop",
        "slash_worker": old_worker,
        "tool_progress_mode": "verbose",
        "tool_started_at": {},
        "transport": None,
        "turn_generation": 3,
        "turn_origin": None,
        "turn_reservation_token": None,
        "turn_state_revision": 9,
        "turn_state_running": False,
        "deferred_notification_texts": [],
        "deferred_notification_event_ids": set(),
        "defer_notifications_until_user": False,
        "_background_prompt_task_ids": set(),
        "_background_prompt_owned_task_ids": {"finished-hidden-task"},
        "rollover_local_capable": True,
        "rollover_consumed_compression_count": 1,
        "_rollover_offer": None,
        "_rollover_runtime_initialized": True,
    }


def _facts(final_message_id: int, **overrides) -> dict:
    facts = {
        "status": "complete",
        "final_content": "The exact final assistant response.",
        "final_message_id": final_message_id,
        "history_adopted": True,
        "settled_history_version": 7,
        "settled_turn_generation": 3,
        "persistence_error": False,
        "goal_followup": None,
        "claimed_notification_ids": set(),
    }
    facts.update(overrides)
    return facts


@pytest.fixture
def rollover_env(db: SessionDB, monkeypatch: pytest.MonkeyPatch):
    final_id = _seed_parent(db)
    session = _session(db)
    session["agent_ready"].set()
    events: list[tuple[str, str, dict]] = []
    build_calls: list[dict] = []
    cwd_registrations: list[str] = []
    context_calls: list[tuple[str, str | None, str]] = []
    approval_registered: list[str] = []
    approval_unregistered: list[str] = []

    monkeypatch.setattr(server, "_sessions", {"runtime-1": session})
    monkeypatch.setattr(server, "_session_rollover_enabled", lambda: True)
    monkeypatch.setattr(
        server, "_session_db", lambda _session: contextlib.nullcontext(db)
    )
    monkeypatch.setattr(server, "_rollover_bridge_blockers", lambda *_a: ())
    monkeypatch.setattr(server, "_rollover_process_blockers", lambda *_a: ())
    monkeypatch.setattr(server, "_rollover_delegation_blockers", lambda *_a: ())
    monkeypatch.setattr(server, "_rollover_goal_blockers", lambda *_a: ())
    monkeypatch.setattr(server, "_rollover_notification_blockers", lambda *_a: ())
    monkeypatch.setattr(server, "_emit", lambda *args: events.append(args))
    monkeypatch.setattr(server, "_new_session_key", lambda: "successor-1")
    monkeypatch.setattr(server, "_SlashWorker", _Worker)
    def set_session_context(
        session_key: str,
        cwd: str | None = None,
        *,
        ui_session_id: str = "",
    ) -> list[str]:
        context_calls.append((session_key, cwd, ui_session_id))
        return ["ctx"]

    monkeypatch.setattr(server, "_set_session_context", set_session_context)
    monkeypatch.setattr(server, "_clear_session_context", lambda _tokens: None)
    monkeypatch.setattr(server, "set_hermes_home_override", lambda _home: "home")
    monkeypatch.setattr(server, "reset_hermes_home_override", lambda _token: None)
    monkeypatch.setattr(server, "_wire_callbacks", lambda _sid: None)
    monkeypatch.setattr(server, "_schedule_mcp_late_refresh", lambda *_a: None)
    monkeypatch.setattr(server, "_notify_session_boundary", lambda *_a: None)
    monkeypatch.setattr(server, "_load_memory_notifications", lambda: "verbose")

    def register_cwd(current: dict) -> None:
        cwd_registrations.append(current["session_key"])

    monkeypatch.setattr(server, "_register_session_cwd", register_cwd)
    def session_info(_agent, current: dict, **_kwargs) -> dict:
        turn_state = server._turn_state_snapshot_locked(current)
        return {
            "stored_session_id": current["session_key"],
            "cwd": current["cwd"],
            "title": (db.get_session(current["session_key"]) or {}).get("title", ""),
            "running": turn_state["running"],
            "turn_started_at": turn_state["turn_started_at"],
            "turn_generation": turn_state["turn_generation"],
            "turn_state_revision": turn_state["turn_state_revision"],
        }

    monkeypatch.setattr(server, "_session_info", session_info)

    import tools.approval as approval

    monkeypatch.setattr(
        approval,
        "register_gateway_notify",
        lambda key, _cb: approval_registered.append(key),
    )
    monkeypatch.setattr(
        approval,
        "unregister_gateway_notify",
        lambda key: approval_unregistered.append(key),
    )
    monkeypatch.setattr(approval, "load_permanent_allowlist", lambda: None)

    def make_agent(sid: str, key: str, **kwargs):
        assert sid == "runtime-1"
        assert key == "successor-1"
        assert db.get_session("successor-1") is None
        assert context_calls == [
            ("successor-1", "/workspace/project", "runtime-1")
        ]
        build_calls.append(kwargs)
        return _Agent("successor-1", db, compression_count=0)

    monkeypatch.setattr(server, "_make_agent", make_agent)

    allowed = server._maybe_emit_session_rollover_offer(
        "runtime-1", session, **_facts(final_id)
    )
    assert allowed.allowed is True
    token = session["_rollover_offer"].token
    events.clear()

    return types.SimpleNamespace(
        db=db,
        final_id=final_id,
        session=session,
        token=token,
        events=events,
        build_calls=build_calls,
        context_calls=context_calls,
        cwd_registrations=cwd_registrations,
        approval_registered=approval_registered,
        approval_unregistered=approval_unregistered,
    )


def _commit(token: object, *, runtime_id: str = "runtime-1") -> dict:
    return server.handle_request(
        {
            "id": "commit",
            "method": "session.rollover.commit",
            "params": {"session_id": runtime_id, "token": token},
        }
    )


def _child_count(db: SessionDB) -> int:
    return int(
        db._conn.execute(
            "SELECT COUNT(*) FROM sessions WHERE parent_session_id = 'predecessor'"
        ).fetchone()[0]
    )


def _assert_parent_unchanged(env) -> None:
    parent = env.db.get_session("predecessor")
    assert parent is not None
    assert parent["ended_at"] is None
    assert parent["end_reason"] is None
    assert env.db.get_session("successor-1") is None
    assert _child_count(env.db) == 0
    assert env.session["session_key"] == "predecessor"
    assert env.session["agent"].session_id == "predecessor"
    assert env.session["active_session_lease"].session_id == "predecessor"
    assert not any(event == "session.rollover.complete" for event, *_ in env.events)


def test_commit_is_slow_and_default_off_after_offer(rollover_env, monkeypatch):
    assert "session.rollover.commit" in server._LONG_HANDLERS
    monkeypatch.setattr(server, "_session_rollover_enabled", lambda: False)

    response = _commit(rollover_env.token)

    assert "error" in response
    _assert_parent_unchanged(rollover_env)
    assert rollover_env.build_calls == []


@pytest.mark.parametrize(
    "case",
    ["wrong", "malformed", "cross-runtime", "predecessor-mismatch", "superseded"],
)
def test_invalid_or_stale_offer_fails_without_mutation_or_event(
    rollover_env, monkeypatch, case
):
    env = rollover_env
    runtime_id = "runtime-1"
    token: object = env.token
    if case == "wrong":
        token = "not-the-current-token"
    elif case == "malformed":
        token = [env.token]
    elif case == "cross-runtime":
        other = _session(env.db)
        other["session_key"] = "other-predecessor"
        other["agent"].session_id = "other-predecessor"
        other["_rollover_offer"] = env.session["_rollover_offer"]
        server._sessions["runtime-2"] = other
        runtime_id = "runtime-2"
    elif case == "predecessor-mismatch":
        env.session["session_key"] = "other-predecessor"
    else:
        old_token = token
        env.session["history"].extend(
            [
                {"role": "user", "content": "A later question."},
                {"role": "assistant", "content": "A later answer."},
            ]
        )
        env.session["history_version"] = 8
        env.session["turn_generation"] = 4
        env.db.append_message("predecessor", role="user", content="A later question.")
        later_id = env.db.append_message(
            "predecessor", role="assistant", content="A later answer."
        )
        result = server._maybe_emit_session_rollover_offer(
            "runtime-1",
            env.session,
            **_facts(
                later_id,
                final_content="A later answer.",
                settled_history_version=8,
                settled_turn_generation=4,
            ),
        )
        assert result.allowed is True
        env.events.clear()
        token = old_token

    response = _commit(token, runtime_id=runtime_id)

    assert "error" in response
    assert env.db.get_session("successor-1") is None
    assert _child_count(env.db) == 0
    assert not any(event == "session.rollover.complete" for event, *_ in env.events)
    assert env.build_calls == []


@pytest.mark.parametrize(
    "change",
    ["running", "history", "memory-tail", "db-tail", "compression"],
)
def test_post_offer_proof_change_aborts_commit(rollover_env, change):
    env = rollover_env
    if change == "running":
        env.session["running"] = True
    elif change == "history":
        env.session["history_version"] += 1
    elif change == "memory-tail":
        env.session["history"][-1]["content"] = "Changed in memory."
    elif change == "db-tail":
        env.db.append_message("predecessor", role="user", content="Late durable input.")
    else:
        env.session["agent"].context_compressor.compression_count += 1

    response = _commit(env.token)

    assert "error" in response
    _assert_parent_unchanged(env)


def test_commit_revalidation_rejects_unpublished_process_completion(
    rollover_env, monkeypatch
):
    env = rollover_env
    monkeypatch.setattr(
        server,
        "_rollover_process_blockers",
        lambda *_args: ("background_completion_publication_pending",),
    )

    response = _commit(env.token)

    assert "error" in response
    assert "background_completion_publication_pending" in response["error"]["message"]
    _assert_parent_unchanged(env)


def test_poller_delivery_handoff_blocks_offer_and_commit_revalidation(
    rollover_env, monkeypatch
):
    from tools import async_delegation
    from tools.process_registry import process_registry

    env = rollover_env
    isolated_queue = queue.Queue()
    isolated_queue.put(
        {
            "type": "completion",
            "session_id": "proc-handoff",
            "session_key": "predecessor",
            "command": "build",
            "exit_code": 0,
            "output": "done",
        }
    )
    monkeypatch.setattr(process_registry, "completion_queue", isolated_queue)
    monkeypatch.setattr(process_registry, "is_completion_consumed", lambda _sid: False)
    monkeypatch.setattr(
        async_delegation,
        "load_deferred_notifications",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        server,
        "_rollover_notification_blockers",
        _REAL_ROLLOVER_NOTIFICATION_BLOCKERS,
    )
    monkeypatch.setattr(server, "_notification_event_belongs_elsewhere", lambda *_a: False)

    dispatch_entered = threading.Event()
    release_dispatch = threading.Event()
    stop = threading.Event()

    def blocked_dispatch(*_args, **_kwargs):
        dispatch_entered.set()
        assert release_dispatch.wait(5)
        return "deferred"

    monkeypatch.setattr(server, "_dispatch_notification_turn", blocked_dispatch)
    poller = threading.Thread(
        target=server._notification_poller_loop,
        args=(stop, "runtime-1", env.session),
    )
    poller.start()
    try:
        assert dispatch_entered.wait(5)
        assert isolated_queue.empty()
        assert server._notification_delivery_handoff_active() is True

        offer_result = server._maybe_emit_session_rollover_offer(
            "runtime-1", env.session, **_facts(env.final_id)
        )
        assert offer_result.allowed is False
        assert offer_result.reasons == ("notification_delivery_inflight",)

        response = _commit(env.token)
        assert "error" in response
        assert "notification_delivery_inflight" in response["error"]["message"]
    finally:
        stop.set()
        release_dispatch.set()
        poller.join(5)

    assert not poller.is_alive()
    assert server._notification_delivery_handoff_active() is False
    _assert_parent_unchanged(env)


@pytest.mark.parametrize("failure_stage", ["agent", "worker"])
def test_fresh_runtime_build_failure_has_no_mutation(
    rollover_env, monkeypatch, failure_stage
):
    env = rollover_env
    candidate = _Agent("successor-1", env.db, compression_count=0)
    if failure_stage == "agent":
        monkeypatch.setattr(
            server,
            "_make_agent",
            lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("agent failed")),
        )
    else:
        monkeypatch.setattr(server, "_make_agent", lambda *_a, **_k: candidate)
        monkeypatch.setattr(
            server,
            "_SlashWorker",
            lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("worker failed")),
        )

    response = _commit(env.token)

    assert "error" in response
    _assert_parent_unchanged(env)
    if failure_stage == "worker":
        assert candidate.closed is True
        assert candidate.memory_shutdown_calls == [None]
        assert candidate.lifecycle == ["memory_shutdown", "close"]


def test_database_fault_restores_parent_runtime_and_active_lease(
    rollover_env, monkeypatch
):
    env = rollover_env
    transfer_targets: list[str] = []
    real_transfer = server._transfer_active_session_slot

    def record_transfer(sid, session, *, new_session_id):
        transfer_targets.append(new_session_id)
        return real_transfer(sid, session, new_session_id=new_session_id)

    monkeypatch.setattr(server, "_transfer_active_session_slot", record_transfer)
    env.db._conn.execute(
        """
        CREATE TEMP TRIGGER abort_commit_child
        BEFORE INSERT ON sessions
        WHEN NEW.parent_session_id = 'predecessor'
        BEGIN
            SELECT RAISE(ABORT, 'simulated commit failure');
        END
        """
    )

    response = _commit(env.token)

    assert "error" in response
    _assert_parent_unchanged(env)
    assert transfer_targets == ["successor-1", "predecessor"]
    assert env.approval_registered == ["successor-1"]
    assert env.approval_unregistered == ["successor-1"]


def test_precompute_failure_leaves_durable_lease_and_runtime_unchanged(
    rollover_env, monkeypatch
):
    env = rollover_env
    candidate = _Agent("successor-1", env.db, compression_count=0)
    transfer_targets: list[str] = []
    real_compression_count = server._compression_count

    monkeypatch.setattr(server, "_make_agent", lambda *_a, **_k: candidate)

    def fail_candidate_compression_count(agent):
        if agent is candidate:
            raise RuntimeError("precompute failed")
        return real_compression_count(agent)

    monkeypatch.setattr(server, "_compression_count", fail_candidate_compression_count)
    monkeypatch.setattr(
        server,
        "_transfer_active_session_slot",
        lambda _sid, _session, *, new_session_id: transfer_targets.append(
            new_session_id
        ),
    )

    response = _commit(env.token)

    assert "error" in response
    _assert_parent_unchanged(env)
    assert transfer_targets == []
    assert candidate.memory_shutdown_calls == [None]
    assert candidate.lifecycle == ["memory_shutdown", "close"]


def test_active_session_lease_transfer_failure_disposes_candidate_without_commit(
    rollover_env, monkeypatch
):
    env = rollover_env
    candidate = _Agent("successor-1", env.db, compression_count=0)
    transfer_targets: list[str] = []

    monkeypatch.setattr(server, "_make_agent", lambda *_a, **_k: candidate)

    def reject_transfer(_sid, _session, *, new_session_id):
        transfer_targets.append(new_session_id)
        return False

    monkeypatch.setattr(server, "_transfer_active_session_slot", reject_transfer)

    response = _commit(env.token)

    assert "error" in response
    _assert_parent_unchanged(env)
    assert transfer_targets == ["successor-1"]
    assert candidate.memory_shutdown_calls == [None]
    assert candidate.lifecycle == ["memory_shutdown", "close"]
    assert env.approval_registered == ["successor-1"]
    assert env.approval_unregistered == ["successor-1"]



def test_success_commits_bounded_child_and_swaps_same_runtime(rollover_env):
    env = rollover_env
    old_agent = env.session["agent"]
    old_worker = env.session["slash_worker"]
    env.db.set_meta("goal:predecessor", json.dumps({"status": "complete"}))
    old_notification_texts = env.session["deferred_notification_texts"]
    old_notification_ids = env.session["deferred_notification_event_ids"]
    old_queues = env.session["queued_prompts"]

    response = _commit(env.token)

    expected = {
        "runtime_id": "runtime-1",
        "predecessor_stored_id": "predecessor",
        "successor_stored_id": "successor-1",
        "token": env.token,
    }
    assert response["result"] == expected
    parent = env.db.get_session("predecessor")
    child = env.db.get_session("successor-1")
    assert parent is not None and parent["end_reason"] == "rollover"
    assert child is not None and child["parent_session_id"] == "predecessor"
    assert child["cwd"] == "/workspace/project"
    assert child["profile_name"] == "coding"
    assert child["git_branch"] == "feature/rollover"
    assert child["git_repo_root"] == "/workspace/project"
    assert child["title"] == "Durable work #2"
    assert json.loads(child["model_config"])["_rollover_from"] == "predecessor"
    durable_handoff = env.db.get_messages("successor-1")
    assert [message["role"] for message in durable_handoff] == ["user", "assistant"]
    assert ("Implement the safe runtime swap." in durable_handoff[0]["content"] and durable_handoff[1]["content"].endswith("The exact final assistant response."))

    assert server._sessions["runtime-1"] is env.session
    assert env.session["session_key"] == "successor-1"
    assert env.session["agent"].session_id == "successor-1"
    assert env.session["active_session_lease"].session_id == "successor-1"
    assert env.session["history"] == [
        {"role": "user", "content": durable_handoff[0]["content"]},
        {"role": "assistant", "content": durable_handoff[1]["content"]},
    ]
    assert env.session["history_version"] == 0
    assert env.session["turn_generation"] == 3
    assert env.session["turn_state_revision"] == 10
    assert server._turn_state_snapshot_locked(env.session)["turn_started_at"] is None
    assert env.session["agent"].context_compressor.compression_count == 0
    assert env.session["agent"]._end_session_on_close is True
    assert env.session["rollover_consumed_compression_count"] == 0
    assert env.session["_background_prompt_task_ids"] == set()
    assert env.session["_background_prompt_owned_task_ids"] == set()
    assert env.session["attached_images"] == []
    assert env.session["edit_snapshots"] == {}
    assert env.session["image_counter"] == 0
    assert env.session["queued_prompt"] is None
    assert env.session["queued_prompts"] == []
    assert env.session["queued_prompts"] is not old_queues
    assert env.session["deferred_notification_texts"] == []
    assert env.session["deferred_notification_texts"] is not old_notification_texts
    assert env.session["deferred_notification_event_ids"] == set()
    assert env.session["deferred_notification_event_ids"] is not old_notification_ids
    assert env.db.get_meta("goal:predecessor") is not None
    assert env.db.get_meta("goal:successor-1") is None

    assert old_agent.background_review_callback is None
    assert old_agent._end_session_on_close is False
    assert old_agent.closed is True
    assert old_agent.memory_shutdown_calls == [_history()]
    assert old_agent.lifecycle == ["memory_shutdown", "close"]
    assert old_worker.closed is True
    assert env.session["slash_worker"].session_key == "successor-1"
    assert env.session["agent"].memory_shutdown_calls == []
    assert env.session["agent"].closed is False
    assert env.approval_registered == ["successor-1"]
    assert env.approval_unregistered == ["predecessor"]
    assert env.cwd_registrations == ["successor-1"]
    assert env.context_calls == [
        ("successor-1", "/workspace/project", "runtime-1")
    ]

    assert env.build_calls and env.build_calls[0]["session_id"] == "successor-1"
    assert env.build_calls[0]["session_db"] is env.db
    assert env.build_calls[0]["platform_override"] == "desktop"
    runtime = env.build_calls[0]["runtime_snapshot"]
    assert runtime["model"] == old_agent.model
    assert runtime["provider"] == old_agent.provider
    assert runtime["base_url"] == old_agent.base_url
    assert runtime["api_key"] == old_agent.api_key
    assert runtime["api_mode"] == old_agent.api_mode
    assert runtime["reasoning_config"] == old_agent.reasoning_config
    assert runtime["service_tier"] == old_agent.service_tier
    assert runtime["request_overrides"] == old_agent.request_overrides

    assert [event for event, *_ in env.events] == [
        "session.rollover.complete",
        "session.info",
    ]
    assert env.events[0] == ("session.rollover.complete", "runtime-1", expected)
    assert env.events[1][1] == "runtime-1"
    assert env.events[1][2]["stored_session_id"] == "successor-1"
    assert env.events[1][2]["turn_generation"] == 3
    assert env.events[1][2]["turn_state_revision"] == 10



def test_success_clears_predecessor_task_env_after_retirement_before_events(
    rollover_env, monkeypatch
):
    import tools.terminal_tool as terminal_tool

    env = rollover_env
    old_agent = env.session["agent"]
    old_worker = env.session["slash_worker"]
    cleared: list[str] = []

    def clear_task_env_overrides(session_key: str) -> None:
        assert env.session["session_key"] == "successor-1"
        assert old_agent.closed is True
        assert old_worker.closed is True
        assert env.events == []
        cleared.append(session_key)

    monkeypatch.setattr(
        terminal_tool, "clear_task_env_overrides", clear_task_env_overrides
    )

    response = _commit(env.token)

    assert "error" not in response
    assert cleared == ["predecessor"]
    assert env.cwd_registrations == ["successor-1"]


def test_commit_failure_clears_only_successor_task_env(
    rollover_env, monkeypatch
):
    import tools.terminal_tool as terminal_tool

    env = rollover_env
    cleared: list[str] = []
    monkeypatch.setattr(
        terminal_tool,
        "clear_task_env_overrides",
        lambda session_key: cleared.append(session_key),
    )
    monkeypatch.setattr(
        env.db,
        "complete_session_rollover",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("commit failed")),
    )

    response = _commit(env.token)

    assert "error" in response
    _assert_parent_unchanged(env)
    assert cleared == ["successor-1"]
    assert env.cwd_registrations == ["successor-1"]


def test_release_failure_after_commit_keeps_successor_canonical(
    rollover_env, monkeypatch
):
    env = rollover_env

    class ReleaseFaultDB:
        def __init__(self, wrapped: SessionDB) -> None:
            self._wrapped = wrapped

        def __getattr__(self, name: str):
            return getattr(self._wrapped, name)

        def release_compression_lock(self, session_id: str, holder: str) -> None:
            raise RuntimeError("release failed")

    fault_db = ReleaseFaultDB(env.db)
    monkeypatch.setattr(
        server, "_session_db", lambda _session: contextlib.nullcontext(fault_db)
    )

    response = _commit(env.token)

    assert response["result"]["successor_stored_id"] == "successor-1"
    parent = env.db.get_session("predecessor")
    child = env.db.get_session("successor-1")
    assert parent is not None and parent["end_reason"] == "rollover"
    assert child is not None and child["parent_session_id"] == "predecessor"
    assert _child_count(env.db) == 1
    assert env.session["session_key"] == "successor-1"
    assert env.session["agent"].session_id == "successor-1"
    assert env.session["active_session_lease"].session_id == "successor-1"
    assert env.session["agent"].closed is False
    assert [event for event, *_ in env.events].count("session.rollover.complete") == 1
    assert [event for event, *_ in env.events].count("session.info") == 1


def test_success_event_observes_durable_and_live_swap_before_info(rollover_env, monkeypatch):
    env = rollover_env
    observations: list[str] = []

    def emit(event: str, sid: str, payload: dict) -> None:
        if event == "session.rollover.complete":
            assert env.db.get_session("successor-1") is not None
            assert env.db.get_session("predecessor")["end_reason"] == "rollover"
            assert server._sessions[sid]["session_key"] == "successor-1"
            assert server._sessions[sid]["agent"].session_id == "successor-1"
        observations.append(event)

    monkeypatch.setattr(server, "_emit", emit)

    response = _commit(env.token)

    assert "error" not in response
    assert observations == ["session.rollover.complete", "session.info"]

def test_rollover_revision_supersedes_delayed_predecessor_and_next_turn_is_monotonic(
    rollover_env,
):
    env = rollover_env
    delayed_predecessor_info = {
        "stored_session_id": "predecessor",
        "running": False,
        "turn_generation": env.session["turn_generation"],
        "turn_state_revision": env.session["turn_state_revision"],
    }

    response = _commit(env.token)

    assert response["result"]["runtime_id"] == "runtime-1"
    successor_info = env.events[-1][2]
    assert successor_info["stored_session_id"] == "successor-1"
    assert (
        successor_info["turn_generation"]
        == delayed_predecessor_info["turn_generation"]
        == 3
    )
    assert (
        successor_info["turn_state_revision"]
        > delayed_predecessor_info["turn_state_revision"]
    )
    assert successor_info["running"] is False
    assert successor_info["turn_started_at"] is None

    with env.session["history_lock"]:
        next_generation = server._set_turn_origin_locked(env.session, "user")
        next_turn = server._turn_state_snapshot_locked(env.session)

    assert next_generation == 4
    assert next_turn["turn_generation"] == 4
    assert next_turn["turn_state_revision"] == 11
    assert next_turn["running"] is True



def test_same_token_retry_returns_successor_without_duplicate_work_or_event(rollover_env):
    env = rollover_env
    first = _commit(env.token)
    first_agent = env.session["agent"]
    first_worker = env.session["slash_worker"]
    first_events = list(env.events)

    second = _commit(env.token)

    assert second["result"] == first["result"]
    assert env.session["agent"] is first_agent
    assert env.session["slash_worker"] is first_worker
    assert _child_count(env.db) == 1
    assert len(env.build_calls) == 1
    assert env.events == first_events


def test_two_concurrent_commits_create_one_child_and_one_swap(
    rollover_env, monkeypatch
):
    env = rollover_env
    builds = 0
    builds_lock = threading.Lock()

    def make_agent(sid: str, key: str, **kwargs):
        nonlocal builds
        with builds_lock:
            builds += 1
        assert env.db.get_session("successor-1") is None
        time.sleep(0.05)
        env.build_calls.append(kwargs)
        return _Agent(key, env.db, compression_count=0)

    monkeypatch.setattr(server, "_make_agent", make_agent)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(_commit, env.token) for _ in range(2)]
        responses = [future.result(timeout=5) for future in futures]

    assert responses[0]["result"] == responses[1]["result"]
    assert responses[0]["result"]["successor_stored_id"] == "successor-1"
    assert builds == 1
    assert _child_count(env.db) == 1
    assert [event for event, *_ in env.events].count("session.rollover.complete") == 1
    assert [event for event, *_ in env.events].count("session.info") == 1
