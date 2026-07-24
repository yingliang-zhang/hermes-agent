"""Slice 2A contracts for local Desktop rollover offers.

These tests deliberately stop at the offer boundary.  No durable successor or
runtime swap is part of this slice.
"""

from __future__ import annotations

import contextlib
import importlib
import queue
import threading
import time
import types
from copy import deepcopy

import pytest

from hermes_cli.config import DEFAULT_CONFIG
from tui_gateway import server


def _rollover():
    return importlib.import_module("tui_gateway.session_rollover")


class _ImmediateThread:
    def __init__(self, target=None, args=(), kwargs=None, **_ignored):
        self._target = target
        self._args = args
        self._kwargs = kwargs or {}

    def start(self):
        self._target(*self._args, **self._kwargs)

    def is_alive(self):
        return False


class _DB:
    def __init__(self, final_id=42, final_content="final answer", goal=None):
        self.final_id = final_id
        self.final_content = final_content
        self.goal = goal
        self.tail_reads = 0
        self.meta_reads = 0

    def get_latest_active_message_identity(self, session_id):
        assert session_id == "stored-1"
        self.tail_reads += 1
        return {
            "id": self.final_id,
            "role": "assistant",
            "content": self.final_content,
        }

    def get_meta(self, key):
        assert key == "goal:stored-1"
        self.meta_reads += 1
        return self.goal


class _Agent:
    def __init__(self, *, session_id="stored-1", compression_count=2):
        self.session_id = session_id
        self.model = "test-model"
        self.provider = "test-provider"
        self.base_url = ""
        self.api_key = ""
        self.service_tier = ""
        self._pending_steer = None
        self._pending_steer_lock = threading.Lock()
        self.context_compressor = types.SimpleNamespace(
            compression_count=compression_count
        )

    def clear_interrupt(self):
        return None

    def run_conversation(self, prompt, conversation_history=None, **_kwargs):
        history = list(conversation_history or [])
        history.extend(
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": "final answer"},
            ]
        )
        return {
            "completed": True,
            "final_response": "final answer",
            "messages": history,
            "turn_exit_reason": "text_response(finish_reason=stop)",
        }


def _session(**overrides):
    session = {
        "agent": _Agent(),
        "session_key": "stored-1",
        "source": "desktop",
        "rollover_local_capable": True,
        "rollover_consumed_compression_count": 1,
        "history": [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "final answer"},
        ],
        "history_lock": threading.Lock(),
        "history_version": 7,
        "running": False,
        "turn_generation": 3,
        "turn_state_running": False,
        "turn_reservation_token": None,
        "inflight_turn": None,
        "queued_prompt": None,
        "queued_prompts": [],
        "tool_started_at": {},
        "_background_prompt_task_ids": set(),
        "_background_prompt_owned_task_ids": set(),
        "deferred_notification_texts": [],
        "deferred_notification_event_ids": set(),
        "defer_notifications_until_user": False,
        "attached_images": [],
        "image_counter": 0,
        "cols": 80,
        "slash_worker": None,
        "show_reasoning": False,
        "tool_progress_mode": "all",
        "transport": None,
    }
    session.update(overrides)
    return session


def _facts(**overrides):
    facts = {
        "status": "complete",
        "final_content": "final answer",
        "final_message_id": 42,
        "history_adopted": True,
        "settled_history_version": 7,
        "settled_turn_generation": 3,
        "persistence_error": False,
        "goal_followup": None,
        "claimed_notification_ids": set(),
    }
    facts.update(overrides)
    return facts

def _all_true(snapshot_type):
    return {name: True for name in snapshot_type.__dataclass_fields__}


def _patch_clear_dependencies(monkeypatch, db=None):
    db = db or _DB()
    monkeypatch.setattr(
        server,
        "_session_db",
        lambda _session: contextlib.nullcontext(db),
    )
    monkeypatch.setattr(server, "_rollover_bridge_blockers", lambda *_a: ())
    monkeypatch.setattr(server, "_rollover_process_blockers", lambda *_a: ())
    monkeypatch.setattr(server, "_rollover_delegation_blockers", lambda *_a: ())
    monkeypatch.setattr(server, "_rollover_goal_blockers", lambda *_a: ())
    monkeypatch.setattr(server, "_rollover_notification_blockers", lambda *_a: ())
    monkeypatch.setattr(server, "_session_rollover_enabled", lambda: True)
    return db


class _CompletionQueue:
    def __init__(self, events=()):
        self.mutex = threading.Lock()
        self.queue = list(events)


def _patch_real_clear_dependencies(monkeypatch, db=None):
    import tools.approval as approval
    import tools.async_delegation as async_delegation
    from tools.process_registry import process_registry

    db = db or _DB()
    monkeypatch.setattr(
        server,
        "_session_db",
        lambda _session: contextlib.nullcontext(db),
    )
    monkeypatch.setattr(server, "_pending", {})
    monkeypatch.setattr(approval, "has_blocking_approval", lambda _key: False)
    monkeypatch.setattr(process_registry, "list_sessions", lambda **_kwargs: [])
    monkeypatch.setattr(process_registry, "completion_queue", _CompletionQueue())
    monkeypatch.setattr(async_delegation, "list_async_delegations", lambda: [])
    monkeypatch.setattr(
        async_delegation,
        "load_deferred_notifications",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(server, "_session_rollover_enabled", lambda: True)
    return db, {
        "approval": approval,
        "async_delegation": async_delegation,
        "process_registry": process_registry,
    }


def test_config_default_is_false_and_only_literal_true_enables(monkeypatch):
    assert DEFAULT_CONFIG["agent"]["session_rollover"]["enabled"] is False

    for value, expected in [
        (True, True),
        (False, False),
        (1, False),
        ("true", False),
        (None, False),
    ]:
        monkeypatch.setattr(
            server,
            "_load_cfg",
            lambda value=value: {"agent": {"session_rollover": {"enabled": value}}},
        )
        assert server._session_rollover_enabled() is expected


@pytest.mark.parametrize(
    ("source", "params", "expected"),
    [
        ("desktop", {"local_rollover_capable": True}, True),
        ("desktop", {"local_rollover_capable": False}, False),
        ("desktop", {}, False),
        ("desktop", {"local_rollover_capable": 1}, False),
        ("desktop", {"local_rollover_capable": "true"}, False),
        ("tui", {"local_rollover_capable": True}, False),
        ("Desktop", {"local_rollover_capable": True}, False),
    ],
)
def test_local_capability_parsing_fails_closed(source, params, expected):
    assert server._parse_rollover_local_capability(params, source) is expected


def test_session_state_defaults_not_capable_and_never_trusts_config():
    session = {}
    server._initialize_rollover_state(
        session,
        params={"local_rollover_capable": True},
        source="tui",
    )
    assert session["rollover_local_capable"] is False
    assert session["rollover_consumed_compression_count"] == 0
    assert session["_rollover_offer"] is None
    assert session["_background_prompt_task_ids"] == set()


@pytest.mark.parametrize(
    ("platform", "capability_params", "expected"),
    [
        ("desktop", {"local_rollover_capable": True}, True),
        ("desktop", {}, False),
        ("desktop", {"local_rollover_capable": 1}, False),
        ("desktop", {"local_rollover_capable": "true"}, False),
        ("tui", {"local_rollover_capable": True}, False),
    ],
)
def test_session_create_uses_resolved_source_for_rollover_capability(
    monkeypatch, platform, capability_params, expected
):
    monkeypatch.setattr(server, "_sessions", {})
    monkeypatch.setattr(server, "_new_session_key", lambda: "stored-created")
    monkeypatch.setattr(server, "_completion_cwd", lambda _params: ".")
    monkeypatch.setattr(server, "_enable_gateway_prompts", lambda: None)
    monkeypatch.setattr(server, "_profile_home", lambda _profile: None)
    monkeypatch.setattr(
        server, "_claim_active_session_slot", lambda *_args, **_kwargs: (None, None)
    )
    monkeypatch.setattr(server, "_load_show_reasoning", lambda: False)
    monkeypatch.setattr(server, "_load_tool_progress_mode", lambda: "all")
    monkeypatch.setattr(server, "_register_session_cwd", lambda _session: None)
    monkeypatch.setattr(
        server, "_restore_activated_profile_completions", lambda _session: None
    )
    monkeypatch.setattr(server, "_schedule_agent_build", lambda _sid: None)
    monkeypatch.setattr(server, "_schedule_session_cap_enforcement", lambda: None)
    monkeypatch.setattr(server, "_resolve_model", lambda: "test-model")
    if platform == "desktop":
        monkeypatch.setenv("HERMES_DESKTOP", "1")
        monkeypatch.delenv("HERMES_DESKTOP_TERMINAL", raising=False)
    else:
        monkeypatch.delenv("HERMES_DESKTOP", raising=False)
        monkeypatch.delenv("HERMES_DESKTOP_TERMINAL", raising=False)

    response = server.handle_request(
        {"id": "create", "method": "session.create", "params": capability_params}
    )

    assert "error" not in response
    created = server._sessions[response["result"]["session_id"]]
    assert created["source"] == platform
    assert created["rollover_local_capable"] is expected


@pytest.mark.parametrize(
    ("session_source", "capability_params", "expected"),
    [
        ("desktop", {"local_rollover_capable": True}, True),
        ("desktop", {}, False),
        ("desktop", {"local_rollover_capable": 1}, False),
        ("desktop", {"local_rollover_capable": "true"}, False),
        ("tui", {"source": "desktop", "local_rollover_capable": True}, False),
    ],
)
def test_session_resume_uses_live_session_source_for_rollover_capability(
    monkeypatch, session_source, capability_params, expected
):
    class _ResumeDB:
        def get_session(self, session_id):
            return {"id": session_id}

        def get_session_by_title(self, _title):
            return None

        def resolve_resume_session_id(self, session_id):
            return session_id

    session = _session(source=session_source, rollover_local_capable=False)
    monkeypatch.setattr(server, "_sessions", {"runtime-1": session})
    monkeypatch.setattr(server, "_get_db", lambda: _ResumeDB())
    monkeypatch.setattr(server, "_profile_home", lambda _profile: None)
    monkeypatch.setattr(server, "_profile_configured_cwd", lambda _home: "")
    monkeypatch.setattr(server, "_live_session_payload", lambda *_args, **_kwargs: {})
    params = {"session_id": "stored-1", **capability_params}

    response = server.handle_request(
        {"id": "resume", "method": "session.resume", "params": params}
    )

    assert "error" not in response
    assert session["rollover_local_capable"] is expected


@pytest.mark.parametrize(
    ("snapshot_name", "field", "code"),
    [
        ("EligibilitySnapshot", "config_enabled", "config_disabled"),
        ("EligibilitySnapshot", "source_desktop", "source_not_desktop"),
        ("EligibilitySnapshot", "local_capable", "local_capability_missing"),
        ("EligibilitySnapshot", "compute_host_clear", "compute_host_active"),
        ("EligibilitySnapshot", "turn_clean", "turn_not_clean"),
        ("EligibilitySnapshot", "history_adopted", "history_not_adopted"),
        ("EligibilitySnapshot", "persistence_clean", "persistence_cleanup_error"),
        ("QuiescenceSnapshot", "running_clear", "running"),
        ("QuiescenceSnapshot", "inflight_clear", "inflight_turn"),
        ("QuiescenceSnapshot", "reservation_clear", "turn_reservation"),
        ("QuiescenceSnapshot", "steer_clear", "pending_steer"),
        ("QuiescenceSnapshot", "queued_prompt_clear", "queued_prompt"),
        ("QuiescenceSnapshot", "queued_prompts_clear", "queued_prompts"),
        ("QuiescenceSnapshot", "tools_clear", "tool_running"),
        ("QuiescenceSnapshot", "background_tasks_clear", "background_prompt_running"),
        ("QuiescenceSnapshot", "bridge_clear", "blocking_input"),
        ("QuiescenceSnapshot", "approval_clear", "approval_pending"),
        ("QuiescenceSnapshot", "bridge_query_known", "approval_query_unknown"),
        ("QuiescenceSnapshot", "processes_clear", "background_process_running"),
        ("QuiescenceSnapshot", "process_query_known", "process_query_unknown"),
        ("QuiescenceSnapshot", "delegations_clear", "async_delegation_running"),
        ("QuiescenceSnapshot", "delegation_query_known", "delegation_query_unknown"),
        ("QuiescenceSnapshot", "goal_active_clear", "goal_active"),
        ("QuiescenceSnapshot", "goal_waiting_clear", "goal_waiting"),
        ("QuiescenceSnapshot", "goal_followup_clear", "goal_followup"),
        ("QuiescenceSnapshot", "goal_query_known", "goal_query_unknown"),
        ("QuiescenceSnapshot", "notification_texts_clear", "deferred_notification_texts"),
        ("QuiescenceSnapshot", "notification_ids_clear", "deferred_notification_ids"),
        ("QuiescenceSnapshot", "notification_adoption_clear", "deferred_notification_adoption"),
        ("QuiescenceSnapshot", "notification_latch_clear", "notification_deferral_latch"),
        ("QuiescenceSnapshot", "addressed_completion_clear", "addressed_completion_pending"),
        ("QuiescenceSnapshot", "notification_delivery_clear", "notification_delivery_inflight"),
        ("QuiescenceSnapshot", "notification_query_known", "notification_query_unknown"),
        ("QuiescenceSnapshot", "history_stable", "history_changed"),
        ("QuiescenceSnapshot", "history_tail_exact", "history_tail_mismatch"),
        ("QuiescenceSnapshot", "agent_identity_current", "agent_identity_mismatch"),
        ("QuiescenceSnapshot", "db_tail_exact", "db_tail_mismatch"),
        ("QuiescenceSnapshot", "db_query_known", "db_query_unknown"),
        ("QuiescenceSnapshot", "compression_boundary_new", "compression_boundary_missing"),
    ],
)
def test_each_rollover_proof_independently_fails_closed(snapshot_name, field, code):
    rollover = _rollover()
    snapshot_type = getattr(rollover, snapshot_name)
    values = _all_true(snapshot_type)
    values[field] = False
    snapshot = snapshot_type(**values)

    evaluator = (
        rollover.evaluate_eligibility
        if snapshot_name == "EligibilitySnapshot"
        else rollover.evaluate_quiescence
    )
    result = evaluator(snapshot)

    assert result.allowed is False
    assert result.reasons == (code,)


@pytest.mark.parametrize(
    "snapshot_name", ["EligibilitySnapshot", "QuiescenceSnapshot"]
)
def test_omitting_any_rollover_proof_is_a_construction_error(snapshot_name):
    rollover = _rollover()
    snapshot_type = getattr(rollover, snapshot_name)
    values = _all_true(snapshot_type)

    for omitted in values:
        incomplete = dict(values)
        incomplete.pop(omitted)
        with pytest.raises(TypeError):
            snapshot_type(**incomplete)


def test_eligibility_schema_missing_compute_host_proof_fails_closed():
    rollover = _rollover()
    values = _all_true(rollover.EligibilitySnapshot)
    values.pop("compute_host_clear")
    legacy_snapshot = types.SimpleNamespace(
        __dataclass_fields__={name: None for name in values},
        **values,
    )

    result = rollover.evaluate_eligibility(legacy_snapshot)

    assert result.allowed is False
    assert result.reasons == ("proof_schema_mismatch",)


def test_quiescence_result_and_offer_identity_are_immutable():
    rollover = _rollover()
    snapshot = rollover.QuiescenceSnapshot(**_all_true(rollover.QuiescenceSnapshot))
    result = rollover.evaluate_quiescence(snapshot)
    assert result.allowed is True
    assert result.reasons == ()
    with pytest.raises((AttributeError, TypeError)):
        result.allowed = False

    fence = rollover.RolloverFence(
        runtime_id="runtime-1",
        predecessor_stored_id="stored-1",
        settled_turn_generation=3,
        history_version=7,
        compression_count=2,
        final_message_id=42,
        final_content="final answer",
    )
    offer = rollover.new_offer(fence)
    assert offer.fence == fence
    assert len(offer.token) >= 32
    assert "final answer" not in offer.token
    with pytest.raises((AttributeError, TypeError)):
        offer.token = "changed"


def test_compute_host_session_suppresses_offer(monkeypatch):
    emitted = []
    session = _session(_compute_host_active=True)
    db = _patch_clear_dependencies(monkeypatch)
    monkeypatch.setattr(server, "_turn_isolation_enabled", lambda _cfg=None: True)
    monkeypatch.setattr(server, "_emit", lambda *args: emitted.append(args))

    result = server._maybe_emit_session_rollover_offer(
        "runtime-1", session, **_facts()
    )

    assert result.allowed is False
    assert result.reasons == ("compute_host_active",)
    assert emitted == []
    assert db.tail_reads == 0
    assert session.get("_rollover_offer") is None


def test_compute_host_query_error_suppresses_offer(monkeypatch):
    emitted = []
    session = _session()
    db = _patch_clear_dependencies(monkeypatch)

    def fail_compute_host_query(_session):
        raise RuntimeError("compute host state unavailable")

    monkeypatch.setattr(server, "_session_uses_compute_host", fail_compute_host_query)
    monkeypatch.setattr(server, "_emit", lambda *args: emitted.append(args))

    result = server._maybe_emit_session_rollover_offer(
        "runtime-1", session, **_facts()
    )

    assert result.allowed is False
    assert result.reasons == ("compute_host_active",)
    assert emitted == []
    assert db.tail_reads == 0
    assert session.get("_rollover_offer") is None


def test_clean_candidate_emits_scoped_payload_and_keeps_full_fence_server_side(
    monkeypatch,
):
    emitted = []
    session = _session()
    db = _patch_clear_dependencies(monkeypatch)
    monkeypatch.setattr(server, "_emit", lambda *args: emitted.append(args))

    result = server._maybe_emit_session_rollover_offer(
        "runtime-1", session, **_facts()
    )

    assert result.allowed is True
    assert len(emitted) == 1
    event, runtime_id, payload = emitted[0]
    assert event == "session.rollover.offer"
    assert runtime_id == "runtime-1"
    assert payload == {
        "token": session["_rollover_offer"].token,
        "runtime_id": "runtime-1",
        "predecessor_stored_id": "stored-1",
        "turn_generation": 3,
        "history_version": 7,
        "compression_count": 2,
        "final_message_id": 42,
    }
    assert "final_content" not in payload
    assert "final answer" not in repr(payload)
    fence = session["_rollover_offer"].fence
    assert fence.final_content == "final answer"
    assert fence.final_message_id == 42
    assert db.tail_reads == 1
    assert session["rollover_consumed_compression_count"] == 1


def test_live_tracked_background_prompt_suppresses_offer(monkeypatch):
    session = _session(_background_prompt_task_ids={"bg_live"})
    _patch_clear_dependencies(monkeypatch)
    emitted = []
    monkeypatch.setattr(server, "_emit", lambda *args: emitted.append(args))

    result = server._maybe_emit_session_rollover_offer(
        "runtime-1", session, **_facts()
    )

    assert result.allowed is False
    assert result.reasons == ("background_prompt_running",)
    assert emitted == []


def test_prompt_background_lifecycle_registers_and_clears(monkeypatch):
    import run_agent

    entered = threading.Event()
    release = threading.Event()
    completed = threading.Event()

    class _BlockingBackgroundAgent:
        def __init__(self, **_kwargs):
            pass

        def run_conversation(self, **_kwargs):
            entered.set()
            assert release.wait(3)
            return {"final_response": "background result"}

    session = _session()
    server._initialize_rollover_state(session)
    monkeypatch.setattr(server, "_sessions", {"runtime-1": session})
    monkeypatch.setattr(server, "_background_agent_kwargs", lambda *_args: {})
    monkeypatch.setattr(run_agent, "AIAgent", _BlockingBackgroundAgent)
    monkeypatch.setattr(
        server,
        "_emit",
        lambda event, *_args: completed.set()
        if event == "background.complete"
        else None,
    )

    response = server.handle_request(
        {
            "id": "background",
            "method": "prompt.background",
            "params": {"session_id": "runtime-1", "text": "work"},
        }
    )
    task_id = response["result"]["task_id"]
    assert entered.wait(3)
    try:
        with session["history_lock"]:
            assert session["_background_prompt_task_ids"] == {task_id}
            assert session["_background_prompt_owned_task_ids"] == {task_id}
    finally:
        release.set()

    assert completed.wait(3)
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        with session["history_lock"]:
            if not session["_background_prompt_task_ids"]:
                break
        time.sleep(0.01)
    with session["history_lock"]:
        assert session["_background_prompt_task_ids"] == set()
        assert session["_background_prompt_owned_task_ids"] == {task_id}


def test_prompt_background_start_failure_clears_registration(monkeypatch):
    class _FailingThread:
        def __init__(self, **_kwargs):
            pass

        def start(self):
            raise RuntimeError("thread unavailable")

    session = _session()
    server._initialize_rollover_state(session)
    monkeypatch.setattr(server, "_sessions", {"runtime-1": session})
    monkeypatch.setattr(server.threading, "Thread", _FailingThread)

    response = server.handle_request(
        {
            "id": "background",
            "method": "prompt.background",
            "params": {"session_id": "runtime-1", "text": "work"},
        }
    )

    assert response["error"] == {
        "code": 5027,
        "message": "background prompt failed to start: thread unavailable",
    }
    with session["history_lock"]:
        assert session["_background_prompt_task_ids"] == set()
        assert session["_background_prompt_owned_task_ids"] == set()



def test_preview_restart_registers_while_running(monkeypatch):
    import run_agent
    from tools import terminal_tool

    entered = threading.Event()
    release = threading.Event()
    completed = threading.Event()

    class _BlockingPreviewAgent:
        def __init__(self, **_kwargs):
            pass

        def run_conversation(self, **_kwargs):
            entered.set()
            assert release.wait(3)
            return {"final_response": "preview restarted"}

    session = _session()
    server._initialize_rollover_state(session)
    monkeypatch.setattr(server, "_sessions", {"runtime-1": session})
    monkeypatch.setattr(server, "_ephemeral_preview_agent_kwargs", lambda *_args: {})
    monkeypatch.setattr(server, "_set_session_context", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(server, "_clear_session_context", lambda _tokens: None)
    monkeypatch.setattr(server, "_session_cwd", lambda _session: "/tmp/work")
    monkeypatch.setattr(run_agent, "AIAgent", _BlockingPreviewAgent)
    monkeypatch.setattr(terminal_tool, "clear_task_env_overrides", lambda _task_id: None)
    monkeypatch.setattr(
        server,
        "_emit",
        lambda event, *_args: completed.set()
        if event == "preview.restart.complete"
        else None,
    )

    response = server.handle_request(
        {
            "id": "preview",
            "method": "preview.restart",
            "params": {"session_id": "runtime-1", "url": "http://localhost:3000"},
        }
    )
    task_id = response["result"]["task_id"]
    assert entered.wait(3)
    try:
        with session["history_lock"]:
            assert session["_background_prompt_task_ids"] == {task_id}
            assert session["_background_prompt_owned_task_ids"] == {task_id}
    finally:
        release.set()

    assert completed.wait(3)
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        with session["history_lock"]:
            if not session["_background_prompt_task_ids"]:
                break
        time.sleep(0.01)
    with session["history_lock"]:
        assert session["_background_prompt_task_ids"] == set()
        assert session["_background_prompt_owned_task_ids"] == {task_id}


def test_preview_restart_normal_completion_clears_registration(monkeypatch):
    import run_agent
    from tools import terminal_tool

    events = []
    context_clears = []
    env_clears = []

    class _CompletingPreviewAgent:
        def __init__(self, **_kwargs):
            pass

        def run_conversation(self, **_kwargs):
            return {"final_response": "preview restarted"}

    session = _session()
    server._initialize_rollover_state(session)
    monkeypatch.setattr(server, "_sessions", {"runtime-1": session})
    monkeypatch.setattr(server.threading, "Thread", _ImmediateThread)
    monkeypatch.setattr(server, "_ephemeral_preview_agent_kwargs", lambda *_args: {})
    monkeypatch.setattr(server, "_set_session_context", lambda *_args, **_kwargs: ["token"])
    monkeypatch.setattr(server, "_clear_session_context", context_clears.append)
    monkeypatch.setattr(server, "_session_cwd", lambda _session: "/tmp/work")
    monkeypatch.setattr(run_agent, "AIAgent", _CompletingPreviewAgent)
    monkeypatch.setattr(terminal_tool, "clear_task_env_overrides", env_clears.append)
    monkeypatch.setattr(server, "_emit", lambda *args: events.append(args))

    response = server.handle_request(
        {
            "id": "preview",
            "method": "preview.restart",
            "params": {"session_id": "runtime-1", "url": "http://localhost:3000"},
        }
    )
    task_id = response["result"]["task_id"]

    with session["history_lock"]:
        assert session["_background_prompt_task_ids"] == set()
        assert session["_background_prompt_owned_task_ids"] == {task_id}
    assert context_clears == [["token"]]
    assert env_clears == [task_id]
    assert [event for event, *_rest in events] == [
        "preview.restart.progress",
        "preview.restart.complete",
    ]
    assert events[-1][2] == {"task_id": task_id, "text": "preview restarted"}


def test_preview_restart_context_setup_failure_clears_registration(monkeypatch):
    from tools import terminal_tool

    events = []
    env_clears = []

    def fail_context(*_args, **_kwargs):
        raise RuntimeError("context unavailable")

    session = _session()
    server._initialize_rollover_state(session)
    monkeypatch.setattr(server, "_sessions", {"runtime-1": session})
    monkeypatch.setattr(server.threading, "Thread", _ImmediateThread)
    monkeypatch.setattr(server, "_set_session_context", fail_context)
    monkeypatch.setattr(
        server,
        "_clear_session_context",
        lambda _tokens: pytest.fail("unset context must not be cleared"),
    )
    monkeypatch.setattr(server, "_session_cwd", lambda _session: "/tmp/work")
    monkeypatch.setattr(terminal_tool, "clear_task_env_overrides", env_clears.append)
    monkeypatch.setattr(server, "_emit", lambda *args: events.append(args))

    response = server.handle_request(
        {
            "id": "preview",
            "method": "preview.restart",
            "params": {"session_id": "runtime-1", "url": "http://localhost:3000"},
        }
    )
    task_id = response["result"]["task_id"]

    with session["history_lock"]:
        assert session["_background_prompt_task_ids"] == set()
        assert session["_background_prompt_owned_task_ids"] == {task_id}
    assert env_clears == [task_id]
    assert events == [
        (
            "preview.restart.complete",
            "runtime-1",
            {"task_id": task_id, "text": "error: context unavailable"},
        )
    ]


def test_preview_restart_start_failure_clears_registration(monkeypatch):
    class _FailingThread:
        def __init__(self, **_kwargs):
            pass

        def start(self):
            raise RuntimeError("thread unavailable")

    session = _session()
    server._initialize_rollover_state(session)
    monkeypatch.setattr(server, "_sessions", {"runtime-1": session})
    monkeypatch.setattr(server.threading, "Thread", _FailingThread)

    response = server.handle_request(
        {
            "id": "preview",
            "method": "preview.restart",
            "params": {"session_id": "runtime-1", "url": "http://localhost:3000"},
        }
    )

    assert response["error"] == {
        "code": 5027,
        "message": "preview restart failed to start: thread unavailable",
    }
    with session["history_lock"]:
        assert session["_background_prompt_task_ids"] == set()
        assert session["_background_prompt_owned_task_ids"] == set()

@pytest.mark.parametrize(
    ("change", "code"),
    [
        ("tail-id", "db_tail_mismatch"),
        ("tail-content", "db_tail_mismatch"),
        ("history", "history_changed"),
        ("history-tail-content", "history_tail_mismatch"),
        ("agent", "agent_identity_mismatch"),
    ],
)
def test_exact_tail_history_and_agent_identity_fences_suppress(
    monkeypatch, change, code
):
    session = _session()
    db = _DB()
    facts = _facts()
    if change == "tail-id":
        db.final_id = 99
    elif change == "tail-content":
        db.final_content = "changed after delivery"
    elif change == "history":
        session["history_version"] = 8
    elif change == "history-tail-content":
        session["history"][-1]["content"] = "mutated in place"
    else:
        session["agent"].session_id = "stale-agent-id"
    _patch_clear_dependencies(monkeypatch, db)
    emitted = []
    monkeypatch.setattr(server, "_emit", lambda *args: emitted.append(args))

    result = server._maybe_emit_session_rollover_offer(
        "runtime-1", session, **facts
    )

    assert result.allowed is False
    assert code in result.reasons
    assert emitted == []
    assert session.get("_rollover_offer") is None


@pytest.mark.parametrize(
    ("status", "final_content"),
    [
        ("error", "final answer"),
        ("interrupted", "final answer"),
        ("complete", ""),
        ("complete", "   "),
    ],
)
def test_turn_clean_override_cannot_admit_failed_or_empty_turn(
    monkeypatch, status, final_content
):
    session = _session()
    _patch_clear_dependencies(monkeypatch)
    emitted = []
    monkeypatch.setattr(server, "_emit", lambda *args: emitted.append(args))

    result = server._maybe_emit_session_rollover_offer(
        "runtime-1",
        session,
        **_facts(status=status, final_content=final_content),
        turn_clean=True,
    )

    assert result.allowed is False
    assert "turn_not_clean" in result.reasons
    assert emitted == []


def test_real_bridge_helper_reports_blocking_input_and_approval(monkeypatch):
    _db, dependencies = _patch_real_clear_dependencies(monkeypatch)
    monkeypatch.setattr(
        server,
        "_pending",
        {"request-1": ("runtime-1", threading.Event())},
    )
    monkeypatch.setattr(
        dependencies["approval"], "has_blocking_approval", lambda _key: True
    )

    assert server._rollover_bridge_blockers("runtime-1", _session()) == (
        "blocking_input",
        "approval_pending",
    )


@pytest.mark.parametrize(
    ("rows", "expected"),
    [
        ([{"status": "running"}], ("background_process_running",)),
        ([{"status": "exited"}], ()),
        ([], ()),
        ([{"status": "exited", "notify_on_complete": False}], ()),
        ([{"status": "exited", "notify_on_complete": True}], ("background_completion_publication_pending",)),
        ([{"status": "exited", "notify_on_complete": True, "completion_notification_enqueued": False}], ("background_completion_publication_pending",)),
        ([{"status": "exited", "notify_on_complete": True, "completion_notification_enqueued": "yes"}], ("background_completion_publication_pending",)),
        ([{"status": "exited", "notify_on_complete": True, "completion_notification_enqueued": True}], ()),
    ],
)
def test_real_process_helper_distinguishes_running_and_exited(
    monkeypatch, rows, expected
):
    _db, dependencies = _patch_real_clear_dependencies(monkeypatch)
    monkeypatch.setattr(
        dependencies["process_registry"],
        "list_sessions",
        lambda **_kwargs: rows,
    )

    assert server._rollover_process_blockers(_session()) == expected


def test_exited_unpublished_process_denies_offer_and_emits_nothing(monkeypatch):
    _db, dependencies = _patch_real_clear_dependencies(monkeypatch)
    monkeypatch.setattr(
        dependencies["process_registry"],
        "list_sessions",
        lambda **_kwargs: [
            {
                "status": "exited",
                "notify_on_complete": True,
                "completion_notification_enqueued": False,
            }
        ],
    )
    emitted = []
    monkeypatch.setattr(server, "_emit", lambda *args: emitted.append(args))
    session = _session()

    result = server._maybe_emit_session_rollover_offer(
        "runtime-1", session, **_facts()
    )

    assert result.allowed is False
    assert result.reasons == ("background_completion_publication_pending",)
    assert emitted == []
    assert session.get("_rollover_offer") is None


def test_notification_delivery_handoff_denies_offer_with_empty_queue(monkeypatch):
    _db, dependencies = _patch_real_clear_dependencies(monkeypatch)
    emitted = []
    monkeypatch.setattr(server, "_emit", lambda *args: emitted.append(args))
    session = _session()

    with server._notification_delivery_handoff():
        assert dependencies["process_registry"].completion_queue.queue == []
        result = server._maybe_emit_session_rollover_offer(
            "runtime-1", session, **_facts()
        )

    assert result.allowed is False
    assert result.reasons == ("notification_delivery_inflight",)
    assert emitted == []
    assert session.get("_rollover_offer") is None



def test_process_completion_publication_handoff_has_no_rollover_gap(monkeypatch):
    import tools.async_delegation as async_delegation
    import tools.process_registry as process_registry_module
    from tools.process_registry import ProcessRegistry, ProcessSession

    publication_started = threading.Event()
    allow_publication = threading.Event()

    class PublicationBarrierQueue(queue.Queue):
        def put(self, item, block=True, timeout=None):
            publication_started.set()
            assert allow_publication.wait(timeout=5)
            return super().put(item, block=block, timeout=timeout)

    registry = ProcessRegistry()
    registry.completion_queue = PublicationBarrierQueue()
    process = ProcessSession(
        id="proc-rollover-gap",
        command="build",
        session_key="stored-1",
        started_at=time.time(),
        exited=True,
        exit_code=0,
        notify_on_complete=True,
    )
    registry._running[process.id] = process
    monkeypatch.setattr(registry, "_write_checkpoint", lambda: None)
    monkeypatch.setattr(process_registry_module, "process_registry", registry)
    monkeypatch.setattr(
        async_delegation,
        "load_deferred_notifications",
        lambda *_args, **_kwargs: [],
    )

    finisher = threading.Thread(target=registry._move_to_finished, args=(process,))
    finisher.start()
    assert publication_started.wait(timeout=5)
    try:
        assert server._rollover_process_blockers(_session()) == (
            "background_completion_publication_pending",
        )
        assert server._rollover_notification_blockers(
            "runtime-1", _session()
        ) == ()
    finally:
        allow_publication.set()
        finisher.join(timeout=5)
    assert not finisher.is_alive()

    assert server._rollover_process_blockers(_session()) == ()
    assert server._rollover_notification_blockers(
        "runtime-1", _session()
    ) == ("addressed_completion_pending",)

    delivered = registry.drain_notifications(
        session_key="stored-1", skip_poll_observed=False
    )
    assert len(delivered) == 1
    assert server._rollover_process_blockers(_session()) == ()
    assert server._rollover_notification_blockers("runtime-1", _session()) == ()


@pytest.mark.parametrize("task_id", ["bg_done", "preview_done"])
def test_real_process_helper_checks_completed_hidden_task_ownership(
    monkeypatch, task_id
):
    _db, dependencies = _patch_real_clear_dependencies(monkeypatch)
    calls = []

    def list_sessions(**filters):
        calls.append(filters)
        if filters == {"task_id": task_id}:
            return [{"status": "running"}]
        return []

    monkeypatch.setattr(
        dependencies["process_registry"], "list_sessions", list_sessions
    )
    session = _session(_background_prompt_owned_task_ids={task_id})

    assert server._rollover_process_blockers(session) == (
        "background_process_running",
    )
    assert calls == [
        {"session_key": "stored-1"},
        {"task_id": task_id},
    ]


@pytest.mark.parametrize(
    ("rows", "expected"),
    [
        ([{"origin_ui_session_id": "runtime-1", "status": "running"}], ("async_delegation_running",)),
        ([{"session_key": "stored-1", "status": "finalizing"}], ("async_delegation_running",)),
        ([{"parent_session_id": "stored-1", "status": "completed"}], ()),
        ([{"origin_ui_session_id": "other", "status": "running"}], ()),
        ([{"origin_ui_session_id": "other"}], ()),
    ],
)
def test_real_delegation_helper_checks_ownership_and_state(
    monkeypatch, rows, expected
):
    _db, dependencies = _patch_real_clear_dependencies(monkeypatch)
    monkeypatch.setattr(
        dependencies["async_delegation"],
        "list_async_delegations",
        lambda: rows,
    )

    assert server._rollover_delegation_blockers("runtime-1", _session()) == expected


@pytest.mark.parametrize(
    ("field", "task_id", "status"),
    [
        ("origin_session", "bg_done", "running"),
        ("session_key", "preview_done", "finalizing"),
        ("parent_session_id", "bg_done", "running"),
    ],
)
def test_real_delegation_helper_checks_completed_hidden_task_ownership(
    monkeypatch, field, task_id, status
):
    _db, dependencies = _patch_real_clear_dependencies(monkeypatch)
    monkeypatch.setattr(
        dependencies["async_delegation"],
        "list_async_delegations",
        lambda: [{field: task_id, "status": status}],
    )
    session = _session(_background_prompt_owned_task_ids={task_id})

    assert server._rollover_delegation_blockers("runtime-1", session) == (
        "async_delegation_running",
    )


@pytest.mark.parametrize(
    ("goal", "expected"),
    [
        (None, ()),
        ('{"status": "complete"}', ()),
        ('{"status": "active"}', ("goal_active",)),
        ('{"status": "paused", "waiting_on_pid": 123}', ("goal_waiting",)),
    ],
)
def test_real_goal_helper_handles_active_waiting_and_no_goal_json(
    monkeypatch, goal, expected
):
    _patch_real_clear_dependencies(monkeypatch, _DB(goal=goal))

    assert server._rollover_goal_blockers(_session()) == expected


@pytest.mark.parametrize(
    ("rows", "events", "expected"),
    [
        ([{"delegation_id": "d1"}], [], ("deferred_notification_adoption",)),
        ([], [{"origin_ui_session_id": "runtime-1"}], ("addressed_completion_pending",)),
        ([], [{"session_key": "stored-1"}], ("addressed_completion_pending",)),
        ([], [{"session_key": "other"}], ()),
        ([], [{}], ()),
    ],
)
def test_real_notification_helper_checks_adoption_and_completion_ownership(
    monkeypatch, rows, events, expected
):
    _db, dependencies = _patch_real_clear_dependencies(monkeypatch)
    monkeypatch.setattr(
        dependencies["async_delegation"],
        "load_deferred_notifications",
        lambda *_args, **_kwargs: rows,
    )
    monkeypatch.setattr(
        dependencies["process_registry"],
        "completion_queue",
        _CompletionQueue(events),
    )

    assert server._rollover_notification_blockers("runtime-1", _session()) == expected


@pytest.mark.parametrize(
    ("failure", "unknown_code"),
    [
        ("bridge-error", "approval_query_unknown"),
        ("process-malformed", "process_query_unknown"),
        ("process-error", "process_query_unknown"),
        ("delegation-malformed", "delegation_query_unknown"),
        ("delegation-error", "delegation_query_unknown"),
        ("goal-malformed", "goal_query_unknown"),
        ("goal-error", "goal_query_unknown"),
        ("notification-malformed", "notification_query_unknown"),
        ("notification-error", "notification_query_unknown"),
    ],
)
def test_real_dependency_malformed_or_error_suppresses_offer(
    monkeypatch, failure, unknown_code
):
    db, dependencies = _patch_real_clear_dependencies(monkeypatch)
    if failure == "bridge-error":
        monkeypatch.setattr(
            dependencies["approval"],
            "has_blocking_approval",
            lambda _key: (_ for _ in ()).throw(RuntimeError("approval failed")),
        )
    elif failure == "process-malformed":
        monkeypatch.setattr(
            dependencies["process_registry"], "list_sessions", lambda **_kwargs: [{}]
        )
    elif failure == "process-error":
        monkeypatch.setattr(
            dependencies["process_registry"],
            "list_sessions",
            lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("process failed")),
        )
    elif failure == "delegation-malformed":
        monkeypatch.setattr(
            dependencies["async_delegation"], "list_async_delegations", lambda: [None]
        )
    elif failure == "delegation-error":
        monkeypatch.setattr(
            dependencies["async_delegation"],
            "list_async_delegations",
            lambda: (_ for _ in ()).throw(RuntimeError("delegation failed")),
        )
    elif failure == "goal-malformed":
        db.goal = "{"
    elif failure == "goal-error":
        monkeypatch.setattr(
            db,
            "get_meta",
            lambda _key: (_ for _ in ()).throw(RuntimeError("goal failed")),
        )
    elif failure == "notification-malformed":
        monkeypatch.setattr(
            dependencies["async_delegation"],
            "load_deferred_notifications",
            lambda *_args, **_kwargs: [None],
        )
    else:
        monkeypatch.setattr(
            dependencies["async_delegation"],
            "load_deferred_notifications",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RuntimeError("notification failed")
            ),
        )
    emitted = []
    monkeypatch.setattr(server, "_emit", lambda *args: emitted.append(args))

    result = server._maybe_emit_session_rollover_offer(
        "runtime-1", _session(), **_facts()
    )

    assert result.allowed is False
    assert unknown_code in result.reasons
    assert emitted == []


def test_same_candidate_deduplicates_but_later_settled_turn_can_offer_again(
    monkeypatch,
):
    session = _session()
    db = _patch_clear_dependencies(monkeypatch)
    emitted = []
    monkeypatch.setattr(server, "_emit", lambda *args: emitted.append(args))

    first = server._maybe_emit_session_rollover_offer(
        "runtime-1", session, **_facts()
    )
    duplicate = server._maybe_emit_session_rollover_offer(
        "runtime-1", session, **_facts()
    )
    first_token = emitted[0][2]["token"]

    assert first.allowed and duplicate.allowed
    assert len(emitted) == 1

    db.final_id = 43
    db.final_content = "later answer"
    session["history"][-1]["content"] = "later answer"
    session["history_version"] = 8
    session["turn_generation"] = 4
    later = server._maybe_emit_session_rollover_offer(
        "runtime-1",
        session,
        **_facts(
            final_content="later answer",
            final_message_id=43,
            settled_history_version=8,
            settled_turn_generation=4,
        ),
    )

    assert later.allowed
    assert len(emitted) == 2
    assert emitted[1][2]["token"] != first_token
    assert session["rollover_consumed_compression_count"] == 1



def test_offer_generation_mutates_no_db_queue_goal_or_notification_state(monkeypatch):
    session = _session(
        queued_prompt=None,
        queued_prompts=[],
        deferred_notification_texts=[],
        deferred_notification_event_ids=set(),
        defer_notifications_until_user=False,
    )
    db = _patch_clear_dependencies(monkeypatch)
    monkeypatch.setattr(server, "_emit", lambda *_a: None)
    before = {
        "history": deepcopy(session["history"]),
        "history_version": session["history_version"],
        "session_key": session["session_key"],
        "queued_prompt": session["queued_prompt"],
        "queued_prompts": deepcopy(session["queued_prompts"]),
        "notification_texts": deepcopy(session["deferred_notification_texts"]),
        "notification_ids": set(session["deferred_notification_event_ids"]),
        "notification_latch": session["defer_notifications_until_user"],
        "compression_consumed": session["rollover_consumed_compression_count"],
    }

    server._maybe_emit_session_rollover_offer("runtime-1", session, **_facts())

    assert session["history"] == before["history"]
    assert session["history_version"] == before["history_version"]
    assert session["session_key"] == before["session_key"]
    assert session["queued_prompt"] is before["queued_prompt"]
    assert session["queued_prompts"] == before["queued_prompts"]
    assert session["deferred_notification_texts"] == before["notification_texts"]
    assert session["deferred_notification_event_ids"] == before["notification_ids"]
    assert session["defer_notifications_until_user"] is before["notification_latch"]
    assert session["rollover_consumed_compression_count"] == before["compression_consumed"]
    assert db.tail_reads == 1


def test_offer_hook_runs_after_complete_settled_info_and_every_post_turn_drain(
    monkeypatch,
):
    order = []
    session = _session(agent=_Agent(), running=True, history=[], history_version=0)
    monkeypatch.setattr(server.threading, "Thread", _ImmediateThread)
    monkeypatch.setattr(server, "_emit", lambda event, *_a: order.append(event))
    monkeypatch.setattr(server, "_wire_callbacks", lambda *_a: None)
    monkeypatch.setattr(server, "_sync_agent_model_with_config", lambda *_a: None)
    monkeypatch.setattr(server, "_register_session_cwd", lambda *_a: None)
    monkeypatch.setattr(server, "_session_cwd", lambda *_a: ".")
    monkeypatch.setattr(server, "make_stream_renderer", lambda *_a: None)
    monkeypatch.setattr(server, "render_message", lambda *_a: None)
    monkeypatch.setattr(server, "_get_usage", lambda *_a: {})
    monkeypatch.setattr(server, "_get_db", lambda: None)
    monkeypatch.setattr(server, "_sync_session_key_after_compress", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_voice_tts_enabled", lambda: False)
    monkeypatch.setattr(server, "_load_cfg", lambda: {})
    monkeypatch.setattr(
        server,
        "_durable_final_assistant_identity",
        lambda *_a: {"id": 42, "role": "assistant", "content": "final answer"},
    )
    monkeypatch.setattr(
        server,
        "_drain_queued_prompt",
        lambda *_a: order.append("queue.drain") or False,
    )
    monkeypatch.setattr(
        server,
        "_drain_post_turn_notifications",
        lambda *_a: order.append("notification.drain") or False,
    )
    monkeypatch.setattr(
        server,
        "_dispatch_goal_followup",
        lambda *_a: order.append("goal.dispatch") or False,
    )
    monkeypatch.setattr(
        server,
        "_maybe_emit_session_rollover_offer",
        lambda *_a, **_k: order.append("offer.hook"),
    )
    from hermes_cli.goals import GoalManager

    monkeypatch.setattr(GoalManager, "is_active", lambda *_a: True)
    monkeypatch.setattr(
        GoalManager,
        "evaluate_after_turn",
        lambda *_a, **_k: {
            "should_continue": True,
            "continuation_prompt": "continue",
            "message": "continuing",
        },
    )

    server._run_prompt_submit("rid", "runtime-1", session, "question")

    complete = order.index("message.complete")
    settled_info = max(i for i, event in enumerate(order) if event == "session.info")
    assert complete < settled_info < order.index("queue.drain")
    assert order.index("queue.drain") < order.index("goal.dispatch")
    assert order.index("goal.dispatch") < order.index("notification.drain")
    assert order.index("notification.drain") < order.index("offer.hook")


@pytest.mark.parametrize(
    "work_source",
    ["queued prompt drain", "goal follow-up", "notification follow-up"],
)
def test_live_post_turn_work_reservation_blocks_rollover_offer(
    monkeypatch, work_source
):
    emitted = []
    session = _session(
        running=True,
        turn_reservation_token=f"{work_source}-reservation",
    )
    _patch_clear_dependencies(monkeypatch)
    monkeypatch.setattr(server, "_emit", lambda *args: emitted.append(args))

    result = server._maybe_emit_session_rollover_offer(
        "runtime-1", session, **_facts()
    )

    assert result.allowed is False
    assert "running" in result.reasons
    assert "turn_reservation" in result.reasons
    assert emitted == []
