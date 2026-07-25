"""Session-scoped coding-workflow persistence and projection contracts."""

from __future__ import annotations

import json
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from hermes_cli.coding_workflow import workflow_presets
from tui_gateway import server


def _agent(workflow: str = "hybrid-v1") -> SimpleNamespace:
    return SimpleNamespace(
        coding_workflow=workflow,
        model="gpt-5.6-sol",
        provider="custom:sudo",
        reasoning_config=None,
        service_tier=None,
        session_id="stored-1",
        tools=[],
    )


def test_runtime_model_config_roundtrip_and_malformed_stored_value_fails_closed(monkeypatch):
    agent = _agent()
    agent.provider = "anthropic"

    config = server._runtime_model_config(agent)
    assert config["coding_workflow"] == "hybrid-v1"

    row = {"model": agent.model, "model_config": json.dumps(config)}
    assert server._stored_session_runtime_overrides(row)["coding_workflow"] == "hybrid-v1"

    row["model_config"] = json.dumps({"coding_workflow": "unknown-v9"})
    with pytest.raises(ValueError, match="coding workflow"):
        server._stored_session_runtime_overrides(row)


def test_deferred_record_lazy_info_and_session_info_project_workflow(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "_load_show_reasoning", lambda: False)
    monkeypatch.setattr(server, "_load_tool_progress_mode", lambda: "off")
    monkeypatch.setattr(server, "_hydrate_deferred_notification_state", lambda _record: None)

    record = server._deferred_session_record(
        "stored-1",
        cols=80,
        cwd=str(tmp_path),
        history=[],
        lease=None,
        coding_workflow="hybrid-v1",
    )
    assert record["coding_workflow"] == "hybrid-v1"
    assert server._lazy_resume_info(
        str(tmp_path), coding_workflow="hybrid-v1"
    )["coding_workflow"] == "hybrid-v1"

    monkeypatch.setattr(server, "_load_cfg", lambda: {"display": {}})
    monkeypatch.setattr(server, "_display_session_cwd", lambda _session: str(tmp_path))
    monkeypatch.setattr(server, "_git_branch_for_cwd", lambda _cwd: "")
    monkeypatch.setattr(server, "_project_info_for_cwd", lambda _cwd: None)
    monkeypatch.setattr(server, "_load_approval_mode", lambda: "manual")
    monkeypatch.setattr(server, "_probe_credentials", lambda _agent: None)
    monkeypatch.setattr(server, "_current_profile_name", lambda: "orchestrator")
    info = server._session_info(_agent(), {**record, "agent": _agent()})
    assert info["coding_workflow"] == "hybrid-v1"


def test_set_session_context_binds_workflow(monkeypatch):
    from gateway import session_context

    captured = {}

    def fake_set_session_vars(**kwargs):
        captured.update(kwargs)
        return ["token"]

    monkeypatch.setattr(session_context, "set_session_vars", fake_set_session_vars)
    monkeypatch.setattr(
        server,
        "_sessions",
        {"runtime-1": {"session_key": "stored-1", "source": "desktop", "coding_workflow": "hybrid-v1"}},
    )
    monkeypatch.setattr(server, "_cwd_for_session_key", lambda _key: "/tmp")

    assert server._set_session_context("stored-1") == ["token"]
    assert captured["coding_workflow"] == "hybrid-v1"


def test_session_create_snapshots_manual_workflow_and_rejects_unknown(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "_sessions", {})
    monkeypatch.setattr(server, "_new_session_key", lambda: "stored-1")
    monkeypatch.setattr(server, "_completion_cwd", lambda _params: str(tmp_path))
    monkeypatch.setattr(server, "_resolve_session_source", lambda _source: "desktop")
    monkeypatch.setattr(server, "_enable_gateway_prompts", lambda: None)
    monkeypatch.setattr(server, "_profile_home", lambda _profile: None)
    monkeypatch.setattr(server, "_claim_active_session_slot", lambda *_args, **_kwargs: (None, None))
    monkeypatch.setattr(server, "_initialize_rollover_state", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(server, "_register_session_cwd", lambda _session: None)
    monkeypatch.setattr(server, "_restore_activated_profile_completions", lambda _session: None)
    monkeypatch.setattr(server, "_schedule_agent_build", lambda _sid: None)
    monkeypatch.setattr(server, "_schedule_session_cap_enforcement", lambda: None)
    monkeypatch.setattr(server, "_git_branch_for_cwd", lambda _cwd: "")
    monkeypatch.setattr(server, "_project_info_for_cwd", lambda _cwd: None)
    monkeypatch.setattr(server, "_current_profile_name", lambda: "orchestrator")

    create = server._methods["session.create"]
    response = create("r1", {"source": "desktop", "coding_workflow": "hybrid-v1"})
    result = response["result"]
    session = server._sessions[result["session_id"]]
    assert session["coding_workflow"] == "hybrid-v1"
    assert result["info"]["coding_workflow"] == "hybrid-v1"

    rejected = create("r2", {"source": "desktop", "coding_workflow": "unknown-v9"})
    assert rejected["error"]["code"] == 4002


def test_session_create_canonicalizes_hybrid_route_before_deferred_build(monkeypatch, tmp_path):
    claimed = []
    scheduled = []
    captured = {}

    class FakeWorker:
        def __init__(self, *_args, **_kwargs):
            pass

        def close(self):
            pass

    def claim(*args, **kwargs):
        claimed.append((args, kwargs))
        return (None, None)

    def fake_make_agent(_sid, _key, **kwargs):
        captured.update(kwargs)
        return _agent()

    monkeypatch.setattr(server, "_sessions", {})
    monkeypatch.setattr(server, "_new_session_key", lambda: "stored-1")
    monkeypatch.setattr(server, "_completion_cwd", lambda _params: str(tmp_path))
    monkeypatch.setattr(server, "_resolve_session_source", lambda _source: "desktop")
    monkeypatch.setattr(server, "_enable_gateway_prompts", lambda: None)
    monkeypatch.setattr(server, "_profile_home", lambda _profile: None)
    monkeypatch.setattr(server, "_claim_active_session_slot", claim)
    monkeypatch.setattr(server, "_initialize_rollover_state", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(server, "_register_session_cwd", lambda _session: None)
    monkeypatch.setattr(server, "_restore_activated_profile_completions", lambda _session: None)
    monkeypatch.setattr(server, "_schedule_agent_build", lambda sid: scheduled.append(sid))
    monkeypatch.setattr(server, "_schedule_session_cap_enforcement", lambda: None)
    monkeypatch.setattr(server, "_git_branch_for_cwd", lambda _cwd: "")
    monkeypatch.setattr(server, "_project_info_for_cwd", lambda _cwd: None)
    monkeypatch.setattr(server, "_current_profile_name", lambda: "orchestrator")

    create = server._methods["session.create"]
    rejected = create(
        "r1",
        {
            "source": "desktop",
            "coding_workflow": "hybrid-v1",
            "provider": "anthropic",
            "model": "claude-sonnet-4.6",
        },
    )
    assert rejected["error"]["code"] == 4002
    assert claimed == []
    assert server._sessions == {}
    assert scheduled == []

    created = create("r2", {"source": "desktop", "coding_workflow": "hybrid-v1"})
    sid = created["result"]["session_id"]
    session = server._sessions[sid]
    assert session["model_override"] == {
        "provider": "custom:sudo",
        "model": "gpt-5.6-sol",
    }
    assert session["create_reasoning_override"] == {
        "enabled": True,
        "effort": "xhigh",
    }

    monkeypatch.setattr(server, "_set_session_context", lambda _key: [])
    monkeypatch.setattr(server, "_clear_session_context", lambda _tokens: None)
    monkeypatch.setattr(server, "_make_agent", fake_make_agent)
    monkeypatch.setattr(server, "_SlashWorker", FakeWorker)
    monkeypatch.setattr(server, "_attach_worker", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(server, "_wire_callbacks", lambda _sid: None)
    monkeypatch.setattr(server, "_emit", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(server, "_session_info", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(server, "_start_notification_poller", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(server, "_notify_session_boundary", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(server, "_probe_config_health", lambda *_args: None)
    monkeypatch.setattr(server, "_schedule_mcp_late_refresh", lambda *_args: None)
    monkeypatch.setattr(server, "_config_model_target", lambda: ("", ""))

    server._start_agent_build(sid, session)
    assert session["agent_ready"].wait(timeout=3), "agent build did not finish"
    assert captured["coding_workflow"] == "hybrid-v1"
    assert captured["model_override"] == {
        "provider": "custom:sudo",
        "model": "gpt-5.6-sol",
    }
    assert captured["reasoning_config_override"] == {
        "enabled": True,
        "effort": "xhigh",
    }


def test_model_options_projects_effective_workflow_and_presets(monkeypatch):
    from hermes_cli import inventory

    monkeypatch.setattr(server, "_sessions", {"runtime-1": {"agent": _agent(), "coding_workflow": "hybrid-v1"}})
    monkeypatch.setattr(server, "_model_picker_context", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(inventory, "build_models_payload", lambda *_args, **_kwargs: {"providers": []})

    response = server._methods["model.options"]("r1", {"session_id": "runtime-1"})
    payload = response["result"]
    assert payload["coding_workflow"] == "hybrid-v1"
    assert payload["workflow_presets"] == workflow_presets()
    assert all(row.get("provider") != "hybrid" for row in payload["workflow_presets"])


class _RouteAgent:
    def __init__(self):
        self.model = "old-model"
        self.provider = "old-provider"
        self.api_key = "old-key"
        self.base_url = "https://old.invalid/v1"
        self.api_mode = "chat_completions"
        self._primary_runtime = None
        self.coding_workflow = "coupled-v1"

    def switch_model(self, *, new_model, new_provider, api_key, base_url, api_mode):
        self.model = new_model
        self.provider = new_provider
        self.api_key = api_key
        self.base_url = base_url
        self.api_mode = api_mode


def _fake_route_switch(_sid, session, _raw, **kwargs):
    assert kwargs["defer_session_commit"] is True
    agent = session["agent"]
    agent.switch_model(
        new_model="gpt-5.6-sol",
        new_provider="custom:sudo",
        api_key="new-key",
        base_url="https://new.invalid/v1",
        api_mode="chat_completions",
    )
    session["model_override"] = {
        "model": agent.model,
        "provider": agent.provider,
        "api_key": agent.api_key,
        "base_url": agent.base_url,
        "api_mode": agent.api_mode,
    }
    return {"value": agent.model, "warning": "", "confirm_required": False, "scope": "session"}


class _TrackingRLock:
    def __init__(self, route_attempted: threading.Event, on_route_attempt=None):
        self._lock = threading.RLock()
        self._route_attempted = route_attempted
        self._on_route_attempt = on_route_attempt
        self._owner = None

    def __enter__(self):
        thread_name = threading.current_thread().name
        if thread_name == "route-rpc":
            self._route_attempted.set()
            if self._on_route_attempt is not None:
                self._on_route_attempt(self._owner)
        self._lock.acquire()
        self._owner = thread_name
        return self

    def __exit__(self, _exc_type, _exc, _tb):
        self._owner = None
        self._lock.release()


def test_route_cannot_mutate_after_prompt_admission(monkeypatch):
    prompt_in_admission = threading.Event()
    release_prompt = threading.Event()
    route_attempted = threading.Event()
    transition_lock = _TrackingRLock(route_attempted)
    ready = threading.Event()
    ready.set()
    agent = _RouteAgent()
    session = {
        "agent": agent,
        "agent_ready": ready,
        "coding_workflow": "coupled-v1",
        "history": [],
        "history_lock": threading.Lock(),
        "route_lock": transition_lock,
        "transition_lock": transition_lock,
        "running": False,
        "session_key": "stored-1",
    }
    monkeypatch.setattr(server, "_sessions", {"runtime-1": session})
    monkeypatch.setattr(server, "_load_dashboard_process_isolation_config", lambda: {})
    monkeypatch.setattr(server, "_session_uses_compute_host", lambda *_args: False)
    monkeypatch.setattr(server, "_ensure_session_db_row", lambda _session: None)
    monkeypatch.setattr(server, "_persist_branch_seed", lambda _session: None)
    monkeypatch.setattr(server, "_start_agent_build", lambda *_args: None)
    monkeypatch.setattr(server, "_run_prompt_submit", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(server, "_apply_model_switch", _fake_route_switch)
    monkeypatch.setattr(server, "_persist_route_runtime_strict", lambda _session: None)
    monkeypatch.setattr(server, "_restart_slash_worker", lambda *_args: None)
    monkeypatch.setattr(server, "_session_info", lambda *_args: {})
    monkeypatch.setattr(server, "_emit", lambda *_args: None)

    message_checks = 0

    def has_prompt_message_id(_session, _message_id):
        nonlocal message_checks
        message_checks += 1
        if message_checks == 2:
            prompt_in_admission.set()
            assert release_prompt.wait(timeout=3), "prompt admission was not released"
        return False

    monkeypatch.setattr(server, "_has_prompt_message_id", has_prompt_message_id)

    prompt_response = {}
    route_response = {}

    def submit_prompt():
        prompt_response.update(
            server._methods["prompt.submit"](
                "prompt",
                {
                    "session_id": "runtime-1",
                    "text": "hello",
                    "message_id": "message-1",
                },
            )
        )

    def switch_route():
        route_response.update(
            server._methods["config.set"](
                "route",
                {
                    "session_id": "runtime-1",
                    "key": "route",
                    "value": {
                        "model": "gpt-5.6-sol",
                        "provider": "custom:sudo",
                        "coding_workflow": "hybrid-v1",
                    },
                },
            )
        )

    prompt_thread = threading.Thread(target=submit_prompt, name="prompt-rpc")
    route_thread = threading.Thread(target=switch_route, name="route-rpc")
    prompt_thread.start()
    assert prompt_in_admission.wait(timeout=3), "prompt did not reach admission"
    route_thread.start()
    assert route_attempted.wait(timeout=3), "route did not reach the transition lock"
    release_prompt.set()
    prompt_thread.join(timeout=3)
    route_thread.join(timeout=3)

    assert not prompt_thread.is_alive()
    assert not route_thread.is_alive()
    assert prompt_response["result"]["status"] == "streaming"
    assert route_response["error"]["code"] == 4009
    assert (agent.model, agent.provider, agent.coding_workflow) == (
        "old-model",
        "old-provider",
        "coupled-v1",
    )


def test_route_cannot_race_deferred_build_snapshot(monkeypatch):
    snapshot_entered = threading.Event()
    release_snapshot = threading.Event()
    route_attempted = threading.Event()

    def release_build_if_locked(owner):
        if owner is not None:
            release_snapshot.set()

    transition_lock = _TrackingRLock(route_attempted, release_build_if_locked)
    initial_override = {"model": "old-model", "provider": "old-provider"}

    class BlockingSession(dict):
        snapshot_blocked = False

        def get(self, key, default=None):
            if key == "model_override" and not self.snapshot_blocked:
                self.snapshot_blocked = True
                snapshot_entered.set()
                assert release_snapshot.wait(timeout=3), "build snapshot was not released"
            return super().get(key, default)

    ready = threading.Event()
    session = BlockingSession(
        agent=None,
        agent_build_lock=threading.Lock(),
        agent_error=None,
        agent_ready=ready,
        coding_workflow="coupled-v1",
        history=[],
        history_lock=threading.Lock(),
        model_override=dict(initial_override),
        profile_home=None,
        route_lock=transition_lock,
        transition_lock=transition_lock,
        running=False,
        session_key="stored-1",
    )
    captured = {}

    class FakeWorker:
        def __init__(self, *_args, **_kwargs):
            pass

        def close(self):
            pass

    def fake_make_agent(_sid, _key, **kwargs):
        captured.update(kwargs)
        agent = _agent("coupled-v1")
        agent.model = kwargs["model_override"]["model"]
        agent.provider = kwargs["model_override"]["provider"]
        return agent

    def mutate_unbuilt_route(_sid, current, _raw, **kwargs):
        assert kwargs["defer_session_commit"] is True
        current["model_override"] = {
            "model": "gpt-5.6-sol",
            "provider": "custom:sudo",
        }
        release_snapshot.set()
        return {
            "value": "gpt-5.6-sol",
            "warning": "",
            "confirm_required": False,
            "scope": "session",
        }

    monkeypatch.setattr(server, "_sessions", {"runtime-1": session})
    monkeypatch.setattr(server, "_set_session_context", lambda _key: [])
    monkeypatch.setattr(server, "_clear_session_context", lambda _tokens: None)
    monkeypatch.setattr(server, "_make_agent", fake_make_agent)
    monkeypatch.setattr(server, "_SlashWorker", FakeWorker)
    monkeypatch.setattr(server, "_attach_worker", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(server, "_initialize_rollover_state", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(server, "_wire_callbacks", lambda _sid: None)
    monkeypatch.setattr(server, "_emit", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(server, "_session_info", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(server, "_lazy_resume_info", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(server, "_start_notification_poller", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(server, "_notify_session_boundary", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(server, "_probe_config_health", lambda *_args: None)
    monkeypatch.setattr(server, "_schedule_mcp_late_refresh", lambda *_args: None)
    monkeypatch.setattr(server, "_config_model_target", lambda: ("", ""))
    monkeypatch.setattr(server, "_apply_model_switch", mutate_unbuilt_route)
    monkeypatch.setattr(server, "_persist_route_runtime_strict", lambda _session: None)

    route_response = {}

    def switch_route():
        route_response.update(
            server._methods["config.set"](
                "route",
                {
                    "session_id": "runtime-1",
                    "key": "route",
                    "value": {
                        "model": "gpt-5.6-sol",
                        "provider": "custom:sudo",
                        "coding_workflow": "hybrid-v1",
                    },
                },
            )
        )

    build_thread = threading.Thread(
        target=server._start_agent_build,
        args=("runtime-1", session),
        name="build-start",
    )
    route_thread = threading.Thread(target=switch_route, name="route-rpc")
    build_thread.start()
    assert snapshot_entered.wait(timeout=3), "deferred build did not reach its route snapshot"
    route_thread.start()
    assert route_attempted.wait(timeout=3), "route did not reach the transition lock"
    build_thread.join(timeout=3)
    route_thread.join(timeout=3)
    assert ready.wait(timeout=3), "deferred build did not finish"

    assert not build_thread.is_alive()
    assert not route_thread.is_alive()
    assert route_response["error"]["code"] == 4009
    assert captured["model_override"] == initial_override
    assert captured["coding_workflow"] == "coupled-v1"


def test_route_commit_is_atomic_emits_once_and_adds_no_marker(monkeypatch):
    agent = _RouteAgent()
    history = [{"role": "user", "content": "keep me byte-stable"}]
    session = {
        "agent": agent,
        "coding_workflow": "coupled-v1",
        "history": history.copy(),
        "history_lock": __import__("threading").Lock(),
        "running": False,
        "session_key": "stored-1",
    }
    monkeypatch.setattr(server, "_sessions", {"runtime-1": session})
    monkeypatch.setattr(server, "_apply_model_switch", _fake_route_switch)
    committed = []
    monkeypatch.setattr(server, "_persist_route_runtime_strict", lambda current: committed.append((current["agent"].model, current["coding_workflow"])))
    monkeypatch.setattr(server, "_restart_slash_worker", lambda *_args: None)
    monkeypatch.setattr(server, "_session_info", lambda current_agent, current: {"model": current_agent.model, "coding_workflow": current["coding_workflow"]})
    emitted = []
    monkeypatch.setattr(server, "_emit", lambda event, sid, payload: emitted.append((event, sid, payload)))

    response = server._methods["config.set"](
        "r1",
        {
            "session_id": "runtime-1",
            "key": "route",
            "value": {
                "model": "gpt-5.6-sol",
                "provider": "custom:sudo",
                "coding_workflow": "hybrid-v1",
            },
        },
    )

    assert "error" not in response
    assert committed == [("gpt-5.6-sol", "hybrid-v1")]
    assert agent.coding_workflow == session["coding_workflow"] == "hybrid-v1"
    assert emitted == [("session.info", "runtime-1", {"model": "gpt-5.6-sol", "coding_workflow": "hybrid-v1"})]
    assert session["history"] == history


def test_hybrid_route_rejects_non_controller_model_without_mutation(monkeypatch):
    agent = _RouteAgent()
    session = {"agent": agent, "coding_workflow": "coupled-v1", "running": False}
    monkeypatch.setattr(server, "_sessions", {"runtime-1": session})

    response = server._methods["config.set"](
        "r1",
        {
            "session_id": "runtime-1",
            "key": "route",
            "value": {
                "model": "glm-5.2-heavy",
                "provider": "custom:sudo",
                "coding_workflow": "hybrid-v1",
            },
        },
    )

    assert response["error"]["code"] == 4002
    assert (agent.model, agent.provider, agent.coding_workflow) == (
        "old-model",
        "old-provider",
        "coupled-v1",
    )


def test_route_persist_failure_rolls_back_model_provider_and_workflow(monkeypatch):
    agent = _RouteAgent()
    session = {
        "agent": agent,
        "coding_workflow": "coupled-v1",
        "model_override": {"model": "old-model", "provider": "old-provider"},
        "running": False,
        "session_key": "stored-1",
    }
    monkeypatch.setattr(server, "_sessions", {"runtime-1": session})
    monkeypatch.setattr(server, "_apply_model_switch", _fake_route_switch)
    monkeypatch.setattr(server, "_persist_route_runtime_strict", lambda _session: (_ for _ in ()).throw(OSError("disk full")))
    monkeypatch.setattr(server, "_restart_slash_worker", lambda *_args: None)
    emitted = []
    monkeypatch.setattr(server, "_emit", lambda *args: emitted.append(args))

    response = server._methods["config.set"](
        "r1",
        {
            "session_id": "runtime-1",
            "key": "route",
            "value": {
                "model": "gpt-5.6-sol",
                "provider": "custom:sudo",
                "coding_workflow": "hybrid-v1",
            },
        },
    )

    assert response["error"]["code"] == 5001
    assert (agent.model, agent.provider, agent.coding_workflow) == (
        "old-model",
        "old-provider",
        "coupled-v1",
    )
    assert session["coding_workflow"] == "coupled-v1"
    assert session["model_override"] == {"model": "old-model", "provider": "old-provider"}
    assert emitted == []


def test_route_rejects_while_deferred_agent_build_is_inflight(monkeypatch):
    session = {
        "agent": None,
        "agent_build_started": True,
        "agent_ready": threading.Event(),
        "coding_workflow": "coupled-v1",
        "running": False,
    }
    monkeypatch.setattr(server, "_sessions", {"runtime-1": session})

    response = server._methods["config.set"](
        "r1",
        {
            "session_id": "runtime-1",
            "key": "route",
            "value": {
                "model": "gpt-5.6-sol",
                "provider": "custom:sudo",
                "coding_workflow": "hybrid-v1",
            },
        },
    )

    assert response["error"]["code"] == 4009
    assert session["coding_workflow"] == "coupled-v1"
    assert "model_override" not in session


def test_route_rejects_malformed_existing_workflow_without_model_mutation(monkeypatch):
    agent = _RouteAgent()
    session = {"agent": agent, "coding_workflow": "bogus", "running": False}
    monkeypatch.setattr(server, "_sessions", {"runtime-1": session})
    apply_switch = Mock()
    monkeypatch.setattr(server, "_apply_model_switch", apply_switch)

    response = server._methods["config.set"](
        "r1",
        {
            "session_id": "runtime-1",
            "key": "route",
            "value": {
                "model": "gpt-5.6-sol",
                "provider": "custom:sudo",
                "coding_workflow": "hybrid-v1",
            },
        },
    )

    assert response["error"]["code"] == 5001
    apply_switch.assert_not_called()
    assert (agent.model, agent.provider, agent.coding_workflow) == (
        "old-model",
        "old-provider",
        "coupled-v1",
    )
