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
        self.reasoning_config = {"enabled": True, "effort": "low"}
        self.service_tier = None
        self._cached_system_prompt = "stable cached system prompt"
        self._primary_runtime = {
            "model": self.model,
            "provider": self.provider,
            "reasoning_config": dict(self.reasoning_config),
        }
        self.coding_workflow = "coupled-v1"

    def switch_model(self, *, new_model, new_provider, api_key, base_url, api_mode):
        self.model = new_model
        self.provider = new_provider
        self.api_key = api_key
        self.base_url = base_url
        self.api_mode = api_mode
        self.reasoning_config = {"enabled": True, "effort": "medium"}
        self._cached_system_prompt = None
        self._primary_runtime = {
            "model": new_model,
            "provider": new_provider,
            "reasoning_config": dict(self.reasoning_config),
        }


def test_live_hybrid_route_installs_and_persists_xhigh_reasoning(monkeypatch):
    updates = []

    class RouteDB:
        def get_session(self, session_key):
            assert session_key == "stored-1"
            return {"model_config": json.dumps({"retained": "value"})}

        def update_session_meta(self, session_key, model_config, model):
            updates.append((session_key, json.loads(model_config), model))

    agent = _RouteAgent()
    agent._session_db = RouteDB()
    session = {
        "agent": agent,
        "coding_workflow": "coupled-v1",
        "create_reasoning_override": {"enabled": True, "effort": "low"},
        "running": False,
        "session_key": "stored-1",
    }
    monkeypatch.setattr(server, "_sessions", {"runtime-1": session})
    monkeypatch.setattr(server, "_apply_model_switch", _fake_route_switch)
    monkeypatch.setattr(server, "_restart_slash_worker", lambda *_args: None)
    monkeypatch.setattr(server, "_session_info", lambda *_args: {})
    monkeypatch.setattr(server, "_emit", lambda *_args: None)

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

    expected_reasoning = {"enabled": True, "effort": "xhigh"}
    assert "error" not in response, response
    assert agent.reasoning_config == expected_reasoning
    assert session["create_reasoning_override"] == expected_reasoning
    assert agent._primary_runtime["reasoning_config"] == expected_reasoning
    assert updates == [
        (
            "stored-1",
            {
                "retained": "value",
                "model": "gpt-5.6-sol",
                "provider": "custom:sudo",
                "base_url": "https://new.invalid/v1",
                "api_mode": "chat_completions",
                "reasoning_config": expected_reasoning,
                "coding_workflow": "hybrid-v1",
            },
            "gpt-5.6-sol",
        )
    ]


def test_workflow_only_hybrid_switch_skips_model_switch_and_preserves_cache(monkeypatch):
    agent = _RouteAgent()
    agent.model = "gpt-5.6-sol"
    agent.provider = "custom:sudo"
    agent._primary_runtime.update(
        {"model": agent.model, "provider": agent.provider}
    )
    cache_before = agent._cached_system_prompt
    history = [{"role": "user", "content": "keep history stable"}]
    session = {
        "agent": agent,
        "coding_workflow": "coupled-v1",
        "history": list(history),
        "running": False,
        "session_key": "stored-1",
    }
    apply_switch = Mock(side_effect=AssertionError("workflow-only switch called model switch"))
    persist_route = Mock()
    monkeypatch.setattr(server, "_sessions", {"runtime-1": session})
    monkeypatch.setattr(server, "_apply_model_switch", apply_switch)
    monkeypatch.setattr(server, "_persist_route_runtime_strict", persist_route)
    monkeypatch.setattr(server, "_restart_slash_worker", lambda *_args: None)
    monkeypatch.setattr(server, "_session_info", lambda *_args: {})
    monkeypatch.setattr(server, "_emit", lambda *_args: None)

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

    expected_reasoning = {"enabled": True, "effort": "xhigh"}
    assert "error" not in response, response
    apply_switch.assert_not_called()
    persist_route.assert_called_once_with(session)
    assert agent._cached_system_prompt is cache_before
    assert session["history"] == history
    assert agent.reasoning_config == expected_reasoning
    assert agent._primary_runtime["reasoning_config"] == expected_reasoning
    assert agent.coding_workflow == session["coding_workflow"] == "hybrid-v1"


def test_named_custom_transport_matches_durable_hybrid_route_without_cache_break(
    monkeypatch,
):
    from hermes_cli import runtime_provider

    sudo_url = "https://sudo.example/v1"
    monkeypatch.setattr(
        runtime_provider,
        "load_config",
        lambda: {"providers": {"sudo": {"api": sudo_url}}},
    )
    agent = _RouteAgent()
    agent.model = "gpt-5.6-sol"
    agent.provider = "custom"
    agent.requested_provider = "custom:sudo"
    agent.base_url = sudo_url
    agent._primary_runtime.update(
        {
            "model": agent.model,
            "provider": agent.provider,
            "requested_provider": agent.requested_provider,
            "base_url": agent.base_url,
        }
    )
    cache_before = agent._cached_system_prompt
    session = {
        "agent": agent,
        "coding_workflow": "coupled-v1",
        "running": False,
        "session_key": "stored-1",
    }
    apply_switch = Mock(
        side_effect=AssertionError("same named-custom route called model switch")
    )
    monkeypatch.setattr(server, "_sessions", {"runtime-1": session})
    monkeypatch.setattr(server, "_apply_model_switch", apply_switch)
    monkeypatch.setattr(server, "_persist_route_runtime_strict", Mock())
    monkeypatch.setattr(server, "_restart_slash_worker", lambda *_args: None)
    monkeypatch.setattr(server, "_session_info", lambda *_args: {})
    monkeypatch.setattr(server, "_emit", lambda *_args: None)

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

    assert "error" not in response, response
    apply_switch.assert_not_called()
    assert agent.provider == "custom"
    assert agent.requested_provider == "custom:sudo"
    assert agent._cached_system_prompt is cache_before
    assert session["model_override"]["provider"] == "custom:sudo"
    assert server._runtime_model_config(agent)["provider"] == "custom:sudo"
    runtime_snapshot = server._rollover_runtime_snapshot(agent, session)
    assert runtime_snapshot["provider"] == "custom:sudo"
    assert response["result"]["value"]["provider"] == "custom:sudo"


def test_route_persist_failure_rolls_back_full_runtime_and_emits_nothing(monkeypatch):
    agent = _RouteAgent()
    agent.requested_provider = "custom:old"
    agent.client = object()
    agent._anthropic_client = object()
    agent._anthropic_api_key = "old-anthropic-key"
    agent._anthropic_base_url = None
    agent._is_anthropic_oauth = True
    client_kw_value = object()
    agent._client_kwargs = {"http_client": client_kw_value}
    agent._use_prompt_caching = True
    agent._use_native_cache_layout = False
    agent._fallback_activated = True
    agent._fallback_index = 2
    agent._fallback_chain = [
        {"provider": "fallback-a", "model": "fallback-model-a"},
        {"provider": "fallback-b", "model": "fallback-model-b"},
        {"provider": "fallback-c", "model": "fallback-model-c"},
    ]
    agent._fallback_model = agent._fallback_chain[0]
    agent._rate_limited_until = 9876.5
    agent._consecutive_stale_streams = 7
    transport = object()
    agent._transport_cache = {"chat_completions": transport}
    agent._credential_pool = object()
    agent._config_context_length = 123_456
    agent.context_compressor = SimpleNamespace(
        model="old-model",
        base_url="https://old.invalid/v1",
        api_key="old-key",
        provider="old-provider",
        api_mode="chat_completions",
        context_length=123_456,
        _configured_threshold_percent=0.61,
        _config_threshold_percent=0.62,
        _base_threshold_percent=0.63,
        threshold_percent=0.64,
        threshold_tokens=78_000,
        tail_token_budget=39_000,
        max_summary_tokens=6_000,
        last_prompt_tokens=111,
        last_completion_tokens=22,
        last_total_tokens=133,
        last_real_prompt_tokens=111,
        last_rough_tokens_when_real_prompt_fit=222,
        last_compression_rough_tokens=333,
        awaiting_real_usage_after_compression=True,
        _ineffective_compression_count=4,
        _fallback_compression_streak=5,
        _summary_failure_cooldown_until=4321.5,
        _last_summary_error="compressor failure sentinel",
        _consecutive_timeout_failures=6,
        _cooldown_persist_failed=True,
        _verify_compaction_cleared_threshold=True,
        _last_compression_made_progress=True,
    )
    reasoning_before = dict(agent.reasoning_config)
    cache_before = agent._cached_system_prompt
    primary_before_ref = agent._primary_runtime
    primary_before = json.loads(json.dumps(agent._primary_runtime))
    client_before = agent.client
    anthropic_client_before = agent._anthropic_client
    client_kwargs_before = agent._client_kwargs
    fallback_chain_before = agent._fallback_chain
    fallback_model_before = agent._fallback_model
    transport_cache_before = agent._transport_cache
    credential_pool_before = agent._credential_pool
    compressor_before = agent.context_compressor
    compressor_state_before = dict(vars(compressor_before))
    session = {
        "agent": agent,
        "coding_workflow": "coupled-v1",
        "model_override": {"model": "old-model", "provider": "old-provider"},
        "create_reasoning_override": dict(reasoning_before),
        "running": False,
        "session_key": "stored-1",
    }

    def mutate_every_switch_field(_sid, current, _raw, **kwargs):
        assert kwargs["defer_session_commit"] is True
        switched = current["agent"]
        switched.model = "gpt-5.6-sol"
        switched.provider = "custom"
        switched.requested_provider = "custom:sudo"
        switched.api_key = "new-key"
        switched.base_url = "https://sudo.example/v1"
        switched.api_mode = "codex_responses"
        switched.client = object()
        switched._anthropic_client = object()
        switched._anthropic_api_key = "new-anthropic-key"
        switched._anthropic_base_url = "https://new.invalid"
        switched._is_anthropic_oauth = False
        switched._client_kwargs = {"new": object()}
        switched._use_prompt_caching = False
        switched._use_native_cache_layout = True
        switched._cached_system_prompt = None
        switched.reasoning_config = {"enabled": True, "effort": "medium"}
        switched._primary_runtime = {
            "model": switched.model,
            "provider": switched.provider,
            "requested_provider": switched.requested_provider,
            "reasoning_config": dict(switched.reasoning_config),
        }
        switched._fallback_activated = False
        switched._fallback_index = 0
        switched._fallback_chain = []
        switched._fallback_model = None
        switched._rate_limited_until = 0
        switched._consecutive_stale_streams = 0
        switched._transport_cache.clear()
        switched._credential_pool = object()
        switched._config_context_length = None
        switched._compression_global_threshold = 0.85
        for field, value in {
            "model": switched.model,
            "base_url": switched.base_url,
            "api_key": switched.api_key,
            "provider": switched.provider,
            "api_mode": switched.api_mode,
            "context_length": 64_000,
            "_configured_threshold_percent": 0.85,
            "_config_threshold_percent": 0.85,
            "_base_threshold_percent": 0.85,
            "threshold_percent": 0.85,
            "threshold_tokens": 54_400,
            "tail_token_budget": 27_200,
            "max_summary_tokens": 3_200,
            "last_prompt_tokens": 0,
            "last_completion_tokens": 0,
            "last_total_tokens": 0,
            "last_real_prompt_tokens": 0,
            "last_rough_tokens_when_real_prompt_fit": 0,
            "last_compression_rough_tokens": 0,
            "awaiting_real_usage_after_compression": False,
            "_ineffective_compression_count": 0,
            "_fallback_compression_streak": 0,
            "_summary_failure_cooldown_until": 0.0,
            "_last_summary_error": None,
            "_consecutive_timeout_failures": 0,
            "_cooldown_persist_failed": False,
            "_verify_compaction_cleared_threshold": False,
            "_last_compression_made_progress": False,
        }.items():
            setattr(switched.context_compressor, field, value)
        current["model_override"] = {
            "model": switched.model,
            "provider": "custom:sudo",
        }
        return {
            "value": switched.model,
            "warning": "",
            "confirm_required": False,
            "scope": "session",
        }

    monkeypatch.setattr(server, "_sessions", {"runtime-1": session})
    monkeypatch.setattr(server, "_apply_model_switch", mutate_every_switch_field)
    monkeypatch.setattr(
        server,
        "_persist_route_runtime_strict",
        lambda _session: (_ for _ in ()).throw(OSError("disk full")),
    )
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

    assert response["error"] == {"code": 5001, "message": "disk full"}
    assert (agent.model, agent.provider, agent.coding_workflow) == (
        "old-model",
        "old-provider",
        "coupled-v1",
    )
    assert agent.requested_provider == "custom:old"
    assert agent.reasoning_config == reasoning_before
    assert agent._cached_system_prompt is cache_before
    assert agent.client is client_before
    assert agent._anthropic_client is anthropic_client_before
    assert agent._anthropic_api_key == "old-anthropic-key"
    assert agent._anthropic_base_url is None
    assert agent._is_anthropic_oauth is True
    assert agent._client_kwargs is client_kwargs_before
    assert agent._client_kwargs == {"http_client": client_kw_value}
    assert agent._use_prompt_caching is True
    assert agent._use_native_cache_layout is False
    assert agent._primary_runtime is primary_before_ref
    assert agent._primary_runtime == primary_before
    assert agent._fallback_activated is True
    assert agent._fallback_index == 2
    assert agent._fallback_chain is fallback_chain_before
    assert agent._fallback_model is fallback_model_before
    assert agent._rate_limited_until == 9876.5
    assert agent._consecutive_stale_streams == 7
    assert agent._transport_cache is transport_cache_before
    assert agent._transport_cache == {"chat_completions": transport}
    assert agent._credential_pool is credential_pool_before
    assert agent._config_context_length == 123_456
    assert not hasattr(agent, "_compression_global_threshold")
    assert agent.context_compressor is compressor_before
    assert vars(agent.context_compressor) == compressor_state_before
    assert session["coding_workflow"] == "coupled-v1"
    assert session["model_override"] == {
        "model": "old-model",
        "provider": "old-provider",
    }
    assert session["create_reasoning_override"] == reasoning_before
    assert emitted == []


class _BranchLease:
    def __init__(self):
        self.released = False

    def release(self):
        self.released = True


class _BranchCandidate:
    def __init__(self, runtime_snapshot, session_db):
        self.model = runtime_snapshot["model"]
        self.coding_workflow = runtime_snapshot["coding_workflow"]
        self._session_db = session_db
        self.closed = False
        self.memory_shutdown = False

    def shutdown_memory_provider(self):
        self.memory_shutdown = True

    def close(self):
        self.closed = True


class _BranchDBProxy:
    def __init__(self, db, *, failure_point=None, transition_lock=None):
        self.db = db
        self.failure_point = failure_point
        self.transition_lock = transition_lock
        self.create_calls = 0
        self.append_calls = 0
        self.title_calls = 0
        self.delete_sessions_dirs = []

    def _assert_unlocked(self):
        if self.transition_lock is not None:
            assert not self.transition_lock.held, "slow DB work held the parent transition lock"

    def get_session_title(self, session_id):
        self._assert_unlocked()
        return self.db.get_session_title(session_id)

    def get_next_title_in_lineage(self, title):
        self._assert_unlocked()
        return self.db.get_next_title_in_lineage(title)

    def create_session(self, session_id, **kwargs):
        self._assert_unlocked()
        self.create_calls += 1
        return self.db.create_session(session_id, **kwargs)

    def append_message(self, **kwargs):
        self._assert_unlocked()
        self.append_calls += 1
        result = self.db.append_message(**kwargs)
        if self.failure_point == "append":
            raise RuntimeError("controlled append failure")
        return result

    def set_session_title(self, session_id, title):
        self._assert_unlocked()
        self.title_calls += 1
        result = self.db.set_session_title(session_id, title)
        if self.failure_point == "title":
            raise RuntimeError("controlled title failure")
        return result

    def delete_session(self, session_id, sessions_dir=None):
        self._assert_unlocked()
        self.delete_sessions_dirs.append(sessions_dir)
        return self.db.delete_session(session_id, sessions_dir=sessions_dir)

    def __getattr__(self, name):
        return getattr(self.db, name)


def test_session_branch_uses_parent_profile_scope_and_persists_full_route(
    monkeypatch, tmp_path
):
    from hermes_state import SessionDB

    class TrackingLock:
        def __init__(self):
            self.held = False

        def __enter__(self):
            assert not self.held
            self.held = True
            return self

        def __exit__(self, _exc_type, _exc, _tb):
            self.held = False

    profile_home = tmp_path / "profile"
    profile_home.mkdir()
    durable_db = SessionDB(db_path=profile_home / "state.db")
    parent_key = "profile-parent"
    child_key = "profile-child"
    durable_db.create_session(parent_key, source="desktop", model="gpt-5.6-sol")
    durable_db.set_session_title(parent_key, "parent")
    lock = TrackingLock()
    scoped_db = _BranchDBProxy(durable_db, transition_lock=lock)
    parent_agent = _RouteAgent()
    parent_agent.model = "gpt-5.6-sol"
    parent_agent.provider = "custom"
    parent_agent.requested_provider = "custom:sudo"
    parent_agent.base_url = "https://sudo.example/v1"
    parent_agent.api_key = "profile-secret"
    parent_agent.api_mode = "codex_responses"
    parent_agent.reasoning_config = {"enabled": True, "effort": "xhigh"}
    parent_agent.service_tier = "priority"
    parent_agent.coding_workflow = "hybrid-v1"
    parent_agent._session_db = scoped_db
    original_timestamps = [1_700_000_000.0, 1_700_000_020.0]
    history = [
        {
            "role": "system",
            "content": "internal branch marker",
            "timestamp": original_timestamps[0],
            server.HERMES_INTERNAL_SYSTEM_MARKER_KEY: True,
        },
        {
            "role": "user",
            "content": "branch this",
            "timestamp": original_timestamps[1],
        },
    ]
    parent_cwd = str(tmp_path / "project")
    session = {
        "agent": parent_agent,
        "coding_workflow": "hybrid-v1",
        "history": history,
        "history_lock": threading.RLock(),
        "transition_lock": lock,
        "running": False,
        "session_key": parent_key,
        "source": "desktop",
        "cwd": parent_cwd,
        "profile_home": str(profile_home),
        "cols": 100,
    }
    lease = _BranchLease()
    captured = {}

    def global_db_sentinel():
        raise AssertionError("profile branch touched the launch-profile DB")

    def set_context(key, **kwargs):
        assert server.get_hermes_home_override() == str(profile_home)
        captured["context"] = (key, kwargs)
        return ["context-token"]

    def make_agent(_sid, _key, **kwargs):
        assert not lock.held, "agent construction held the parent transition lock"
        assert server.get_hermes_home_override() == str(profile_home)
        captured["make"] = kwargs
        return _BranchCandidate(kwargs["runtime_snapshot"], kwargs["session_db"])

    def init_session(sid, key, agent, copied_history, **kwargs):
        assert not lock.held, "session initialization held the parent transition lock"
        assert server.get_hermes_home_override() == str(profile_home)
        captured["init"] = kwargs
        server._sessions[sid] = {
            "agent": agent,
            "session_key": key,
            "history": copied_history,
            "model_override": None,
            "coding_workflow": agent.coding_workflow,
        }

    monkeypatch.setattr(server, "_sessions", {"runtime-1": session})
    monkeypatch.setattr(server, "_get_db", global_db_sentinel)
    monkeypatch.setattr(server, "_new_session_key", lambda: child_key)
    monkeypatch.setattr(
        server, "_claim_active_session_slot", lambda *_args, **_kwargs: (lease, None)
    )
    monkeypatch.setattr(server, "_session_cwd", lambda current: current["cwd"])
    monkeypatch.setattr(server, "_make_agent", make_agent)
    monkeypatch.setattr(server, "_init_session", init_session)
    monkeypatch.setattr(server, "_set_session_context", set_context)
    monkeypatch.setattr(server, "_clear_session_context", lambda tokens: tokens == ["context-token"])

    response = server._methods["session.branch"](
        "r1", {"session_id": "runtime-1", "name": "profile branch"}
    )

    assert "error" not in response, response
    child_sid = response["result"]["session_id"]
    runtime_snapshot = captured["make"]["runtime_snapshot"]
    assert captured["make"]["session_db"] is scoped_db
    assert captured["make"]["platform_override"] == "desktop"
    assert captured["make"]["reasoning_config_override"] == {
        "enabled": True,
        "effort": "xhigh",
    }
    assert captured["make"]["service_tier_override"] == "priority"
    assert captured["context"] == (
        child_key,
        {
            "cwd": parent_cwd,
            "ui_session_id": child_sid,
            "coding_workflow": "hybrid-v1",
        },
    )
    assert captured["init"] == {
        "cols": 100,
        "cwd": parent_cwd,
        "session_db": scoped_db,
        "source": "desktop",
        "profile_home": profile_home,
    }
    assert runtime_snapshot["provider"] == "custom:sudo"
    child = server._sessions[child_sid]
    assert child["model_override"] == {
        "model": "gpt-5.6-sol",
        "provider": "custom:sudo",
        "base_url": "https://sudo.example/v1",
        "api_key": "profile-secret",
        "api_mode": "codex_responses",
    }
    assert child["create_reasoning_override"] == parent_agent.reasoning_config
    assert child["create_service_tier_override"] == "priority"
    assert child["coding_workflow"] == "hybrid-v1"
    assert child["active_session_lease"] is lease
    row = durable_db.get_session(child_key)
    assert row is not None
    assert row["source"] == "desktop"
    assert row["cwd"] == parent_cwd
    model_config = json.loads(row["model_config"])
    assert model_config == {
        "_branched_from": parent_key,
        "model": "gpt-5.6-sol",
        "provider": "custom:sudo",
        "base_url": "https://sudo.example/v1",
        "api_mode": "codex_responses",
        "reasoning_config": {"enabled": True, "effort": "xhigh"},
        "service_tier": "priority",
        "coding_workflow": "hybrid-v1",
    }
    assert "api_key" not in model_config
    copied = durable_db.get_messages(child_key)
    assert [message["timestamp"] for message in copied] == original_timestamps
    assert bool(copied[0]["internal_system_marker"])
    assert durable_db.get_session_title(child_key) == "profile branch"
    assert not lease.released
    durable_db.close()


def test_session_branch_make_agent_failure_writes_nothing_and_closes_owned_db(
    monkeypatch, tmp_path
):
    import hermes_state

    class OwnedDB:
        def __init__(self):
            self.closed = False
            self.create_calls = 0
            self.append_calls = 0
            self.title_calls = 0

        def create_session(self, *_args, **_kwargs):
            self.create_calls += 1

        def append_message(self, **_kwargs):
            self.append_calls += 1

        def set_session_title(self, *_args):
            self.title_calls += 1

        def close(self):
            self.closed = True

    profile_home = tmp_path / "owned-profile"
    owned_db = OwnedDB()
    lease = _BranchLease()
    parent_agent = _RouteAgent()
    parent_agent.model = "gpt-5.6-sol"
    parent_agent.provider = "custom:sudo"
    parent_agent.coding_workflow = "hybrid-v1"
    parent = {
        "agent": parent_agent,
        "coding_workflow": "hybrid-v1",
        "history": [{"role": "user", "content": "branch"}],
        "history_lock": threading.RLock(),
        "session_key": "parent-key",
        "source": "desktop",
        "cwd": str(tmp_path / "project"),
        "profile_home": str(profile_home),
        "cols": 80,
    }

    monkeypatch.setattr(server, "_sessions", {"parent-sid": parent})
    monkeypatch.setattr(
        server,
        "_get_db",
        lambda: (_ for _ in ()).throw(AssertionError("global DB touched")),
    )
    monkeypatch.setattr(server, "_new_session_key", lambda: "child-key")
    monkeypatch.setattr(
        server, "_claim_active_session_slot", lambda *_args, **_kwargs: (lease, None)
    )
    monkeypatch.setattr(hermes_state, "SessionDB", lambda db_path: owned_db)
    monkeypatch.setattr(server, "_set_session_context", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(server, "_clear_session_context", lambda _tokens: None)

    def fail_make_agent(_sid, _key, **kwargs):
        assert kwargs["session_db"] is owned_db
        assert server.get_hermes_home_override() == str(profile_home)
        raise RuntimeError("controlled make-agent failure")

    monkeypatch.setattr(server, "_make_agent", fail_make_agent)

    response = server._methods["session.branch"](
        "r2", {"session_id": "parent-sid", "name": "never durable"}
    )

    assert response["error"]["code"] == 5000
    assert (owned_db.create_calls, owned_db.append_calls, owned_db.title_calls) == (0, 0, 0)
    assert owned_db.closed
    assert lease.released
    assert server._sessions == {"parent-sid": parent}


@pytest.mark.parametrize("failure_point", ["append", "title", "init"])
def test_session_branch_post_create_failure_compensates_durable_and_live_state(
    monkeypatch, tmp_path, failure_point
):
    from hermes_state import SessionDB

    profile_home = tmp_path / failure_point
    profile_home.mkdir()
    durable_db = SessionDB(db_path=profile_home / "state.db")
    parent_key = f"parent-{failure_point}"
    child_key = f"child-{failure_point}"
    durable_db.create_session(parent_key, source="desktop", model="gpt-5.6-sol")
    failing_db = _BranchDBProxy(durable_db, failure_point=failure_point)
    parent_agent = _RouteAgent()
    parent_agent.model = "gpt-5.6-sol"
    parent_agent.provider = "custom:sudo"
    parent_agent.coding_workflow = "hybrid-v1"
    parent_agent._session_db = failing_db
    parent = {
        "agent": parent_agent,
        "coding_workflow": "hybrid-v1",
        "history": [
            {"role": "user", "content": "branch", "timestamp": 100.0},
            {"role": "assistant", "content": "ready", "timestamp": 200.0},
        ],
        "history_lock": threading.RLock(),
        "session_key": parent_key,
        "source": "desktop",
        "cwd": str(tmp_path / "project"),
        "profile_home": str(profile_home),
        "cols": 90,
    }
    lease = _BranchLease()
    candidate_holder = {}
    init_called = []
    partial_worker = SimpleNamespace(closed=False)
    partial_worker.close = lambda: setattr(partial_worker, "closed", True)
    partial_stop = threading.Event()

    def make_agent(_sid, _key, **kwargs):
        candidate = _BranchCandidate(kwargs["runtime_snapshot"], kwargs["session_db"])
        candidate_holder["agent"] = candidate
        return candidate

    def init_session(sid, key, agent, history, **_kwargs):
        init_called.append(sid)
        if failure_point == "init":
            server._sessions[sid] = {
                "agent": agent,
                "session_key": key,
                "history": history,
                "history_lock": threading.RLock(),
                "slash_worker": partial_worker,
                "_notif_stop": partial_stop,
            }
            raise RuntimeError("controlled init failure")
        raise AssertionError("init reached after a durable write failure")

    monkeypatch.setattr(server, "_sessions", {"parent-sid": parent})
    monkeypatch.setattr(
        server,
        "_get_db",
        lambda: (_ for _ in ()).throw(AssertionError("global DB touched")),
    )
    monkeypatch.setattr(server, "_new_session_key", lambda: child_key)
    monkeypatch.setattr(
        server, "_claim_active_session_slot", lambda *_args, **_kwargs: (lease, None)
    )
    monkeypatch.setattr(server, "_make_agent", make_agent)
    monkeypatch.setattr(server, "_init_session", init_session)
    monkeypatch.setattr(server, "_set_session_context", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(server, "_clear_session_context", lambda _tokens: None)

    response = server._methods["session.branch"](
        "r3", {"session_id": "parent-sid", "name": "will roll back"}
    )

    assert "error" in response
    assert candidate_holder["agent"].closed
    assert candidate_holder["agent"].memory_shutdown
    assert candidate_holder["agent"]._end_session_on_close is False
    assert durable_db.get_session(child_key) is None
    assert durable_db.get_messages(child_key) == []
    assert failing_db.delete_sessions_dirs == [None]
    assert lease.released
    assert server._sessions == {"parent-sid": parent}
    if failure_point == "init":
        assert init_called
        assert partial_worker.closed
        assert partial_stop.is_set()
    else:
        assert not init_called
    assert durable_db.get_session(parent_key) is not None
    durable_db.close()


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
