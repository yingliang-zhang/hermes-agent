"""Focused REST tests for Hybrid v1 coding-workflow routing (Slice 2).

Pins the ``/api/model/set`` + ``/api/model/info`` + ``/api/model/options``
contract for the profile-global coding-workflow default:

* ``coding_workflow`` is optional on ``scope=main`` and validated against the
  frozen allowlist (``coupled-v1`` / ``hybrid-v1``); unknown values fail closed
  with 400 and leave config untouched.
* ``hybrid-v1`` forces the real Sol controller (``custom:sudo`` /
  ``gpt-5.6-sol``) and persists ``coding_workflow.default`` in the SAME
  ``save_config`` call as the model provider/model (atomic global save).
* ``coupled-v1`` preserves the selected ordinary provider/model.
* Auxiliary task assignments are never mutated by a main-slot workflow switch.
* ``/api/model/info`` and ``/api/model/options`` expose the current
  ``coding_workflow`` plus the workflow-preset metadata, and Hybrid is NOT
  surfaced as a virtual model provider.
"""

import yaml

import pytest


@pytest.fixture
def client(tmp_path, monkeypatch, _isolate_hermes_home):
    try:
        from starlette.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi/starlette not installed")

    from hermes_constants import get_hermes_home
    import hermes_state
    from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN

    home = get_hermes_home()
    (home / "config.yaml").write_text("model:\n  provider: openrouter\n  default: start/model\n", encoding="utf-8")
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", home / "state.db")
    c = TestClient(app)
    c.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    return c


def _cfg():
    from hermes_constants import get_hermes_home

    return yaml.safe_load((get_hermes_home() / "config.yaml").read_text()) or {}


def _hybrid_config():
    return {
        "model": {
            "provider": "custom:sudo",
            "default": "gpt-5.6-sol",
            "base_url": "https://controller.example/v1",
            "api_key": "sk-controller",
            "temperature": 0.25,
        },
        "coding_workflow": {
            "default": "hybrid-v1",
            "review_policy": "keep-review-policy",
        },
        "providers": {
            "keep-provider": {
                "base_url": "https://keep.example/v1",
                "api_key": "sk-keep-provider",
                "models": {"keep/model": {}},
            }
        },
        "auxiliary": {
            "vision": {"provider": "nous", "model": "pinned-vision"},
        },
    }


def _write_hybrid_config(path=None, *, providers=None):
    from hermes_constants import get_hermes_home

    cfg = _hybrid_config()
    if providers:
        cfg["providers"].update(providers)
    config_path = path or (get_hermes_home() / "config.yaml")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return cfg


def _spy_saved_configs(monkeypatch):
    from copy import deepcopy

    import hermes_cli.web_server as web_server

    real_save_config = web_server.save_config
    saves = []

    def save_config_spy(cfg, *args, **kwargs):
        saves.append((deepcopy(cfg), deepcopy(kwargs)))
        return real_save_config(cfg, *args, **kwargs)

    monkeypatch.setattr(web_server, "save_config", save_config_spy)
    return saves


class TestModelSetCodingWorkflow:
    def test_invalid_coding_workflow_rejected(self, client):
        before = _cfg()
        resp = client.post(
            "/api/model/set",
            json={
                "scope": "main",
                "provider": "openrouter",
                "model": "test/model-1",
                "coding_workflow": "bogus-v9",
                "confirm_expensive_model": True,
            },
        )
        assert resp.status_code == 400
        # Config is untouched — the invalid request was a no-op.
        assert _cfg() == before

    def test_hybrid_contradictory_route_rejected_without_write(self, client):
        before = _cfg()
        resp = client.post(
            "/api/model/set",
            json={
                "scope": "main",
                "provider": "custom:sudo",
                "model": "gpt-5.5",
                "coding_workflow": "hybrid-v1",
            },
        )
        assert resp.status_code == 400
        assert _cfg() == before

    def test_hybrid_forces_sol_controller_and_persists_workflow(self, client):
        resp = client.post(
            "/api/model/set",
            json={
                "scope": "main",
                "provider": "custom:sudo",
                "model": "gpt-5.6-sol",
                "coding_workflow": "hybrid-v1",
            },
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body.get("coding_workflow") == "hybrid-v1"
        assert body.get("provider") == "custom:sudo"
        assert body.get("model") == "gpt-5.6-sol"

        cfg = _cfg()
        # Atomic global save: model + workflow land in one config write.
        assert cfg["model"]["provider"] == "custom:sudo"
        assert cfg["model"]["default"] == "gpt-5.6-sol"
        assert cfg["coding_workflow"]["default"] == "hybrid-v1"

    def test_coupled_preserves_selected_ordinary_model(self, client):
        resp = client.post(
            "/api/model/set",
            json={
                "scope": "main",
                "provider": "openrouter",
                "model": "test/model-1",
                "coding_workflow": "coupled-v1",
                "confirm_expensive_model": True,
            },
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body.get("coding_workflow") == "coupled-v1"
        assert body.get("provider") == "openrouter"
        assert body.get("model") == "test/model-1"

        cfg = _cfg()
        assert cfg["model"]["provider"] == "openrouter"
        assert cfg["model"]["default"] == "test/model-1"
        assert cfg.get("coding_workflow", {}).get("default") == "coupled-v1"

    def test_ordinary_main_write_without_workflow_forces_coupled(self, client):
        from hermes_constants import get_hermes_home

        (get_hermes_home() / "config.yaml").write_text(
            "model:\n  provider: custom:sudo\n  default: gpt-5.6-sol\n"
            "coding_workflow:\n  default: hybrid-v1\n",
            encoding="utf-8",
        )
        resp = client.post(
            "/api/model/set",
            json={
                "scope": "main",
                "provider": "openrouter",
                "model": "test/model-1",
                "confirm_expensive_model": True,
            },
        )
        assert resp.status_code == 200, resp.text
        assert resp.json().get("coding_workflow") == "coupled-v1"
        cfg = _cfg()
        assert cfg["model"]["provider"] == "openrouter"
        assert cfg["model"]["default"] == "test/model-1"
        assert cfg["coding_workflow"]["default"] == "coupled-v1"

    def test_auxiliary_unchanged_on_main_workflow_switch(self, client, tmp_path, monkeypatch):
        from hermes_constants import get_hermes_home

        home = get_hermes_home()
        (home / "config.yaml").write_text(
            "model:\n  provider: openrouter\n  default: start/model\n"
            "auxiliary:\n  vision:\n    provider: nous\n    model: pinned-vision\n"
            "  mcp:\n    provider: openrouter\n    model: pinned-mcp\n",
            encoding="utf-8",
        )
        resp = client.post(
            "/api/model/set",
            json={
                "scope": "main",
                "provider": "custom:sudo",
                "model": "gpt-5.6-sol",
                "coding_workflow": "hybrid-v1",
            },
        )
        assert resp.status_code == 200, resp.text
        cfg = _cfg()
        aux = cfg.get("auxiliary", {})
        # Auxiliary pins survive the main+workflow switch unchanged.
        assert aux["vision"]["provider"] == "nous"
        assert aux["vision"]["model"] == "pinned-vision"
        assert aux["mcp"]["provider"] == "openrouter"
        assert aux["mcp"]["model"] == "pinned-mcp"



class TestSiblingMainModelWrites:
    def test_custom_endpoint_make_default_forces_coupled_in_single_save(
        self, client, monkeypatch
    ):
        _write_hybrid_config()
        saves = _spy_saved_configs(monkeypatch)

        resp = client.post(
            "/api/providers/custom-endpoints",
            json={
                "id": "ordinary-proxy",
                "name": "Ordinary Proxy",
                "base_url": "https://ordinary.example/v1",
                "model": "ordinary/model",
                "api_key": "sk-ordinary",
                "make_default": True,
            },
        )

        assert resp.status_code == 200, resp.text
        assert len(saves) == 1
        saved, _save_kwargs = saves[0]
        assert saved["model"]["provider"] == "ordinary-proxy"
        assert saved["model"]["default"] == "ordinary/model"
        assert saved["model"]["api_key"] == "sk-ordinary"
        assert saved["coding_workflow"]["default"] == "coupled-v1"
        assert saved["coding_workflow"]["review_policy"] == "keep-review-policy"
        assert saved["model"]["temperature"] == 0.25
        assert saved["providers"]["keep-provider"]["api_key"] == "sk-keep-provider"
        assert saved["auxiliary"]["vision"]["model"] == "pinned-vision"
        assert _cfg()["coding_workflow"]["default"] == "coupled-v1"

    def test_custom_endpoint_activate_forces_coupled_in_single_save(
        self, client, monkeypatch
    ):
        _write_hybrid_config(
            providers={
                "ordinary-proxy": {
                    "name": "Ordinary Proxy",
                    "base_url": "https://ordinary.example/v1",
                    "model": "ordinary/model",
                    "api_key": "sk-ordinary",
                    "models": {"ordinary/model": {}},
                }
            }
        )
        saves = _spy_saved_configs(monkeypatch)

        resp = client.post("/api/providers/custom-endpoints/ordinary-proxy/activate", json={})

        assert resp.status_code == 200, resp.text
        assert len(saves) == 1
        saved, _save_kwargs = saves[0]
        assert saved["model"]["provider"] == "ordinary-proxy"
        assert saved["model"]["default"] == "ordinary/model"
        assert saved["model"]["base_url"] == "https://ordinary.example/v1"
        assert saved["model"]["api_key"] == "sk-ordinary"
        assert saved["coding_workflow"]["default"] == "coupled-v1"
        assert saved["coding_workflow"]["review_policy"] == "keep-review-policy"
        assert saved["providers"]["keep-provider"]["api_key"] == "sk-keep-provider"
        assert _cfg()["coding_workflow"]["default"] == "coupled-v1"

    def test_flat_config_model_change_forces_coupled_and_preserves_siblings(
        self, client, monkeypatch
    ):
        _write_hybrid_config()
        saves = _spy_saved_configs(monkeypatch)
        web_config = client.get("/api/config").json()
        web_config["model"] = "anthropic/claude-sonnet-4.6"

        resp = client.put("/api/config", json={"config": web_config})

        assert resp.status_code == 200, resp.text
        assert len(saves) == 1
        saved, _save_kwargs = saves[0]
        assert saved["model"]["provider"] == "openrouter"
        assert saved["model"]["default"] == "anthropic/claude-sonnet-4.6"
        assert saved["model"]["temperature"] == 0.25
        assert saved["coding_workflow"] == {
            "default": "coupled-v1",
            "review_policy": "keep-review-policy",
        }
        assert saved["providers"]["keep-provider"]["api_key"] == "sk-keep-provider"
        assert saved["auxiliary"]["vision"]["provider"] == "nous"
        assert saved["auxiliary"]["vision"]["model"] == "pinned-vision"
        persisted = _cfg()
        assert persisted["model"]["default"] == "anthropic/claude-sonnet-4.6"
        assert persisted["coding_workflow"]["default"] == "coupled-v1"

    def test_flat_config_bare_string_model_change_forces_coupled(
        self, client, monkeypatch
    ):
        """A bare-string on-disk model with Hybrid must force coupled-v1
        when the model changes through the flat Config page — the bare
        string path must not bypass _apply_main_route_assignment."""
        from hermes_constants import get_hermes_home

        home = get_hermes_home()
        cfg = {
            "model": "gpt-5.6-sol",
            "coding_workflow": {"default": "hybrid-v1"},
        }
        (home / "config.yaml").write_text(
            yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8"
        )
        saves = _spy_saved_configs(monkeypatch)

        web_config = client.get("/api/config").json()
        web_config["model"] = "anthropic/claude-sonnet-4.6"

        resp = client.put("/api/config", json={"config": web_config})

        assert resp.status_code == 200, resp.text
        assert len(saves) == 1
        saved, _ = saves[0]
        # The bare-string model was upgraded to a dict and the workflow
        # was forced to coupled-v1 — not preserving hybrid-v1.
        assert isinstance(saved["model"], dict)
        assert saved["model"]["default"] == "anthropic/claude-sonnet-4.6"
        assert saved["coding_workflow"]["default"] == "coupled-v1"
        persisted = _cfg()
        assert persisted["coding_workflow"]["default"] == "coupled-v1"

    def test_put_config_cannot_write_coding_workflow(
        self, client, monkeypatch
    ):
        """The generic PUT /api/config must not bypass the validated
        /api/model/set workflow authority — coding_workflow in the PUT
        body is stripped and the existing disk value is preserved."""
        _write_hybrid_config()
        saves = _spy_saved_configs(monkeypatch)

        web_config = client.get("/api/config").json()
        # Attempt to change workflow through the generic config save
        web_config["coding_workflow"] = {"default": "coupled-v1"}

        resp = client.put("/api/config", json={"config": web_config})

        assert resp.status_code == 200, resp.text
        assert len(saves) == 1
        saved, _ = saves[0]
        # The existing hybrid-v1 from disk is preserved — the incoming
        # coupled-v1 from the PUT body was stripped.
        assert saved["coding_workflow"]["default"] == "hybrid-v1"
        assert _cfg()["coding_workflow"]["default"] == "hybrid-v1"

    def test_profile_model_update_forces_coupled_in_target_profile(
        self, client, monkeypatch
    ):
        from hermes_constants import get_hermes_home

        _write_hybrid_config()
        profile_config = get_hermes_home() / "profiles" / "route-prof" / "config.yaml"
        _write_hybrid_config(profile_config)
        saves = _spy_saved_configs(monkeypatch)

        resp = client.put(
            "/api/profiles/route-prof/model",
            json={"provider": "openrouter", "model": "anthropic/claude-sonnet-4.6"},
        )

        assert resp.status_code == 200, resp.text
        assert len(saves) == 1
        saved, _save_kwargs = saves[0]
        assert saved["model"]["provider"] == "openrouter"
        assert saved["model"]["default"] == "anthropic/claude-sonnet-4.6"
        assert saved["model"]["temperature"] == 0.25
        assert saved["coding_workflow"]["default"] == "coupled-v1"
        assert saved["coding_workflow"]["review_policy"] == "keep-review-policy"
        assert saved["providers"]["keep-provider"]["api_key"] == "sk-keep-provider"
        assert saved["auxiliary"]["vision"]["model"] == "pinned-vision"
        persisted = yaml.safe_load(profile_config.read_text(encoding="utf-8"))
        assert persisted["model"]["provider"] == "openrouter"
        assert persisted["coding_workflow"]["default"] == "coupled-v1"
        assert _cfg()["coding_workflow"]["default"] == "hybrid-v1"

    def test_auxiliary_model_write_does_not_change_hybrid_workflow(
        self, client, monkeypatch
    ):
        _write_hybrid_config()
        saves = _spy_saved_configs(monkeypatch)

        resp = client.post(
            "/api/model/set",
            json={
                "scope": "auxiliary",
                "task": "vision",
                "provider": "openrouter",
                "model": "google/gemini-2.5-flash",
            },
        )

        assert resp.status_code == 200, resp.text
        assert len(saves) == 1
        saved, _save_kwargs = saves[0]
        assert saved["model"]["provider"] == "custom:sudo"
        assert saved["model"]["default"] == "gpt-5.6-sol"
        assert saved["coding_workflow"]["default"] == "hybrid-v1"
        assert saved["coding_workflow"]["review_policy"] == "keep-review-policy"
        assert saved["auxiliary"]["vision"]["provider"] == "openrouter"
        assert saved["auxiliary"]["vision"]["model"] == "google/gemini-2.5-flash"
        assert _cfg()["coding_workflow"]["default"] == "hybrid-v1"

class TestModelInfoExposesWorkflow:
    def test_info_has_coding_workflow_and_presets(self, client):
        client.post(
            "/api/model/set",
            json={
                "scope": "main",
                "provider": "custom:sudo",
                "model": "gpt-5.6-sol",
                "coding_workflow": "hybrid-v1",
            },
        )
        resp = client.get("/api/model/info")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body.get("coding_workflow") == "hybrid-v1"
        presets = body.get("workflow_presets")
        assert isinstance(presets, list) and presets
        ids = {p.get("id") for p in presets}
        assert {"coupled-v1", "hybrid-v1"} <= ids
        hybrid = next(p for p in presets if p["id"] == "hybrid-v1")
        ctrl = hybrid.get("controller") or {}
        assert ctrl.get("provider") == "custom:sudo"
        assert ctrl.get("model") == "gpt-5.6-sol"

    def test_info_defaults_to_coupled_when_absent(self, client):
        resp = client.get("/api/model/info")
        assert resp.status_code == 200, resp.text
        assert resp.json().get("coding_workflow") == "coupled-v1"


class TestModelOptionsExposesWorkflow:
    def test_options_has_coding_workflow_and_presets(self, client):
        resp = client.get("/api/model/options")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert "coding_workflow" in body
        presets = body.get("workflow_presets")
        assert isinstance(presets, list) and {"coupled-v1", "hybrid-v1"} <= {
            p.get("id") for p in presets
        }

    def test_no_virtual_hybrid_provider(self, client):
        resp = client.get("/api/model/options")
        assert resp.status_code == 200, resp.text
        slugs = [p.get("slug", "").lower() for p in resp.json().get("providers", [])]
        # Hybrid is a workflow preset, never a fake model provider row.
        assert "hybrid" not in slugs
        assert "coding_workflow" not in slugs
