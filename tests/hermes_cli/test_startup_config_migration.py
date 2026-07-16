"""Config migration contracts for update, serve, and dashboard startup."""

import copy
import os
import subprocess
import sys
import types
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import yaml


def _dashboard_args(*, headless: bool) -> SimpleNamespace:
    return SimpleNamespace(
        status=False,
        stop=False,
        host="127.0.0.1",
        port=0,
        no_open=True,
        insecure=False,
        skip_build=False,
        isolated=False,
        open_profile="",
        headless_backend=headless,
    )


def _wire_dashboard(main_mod, monkeypatch, tmp_path, events, *, profile="default"):
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<html></html>", encoding="utf-8")
    monkeypatch.setenv("HERMES_WEB_DIST", str(dist))
    monkeypatch.setattr(
        "hermes_cli.profiles.get_active_profile_name", lambda: profile
    )
    if profile != "default":
        monkeypatch.setenv("HERMES_DESKTOP", "1")
    monkeypatch.setattr(main_mod, "_sync_bundled_skills_quietly", lambda: None)
    monkeypatch.setitem(sys.modules, "fastapi", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "uvicorn", types.SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "hermes_logging",
        types.SimpleNamespace(setup_logging=lambda **_kwargs: None),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.plugins",
        types.SimpleNamespace(discover_plugins=lambda: None),
    )
    monkeypatch.setattr(
        "hermes_cli.config.apply_terminal_config_to_env",
        lambda: events.append("terminal-config"),
    )
    monkeypatch.setattr(
        "hermes_cli.mcp_startup.start_background_mcp_discovery",
        lambda **_kwargs: None,
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.web_server",
        types.SimpleNamespace(
            start_server=lambda **kwargs: events.append(("start-server", kwargs)),
        ),
    )


def test_profile_migration_uses_context_local_home(tmp_path, monkeypatch):
    from hermes_constants import get_hermes_home
    from hermes_cli.config import DEFAULT_CONFIG
    from hermes_cli.main import _migrate_profile_config

    root = tmp_path / ".hermes"
    profile = root / "profiles" / "worker"
    root.mkdir()
    profile.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

    latest = int(DEFAULT_CONFIG["_config_version"])
    stale = copy.deepcopy(DEFAULT_CONFIG)
    stale["_config_version"] = latest - 1
    for home in (root, profile):
        (home / "config.yaml").write_text(
            yaml.safe_dump(stale, sort_keys=False), encoding="utf-8"
        )

    changed = _migrate_profile_config(SimpleNamespace(name="worker", path=profile))

    assert changed is True
    assert yaml.safe_load((profile / "config.yaml").read_text())["_config_version"] == latest
    assert yaml.safe_load((root / "config.yaml").read_text())["_config_version"] == latest - 1
    assert get_hermes_home() == root
    assert os.environ["HERMES_HOME"] == str(root)


def test_profile_migration_warns_when_settings_need_review(
    tmp_path, monkeypatch, capsys
):
    from hermes_constants import get_hermes_home
    from hermes_cli import config
    from hermes_cli.main import _migrate_profile_config

    root = tmp_path / ".hermes"
    profile = root / "profiles" / "worker"
    profile.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setattr(config, "check_config_version", lambda: (4, 33))
    monkeypatch.setattr(
        config,
        "get_missing_env_vars",
        lambda required_only=False: [{"name": "NEW_API_KEY"}],
    )
    monkeypatch.setattr(config, "get_missing_config_fields", lambda: [])
    monkeypatch.setattr(
        config,
        "migrate_config",
        lambda **_kwargs: pytest.fail("unsafe migration must not run"),
    )

    changed = _migrate_profile_config(SimpleNamespace(name="worker", path=profile))

    assert changed is False
    assert "hermes -p worker config migrate" in capsys.readouterr().out
    assert get_hermes_home() == root
    assert os.environ["HERMES_HOME"] == str(root)


@pytest.mark.parametrize("headless", [False, True], ids=["dashboard", "serve"])
def test_dashboard_and_serve_migrate_before_config_consumers(
    tmp_path, monkeypatch, headless
):
    from hermes_cli import main as main_mod

    events = []
    _wire_dashboard(main_mod, monkeypatch, tmp_path, events)
    monkeypatch.setattr(
        main_mod,
        "_migrate_config_if_safe",
        lambda **kwargs: events.append(("migrate", kwargs["profile_name"])),
    )

    main_mod.cmd_dashboard(_dashboard_args(headless=headless))

    assert events[0] == ("migrate", "default")
    assert events[1] == "terminal-config"
    assert events[2][0] == "start-server"
    assert events[2][1]["headless"] is headless


def test_dashboard_reports_migration_failure_and_still_reaches_server(
    tmp_path, monkeypatch, capsys
):
    from hermes_cli import main as main_mod

    events = []
    _wire_dashboard(main_mod, monkeypatch, tmp_path, events, profile="worker")

    def fail_migration(**_kwargs):
        raise OSError("read-only config")

    monkeypatch.setattr(main_mod, "_migrate_config_if_safe", fail_migration)

    main_mod.cmd_dashboard(_dashboard_args(headless=True))

    error = capsys.readouterr().err
    assert "Config migration failed for profile 'worker'" in error
    assert "hermes -p worker config migrate" in error
    assert events[-1][0] == "start-server"


def _update_run_side_effect(command, **_kwargs):
    joined = " ".join(str(part) for part in command)
    if "rev-parse" in joined and "--abbrev-ref" in joined:
        return subprocess.CompletedProcess(command, 0, stdout="main\n", stderr="")
    if "rev-parse" in joined and "--verify" in joined:
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
    if "rev-list" in joined:
        return subprocess.CompletedProcess(command, 0, stdout="1\n", stderr="")
    return subprocess.CompletedProcess(command, 0, stdout="", stderr="")


def test_update_attempts_every_profile_and_reports_failures(
    tmp_path, monkeypatch, capsys
):
    from hermes_cli import main as main_mod

    profiles = [
        SimpleNamespace(name="default", path=tmp_path / ".hermes"),
        SimpleNamespace(name="worker", path=tmp_path / ".hermes/profiles/worker"),
    ]
    attempted = []

    def migrate(profile):
        attempted.append(profile.name)
        if profile.name == "worker":
            raise OSError("read-only config")

    empty_sync = {
        "copied": [],
        "updated": [],
        "user_modified": [],
        "cleaned": [],
    }
    with (
        patch("shutil.which", return_value=None),
        patch("subprocess.run", side_effect=_update_run_side_effect),
        patch("hermes_cli.managed_uv.ensure_uv", return_value=None),
        patch("hermes_cli.managed_uv.update_managed_uv", return_value=None),
        patch("hermes_cli.profiles.list_profiles", return_value=profiles),
        patch("hermes_cli.profiles.seed_profile_skills", return_value=empty_sync),
        patch("tools.skills_sync.sync_skills", return_value=empty_sync),
        patch("hermes_cli.config.get_missing_env_vars", return_value=[]),
        patch("hermes_cli.config.get_missing_config_fields", return_value=[]),
        patch("hermes_cli.config.check_config_version", return_value=(33, 33)),
        patch.object(main_mod, "_migrate_profile_config", side_effect=migrate),
        patch.object(main_mod, "_update_node_dependencies", return_value=[]),
        patch.object(main_mod, "_build_web_ui", return_value=True),
    ):
        main_mod.cmd_update(SimpleNamespace())

    assert attempted == ["default", "worker"]
    output = capsys.readouterr().out
    assert "Config migration failed for profile 'worker'" in output
    assert "hermes -p worker config migrate" in output
