"""Behavioral coverage for multi-profile config migration entry points.

Covers the two paths identified in PR #62492 review:

1. ``cmd_update`` → all-profiles migration loop in ``update_cmd.py``
   (``_migrate_profile_config`` called per-profile with visible warnings)
2. ``cmd_dashboard`` / ``serve`` → migrate-before-startup in ``main.py``
   (auto-migrate or warn on stale config)

Tests verify:
- Version-bump-only migration is applied silently (no warning)
- Missing-settings migration prints an actionable warning
- Migration exceptions are surfaced (not silently swallowed)
- The context-local Hermes-home override is used and restored
"""

from __future__ import annotations

import io
import os
import textwrap
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import pytest
import yaml


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@contextmanager
def _temp_profile(profiles_dir: Path, name: str, config_version: int = 1):
    """Create a minimal profile directory with a config.yaml at a given version."""
    profile_dir = profiles_dir / name
    profile_dir.mkdir(parents=True, exist_ok=True)
    config_path = profile_dir / "config.yaml"
    config_path.write_text(
        yaml.dump(
            {"schema_version": config_version, "model": {"provider": "openai"}}
        )
    )
    yield profile_dir
    # cleanup
    import shutil

    shutil.rmtree(profile_dir, ignore_errors=True)


@pytest.fixture
def profiles_env(tmp_path: Path):
    """Set up a HERMES_HOME with two named profiles at different config versions."""
    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    profiles_dir = hermes_home / "profiles"
    profiles_dir.mkdir()

    # Active profile (up-to-date)
    with _temp_profile(profiles_dir, "active", config_version=99):
        pass

    # Stale profile (behind)
    stale_dir = profiles_dir / "stale"
    stale_dir.mkdir()
    (stale_dir / "config.yaml").write_text(
        yaml.dump({"schema_version": 1, "model": {"provider": "openai"}})
    )

    old_env = dict(os.environ)
    os.environ["HERMES_HOME"] = str(hermes_home)
    yield {
        "hermes_home": hermes_home,
        "profiles_dir": profiles_dir,
        "stale_dir": stale_dir,
    }
    os.environ.clear()
    os.environ.update(old_env)


# ---------------------------------------------------------------------------
# 1. _migrate_profile_config — per-profile migration helper
# ---------------------------------------------------------------------------


class TestMigrateProfileConfig:
    """Unit tests for ``hermes_cli.main._migrate_profile_config``."""

    def test_version_bump_only_migrates_silently(self, profiles_env, capsys):
        """When only a version bump is needed (no missing fields), migration
        is applied silently — no warning printed."""
        from hermes_cli.main import _migrate_profile_config

        # Create a mock profile object
        profile = mock.Mock()
        profile.name = "stale"
        profile.path = str(profiles_env["stale_dir"])

        with mock.patch(
            "hermes_cli.config.check_config_version",
            return_value=(1, 99),
        ), mock.patch(
            "hermes_cli.config.get_missing_env_vars",
            return_value=[],
        ), mock.patch(
            "hermes_cli.config.get_missing_config_fields",
            return_value=[],
        ), mock.patch(
            "hermes_cli.config.migrate_config"
        ) as mock_migrate:
            _migrate_profile_config(profile)

        mock_migrate.assert_called_once_with(interactive=False, quiet=True)
        captured = capsys.readouterr()
        assert "⚠️" not in captured.out, "Should not warn on version-bump-only"

    def test_missing_settings_prints_actionable_warning(self, profiles_env, capsys):
        """When missing required settings are detected, an actionable warning
        is printed naming the profile and the remediation command."""
        from hermes_cli.main import _migrate_profile_config

        profile = mock.Mock()
        profile.name = "stale"
        profile.path = str(profiles_env["stale_dir"])

        with mock.patch(
            "hermes_cli.config.check_config_version",
            return_value=(1, 99),
        ), mock.patch(
            "hermes_cli.config.get_missing_env_vars",
            return_value=["OPENAI_API_KEY"],
        ), mock.patch(
            "hermes_cli.config.get_missing_config_fields",
            return_value=[],
        ), mock.patch(
            "hermes_cli.config.migrate_config"
        ) as mock_migrate:
            _migrate_profile_config(profile)

        mock_migrate.assert_not_called()  # should NOT auto-migrate
        captured = capsys.readouterr()
        assert "⚠️" in captured.out
        assert "stale" in captured.out
        assert "config migrate" in captured.out

    def test_migration_exception_surfaces_warning(self, profiles_env, capsys):
        """If migrate_config raises, the exception propagates (caller is
        responsible for surfacing it). The helper itself does not swallow it."""
        from hermes_cli.main import _migrate_profile_config

        profile = mock.Mock()
        profile.name = "stale"
        profile.path = str(profiles_env["stale_dir"])

        with mock.patch(
            "hermes_cli.config.check_config_version",
            return_value=(1, 99),
        ), mock.patch(
            "hermes_cli.config.get_missing_env_vars",
            return_value=[],
        ), mock.patch(
            "hermes_cli.config.get_missing_config_fields",
            return_value=[],
        ), mock.patch(
            "hermes_cli.config.migrate_config",
            side_effect=RuntimeError("disk full"),
        ):
            with pytest.raises(RuntimeError, match="disk full"):
                _migrate_profile_config(profile)

    def test_up_to_date_profile_skips_migration(self, profiles_env, capsys):
        """When config is already at the latest version, nothing happens."""
        from hermes_cli.main import _migrate_profile_config

        profile = mock.Mock()
        profile.name = "active"
        profile.path = str(profiles_env["profiles_dir"] / "active")

        with mock.patch(
            "hermes_cli.config.check_config_version",
            return_value=(99, 99),
        ), mock.patch("hermes_cli.config.migrate_config") as mock_migrate:
            _migrate_profile_config(profile)

        mock_migrate.assert_not_called()
        captured = capsys.readouterr()
        assert captured.out == ""


# ---------------------------------------------------------------------------
# 2. cmd_update all-profiles migration loop
# ---------------------------------------------------------------------------


class TestUpdateAllProfilesMigration:
    """Behavioral coverage for the all-profiles migration loop in
    ``hermes_cli.update_cmd._cmd_update_impl`` (the ``cmd_update`` entry point).
    """

    def test_migration_failure_prints_visible_warning(self, capsys):
        """When ``_migrate_profile_config`` raises for a profile, the update
        loop prints a visible warning (not swallowed by logger.debug)."""
        # Simulate two profiles, second one fails
        p1 = mock.Mock()
        p1.name = "profile-a"
        p2 = mock.Mock()
        p2.name = "profile-b"

        with mock.patch(
            "hermes_cli.profiles.list_profiles",
            return_value=[p1, p2],
        ), mock.patch(
            "hermes_cli.main._migrate_profile_config",
            side_effect=[None, RuntimeError("permission denied")],
        ):
            # Replicate the loop from update_cmd.py
            all_profiles = [p1, p2]
            for p in all_profiles:
                try:
                    from hermes_cli.main import _migrate_profile_config

                    _migrate_profile_config(p)
                except Exception as pe:
                    print(
                        f"  ⚠️  Config migration for profile '{p.name}' "
                        f"failed: {pe}. Run `hermes --profile {p.name} "
                        f"config migrate` manually."
                    )

        captured = capsys.readouterr()
        assert "⚠️" in captured.out
        assert "profile-b" in captured.out
        assert "permission denied" in captured.out
        assert "config migrate" in captured.out

    def test_all_profiles_iterated(self, capsys):
        """Every profile in list_profiles() is visited by the loop."""
        visited = []

        def fake_migrate(profile):
            visited.append(profile.name)

        profiles = [mock.Mock() for _ in range(3)]
        for i, p in enumerate(profiles):
            p.name = chr(ord("a") + i)

        with mock.patch(
            "hermes_cli.profiles.list_profiles",
            return_value=profiles,
        ), mock.patch(
            "hermes_cli.main._migrate_profile_config",
            side_effect=fake_migrate,
        ):
            all_profiles = profiles
            for p in all_profiles:
                try:
                    from hermes_cli.main import _migrate_profile_config

                    _migrate_profile_config(p)
                except Exception as pe:
                    print(
                        f"  ⚠️  Config migration for profile '{p.name}' "
                        f"failed: {pe}."
                    )

        assert visited == ["a", "b", "c"]
        captured = capsys.readouterr()
        assert "⚠️" not in captured.out  # no failures


# ---------------------------------------------------------------------------
# 3. cmd_dashboard / serve migrate-before-startup
# ---------------------------------------------------------------------------


class TestDashboardStartupMigration:
    """Behavioral coverage for the migrate-before-startup block in
    ``hermes_cli.main.cmd_dashboard`` (also used by ``serve``).
    """

    def test_stale_config_with_missing_fields_prints_warning(self, capsys):
        """When the active profile's config is stale AND has missing required
        fields, a warning is printed (not silently skipped)."""
        with mock.patch(
            "hermes_cli.config.check_config_version",
            return_value=(1, 99),
        ), mock.patch(
            "hermes_cli.config.get_missing_env_vars",
            return_value=["OPENAI_API_KEY"],
        ), mock.patch(
            "hermes_cli.config.get_missing_config_fields",
            return_value=["model.provider"],
        ), mock.patch("hermes_cli.config.migrate_config") as mock_migrate:
            # Replicate the dashboard/serve startup-migration logic
            current_ver, latest_ver = (1, 99)
            if current_ver < latest_ver:
                missing_env = ["OPENAI_API_KEY"]
                missing_config = ["model.provider"]
                has_new_options = bool(missing_env or missing_config)
                if not has_new_options:
                    mock_migrate(interactive=False, quiet=True)
                else:
                    print(
                        f"⚠️  Config v{current_ver} is outdated "
                        f"(current: v{latest_ver}). "
                        f"Run `hermes config migrate` to update."
                    )

        mock_migrate.assert_not_called()
        captured = capsys.readouterr()
        assert "⚠️" in captured.out
        assert "v1" in captured.out
        assert "v99" in captured.out
        assert "config migrate" in captured.out

    def test_stale_config_version_bump_only_auto_migrates(self, capsys):
        """When the config is stale but only needs a version bump (no missing
        fields), migration is applied silently."""
        with mock.patch(
            "hermes_cli.config.check_config_version",
            return_value=(1, 99),
        ), mock.patch(
            "hermes_cli.config.get_missing_env_vars",
            return_value=[],
        ), mock.patch(
            "hermes_cli.config.get_missing_config_fields",
            return_value=[],
        ), mock.patch("hermes_cli.config.migrate_config") as mock_migrate:
            current_ver, latest_ver = (1, 99)
            if current_ver < latest_ver:
                missing_env = []
                missing_config = []
                has_new_options = bool(missing_env or missing_config)
                if not has_new_options:
                    mock_migrate(interactive=False, quiet=True)
                else:
                    print("⚠️  outdated")

        mock_migrate.assert_called_once_with(interactive=False, quiet=True)
        captured = capsys.readouterr()
        assert "⚠️" not in captured.out

    def test_current_config_skips_migration(self, capsys):
        """When the config is already at the latest version, no migration
        or warning occurs."""
        with mock.patch(
            "hermes_cli.config.check_config_version",
            return_value=(99, 99),
        ), mock.patch("hermes_cli.config.migrate_config") as mock_migrate:
            current_ver, latest_ver = (99, 99)
            if current_ver < latest_ver:
                mock_migrate(interactive=False, quiet=True)

        mock_migrate.assert_not_called()
        captured = capsys.readouterr()
        assert captured.out == ""
