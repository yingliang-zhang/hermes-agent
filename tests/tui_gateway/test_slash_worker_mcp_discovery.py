"""Integration coverage for profile-local MCP discovery in slash workers."""

from __future__ import annotations

import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import textwrap
import threading

import pytest
import yaml

from tui_gateway import slash_worker

pytest.importorskip("mcp.server.fastmcp")


def test_tools_command_waits_for_configured_mcp_discovery(monkeypatch):
    joins: list[object] = []
    monkeypatch.setattr(
        "hermes_cli.mcp_startup.join_mcp_discovery",
        lambda timeout=None: joins.append(timeout) or True,
    )

    slash_worker._wait_for_command_readiness("/tools")
    assert joins == [slash_worker._MCP_COMMAND_READINESS_TIMEOUT_S]

    joins.clear()
    slash_worker._wait_for_command_readiness("/status")
    assert joins == []

def test_tools_command_reports_permanently_stalled_discovery(monkeypatch):
    stalled = threading.Event()
    monkeypatch.setattr(slash_worker, "_MCP_COMMAND_READINESS_TIMEOUT_S", 0.01)
    monkeypatch.setattr(
        "hermes_cli.mcp_startup.join_mcp_discovery",
        lambda timeout=None: stalled.wait(timeout),
    )

    with pytest.raises(RuntimeError, match="MCP tool discovery did not finish") as exc:
        slash_worker._wait_for_command_readiness("/tools")

    message = str(exc.value)
    assert "retry /tools" in message
    assert "/reload-mcp" in message




def test_profile_local_mcp_tool_is_visible_in_slash_worker(tmp_path):
    profile_home = tmp_path / "profile-home"
    profile_home.mkdir()
    marker = "profile-local-61922"
    readiness_gate = tmp_path / "allow-mcp-startup"
    server = tmp_path / "fastmcp_probe.py"
    server.write_text(
        textwrap.dedent(
            f"""
            import time
            from pathlib import Path

            from mcp.server.fastmcp import FastMCP
            mcp = FastMCP("profileprobe")

            @mcp.tool()
            def hermes_61922_profile_probe() -> str:
                return {marker!r}

            if __name__ == "__main__":
                deadline = time.monotonic() + 10
                while not Path({str(readiness_gate)!r}).exists():
                    if time.monotonic() >= deadline:
                        raise RuntimeError("readiness gate was never released")
                    time.sleep(0.01)
                mcp.run(transport="stdio")
            """
        ),
        encoding="utf-8",
    )
    (profile_home / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "mcp_discovery_timeout": 0.01,
                "mcp_servers": {
                    "profileprobe": {
                        "enabled": True,
                        "command": sys.executable,
                        "args": [str(server)],
                        "connect_timeout": 10,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    env = os.environ.copy()
    for key in list(env):
        if key.endswith("_API_KEY") or key.endswith("_TOKEN"):
            env.pop(key)
    env["HERMES_HOME"] = str(profile_home)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
    env["HERMES_SLASH_WATCHDOG_GRACE_S"] = "0"
    env["HERMES_SLASH_WATCHDOG_POLL_S"] = "0.05"
    proc = subprocess.Popen(
        [
            sys.executable,
            "-u",
            "-m",
            "tui_gateway.slash_worker",
            "--session-key",
            "agent:main:tui:dm:mcp-profile-test",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        cwd=tmp_path,
    )
    output: queue.Queue[dict] = queue.Queue()
    try:
        assert proc.stdin is not None
        assert proc.stdout is not None
        stdout = proc.stdout

        def read_output() -> None:
            for line in stdout:
                output.put(json.loads(line))

        threading.Thread(target=read_output, daemon=True).start()
        try:
            ready = output.get(timeout=30)
        except queue.Empty:
            pytest.fail("slash worker did not report readiness within 30 seconds")
        assert ready == {"event": "ready"}

        readiness_gate.touch()
        proc.stdin.write(json.dumps({"id": 1, "command": "/tools"}) + "\n")
        proc.stdin.flush()
        try:
            response = output.get(timeout=10)
        except queue.Empty:
            pytest.fail("ready slash worker produced no /tools response within 10 seconds")
        assert response["ok"] is True
        assert "mcp__profileprobe__hermes_61922_profile_probe" in response["output"]
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
