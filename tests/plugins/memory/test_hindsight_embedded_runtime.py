"""Isolation and lifecycle tests for the split Hindsight embedded runtime."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import sys
from dataclasses import dataclass
from importlib.abc import MetaPathFinder
from pathlib import Path, PurePosixPath
from types import ModuleType

import pytest

from plugins.memory.hindsight import embedded_runtime
from plugins.memory.hindsight.embedded_runtime import (
    DedicatedEmbeddedClient,
    resolve_server_executable,
)


class _RejectServerPackages(MetaPathFinder):
    blocked = {"hindsight", "hindsight_api"}

    def find_spec(self, fullname, path=None, target=None):
        del path, target
        if fullname.split(".", 1)[0] in self.blocked:
            raise AssertionError(
                f"server-only module imported in Hermes core: {fullname}"
            )
        return None


class _FakeHindsight:
    instances: list["_FakeHindsight"] = []

    def __init__(self, *, base_url: str):
        self.base_url = base_url
        self.close_calls = 0
        self.instances.append(self)

    def close(self) -> None:
        self.close_calls += 1

    def ping(self) -> str:
        return self.base_url


class _FakeManager:
    def __init__(self) -> None:
        self.running = False
        self.ensure_calls: list[tuple[dict[str, str], str]] = []
        self.stop_calls: list[str] = []

    def ensure_running(self, config: dict[str, str], profile: str) -> bool:
        self.ensure_calls.append((dict(config), profile))
        self.running = True
        return True

    def get_url(self, profile: str) -> str:
        return f"http://127.0.0.1:8888/{profile}"

    def is_running(self, profile: str) -> bool:
        del profile
        return self.running

    def stop(self, profile: str) -> bool:
        self.stop_calls.append(profile)
        self.running = False
        return True


_MARKER_NAME = ".hermes-hindsight-server.json"
_MARKER_SCHEMA = 2
_HASH_CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True)
class _ManagedServer:
    repo: Path
    store: Path
    stable: Path
    current: Path
    environment: Path
    executable: Path
    python: Path
    resolved_python: Path
    package_file: Path
    package_link: Path | None


def _seal_stat_key(details: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        details.st_dev,
        details.st_ino,
        details.st_mode,
        details.st_size,
        details.st_mtime_ns,
        details.st_ctime_ns,
    )


def _builder_hash_regular_file(path: Path) -> tuple[str, int]:
    descriptor: int | None = None
    try:
        before = path.lstat()
        assert stat.S_ISREG(before.st_mode)
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        opened = os.fstat(descriptor)
        assert stat.S_ISREG(opened.st_mode)
        assert _seal_stat_key(before) == _seal_stat_key(opened)
        digest = hashlib.sha256()
        size = 0
        while True:
            chunk = os.read(descriptor, _HASH_CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
        after = os.fstat(descriptor)
        assert _seal_stat_key(opened) == _seal_stat_key(after)
        assert size == after.st_size
        return digest.hexdigest(), size
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _builder_safe_relative(path: Path, root: Path) -> str:
    relative = path.relative_to(root).as_posix()
    pure = PurePosixPath(relative)
    assert not pure.is_absolute()
    assert all(part not in {"", ".", ".."} for part in pure.parts)
    return relative


def _builder_tree_excludes(relative: str) -> bool:
    parts = PurePosixPath(relative).parts
    if "__pycache__" in parts:
        return True
    name = parts[-1]
    if name.endswith((".pyc", ".pyo")):
        return True
    if len(parts) != 1:
        return False
    return name == _MARKER_NAME or (
        name.startswith(f".{_MARKER_NAME}.") and name.endswith(".tmp")
    )


def _builder_tree_seal(environment: Path) -> tuple[str, int]:
    pending = [environment]
    candidates: list[tuple[str, Path]] = []
    while pending:
        directory = pending.pop()
        entries = sorted(os.scandir(directory), key=lambda entry: entry.name)
        for entry in entries:
            path = Path(entry.path)
            relative = _builder_safe_relative(path, environment)
            if _builder_tree_excludes(relative):
                continue
            details = entry.stat(follow_symlinks=False)
            if stat.S_ISDIR(details.st_mode):
                pending.append(path)
            elif stat.S_ISREG(details.st_mode) or stat.S_ISLNK(details.st_mode):
                candidates.append((relative, path))
            else:
                raise AssertionError(f"unsupported fixture entry: {path}")

    digest = hashlib.sha256()
    count = 0
    for relative, path in sorted(candidates, key=lambda item: item[0]):
        details = path.lstat()
        executable_mode = stat.S_IMODE(details.st_mode) & 0o111
        if stat.S_ISREG(details.st_mode):
            content_sha256, size = _builder_hash_regular_file(path)
            record: dict[str, object] = {
                "type": "file",
                "path": relative,
                "executable_mode": executable_mode,
                "symlink_target": None,
                "sha256": content_sha256,
                "size": size,
            }
        elif stat.S_ISLNK(details.st_mode):
            target = os.readlink(path)
            after = path.lstat()
            assert _seal_stat_key(details) == _seal_stat_key(after)
            target_bytes = os.fsencode(target)
            record = {
                "type": "symlink",
                "path": relative,
                "executable_mode": executable_mode,
                "symlink_target": target,
                "sha256": hashlib.sha256(target_bytes).hexdigest(),
                "size": len(target_bytes),
            }
        else:
            raise AssertionError(f"fixture entry changed type: {path}")
        encoded = json.dumps(
            record,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        count += 1
    return digest.hexdigest(), count


def _write_environment(
    store: Path, fingerprint: str
) -> tuple[Path, Path, Path, Path, Path, Path | None]:
    environment = store / "envs" / fingerprint
    relative_api = (
        Path("Scripts") / "hindsight-api.exe"
        if os.name == "nt"
        else Path("bin") / "hindsight-api"
    )
    relative_python = (
        Path("Scripts") / "python.exe" if os.name == "nt" else Path("bin") / "python"
    )

    executable = environment / relative_api
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"#!/bin/sh\nexit 0\n")
    executable.chmod(0o755)

    python = environment / relative_python
    python.parent.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        shutil.copyfile(sys.executable, python)
        resolved_python = python
    else:
        resolved_python = environment / "runtime" / "python-real"
        resolved_python.parent.mkdir()
        shutil.copyfile(sys.executable, resolved_python)
        python.symlink_to(Path("..") / "runtime" / "python-real")
    resolved_python.chmod(0o755)

    package_dir = environment / "lib" / "hindsight_api"
    package_dir.mkdir(parents=True)
    package_file = package_dir / "service.py"
    package_file.write_bytes(b"SERVER_VERSION = 1\n")
    (package_dir / "payload-a.dat").write_bytes(b"payload-a")
    (package_dir / "payload-b.dat").write_bytes(b"payload-b")
    package_link = package_dir / "active.dat"
    try:
        package_link.symlink_to("payload-a.dat")
    except OSError:
        package_link = None

    tree_sha256, tree_entry_count = _builder_tree_seal(environment)
    payload = {
        "schema": _MARKER_SCHEMA,
        "fingerprint": fingerprint,
        "python": relative_python.as_posix(),
        "executable": relative_api.as_posix(),
        "executable_sha256": _builder_hash_regular_file(executable)[0],
        "python_resolved_sha256": _builder_hash_regular_file(resolved_python)[0],
        "tree_sha256": tree_sha256,
        "tree_entry_count": tree_entry_count,
    }
    (environment / _MARKER_NAME).write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return environment, executable, python, resolved_python, package_file, package_link


def _managed_server(tmp_path: Path) -> _ManagedServer:
    repo = tmp_path / "hermes-agent"
    repo.mkdir()
    store = tmp_path / "hindsight-server"
    fingerprint = "a" * 64
    environment, executable, python, resolved_python, package_file, package_link = (
        _write_environment(store, fingerprint)
    )
    current = store / "current"
    current.symlink_to(Path("envs") / fingerprint, target_is_directory=True)
    relative_api = executable.relative_to(environment)
    return _ManagedServer(
        repo=repo,
        store=store,
        stable=current / relative_api,
        current=current,
        environment=environment,
        executable=executable,
        python=python,
        resolved_python=resolved_python,
        package_file=package_file,
        package_link=package_link,
    )


def _rewrite_marker(server: _ManagedServer, tamper: str) -> None:
    marker = server.environment / _MARKER_NAME
    payload = json.loads(marker.read_text(encoding="utf-8"))
    if tamper == "schema":
        payload["schema"] = 1
    elif tamper == "missing-field":
        payload.pop("python_resolved_sha256")
    elif tamper == "extra-field":
        payload["unexpected"] = True
    elif tamper == "count":
        payload["tree_entry_count"] += 1
    elif tamper == "count-type":
        payload["tree_entry_count"] = "1"
    elif tamper == "tree-digest":
        payload["tree_sha256"] = "0" * 64
    elif tamper == "api-digest":
        payload["executable_sha256"] = "0" * 64
    elif tamper == "python-digest":
        payload["python_resolved_sha256"] = "0" * 64
    elif tamper == "path":
        payload["python"] = "../python"
    else:
        raise AssertionError(f"unknown tamper case: {tamper}")
    marker.write_text(json.dumps(payload), encoding="utf-8")


def _install_lightweight_modules(monkeypatch, manager: _FakeManager) -> None:
    _FakeHindsight.instances.clear()
    client_module = ModuleType("hindsight_client")
    client_module.Hindsight = _FakeHindsight
    embed_module = ModuleType("hindsight_embed")
    embed_module.get_embed_manager = lambda: manager
    monkeypatch.setitem(sys.modules, "hindsight_client", client_module)
    monkeypatch.setitem(sys.modules, "hindsight_embed", embed_module)


def test_schema_two_environment_launches_exact_stable_executable_without_server_imports(
    tmp_path, monkeypatch
):
    server = _managed_server(tmp_path)
    manager = _FakeManager()
    _install_lightweight_modules(monkeypatch, manager)
    monkeypatch.delitem(sys.modules, "hindsight", raising=False)
    monkeypatch.delitem(sys.modules, "hindsight_api", raising=False)
    blocker = _RejectServerPackages()
    monkeypatch.setattr(sys, "meta_path", [blocker, *sys.meta_path])

    client = DedicatedEmbeddedClient(
        profile="hermes",
        server_executable=server.stable,
        llm_provider="openai",
    )
    assert manager.ensure_calls == []

    assert client.ping() == "http://127.0.0.1:8888/hermes"
    assert "hindsight" not in sys.modules
    assert "hindsight_api" not in sys.modules
    assert manager.ensure_calls[0][0]["HINDSIGHT_EMBED_API_EXECUTABLE"] == str(
        server.stable
    )


def test_default_and_environment_resolvers_keep_the_stable_current_path(
    tmp_path, monkeypatch
):
    server = _managed_server(tmp_path)

    assert resolve_server_executable(runtime_repo=server.repo) == server.stable
    monkeypatch.setenv("HINDSIGHT_EMBED_API_EXECUTABLE", str(server.stable))
    assert resolve_server_executable() == server.stable


@pytest.mark.parametrize(
    "failure", ["missing", "escaping", "non-executable", "binary-escape"]
)
def test_server_executable_validation_fails_closed(tmp_path, failure):
    server = _managed_server(tmp_path)

    if failure == "missing":
        server.executable.unlink()
    elif failure == "escaping":
        server.current.unlink()
        server.current.symlink_to(Path("..") / "outside", target_is_directory=True)
    elif failure == "non-executable":
        server.executable.chmod(server.executable.stat().st_mode & ~0o111)
    else:
        outside = tmp_path / "outside-api"
        outside.write_bytes(b"#!/bin/sh\nexit 0\n")
        outside.chmod(0o755)
        server.executable.unlink()
        server.executable.symlink_to(outside)

    with pytest.raises(RuntimeError):
        resolve_server_executable(server.stable, runtime_repo=server.repo)


@pytest.mark.parametrize(
    "tamper",
    [
        "schema",
        "missing-field",
        "extra-field",
        "count",
        "count-type",
        "tree-digest",
        "api-digest",
        "python-digest",
        "path",
    ],
)
def test_schema_two_marker_tamper_is_rejected(tmp_path, tamper):
    server = _managed_server(tmp_path)
    _rewrite_marker(server, tamper)

    with pytest.raises(RuntimeError):
        resolve_server_executable(server.stable)


@pytest.mark.parametrize("change", ["modify", "delete", "add"])
def test_package_tree_tamper_is_rechecked_in_the_same_process(tmp_path, change):
    server = _managed_server(tmp_path)
    assert resolve_server_executable(server.stable) == server.stable

    if change == "modify":
        server.package_file.write_bytes(b"SERVER_VERSION = 2\n")
    elif change == "delete":
        server.package_file.unlink()
    else:
        (server.environment / "lib" / "injected.py").write_bytes(b"INJECTED = True\n")

    with pytest.raises(RuntimeError, match="tree seal"):
        resolve_server_executable(server.stable)


@pytest.mark.skipif(
    os.name == "nt", reason="Windows chmod does not expose POSIX exec bits"
)
def test_package_mode_change_is_rejected(tmp_path):
    server = _managed_server(tmp_path)
    assert resolve_server_executable(server.stable) == server.stable
    server.package_file.chmod(server.package_file.stat().st_mode | stat.S_IXUSR)

    with pytest.raises(RuntimeError, match="tree seal"):
        resolve_server_executable(server.stable)


def test_package_symlink_retarget_is_rejected(tmp_path):
    server = _managed_server(tmp_path)
    if server.package_link is None:
        pytest.skip("symlinks are unavailable")
    assert resolve_server_executable(server.stable) == server.stable
    server.package_link.unlink()
    server.package_link.symlink_to("payload-b.dat")

    with pytest.raises(RuntimeError, match="tree seal"):
        resolve_server_executable(server.stable)


def test_resolved_interpreter_byte_mutation_is_rejected(tmp_path):
    server = _managed_server(tmp_path)
    assert resolve_server_executable(server.stable) == server.stable
    server.resolved_python.write_bytes(server.resolved_python.read_bytes() + b"tamper")

    with pytest.raises(RuntimeError, match="interpreter hash mismatch"):
        resolve_server_executable(server.stable)


@pytest.mark.parametrize(
    "failure", ["missing", "unsafe-type", "non-executable", "dangling"]
)
def test_unsafe_interpreter_is_rejected(tmp_path, failure):
    server = _managed_server(tmp_path)

    if failure == "missing":
        server.python.unlink()
    elif failure == "unsafe-type":
        server.python.unlink()
        server.python.mkdir()
    elif failure == "non-executable":
        server.resolved_python.chmod(server.resolved_python.stat().st_mode & ~0o111)
    else:
        if not server.python.is_symlink():
            pytest.skip("fixture interpreter is not a symlink")
        server.python.unlink()
        server.python.symlink_to(Path("..") / "runtime" / "missing-python")

    with pytest.raises(RuntimeError):
        resolve_server_executable(server.stable)


def test_python_cache_files_do_not_invalidate_the_tree_seal(tmp_path):
    server = _managed_server(tmp_path)
    cache = server.environment / "lib" / "__pycache__"
    cache.mkdir()
    (cache / "service.cpython-313.pyc").write_bytes(b"cache")
    (server.environment / "lib" / "ignored.pyc").write_bytes(b"cache")
    (server.environment / "lib" / "ignored.pyo").write_bytes(b"cache")

    assert resolve_server_executable(server.stable) == server.stable


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO entries require POSIX")
def test_unsupported_tree_entry_is_rejected(tmp_path):
    server = _managed_server(tmp_path)
    os.mkfifo(server.environment / "lib" / "runtime.pipe")

    with pytest.raises(RuntimeError, match="Unsupported.*environment entry"):
        resolve_server_executable(server.stable)


def test_regular_file_stat_change_during_hashing_is_rejected(tmp_path, monkeypatch):
    server = _managed_server(tmp_path)
    real_fstat = os.fstat
    fstat_calls = 0

    def mutate_before_final_stat(descriptor: int) -> os.stat_result:
        nonlocal fstat_calls
        fstat_calls += 1
        if fstat_calls == 2:
            server.package_file.write_bytes(b"changed during hashing")
        return real_fstat(descriptor)

    monkeypatch.setattr(embedded_runtime.os, "fstat", mutate_before_final_stat)

    with pytest.raises(RuntimeError, match="changed while hashing"):
        embedded_runtime._hash_regular_file(server.package_file)


def test_current_pointer_swap_during_validation_is_rejected(tmp_path, monkeypatch):
    server = _managed_server(tmp_path)
    replacement_fingerprint = "b" * 64
    _write_environment(server.store, replacement_fingerprint)
    original_tree_seal = embedded_runtime._environment_tree_seal

    def seal_then_swap(environment: Path) -> tuple[str, int]:
        result = original_tree_seal(environment)
        server.current.unlink()
        server.current.symlink_to(
            Path("envs") / replacement_fingerprint, target_is_directory=True
        )
        return result

    monkeypatch.setattr(embedded_runtime, "_environment_tree_seal", seal_then_swap)

    with pytest.raises(RuntimeError, match="current pointer changed"):
        resolve_server_executable(server.stable)


def test_invalid_tree_never_reaches_the_daemon_manager(tmp_path, monkeypatch):
    server = _managed_server(tmp_path)
    server.package_file.write_bytes(b"tampered")
    manager = _FakeManager()
    _install_lightweight_modules(monkeypatch, manager)

    with pytest.raises(RuntimeError, match="tree seal"):
        DedicatedEmbeddedClient(profile="hermes", server_executable=server.stable)
    assert manager.ensure_calls == []


def test_context_manager_preserves_lazy_start_and_closes_client(tmp_path, monkeypatch):
    server = _managed_server(tmp_path)
    manager = _FakeManager()
    _install_lightweight_modules(monkeypatch, manager)

    with DedicatedEmbeddedClient(
        profile="hermes", server_executable=server.stable
    ) as client:
        inner_client = client._client
        assert inner_client is not None
        assert client.ping() == "http://127.0.0.1:8888/hermes"

    assert inner_client.close_calls == 1
    assert client._client is None


def test_manager_receives_exact_executable_and_reconnect_close_stays_compatible(
    tmp_path, monkeypatch
):
    server = _managed_server(tmp_path)
    stable = server.stable
    manager = _FakeManager()
    _install_lightweight_modules(monkeypatch, manager)
    monkeypatch.setenv("HINDSIGHT_EMBED_API_EXECUTABLE", "relative-is-unsafe")

    client = DedicatedEmbeddedClient(
        profile="hermes",
        server_executable=stable,
        idle_timeout=300,
    )
    client._ensure_started()
    first = client._client
    assert first is _FakeHindsight.instances[0]
    assert manager.ensure_calls == [
        (
            {
                "HINDSIGHT_API_LLM_PROVIDER": "groq",
                "HINDSIGHT_API_LLM_API_KEY": "",
                "HINDSIGHT_API_LLM_MODEL": "openai/gpt-oss-120b",
                "HINDSIGHT_API_LOG_LEVEL": "info",
                "HINDSIGHT_EMBED_DAEMON_IDLE_TIMEOUT": "300",
                "HINDSIGHT_EMBED_API_EXECUTABLE": str(stable),
            },
            "hermes",
        )
    ]

    manager.running = False
    client._ensure_started()
    second = client._client
    assert second is not first
    assert first.close_calls == 1
    assert len(manager.ensure_calls) == 2

    client.close()
    client.close()
    assert second.close_calls == 1
    assert client._client is None
    assert manager.stop_calls == []
    with pytest.raises(RuntimeError, match="after it has been closed"):
        client._ensure_started()
