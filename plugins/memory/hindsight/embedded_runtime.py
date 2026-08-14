"""Lightweight embedded Hindsight client for the split Hermes runtime.

The API/ML server lives in an updater-managed, versioned environment. Hermes
core owns only ``hindsight-client`` and ``hindsight-embed`` and launches the
server exclusively through the stable ``current`` executable.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import stat
import threading
from pathlib import Path, PurePosixPath
from typing import Any

logger = logging.getLogger(__name__)

_API_EXECUTABLE_ENV = "HINDSIGHT_EMBED_API_EXECUTABLE"
_MARKER_NAME = ".hermes-hindsight-server.json"
_MARKER_SCHEMA = 2
_MARKER_FIELDS = frozenset({
    "schema",
    "fingerprint",
    "python",
    "executable",
    "executable_sha256",
    "python_resolved_sha256",
    "tree_sha256",
    "tree_entry_count",
})
_FINGERPRINT = re.compile(r"^[0-9a-f]{64}$")
_HASH_CHUNK_SIZE = 1024 * 1024


def _runtime_repo() -> Path:
    return Path(__file__).resolve().parents[3]


def _api_relative_path() -> Path:
    if os.name == "nt":
        return Path("Scripts") / "hindsight-api.exe"
    return Path("bin") / "hindsight-api"


def _python_relative_path() -> Path:
    if os.name == "nt":
        return Path("Scripts") / "python.exe"
    return Path("bin") / "python"


def _safe_relative(path: Path, root: Path, label: str) -> str:
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError as exc:
        raise RuntimeError(f"{label} escapes environment: {path}") from exc
    pure = PurePosixPath(relative)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise RuntimeError(f"{label} has an unsafe relative path: {relative}")
    return relative


def _seal_stat_key(details: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        details.st_dev,
        details.st_ino,
        details.st_mode,
        details.st_size,
        details.st_mtime_ns,
        details.st_ctime_ns,
    )


def _hash_regular_file(path: Path) -> tuple[str, int]:
    descriptor: int | None = None
    try:
        before = path.lstat()
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeError(f"sealed path is not a regular file: {path}")
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or _seal_stat_key(before) != _seal_stat_key(
            opened
        ):
            raise RuntimeError(f"sealed file changed while opening: {path}")
        digest = hashlib.sha256()
        size = 0
        while True:
            chunk = os.read(descriptor, _HASH_CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
        after = os.fstat(descriptor)
        if _seal_stat_key(opened) != _seal_stat_key(after) or size != after.st_size:
            raise RuntimeError(f"sealed file changed while hashing: {path}")
        return digest.hexdigest(), size
    except RuntimeError:
        raise
    except OSError as exc:
        raise RuntimeError(f"cannot hash sealed file {path}: {exc}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _sha256(path: Path) -> str:
    return _hash_regular_file(path)[0]


def _tree_seal_excludes(relative: str) -> bool:
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


def _environment_tree_seal(environment: Path) -> tuple[str, int]:
    if environment.is_symlink() or not environment.is_dir():
        raise RuntimeError(
            f"Hindsight server environment is not a real directory: {environment}"
        )
    pending = [environment]
    candidates: list[tuple[str, Path]] = []
    while pending:
        directory = pending.pop()
        try:
            entries = sorted(os.scandir(directory), key=lambda entry: entry.name)
        except OSError as exc:
            raise RuntimeError(
                f"Cannot scan Hindsight server environment {directory}: {exc}"
            ) from exc
        for entry in entries:
            path = Path(entry.path)
            relative = _safe_relative(
                path, environment, "Hindsight server environment entry"
            )
            if _tree_seal_excludes(relative):
                continue
            try:
                details = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise RuntimeError(
                    f"Cannot inspect Hindsight server environment entry {path}: {exc}"
                ) from exc
            if stat.S_ISDIR(details.st_mode):
                pending.append(path)
            elif stat.S_ISREG(details.st_mode) or stat.S_ISLNK(details.st_mode):
                candidates.append((relative, path))
            else:
                raise RuntimeError(
                    f"Unsupported Hindsight server environment entry: {path}"
                )

    digest = hashlib.sha256()
    count = 0
    for relative, path in sorted(candidates, key=lambda item: item[0]):
        try:
            details = path.lstat()
        except OSError as exc:
            raise RuntimeError(
                f"Cannot inspect Hindsight server environment entry {path}: {exc}"
            ) from exc
        executable_mode = stat.S_IMODE(details.st_mode) & 0o111
        if stat.S_ISREG(details.st_mode):
            content_sha256, size = _hash_regular_file(path)
            record: dict[str, object] = {
                "type": "file",
                "path": relative,
                "executable_mode": executable_mode,
                "symlink_target": None,
                "sha256": content_sha256,
                "size": size,
            }
        elif stat.S_ISLNK(details.st_mode):
            try:
                target = os.readlink(path)
                after = path.lstat()
            except OSError as exc:
                raise RuntimeError(
                    f"Cannot read Hindsight server environment symlink {path}: {exc}"
                ) from exc
            if _seal_stat_key(details) != _seal_stat_key(after):
                raise RuntimeError(
                    f"Hindsight server environment symlink changed while sealing: {path}"
                )
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
            raise RuntimeError(
                f"Hindsight server environment entry changed type: {path}"
            )
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


def _read_marker(environment: Path, fingerprint: str) -> dict[str, object]:
    marker = environment / _MARKER_NAME
    if marker.is_symlink() or not marker.is_file():
        raise RuntimeError(f"Hindsight server marker is missing or unsafe: {marker}")
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Hindsight server marker is invalid: {exc}") from exc
    if not isinstance(payload, dict) or set(payload) != _MARKER_FIELDS:
        raise RuntimeError("Hindsight server marker fields are invalid")
    if payload["schema"] != _MARKER_SCHEMA or payload["fingerprint"] != fingerprint:
        raise RuntimeError("Hindsight server marker fingerprint or schema is invalid")
    for field in (
        "executable_sha256",
        "python_resolved_sha256",
        "tree_sha256",
    ):
        value = payload[field]
        if not isinstance(value, str) or _FINGERPRINT.fullmatch(value) is None:
            raise RuntimeError(f"Hindsight server marker {field} is invalid")
    if type(payload["tree_entry_count"]) is not int or payload["tree_entry_count"] < 0:
        raise RuntimeError("Hindsight server marker tree entry count is invalid")
    return payload


def resolve_server_executable(
    explicit: str | os.PathLike[str] | None = None,
    *,
    runtime_repo: Path | None = None,
) -> Path:
    """Return a fully sealed stable executable or fail closed.

    An explicit config value wins over the environment. Without either, the
    updater-owned ``<repo.parent>/hindsight-server/current`` pointer is used.
    The returned path intentionally retains ``current`` so daemon launches do
    not pin an obsolete versioned environment across updater promotions.
    """

    configured = explicit
    if configured is None or not str(configured).strip():
        configured = os.environ.get(_API_EXECUTABLE_ENV)

    if configured is None or not str(configured).strip():
        repo = (runtime_repo or _runtime_repo()).expanduser().resolve()
        candidate = repo.parent / "hindsight-server" / "current" / _api_relative_path()
    else:
        candidate = Path(configured).expanduser()

    if not candidate.is_absolute():
        raise RuntimeError(
            f"{_API_EXECUTABLE_ENV} must identify an absolute stable executable"
        )

    relative_api = _api_relative_path()
    if (
        candidate.name != relative_api.name
        or candidate.parent.name != relative_api.parent.name
        or candidate.parent.parent.name != "current"
    ):
        raise RuntimeError(
            "Hindsight server executable must use a <store>/current/"
            f"{relative_api.as_posix()} stable path"
        )

    current = candidate.parent.parent
    store = current.parent
    envs = store / "envs"
    if store.is_symlink() or not store.is_dir():
        raise RuntimeError(f"Hindsight server store is missing or unsafe: {store}")
    if envs.is_symlink() or not envs.is_dir():
        raise RuntimeError(
            f"Hindsight server environment root is missing or unsafe: {envs}"
        )
    if not current.is_symlink():
        raise RuntimeError(
            f"Hindsight server current pointer is missing or unsafe: {current}"
        )

    try:
        target_text = os.readlink(current)
    except OSError as exc:
        raise RuntimeError(
            f"Cannot read Hindsight server current pointer: {exc}"
        ) from exc
    target = Path(target_text)
    if (
        target.is_absolute()
        or len(target.parts) != 2
        or target.parts[0] != "envs"
        or _FINGERPRINT.fullmatch(target.parts[1]) is None
    ):
        raise RuntimeError(
            f"Hindsight server current pointer would escape its store: {target_text}"
        )

    fingerprint = target.parts[1]
    environment = envs / fingerprint
    if environment.is_symlink() or not environment.is_dir():
        raise RuntimeError(
            f"Hindsight versioned server environment is missing or unsafe: {environment}"
        )
    payload = _read_marker(environment, fingerprint)

    expected_api = _api_relative_path().as_posix()
    expected_python = _python_relative_path().as_posix()
    if payload["executable"] != expected_api or payload["python"] != expected_python:
        raise RuntimeError("Hindsight server marker executable paths are invalid")

    versioned_executable = environment.joinpath(*PurePosixPath(expected_api).parts)
    if versioned_executable.is_symlink() or not versioned_executable.is_file():
        raise RuntimeError(
            f"Hindsight server executable is missing or unsafe: {versioned_executable}"
        )
    if not os.access(versioned_executable, os.X_OK):
        raise RuntimeError(
            f"Hindsight server executable is not executable: {versioned_executable}"
        )
    if _sha256(versioned_executable) != payload["executable_sha256"]:
        raise RuntimeError("Hindsight server executable hash mismatch")

    python = environment.joinpath(*PurePosixPath(expected_python).parts)
    try:
        python_details = python.lstat()
    except OSError as exc:
        raise RuntimeError(
            f"Hindsight server interpreter is missing: {python}"
        ) from exc
    if not (
        stat.S_ISREG(python_details.st_mode) or stat.S_ISLNK(python_details.st_mode)
    ):
        raise RuntimeError(f"Hindsight server interpreter is unsafe: {python}")
    try:
        resolved_python = python.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError(
            f"Cannot resolve Hindsight server interpreter: {python}"
        ) from exc
    if not resolved_python.is_file() or not os.access(resolved_python, os.X_OK):
        raise RuntimeError(
            f"Hindsight server interpreter is missing or not executable: {resolved_python}"
        )
    if _sha256(resolved_python) != payload["python_resolved_sha256"]:
        raise RuntimeError("Hindsight server interpreter hash mismatch")

    tree_sha256, tree_entry_count = _environment_tree_seal(environment)
    if (
        tree_sha256 != payload["tree_sha256"]
        or tree_entry_count != payload["tree_entry_count"]
    ):
        raise RuntimeError("Hindsight server environment tree seal mismatch")

    try:
        resolved_environment = environment.resolve(strict=True)
        resolved_current = current.resolve(strict=True)
        resolved_candidate = candidate.resolve(strict=True)
        resolved_executable = versioned_executable.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError(
            f"Cannot resolve Hindsight server executable: {exc}"
        ) from exc
    if resolved_current != resolved_environment:
        raise RuntimeError("Hindsight server current pointer changed during validation")
    if resolved_candidate != resolved_executable:
        raise RuntimeError(
            "Hindsight server executable escapes the current versioned environment"
        )
    try:
        final_target_text = os.readlink(current)
    except OSError as exc:
        raise RuntimeError(
            f"Cannot re-read Hindsight server current pointer: {exc}"
        ) from exc
    if final_target_text != target_text:
        raise RuntimeError("Hindsight server current pointer changed during validation")
    return candidate


class DedicatedEmbeddedClient:
    """Hindsight client with a dedicated, fail-closed daemon executable."""

    def __init__(
        self,
        *,
        profile: str = "default",
        llm_provider: str = "groq",
        llm_api_key: str = "",
        llm_model: str = "openai/gpt-oss-120b",
        llm_base_url: str | None = None,
        database_url: str | None = None,
        idle_timeout: int = 0,
        log_level: str = "info",
        server_executable: str | os.PathLike[str] | None = None,
    ) -> None:
        from hindsight_client import Hindsight
        from hindsight_embed import get_embed_manager

        stable_executable = resolve_server_executable(server_executable)
        self.profile = profile
        self.config = {
            "HINDSIGHT_API_LLM_PROVIDER": llm_provider,
            "HINDSIGHT_API_LLM_API_KEY": llm_api_key,
            "HINDSIGHT_API_LLM_MODEL": llm_model,
            "HINDSIGHT_API_LOG_LEVEL": log_level,
            "HINDSIGHT_EMBED_DAEMON_IDLE_TIMEOUT": str(idle_timeout),
            _API_EXECUTABLE_ENV: str(stable_executable),
        }
        if llm_base_url:
            self.config["HINDSIGHT_API_LLM_BASE_URL"] = llm_base_url
        if database_url:
            self.config["HINDSIGHT_EMBED_API_DATABASE_URL"] = database_url

        self._client_type = Hindsight
        self._client: Any | None = None
        self._lock = threading.Lock()
        self._started = False
        self._closed = False
        self._manager = get_embed_manager()

    def _discard_stale_client(self) -> None:
        client = self._client
        if client is not None:
            try:
                client.close()
            except Exception:
                logger.debug("Error closing stale Hindsight client", exc_info=True)
        self._client = None
        self._started = False

    def _ensure_started(self) -> None:
        """Start on first use and restart an unhealthy daemon under one lock."""
        if self._started and self._client is not None:
            if self._manager.is_running(self.profile):
                return
            logger.warning(
                "Hindsight daemon for profile %r is unhealthy; restarting", self.profile
            )

        with self._lock:
            if self._started and self._client is not None:
                if self._manager.is_running(self.profile):
                    return
                self._discard_stale_client()
            elif self._started:
                self._started = False

            if self._closed:
                raise RuntimeError(
                    "Cannot use DedicatedEmbeddedClient after it has been closed"
                )
            if not self._manager.ensure_running(self.config, self.profile):
                raise RuntimeError(
                    f"Failed to start Hindsight daemon for profile {self.profile!r}"
                )
            daemon_url = self._manager.get_url(self.profile)
            self._client = self._client_type(base_url=daemon_url)
            self._started = True
            logger.info("Connected to Hindsight daemon at %s", daemon_url)

    def close(self, stop_daemon: bool = False) -> None:
        """Close once; leave the shared daemon to its configured idle timeout."""
        if self._closed:
            return
        acquired = self._lock.acquire(timeout=5.0)
        if not acquired:
            logger.warning(
                "Hindsight cleanup lock timed out for profile %r; daemon will idle-stop",
                self.profile,
            )
            self._closed = True
            return
        try:
            if self._closed:
                return
            client = self._client
            if client is not None:
                try:
                    client.close()
                except Exception:
                    logger.debug("Error closing Hindsight client", exc_info=True)
                self._client = None
            if stop_daemon and self._started:
                self._manager.stop(self.profile)
            self._closed = True
        finally:
            self._lock.release()

    def __getattr__(self, name: str) -> Any:
        self._ensure_started()
        return getattr(self._client, name)

    def __enter__(self) -> "DedicatedEmbeddedClient":
        self._ensure_started()
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()

    def __del__(self) -> None:
        if "_closed" not in self.__dict__:
            return
        try:
            self.close()
        except Exception:
            pass
