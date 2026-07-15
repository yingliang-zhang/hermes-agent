"""Behavior tests for the built-in memory → external provider bridge.

The bridge lives behind the MemoryManager interface
(``MemoryManager.notify_memory_tool_write``): the agent loop hands over the raw
built-in memory tool result + args, and the manager decides whether/what to
mirror to external providers. These tests drive that method with a fake
external provider and assert which ``on_memory_write`` calls land.
"""

import json
import logging
import threading

import pytest

from agent.memory_manager import MemoryManager
from agent.memory_provider import MemoryProvider


class _RecordingProvider(MemoryProvider):
    """Minimal external provider that records on_memory_write calls."""

    def __init__(self) -> None:
        self.calls = []

    @property
    def name(self) -> str:
        return "recording"

    def is_available(self) -> bool:
        return True

    def initialize(self, session_id: str, **kwargs) -> None:
        pass

    def get_tool_schemas(self):
        return []

    def shutdown(self) -> None:
        pass

    def on_memory_write(self, action, target, content, metadata=None):
        self.calls.append({
            "action": action,
            "target": target,
            "content": content,
            "metadata": dict(metadata or {}),
        })


def _manager_with_provider():
    mgr = MemoryManager()
    provider = _RecordingProvider()
    mgr.add_provider(provider)
    return mgr, provider


def test_notifies_remove_with_old_text_after_success():
    mgr, provider = _manager_with_provider()
    mgr.notify_memory_tool_write(
        json.dumps({"success": True}),
        {"action": "remove", "target": "memory", "old_text": "stale preference entry"},
    )
    assert mgr.flush_pending(timeout=2)
    assert provider.calls == [
        {
            "action": "remove",
            "target": "memory",
            "content": "",
            "metadata": {"old_text": "stale preference entry"},
        }
    ]


def test_skips_failed_memory_write():
    mgr, provider = _manager_with_provider()
    mgr.notify_memory_tool_write(
        json.dumps({"success": False, "error": "No entry matched"}),
        {"action": "remove", "target": "memory", "old_text": "stale preference entry"},
    )
    assert provider.calls == []


def test_skips_staged_memory_write():
    mgr, provider = _manager_with_provider()
    mgr.notify_memory_tool_write(
        json.dumps({"success": True, "staged": True, "pending_id": "abc123"}),
        {"action": "remove", "target": "memory", "old_text": "stale preference entry"},
    )
    assert provider.calls == []


@pytest.mark.parametrize("tool_result", [None, [], object(), "not-json"])
def test_skips_unrecognized_tool_result_shape(tool_result):
    mgr, provider = _manager_with_provider()
    mgr.notify_memory_tool_write(
        tool_result,
        {"action": "add", "target": "memory", "content": "new fact"},
    )
    assert provider.calls == []


def test_preserves_old_text_for_replace_and_remove_batch():
    mgr, provider = _manager_with_provider()
    mgr.notify_memory_tool_write(
        json.dumps({"success": True}),
        {
            "target": "user",
            "operations": [
                {"action": "replace", "old_text": "old preference", "content": "updated"},
                {"action": "remove", "old_text": "obsolete preference"},
                {"action": "add", "content": "new fact"},
            ],
        },
    )
    assert mgr.flush_pending(timeout=2)
    assert provider.calls == [
        {"action": "replace", "target": "user", "content": "updated",
         "metadata": {"old_text": "old preference"}},
        {"action": "remove", "target": "user", "content": "",
         "metadata": {"old_text": "obsolete preference"}},
        {"action": "add", "target": "user", "content": "new fact", "metadata": {}},
    ]


def test_non_mutating_actions_are_not_mirrored():
    mgr, provider = _manager_with_provider()
    mgr.notify_memory_tool_write(
        json.dumps({"success": True}),
        {"action": "read", "target": "memory"},
    )
    assert provider.calls == []


def test_build_metadata_callback_is_merged_per_op():
    mgr, provider = _manager_with_provider()
    mgr.notify_memory_tool_write(
        json.dumps({"success": True}),
        {"action": "add", "target": "memory", "content": "fact"},
        build_metadata=lambda: {"session_id": "s1", "tool_name": "memory"},
    )
    assert mgr.flush_pending(timeout=2)
    assert provider.calls == [
        {
            "action": "add",
            "target": "memory",
            "content": "fact",
            "metadata": {"session_id": "s1", "tool_name": "memory"},
        }
    ]


def test_write_racing_boundary_begin_stays_on_serialized_worker():
    """A write accepted before a boundary stays FIFO ahead of extraction."""
    mgr, provider = _manager_with_provider()
    worker_started = threading.Event()
    release_worker = threading.Event()
    events = []

    def blocking_sync(*_args, **_kwargs):
        worker_started.set()
        release_worker.wait(timeout=5)
        events.append("sync")

    provider.sync_turn = blocking_sync  # type: ignore[method-assign]
    provider.on_memory_write = (  # type: ignore[method-assign]
        lambda *_args, **_kwargs: events.append("write")
    )
    provider.on_session_end = (  # type: ignore[method-assign]
        lambda _messages: events.append("end")
    )
    provider.on_session_switch = (  # type: ignore[method-assign]
        lambda _session_id, **_kwargs: events.append("switch")
    )

    mgr.sync_all("before boundary", "reply", session_id="A")
    assert worker_started.wait(timeout=1)
    mgr.on_memory_write("add", "memory", "fact")

    # The write must not bypass the occupied worker and touch the provider inline.
    assert events == []
    mgr.commit_session_boundary_async(
        [{"role": "user", "content": "A history"}],
        new_session_id="B",
        parent_session_id="A",
    )
    release_worker.set()

    assert mgr.flush_pending(timeout=2)
    assert events == ["sync", "write", "end", "switch"]


def test_mutations_queued_after_failed_rebind_fail_closed(caplog):
    """No queued write, sync, or prefetch may touch an old provider bank."""
    mgr, provider = _manager_with_provider()
    sync_calls = []
    prefetch_calls = []
    caplog.set_level(logging.DEBUG, logger="agent.memory_manager")

    def fail_switch(*_args, **_kwargs):
        raise RuntimeError("rebind failed")

    provider.on_session_switch = fail_switch  # type: ignore[method-assign]
    provider.sync_turn = (  # type: ignore[method-assign]
        lambda user, *_args, **_kwargs: sync_calls.append(user)
    )
    provider.queue_prefetch = (  # type: ignore[method-assign]
        lambda query, **_kwargs: prefetch_calls.append(query)
    )

    mgr.commit_session_boundary_async([], new_session_id="B", parent_session_id="A")
    assert mgr.flush_pending(timeout=2)
    assert not mgr.session_boundary_pending
    assert not mgr.memory_enabled

    mgr.sync_all("must skip", "reply", session_id="B")
    mgr.queue_prefetch_all("must skip", session_id="B")
    mgr.on_memory_write("add", "memory", "must skip")

    assert mgr.flush_pending(timeout=2)
    assert sync_calls == []
    assert prefetch_calls == []
    assert provider.calls == []
    assert "rebind" in caplog.text or "boundary" in caplog.text


def test_mutations_skip_when_a_newer_boundary_is_pending():
    """Work queued for B must not mutate B once an overlapping B→C is pending."""
    mgr, provider = _manager_with_provider()
    extraction_started = threading.Event()
    release_extraction = threading.Event()
    sync_calls = []
    prefetch_calls = []

    def blocking_end(_messages):
        extraction_started.set()
        release_extraction.wait(timeout=5)

    provider.on_session_end = blocking_end  # type: ignore[method-assign]
    provider.sync_turn = (  # type: ignore[method-assign]
        lambda user, *_args, **_kwargs: sync_calls.append(user)
    )
    provider.queue_prefetch = (  # type: ignore[method-assign]
        lambda query, **_kwargs: prefetch_calls.append(query)
    )

    mgr.commit_session_boundary_async(
        [{"role": "user", "content": "A history"}],
        new_session_id="B",
        parent_session_id="A",
    )
    assert extraction_started.wait(timeout=1)
    mgr.sync_all("B turn", "reply", session_id="B")
    mgr.queue_prefetch_all("B turn", session_id="B")
    mgr.on_memory_write("add", "memory", "B fact")
    mgr.commit_session_boundary_async([], new_session_id="C", parent_session_id="B")
    release_extraction.set()

    assert mgr.flush_pending(timeout=2)
    assert sync_calls == []
    assert prefetch_calls == []
    assert provider.calls == []
