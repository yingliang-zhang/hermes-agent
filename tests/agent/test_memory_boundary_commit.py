"""Tests for MemoryManager.commit_session_boundary_async.

The /new session boundary must deliver on_session_end (old-session
extraction) strictly BEFORE on_session_switch (provider rebinding to the
new session), without blocking the caller. Both hooks run as one task on
the manager's single serialized background worker.
"""

from __future__ import annotations
import json

import threading
import time
from typing import Any, Dict, List

import pytest

from agent.memory_manager import MemoryManager
from agent.memory_provider import MemoryProvider


class _RecordingProvider(MemoryProvider):
    """Provider that records hook invocations with thread identity."""

    def __init__(self, end_delay: float = 0.0):
        self.calls: List[tuple] = []
        self._end_delay = end_delay
        self._caller_thread_ids: List[int] = []

    # Required ABC surface (minimal no-ops)
    @property
    def name(self) -> str:
        return "recorder"

    def is_available(self) -> bool:
        return True

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return []

    def initialize(self, agent: Any = None, **kwargs) -> bool:  # type: ignore[override]
        return True

    def build_system_prompt(self) -> str:  # type: ignore[override]
        return ""

    def sync_turn(self, user_content: str, assistant_content: str, **kwargs) -> None:  # type: ignore[override]
        self.calls.append(("sync_turn", kwargs.get("session_id", "")))

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        if self._end_delay:
            time.sleep(self._end_delay)
        self._caller_thread_ids.append(threading.get_ident())
        self.calls.append(("end", list(messages)))

    def on_session_switch(self, new_session_id: str, **kwargs) -> None:
        self.calls.append(("switch", new_session_id, kwargs.get("reset")))


def _make_manager(provider: _RecordingProvider) -> MemoryManager:
    mm = MemoryManager()
    mm._providers.append(provider)  # bypass add_provider validation for the stub
    return mm


def test_boundary_commit_delivers_end_strictly_before_switch():
    """Even with a slow (LLM-like) extraction, switch waits for end."""
    provider = _RecordingProvider(end_delay=0.15)
    mm = _make_manager(provider)

    msgs = [{"role": "user", "content": "old turn"}]
    t0 = time.monotonic()
    mm.commit_session_boundary_async(
        msgs, new_session_id="new-sid", parent_session_id="old-sid"
    )
    # Caller returns immediately — the slow extraction must not block /new.
    assert time.monotonic() - t0 < 0.1

    assert mm.flush_pending(timeout=5)

    kinds = [c[0] for c in provider.calls]
    assert kinds == ["end", "switch"], f"ordering violated: {provider.calls}"
    assert provider.calls[0] == ("end", msgs)
    assert provider.calls[1] == ("switch", "new-sid", True)
    # And it genuinely ran off the caller's thread.
    assert provider._caller_thread_ids[0] != threading.get_ident()


def test_boundary_commit_serializes_against_turn_syncs():
    """The boundary task shares the single worker with sync_all — FIFO order
    means a queued boundary can't interleave into a later turn's sync."""
    provider = _RecordingProvider(end_delay=0.05)
    mm = _make_manager(provider)

    mm.commit_session_boundary_async(
        [{"role": "user", "content": "old"}],
        new_session_id="new-sid",
    )
    mm.sync_all("next-session user msg", "assistant reply", session_id="new-sid")

    assert mm.flush_pending(timeout=5)

    kinds = [c[0] for c in provider.calls]
    assert kinds == ["end", "switch", "sync_turn"], f"unexpected order: {provider.calls}"


def test_boundary_commit_switch_still_fires_when_end_raises():
    """A failing provider extraction must not strand providers on the old sid."""

    class _ExplodingEndProvider(_RecordingProvider):
        def on_session_end(self, messages):  # type: ignore[override]
            raise RuntimeError("provider extraction blew up")

    provider = _ExplodingEndProvider()
    mm = _make_manager(provider)

    mm.commit_session_boundary_async([{"role": "user", "content": "x"}], new_session_id="new-sid")
    assert mm.flush_pending(timeout=5)

    assert ("switch", "new-sid", True) in provider.calls


def test_boundary_commit_switch_failure_keeps_memory_fail_closed():
    """A failed provider rebind must not expose old-session state."""

    class _ExplodingSwitchProvider(_RecordingProvider):
        def on_session_switch(self, new_session_id: str, **kwargs) -> None:
            del new_session_id, kwargs
            raise RuntimeError("provider rebind blew up")

        def prefetch(self, query: str, *, session_id: str = "") -> str:
            del query, session_id
            return "old-session memory"

    provider = _ExplodingSwitchProvider()
    mm = _make_manager(provider)

    mm.commit_session_boundary_async(
        [{"role": "user", "content": "old"}], new_session_id="new-sid"
    )
    assert mm.flush_pending(timeout=5)

    assert not mm.session_boundary_pending
    assert not mm.memory_enabled
    assert mm.prefetch_all("new-session query", session_id="new-sid") == ""


def test_async_unexpected_worker_failure_completes_prompt_wait(
    monkeypatch, caplog
):
    class _PromptProvider(_RecordingProvider):
        def system_prompt_block(self):
            return "old memory"

    switch_started = threading.Event()
    release_switch = threading.Event()
    mm = _make_manager(_PromptProvider())

    def fail_switch(*_args, **_kwargs):
        switch_started.set()
        release_switch.wait(timeout=5)
        raise RuntimeError("unexpected manager worker failure")

    monkeypatch.setattr(mm, "_switch_providers_on_worker", fail_switch)
    with caplog.at_level("ERROR"):
        mm.commit_session_boundary_async([], new_session_id="B")
        assert switch_started.wait(timeout=1)
        prompt_results = []
        prompt_waiter = threading.Thread(
            target=lambda: prompt_results.append(mm.build_system_prompt())
        )
        prompt_waiter.start()
        prompt_waiter.join(timeout=0.05)
        waited_for_boundary = prompt_waiter.is_alive()

        release_switch.set()
        prompt_waiter.join(timeout=2)
        assert mm.flush_pending(timeout=2)

    assert waited_for_boundary
    assert not prompt_waiter.is_alive()
    assert prompt_results == [""]
    assert mm.build_system_prompt() == ""
    assert not mm.session_boundary_pending
    assert not mm.memory_enabled
    assert "failed unexpectedly; memory remains disabled" in caplog.text


def test_sync_unexpected_worker_failure_completes_prompt_wait(
    monkeypatch, caplog
):
    class _PromptProvider(_RecordingProvider):
        def system_prompt_block(self):
            return "old memory"

    switch_started = threading.Event()
    release_switch = threading.Event()
    mm = _make_manager(_PromptProvider())

    def fail_switch(*_args, **_kwargs):
        switch_started.set()
        release_switch.wait(timeout=5)
        raise RuntimeError("unexpected manager worker failure")

    monkeypatch.setattr(mm, "_switch_providers_on_worker", fail_switch)
    switch_results = []
    with caplog.at_level("ERROR"):
        switcher = threading.Thread(
            target=lambda: switch_results.append(mm.on_session_switch("B"))
        )
        switcher.start()
        assert switch_started.wait(timeout=1)
        prompt_results = []
        prompt_waiter = threading.Thread(
            target=lambda: prompt_results.append(mm.build_system_prompt())
        )
        prompt_waiter.start()
        prompt_waiter.join(timeout=0.05)
        waited_for_boundary = prompt_waiter.is_alive()

        release_switch.set()
        switcher.join(timeout=2)
        prompt_waiter.join(timeout=2)
        assert mm.flush_pending(timeout=2)

    assert waited_for_boundary
    assert not switcher.is_alive()
    assert not prompt_waiter.is_alive()
    assert switch_results == [False]
    assert prompt_results == [""]
    assert mm.build_system_prompt() == ""
    assert not mm.session_boundary_pending
    assert not mm.memory_enabled
    assert "failed unexpectedly; memory remains disabled" in caplog.text


def test_failed_rebind_blocks_all_new_session_lifecycle_payloads():
    class _FailedRebindProvider(_RecordingProvider):
        def __init__(self):
            super().__init__()
            self.session_id = "A"

        def initialize(self, session_id="", **kwargs):
            del kwargs
            self.session_id = session_id
            return True

        def on_session_switch(self, new_session_id, **kwargs):
            del kwargs
            self.calls.append(("switch_failed", self.session_id, new_session_id))
            raise RuntimeError("rebind failed")

        def on_turn_start(self, turn_number, message, **kwargs):
            del turn_number, kwargs
            self.calls.append(("turn", self.session_id, message))

        def on_pre_compress(self, messages):
            self.calls.append(("pre_compress", self.session_id, list(messages)))
            return "old-bank-context"

        def on_session_end(self, messages):
            self.calls.append(("session_end", self.session_id, list(messages)))

        def on_delegation(self, task, result, **kwargs):
            del kwargs
            self.calls.append(("delegation", self.session_id, task, result))

    provider = _FailedRebindProvider()
    mm = _make_manager(provider)
    mm.initialize_all("A")

    assert mm.on_session_switch("B") is False
    assert not mm.memory_enabled
    assert mm._bound_session_id == "A"

    payload = [{"role": "user", "content": "B transcript"}]
    mm.on_turn_start(1, "B turn")
    assert mm.on_pre_compress(payload) == ""
    mm.on_session_end(payload)
    mm.on_delegation("B task", "B result", child_session_id="B-child")

    assert provider.calls == [("switch_failed", "A", "B")]


def test_boundary_switch_succeeds_only_when_every_provider_rebinds():
    class _ExplodingSwitchProvider(_RecordingProvider):
        def on_session_switch(self, new_session_id: str, **kwargs) -> None:
            del new_session_id, kwargs
            raise RuntimeError("provider rebind blew up")

    good = _RecordingProvider()
    broken = _ExplodingSwitchProvider()
    mm = MemoryManager()
    mm._providers.extend([good, broken])
    mm._bound_session_id = "old-sid"

    assert mm.on_session_switch("new-sid") is False
    assert mm._bound_session_id == "old-sid"
    assert not mm.session_boundary_pending
    assert not mm.memory_enabled
    assert good.calls == [("switch", "new-sid", False)]

def test_boundary_superseded_before_start_preserves_valid_extraction():
    """Supersession skips obsolete rebinding, not accepted A extraction."""
    provider = _RecordingProvider()
    worker_started = threading.Event()
    release_worker = threading.Event()

    def blocking_sync(*_args, **_kwargs):
        worker_started.set()
        release_worker.wait(timeout=5)

    provider.sync_turn = blocking_sync  # type: ignore[method-assign]
    mm = _make_manager(provider)
    mm.sync_all("block worker", "reply", session_id="A")
    assert worker_started.wait(timeout=1)

    mm.commit_session_boundary_async(
        [{"role": "user", "content": "A history"}],
        new_session_id="B",
        parent_session_id="A",
    )
    mm.commit_session_boundary_async(
        [],
        new_session_id="C",
        parent_session_id="B",
    )
    release_worker.set()

    assert mm.flush_pending(timeout=2)
    assert provider.calls == [
        ("end", [{"role": "user", "content": "A history"}]),
        ("switch", "C", True),
    ]


def test_running_extraction_finishes_but_stale_switch_is_skipped():
    """A newer boundary supersedes only the stale extraction's commit phase."""
    extraction_started = threading.Event()
    release_extraction = threading.Event()

    class _BlockingEndProvider(_RecordingProvider):
        def on_session_end(self, messages):  # type: ignore[override]
            extraction_started.set()
            release_extraction.wait(timeout=5)
            self.calls.append(("end", list(messages)))

    provider = _BlockingEndProvider()
    mm = _make_manager(provider)
    mm.commit_session_boundary_async(
        [{"role": "user", "content": "A history"}],
        new_session_id="B",
        parent_session_id="A",
    )
    assert extraction_started.wait(timeout=1)

    mm.commit_session_boundary_async(
        [],
        new_session_id="C",
        parent_session_id="B",
    )
    release_extraction.set()

    assert mm.flush_pending(timeout=2)
    assert provider.calls == [
        ("end", [{"role": "user", "content": "A history"}]),
        ("switch", "C", True),
    ]


def test_intermediate_snapshot_is_skipped_when_parent_was_never_bound():
    """B history must not be extracted through a provider still bound to A."""
    first_extraction_started = threading.Event()
    release_first_extraction = threading.Event()

    class _BoundProvider(_RecordingProvider):
        def __init__(self):
            super().__init__()
            self.session_id = "A"

        def on_session_end(self, messages):  # type: ignore[override]
            self.calls.append(("end", self.session_id, list(messages)))
            if messages[0]["content"] == "A history":
                first_extraction_started.set()
                release_first_extraction.wait(timeout=5)

        def on_session_switch(self, new_session_id, **kwargs):  # type: ignore[override]
            self.session_id = new_session_id
            self.calls.append(("switch", new_session_id, kwargs.get("reset")))

    provider = _BoundProvider()
    mm = _make_manager(provider)
    mm._bound_session_id = "A"
    mm.commit_session_boundary_async(
        [{"role": "user", "content": "A history"}],
        new_session_id="B",
        parent_session_id="A",
    )
    assert first_extraction_started.wait(timeout=1)

    mm.commit_session_boundary_async(
        [{"role": "user", "content": "B history"}],
        new_session_id="C",
        parent_session_id="B",
    )
    release_first_extraction.set()

    assert mm.flush_pending(timeout=2)
    assert provider.calls == [
        ("end", "A", [{"role": "user", "content": "A history"}]),
        ("switch", "C", True),
    ]
    assert provider.session_id == "C"






def test_rejected_boundary_keeps_gate_closed_and_logs(caplog):
    class _RejectingExecutor:
        def submit(self, _fn):
            raise RuntimeError("executor rejected task")

    provider = _RecordingProvider()
    mm = _make_manager(provider)
    mm._sync_executor = _RejectingExecutor()  # type: ignore[assignment]

    with caplog.at_level("WARNING"):
        mm.commit_session_boundary_async([], new_session_id="B")

    assert not mm.session_boundary_pending
    assert not mm.memory_enabled
    assert "could not be queued; memory remains disabled" in caplog.text


class _OverlappingTransitionProvider(_RecordingProvider):
    def __init__(self):
        super().__init__()
        self.session_id = "A"
        self.extraction_started = threading.Event()
        self.release_extraction = threading.Event()
        self.overlap = False
        self._active_hooks = 0
        self._active_lock = threading.Lock()

    def _enter_hook(self) -> None:
        with self._active_lock:
            self.overlap = self.overlap or self._active_hooks > 0
            self._active_hooks += 1

    def _leave_hook(self) -> None:
        with self._active_lock:
            self._active_hooks -= 1

    def on_session_end(self, messages):  # type: ignore[override]
        self._enter_hook()
        try:
            self.extraction_started.set()
            self.release_extraction.wait(timeout=5)
            self.calls.append(("end", list(messages)))
        finally:
            self._leave_hook()

    def on_session_switch(self, new_session_id, **kwargs):  # type: ignore[override]
        self._enter_hook()
        try:
            self.session_id = new_session_id
            self.calls.append(
                (
                    "switch",
                    new_session_id,
                    kwargs.get("reason"),
                    kwargs.get("rewound", False),
                )
            )
        finally:
            self._leave_hook()


def _assert_pending_new_superseded_by_direct_transition(
    *, target: str, transition_kwargs: Dict[str, Any]
) -> None:
    provider = _OverlappingTransitionProvider()
    mm = _make_manager(provider)
    mm._bound_session_id = "A"

    mm.commit_session_boundary_async(
        [{"role": "user", "content": "A history"}],
        new_session_id="B",
        parent_session_id="A",
    )
    assert provider.extraction_started.wait(timeout=1)

    transition_reserved = threading.Event()
    original_reserve = mm.reserve_session_boundary

    def observed_reserve() -> int:
        generation = original_reserve()
        transition_reserved.set()
        return generation

    mm.reserve_session_boundary = observed_reserve  # type: ignore[method-assign]
    results = []
    transition = threading.Thread(
        target=lambda: results.append(mm.on_session_switch(target, **transition_kwargs))
    )
    transition.start()
    reserved_before_release = transition_reserved.wait(timeout=1)
    waited_for_extraction = transition.is_alive()

    provider.release_extraction.set()
    transition.join(timeout=2)

    assert reserved_before_release
    assert waited_for_extraction

    assert not transition.is_alive()
    assert results == [True]
    assert mm.flush_pending(timeout=2)
    assert provider.session_id == target
    assert provider.calls == [
        ("end", [{"role": "user", "content": "A history"}]),
        (
            "switch",
            target,
            transition_kwargs.get("reason"),
            transition_kwargs.get("rewound", False),
        ),
    ]
    assert not provider.overlap


@pytest.mark.parametrize("reason", ["resume", "branch"])
def test_pending_new_is_superseded_by_resume_or_branch(reason):
    _assert_pending_new_superseded_by_direct_transition(
        target=f"latest-{reason}", transition_kwargs={"reason": reason}
    )


@pytest.mark.parametrize(
    ("target", "transition_kwargs"),
    [
        ("compressed", {"reason": "compression"}),
        ("A", {"rewound": True}),
    ],
)
def test_pending_new_is_superseded_by_compression_or_rewind(
    target, transition_kwargs
):
    _assert_pending_new_superseded_by_direct_transition(
        target=target, transition_kwargs=transition_kwargs
    )


class _BarrierReadProvider(_RecordingProvider):
    def __init__(self, operation: str):
        super().__init__()
        self.operation = operation
        self.session_id = "A"
        self.read_started = threading.Event()
        self.release_read = threading.Event()
        self.switch_started = threading.Event()
        self.overlap = False
        self._active_hooks = 0
        self._active_lock = threading.Lock()
        self._blocked_once = False

    def get_tool_schemas(self):
        return [
            {"name": "barrier_recall", "description": "Recall", "parameters": {}}
        ]

    def _enter_hook(self) -> None:
        with self._active_lock:
            self.overlap = self.overlap or self._active_hooks > 0
            self._active_hooks += 1

    def _leave_hook(self) -> None:
        with self._active_lock:
            self._active_hooks -= 1

    def _read(self, operation: str) -> str:
        self._enter_hook()
        try:
            session_id = self.session_id
            if operation == self.operation and not self._blocked_once:
                self._blocked_once = True
                self.read_started.set()
                self.release_read.wait(timeout=5)
            self.calls.append(("read", operation, session_id))
            if operation == "tool":
                return json.dumps({"bank": session_id})
            return f"bank:{session_id}"
        finally:
            self._leave_hook()

    def system_prompt_block(self):
        return self._read("prompt")

    def prefetch(self, query, *, session_id=""):
        del query, session_id
        return self._read("prefetch")

    def handle_tool_call(self, tool_name, args, **kwargs):
        del tool_name, args, kwargs
        return self._read("tool")

    def on_session_switch(self, new_session_id, **kwargs):  # type: ignore[override]
        del kwargs
        self._enter_hook()
        try:
            self.switch_started.set()
            self.session_id = new_session_id
            self.calls.append(("switch", new_session_id))
        finally:
            self._leave_hook()


@pytest.mark.parametrize("operation", ["prompt", "prefetch", "tool"])
def test_synchronous_read_discards_stale_generation_without_hook_overlap(operation):
    provider = _BarrierReadProvider(operation)
    mm = _make_manager(provider)
    mm._bound_session_id = "A"
    mm._tool_to_provider["barrier_recall"] = provider
    results = []

    if operation == "prompt":
        invoke = mm.build_system_prompt
    elif operation == "prefetch":
        invoke = lambda: mm.prefetch_all("query", session_id="A")
    else:
        invoke = lambda: mm.handle_tool_call("barrier_recall", {}, session_id="A")

    reader = threading.Thread(target=lambda: results.append(invoke()))
    reader.start()
    assert provider.read_started.wait(timeout=1)

    mm.commit_session_boundary_async(
        [], new_session_id="B", parent_session_id="A"
    )
    switch_overlapped_read = provider.switch_started.wait(timeout=0.05)
    provider.release_read.set()

    reader.join(timeout=2)
    assert not reader.is_alive()
    assert mm.flush_pending(timeout=2)
    assert provider.session_id == "B"
    assert not provider.overlap
    assert not switch_overlapped_read
    if operation == "prompt":
        assert results == ["bank:B"]
    elif operation == "prefetch":
        assert results == [""]
    else:
        assert "error" in json.loads(results[0])


def test_failed_rebind_completes_prompt_wait_but_keeps_memory_disabled():
    switch_started = threading.Event()
    release_switch = threading.Event()

    class _BlockingFailedSwitchProvider(_RecordingProvider):
        def on_session_switch(self, new_session_id, **kwargs):  # type: ignore[override]
            del new_session_id, kwargs
            switch_started.set()
            release_switch.wait(timeout=5)
            raise RuntimeError("rebind failed")

        def system_prompt_block(self):
            return "old memory"

    mm = _make_manager(_BlockingFailedSwitchProvider())
    mm.commit_session_boundary_async([], new_session_id="B", parent_session_id="A")
    assert switch_started.wait(timeout=1)
    results = []
    prompt_builder = threading.Thread(
        target=lambda: results.append(mm.build_system_prompt())
    )
    prompt_builder.start()
    prompt_builder.join(timeout=0.05)
    waited_for_switch = prompt_builder.is_alive()

    release_switch.set()
    prompt_builder.join(timeout=2)

    assert waited_for_switch

    assert not prompt_builder.is_alive()
    assert results == [""]
    assert not mm.session_boundary_pending
    assert not mm.memory_enabled

def test_boundary_commit_noop_without_providers():
    mm = MemoryManager()
    # Must not create the executor or raise.
    mm.commit_session_boundary_async([{"role": "user", "content": "x"}], new_session_id="s")
    assert mm._sync_executor is None
