"""Regression guard: end-of-turn memory sync must not block the turn.

Before this fix, ``MemoryManager.sync_all`` / ``queue_prefetch_all`` looped
``provider.sync_turn`` / ``provider.queue_prefetch`` INLINE on the
turn-completion path. A provider making a blocking network/daemon call (a
misconfigured Hindsight daemon was observed blocking ~298s before failing)
held ``run_conversation`` open long after the user saw their response, so
every interface (CLI, TUI, gateway) kept the agent marked "running" for
minutes and any follow-up message triggered an aggressive interrupt that
dropped the message.

The fix dispatches provider work to a single-worker background executor.
``sync_all`` / ``queue_prefetch_all`` return immediately; the work completes
(or fails, logged) in the background. ``flush_pending`` provides a barrier
for session boundaries and deterministic tests. ``shutdown_all`` gives accepted
work a bounded drain window, then reports cancellation and process-exit risk.
"""
import json
import logging
import os
import subprocess
import sys
import textwrap
import threading
import time
from concurrent.futures import Future
from pathlib import Path

import pytest

from agent.memory_provider import MemoryProvider
from agent.memory_manager import MemoryManager


class _SlowProvider(MemoryProvider):
    """Provider whose sync/prefetch block, simulating a slow backend."""

    _name = "slow"

    def __init__(self, delay: float = 1.0):
        self._delay = delay
        self.sync_done = False
        self.prefetch_done = False

    @property
    def name(self) -> str:
        return self._name

    def initialize(self, session_id: str = "", **kwargs) -> None:
        pass

    def is_available(self) -> bool:
        return True

    def system_prompt_block(self) -> str:
        return ""

    def prefetch(self, query, *, session_id: str = "") -> str:
        return ""

    def queue_prefetch(self, query, *, session_id: str = "") -> None:
        time.sleep(self._delay)
        self.prefetch_done = True

    def sync_turn(self, user_content, assistant_content, *, session_id: str = "", messages=None) -> None:
        time.sleep(self._delay)
        self.sync_done = True

    def get_tool_schemas(self):
        return []

    def handle_tool_call(self, tool_name, args, **kwargs) -> str:
        return ""


def test_sync_all_does_not_block_on_slow_provider():
    """The crux of the fix: a slow provider must NOT stall the caller."""
    mgr = MemoryManager()
    mgr.add_provider(_SlowProvider(delay=2.0))

    t0 = time.time()
    mgr.sync_all("hi", "hey", session_id="s1")
    mgr.queue_prefetch_all("hi", session_id="s1")
    elapsed = time.time() - t0

    # Provider blocks 2s per call inline; off-thread dispatch returns ~instantly.
    assert elapsed < 0.5, f"turn-completion path blocked {elapsed:.2f}s"


def test_background_work_still_completes():
    """Dispatching off-thread must not silently drop the write."""
    mgr = MemoryManager()
    p = _SlowProvider(delay=0.1)
    mgr.add_provider(p)

    mgr.sync_all("hi", "hey", session_id="s1")
    mgr.queue_prefetch_all("hi", session_id="s1")

    assert mgr.flush_pending(timeout=10) is True
    assert p.sync_done is True
    assert p.prefetch_done is True


def test_flush_pending_no_executor_is_true():
    """flush_pending must be a no-op (return True) before any sync ran."""
    mgr = MemoryManager()
    assert mgr.flush_pending(timeout=1) is True


def test_no_providers_does_not_create_executor():
    """Builtin-only / no-provider sessions must not spawn an executor."""
    mgr = MemoryManager()
    mgr.sync_all("hi", "hey")
    mgr.queue_prefetch_all("hi")
    assert mgr._sync_executor is None


def test_shutdown_all_is_bounded_with_wedged_provider():
    """A provider that never returns must not hang teardown."""
    mgr = MemoryManager()
    mgr.add_provider(_SlowProvider(delay=30.0))
    mgr.sync_all("hi", "hey")

    t0 = time.time()
    mgr.shutdown_all()
    elapsed = time.time() - t0

    # Bounded by _SYNC_DRAIN_TIMEOUT_S (5s) plus a little slack.
    assert elapsed < 8.0, f"shutdown blocked {elapsed:.1f}s on wedged provider"


def test_writes_are_serialized_in_order():
    """Single-worker executor must preserve turn ordering (N before N+1)."""
    order = []

    class _OrderProvider(_SlowProvider):
        _name = "order"

        def sync_turn(self, user_content, assistant_content, *, session_id="", messages=None):
            order.append(user_content)

    mgr = MemoryManager()
    mgr.add_provider(_OrderProvider(delay=0.0))
    for i in range(5):
        mgr.sync_all(f"turn-{i}", "resp", session_id="s1")
    assert mgr.flush_pending(timeout=10) is True
    assert order == [f"turn-{i}" for i in range(5)]


def test_shutdown_drains_queued_writes_and_boundary_in_fifo_order():
    """Accepted writes and a boundary remain FIFO through shutdown."""
    started = threading.Event()
    release = threading.Event()
    calls = []

    class _BlockingProvider(_SlowProvider):
        def sync_turn(
            self,
            user_content,
            assistant_content,
            *,
            session_id="",
            messages=None,
        ):
            del assistant_content, session_id, messages
            if user_content == "turn-0":
                started.set()
                assert release.wait(timeout=2)
            calls.append(("sync", user_content))

        def on_session_end(self, messages):
            calls.append(("end", messages[0]["content"]))

        def on_session_switch(self, new_session_id, **kwargs):
            del kwargs
            calls.append(("switch", new_session_id))

    mgr = MemoryManager()
    mgr.add_provider(_BlockingProvider(delay=0))
    mgr.sync_all("turn-0", "response")
    assert started.wait(timeout=1)
    mgr.sync_all("turn-1", "response")
    mgr.commit_session_boundary_async(
        [{"role": "user", "content": "old-session"}],
        new_session_id="new-session",
    )

    threading.Timer(0.05, release.set).start()
    mgr.shutdown_all()

    assert calls == [
        ("sync", "turn-0"),
        ("sync", "turn-1"),
        ("end", "old-session"),
        ("switch", "new-session"),
    ]
    assert mgr.shutdown_drain_state == {
        "status": "drained",
        "abandoned_writes": 0,
        "abandoned_prefetches": 0,
        "active_tasks": 0,
    }


class _BlockingLifecycleProvider(_SlowProvider):
    """Provider with observable work and shutdown ordering."""

    def __init__(self):
        super().__init__(delay=0.0)
        self.first_started = threading.Event()
        self.release_first = threading.Event()
        self.shutdown_called = threading.Event()
        self.operations = []

    def sync_turn(
        self,
        user_content,
        assistant_content,
        *,
        session_id="",
        messages=None,
    ):
        del assistant_content, session_id, messages
        self.operations.append(f"sync:{user_content}:start")
        if user_content == "first":
            self.first_started.set()
            self.release_first.wait(timeout=5)
        self.operations.append(f"sync:{user_content}:done")

    def queue_prefetch(self, query, *, session_id=""):
        del session_id
        self.operations.append(f"prefetch:{query}")

    def shutdown(self):
        self.operations.append("shutdown")
        self.shutdown_called.set()



def test_flush_submission_linearizes_before_executor_detach():
    """A barrier accepted before detach remains part of the shutdown drain."""
    mgr = MemoryManager()
    provider = _BlockingLifecycleProvider()
    mgr.add_provider(provider)
    mgr.sync_all("first", "response")
    assert provider.first_started.wait(timeout=2)

    executor = mgr._sync_executor
    assert executor is not None
    original_submit = executor.submit
    original_shutdown = executor.shutdown
    barrier_submit_started = threading.Event()
    release_barrier_submit = threading.Event()
    detach_started = threading.Event()
    shutdown_calling = threading.Event()
    flush_result = []

    def controlled_submit(fn, *args, **kwargs):
        if threading.current_thread().name == "memory-flush":
            barrier_submit_started.set()
            if not release_barrier_submit.wait(timeout=5):
                raise AssertionError("barrier submission was not released")
        return original_submit(fn, *args, **kwargs)

    def observed_shutdown(*args, **kwargs):
        if kwargs.get("wait") is False:
            detach_started.set()
        return original_shutdown(*args, **kwargs)

    executor.submit = controlled_submit
    executor.shutdown = observed_shutdown

    flush_thread = threading.Thread(
        target=lambda: flush_result.append(mgr.flush_pending(timeout=2)),
        name="memory-flush",
    )
    shutdown_thread = threading.Thread(
        target=lambda: (shutdown_calling.set(), mgr.shutdown_all()),
        name="memory-shutdown-caller",
    )
    flush_thread.start()
    assert barrier_submit_started.wait(timeout=2)
    shutdown_thread.start()
    assert shutdown_calling.wait(timeout=2)
    try:
        assert not detach_started.wait(timeout=0.1)
    finally:
        release_barrier_submit.set()
        provider.release_first.set()

    flush_thread.join(timeout=2)
    shutdown_thread.join(timeout=2)
    assert not flush_thread.is_alive()
    assert not shutdown_thread.is_alive()
    assert detach_started.is_set()
    assert flush_result == [True]


def test_flush_after_detach_waits_for_explicit_drain_completion(monkeypatch):
    """A detached executor is not drained until its final wait succeeds."""
    import agent.memory_manager as memory_manager_module

    monkeypatch.setattr(memory_manager_module, "_SYNC_DRAIN_TIMEOUT_S", 0.02)
    mgr = MemoryManager()
    provider = _BlockingLifecycleProvider()
    mgr.add_provider(provider)
    mgr.sync_all("first", "response")
    assert provider.first_started.wait(timeout=2)

    executor = mgr._sync_executor
    assert executor is not None
    original_shutdown = executor.shutdown
    detach_started = threading.Event()
    drain_wait_started = threading.Event()

    def observed_shutdown(*args, **kwargs):
        if kwargs.get("wait") is False:
            detach_started.set()
        elif kwargs.get("wait") is True:
            drain_wait_started.set()
        return original_shutdown(*args, **kwargs)

    executor.shutdown = observed_shutdown
    shutdown_thread = threading.Thread(target=mgr.shutdown_all)
    shutdown_thread.start()
    try:
        assert detach_started.wait(timeout=2)
        assert drain_wait_started.wait(timeout=2)
        assert mgr.flush_pending(timeout=0.02) is False

        flush_result = []
        flush_returned = threading.Event()

        def flush_until_drained():
            flush_result.append(mgr.flush_pending(timeout=2))
            flush_returned.set()

        flush_thread = threading.Thread(target=flush_until_drained)
        flush_thread.start()
        assert not flush_returned.wait(timeout=0.1)
    finally:
        provider.release_first.set()

    assert flush_returned.wait(timeout=2)
    flush_thread.join(timeout=2)
    shutdown_thread.join(timeout=2)
    assert flush_result == [True]


def test_shutdown_drains_accepted_prefetch_queued_behind_inflight_sync():
    """Shutdown preserves FIFO work accepted behind an in-flight task."""
    mgr = MemoryManager()
    provider = _BlockingLifecycleProvider()
    mgr.add_provider(provider)
    mgr.sync_all("first", "response")
    assert provider.first_started.wait(timeout=2)
    mgr.queue_prefetch_all("second")

    executor = mgr._sync_executor
    assert executor is not None
    shutdown_started = threading.Event()
    original_shutdown = executor.shutdown

    def observed_shutdown(*args, **kwargs):
        if kwargs.get("wait") is False:
            shutdown_started.set()
        return original_shutdown(*args, **kwargs)

    executor.shutdown = observed_shutdown
    shutdown_thread = threading.Thread(target=mgr.shutdown_all)
    shutdown_thread.start()
    try:
        assert shutdown_started.wait(timeout=2)
    finally:
        provider.release_first.set()
    shutdown_thread.join(timeout=5)

    assert not shutdown_thread.is_alive()
    assert provider.operations == [
        "sync:first:start",
        "sync:first:done",
        "prefetch:second",
        "shutdown",
    ]


def test_shutdown_timeout_cancels_queued_work_and_reports_active_risk(
    monkeypatch, caplog
):
    """Timeout cancels queued work and keeps active process-exit risk visible."""
    import agent.memory_manager as memory_manager_module

    monkeypatch.setattr(memory_manager_module, "_SYNC_DRAIN_TIMEOUT_S", 0.02)
    mgr = MemoryManager()
    provider = _BlockingLifecycleProvider()
    mgr.add_provider(provider)
    mgr.sync_all("first", "response")
    assert provider.first_started.wait(timeout=2)
    mgr.sync_all("queued-write", "response")
    mgr.queue_prefetch_all("queued-prefetch")

    started = time.monotonic()
    try:
        with caplog.at_level(logging.WARNING, logger="agent.memory_manager"):
            mgr.shutdown_all()
        elapsed = time.monotonic() - started
        assert elapsed < 0.5
        assert mgr.shutdown_drain_state == {
            "status": "timed_out",
            "abandoned_writes": 1,
            "abandoned_prefetches": 1,
            "active_tasks": 1,
        }
        assert "abandoned 1 queued write(s) and 1 queued prefetch(es)" in caplog.text
        assert "1 active manager task(s)" in caplog.text
        assert "at risk if the process exits" in caplog.text
        assert "no accepted work was cancelled" not in caplog.text
        assert not provider.shutdown_called.is_set()

        # Submission after shutdown began must be rejected, not run inline or
        # accepted by a replacement executor.
        mgr.sync_all("late", "response")
    finally:
        provider.release_first.set()

    assert provider.shutdown_called.wait(timeout=2)
    assert provider.operations == [
        "sync:first:start",
        "sync:first:done",
        "shutdown",
    ]
    assert mgr.shutdown_drain_state == {
        "status": "completed_with_loss",
        "abandoned_writes": 1,
        "abandoned_prefetches": 1,
        "active_tasks": 0,
    }

def test_shutdown_cancellation_linearizes_with_terminal_publication(monkeypatch):
    """A successful cancellation cannot be lost to a racing finalizer."""
    import agent.memory_manager as memory_manager_module

    monkeypatch.setattr(memory_manager_module, "_SYNC_DRAIN_TIMEOUT_S", 0.02)
    mgr = MemoryManager()
    provider = _BlockingLifecycleProvider()
    mgr.add_provider(provider)
    mgr.sync_all("first", "response")
    assert provider.first_started.wait(timeout=2)
    mgr.sync_all("queued-write", "response")

    with mgr._sync_executor_lock:
        queued_future = list(mgr._background_futures)[-1]
    assert not queued_future.running()

    finalizer_published = threading.Event()

    class _ObservedLock:
        def __init__(self, delegate):
            self._delegate = delegate

        def acquire(self, *args, **kwargs):
            return self._delegate.acquire(*args, **kwargs)

        def release(self):
            self._delegate.release()

        def __enter__(self):
            self.acquire()
            return self

        def __exit__(self, *args):
            self.release()
            if threading.current_thread().name == "memory-shutdown-finalizer":
                finalizer_published.set()

    mgr._sync_executor_lock = _ObservedLock(mgr._sync_executor_lock)
    original_cancel = queued_future.cancel
    cancellation_succeeded = threading.Event()

    def controlled_cancel():
        # Future.cancel() must run outside the executor-state lock because its
        # done callback acquires that same lock synchronously.
        assert mgr._sync_executor_lock.acquire(blocking=False)
        mgr._sync_executor_lock.release()
        cancelled = original_cancel()
        assert cancelled
        cancellation_succeeded.set()
        provider.release_first.set()
        # On the broken interleaving the finalizer publishes clean "drained"
        # here, before shutdown_all() records this successful cancellation.
        finalizer_published.wait(timeout=0.1)
        return cancelled

    queued_future.cancel = controlled_cancel
    mgr.shutdown_all()

    assert cancellation_succeeded.is_set()
    assert mgr._sync_drain_complete.wait(timeout=2)
    assert mgr.shutdown_drain_state == {
        "status": "completed_with_loss",
        "abandoned_writes": 1,
        "abandoned_prefetches": 0,
        "active_tasks": 0,
    }


def test_shutdown_state_waits_for_provider_shutdown(monkeypatch):
    import agent.memory_manager as memory_manager_module

    monkeypatch.setattr(memory_manager_module, "_SYNC_DRAIN_TIMEOUT_S", 0.02)
    mgr = MemoryManager()
    provider = _SlowProvider(delay=0)
    shutdown_started = threading.Event()
    release_shutdown = threading.Event()

    def blocking_shutdown():
        shutdown_started.set()
        release_shutdown.wait(timeout=2)

    provider.shutdown = blocking_shutdown
    mgr.add_provider(provider)
    try:
        mgr.shutdown_all()
        assert shutdown_started.is_set()
        assert mgr.shutdown_drain_state == {
            "status": "timed_out",
            "abandoned_writes": 0,
            "abandoned_prefetches": 0,
            "active_tasks": 0,
        }
    finally:
        release_shutdown.set()

    assert mgr._sync_drain_complete.wait(timeout=2)
    assert mgr.shutdown_drain_state["status"] == "drained"


def test_shutdown_aggregates_provider_owned_abandonment():
    mgr = MemoryManager()
    provider = _SlowProvider(delay=0)
    provider.shutdown = lambda: {
        "status": "abandoned",
        "abandoned_writes": 2,
        "abandoned_prefetches": 1,
    }
    mgr.add_provider(provider)

    mgr.shutdown_all()

    assert mgr.shutdown_drain_state == {
        "status": "completed_with_loss",
        "abandoned_writes": 2,
        "abandoned_prefetches": 1,
        "active_tasks": 0,
    }

def test_subprocess_exit_reports_active_write_at_risk(tmp_path):
    """A timed-out daemon write is reported at risk before process exit."""
    marker = tmp_path / "accepted-write-finished"
    script = textwrap.dedent(
        f"""
        import json
        import threading
        import time
        from pathlib import Path

        import agent.memory_manager as memory_manager_module
        from agent.memory_manager import MemoryManager
        from agent.memory_provider import MemoryProvider

        started = threading.Event()
        marker = Path({str(marker)!r})

        class SlowWriteProvider(MemoryProvider):
            @property
            def name(self):
                return "subprocess-slow-write"

            def initialize(self, session_id="", **kwargs):
                del session_id, kwargs

            def is_available(self):
                return True

            def get_tool_schemas(self):
                return []

            def sync_turn(self, user_content, assistant_content, **kwargs):
                del user_content, assistant_content, kwargs
                started.set()
                time.sleep(0.3)
                marker.write_text("finished", encoding="utf-8")

        memory_manager_module._SYNC_DRAIN_TIMEOUT_S = 0.02
        manager = MemoryManager()
        manager.add_provider(SlowWriteProvider())
        manager.sync_all("accepted", "response")
        assert started.wait(timeout=1)
        manager.shutdown_all()
        print(json.dumps(manager.shutdown_drain_state))
        """
    )
    env = os.environ.copy()
    env.update(
        HOME=str(tmp_path),
        HERMES_HOME=str(tmp_path / "hermes-home"),
        PYTHONPATH="",
        HERMES_DESKTOP="",
        HERMES_DESKTOP_TERMINAL="",
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=3,
        check=True,
    )

    state = json.loads(completed.stdout.strip().splitlines()[-1])
    assert state == {
        "status": "timed_out",
        "abandoned_writes": 0,
        "abandoned_prefetches": 0,
        "active_tasks": 1,
    }
    assert "at risk if the process exits" in completed.stderr
    assert "no accepted work was cancelled" not in completed.stderr
    assert not marker.exists()
    time.sleep(0.4)
    assert not marker.exists()


def test_executor_creation_failure_drops_work_without_inline_provider_call(
    monkeypatch, caplog
):
    """Worker creation failure is warned and never runs provider code inline."""
    import tools.daemon_pool as daemon_pool

    def fail_creation(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("no worker")

    monkeypatch.setattr(daemon_pool, "DaemonThreadPoolExecutor", fail_creation)
    mgr = MemoryManager()
    provider = _SlowProvider(delay=0.0)
    mgr.add_provider(provider)

    with caplog.at_level(logging.WARNING):
        mgr.sync_all("write", "response")

    assert provider.sync_done is False
    assert "Failed to create memory sync executor: no worker" in caplog.text


def test_executor_submission_failure_drops_work_without_inline_provider_call(caplog):
    """Worker submission failure is warned and never runs provider code inline."""

    class RejectingExecutor:
        def submit(self, fn):
            del fn
            raise RuntimeError("not accepting")

        def shutdown(self, *args, **kwargs):
            del args, kwargs

    mgr = MemoryManager()
    provider = _SlowProvider(delay=0.0)
    mgr.add_provider(provider)
    mgr._sync_executor = RejectingExecutor()

    with caplog.at_level(logging.WARNING):
        mgr.sync_all("write", "response")

    assert provider.sync_done is False
    assert "Memory background task submission failed: not accepting" in caplog.text


def test_executor_drain_failure_leaves_providers_open(caplog):
    """Failed wait=True cannot authorize closing providers under queued work."""

    class FailingDrainExecutor:
        def __init__(self):
            self.accepted = []
            self.drain_wait_called = threading.Event()

        def submit(self, fn):
            future = Future()
            self.accepted.append((fn, future))
            return future

        def shutdown(self, *, wait=True, **kwargs):
            del kwargs
            if wait:
                self.drain_wait_called.set()
                raise RuntimeError("drain failed")

    mgr = MemoryManager()
    provider = _BlockingLifecycleProvider()
    executor = FailingDrainExecutor()
    mgr.add_provider(provider)
    mgr._sync_executor = executor
    mgr.sync_all("accepted", "response")
    assert len(executor.accepted) == 1

    with caplog.at_level(logging.WARNING):
        mgr.shutdown_all()

    assert executor.drain_wait_called.wait(timeout=2)
    assert provider.first_started.is_set() is False
    assert provider.shutdown_called.is_set() is False
    assert mgr._sync_drain_complete.is_set() is False
    assert "Memory sync executor drain wait failed: drain failed" in caplog.text


def test_concurrent_shutdown_is_single_reversed_and_permanently_closed(monkeypatch):
    """Concurrent callers share one finalizer and close each provider once."""
    import agent.memory_manager as memory_manager_module

    monkeypatch.setattr(memory_manager_module, "_SYNC_DRAIN_TIMEOUT_S", 0.02)
    mgr = MemoryManager()
    shutdown_order = []
    builtin = _BlockingLifecycleProvider()
    builtin._name = "builtin"
    external = _SlowProvider(delay=0.0)
    external._name = "external"
    builtin.shutdown = lambda: shutdown_order.append("builtin")
    external.shutdown = lambda: shutdown_order.append("external")
    mgr.add_provider(builtin)
    mgr.add_provider(external)
    mgr.sync_all("first", "response")
    assert builtin.first_started.wait(timeout=2)

    executor = mgr._sync_executor
    assert executor is not None
    original_shutdown = executor.shutdown
    executor_shutdown_calls = []
    drain_wait_started = threading.Event()

    def observed_shutdown(*args, **kwargs):
        wait = kwargs.get("wait")
        executor_shutdown_calls.append(wait)
        if wait is True:
            drain_wait_started.set()
        return original_shutdown(*args, **kwargs)

    executor.shutdown = observed_shutdown
    caller_gate = threading.Barrier(5)
    caller_errors = []

    def call_shutdown():
        try:
            caller_gate.wait(timeout=2)
            mgr.shutdown_all()
        except BaseException as exc:
            caller_errors.append(exc)

    callers = [threading.Thread(target=call_shutdown) for _ in range(4)]
    for caller in callers:
        caller.start()
    caller_gate.wait(timeout=2)
    try:
        assert drain_wait_started.wait(timeout=2)
        for caller in callers:
            caller.join(timeout=2)
        assert not any(caller.is_alive() for caller in callers)
        assert shutdown_order == []
        finalizer = mgr._shutdown_finalizer
        assert finalizer is not None
        assert finalizer.is_alive()
    finally:
        builtin.release_first.set()

    finalizer.join(timeout=2)
    assert not finalizer.is_alive()
    assert caller_errors == []
    assert executor_shutdown_calls == [False, True]
    assert shutdown_order == ["external", "builtin"]

    mgr.sync_all("late", "response")
    mgr.queue_prefetch_all("late")
    assert mgr.flush_pending(timeout=0.1) is True
    assert mgr._sync_executor is None
