import inspect

import psutil

from tui_gateway import slash_worker


def test_is_orphaned_true_when_ppid_changes():
    # Our parent went away and we were reparented to a subreaper/init.
    assert slash_worker._is_orphaned(1234, 1.0, getppid=lambda: 999999) is True


def test_is_orphaned_true_when_parent_pid_is_reused():
    parent = psutil.Process()
    assert (
        slash_worker._is_orphaned(
            parent.pid,
            parent.create_time() + 1.0,
            getppid=lambda: parent.pid,
        )
        is True
    )


def test_is_orphaned_false_when_direct_parent_identity_is_unchanged():
    parent = psutil.Process()
    assert (
        slash_worker._is_orphaned(
            parent.pid,
            parent.create_time(),
            getppid=lambda: parent.pid,
        )
        is False
    )


def test_parent_death_watchdog_contract_carries_parent_identity():
    assert list(inspect.signature(slash_worker._is_orphaned).parameters) == [
        "original_ppid",
        "parent_create_time",
        "getppid",
    ]
    assert list(inspect.signature(slash_worker._start_parent_death_watchdog).parameters) == [
        "original_ppid",
        "parent_create_time",
    ]
