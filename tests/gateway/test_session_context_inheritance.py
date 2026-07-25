"""Cross-session ContextVar *inheritance* leak guard.

Companion to ``tests/tools/test_local_env_session_leak.py``. That file covers
the ``os.environ``-mirror leak (a subprocess inheriting a foreign *global* when
this task's ContextVar is ``_UNSET``). THIS file covers a distinct, subtler
variant that the ``_UNSET``-strip guard does NOT catch:

    Each gateway message is processed in its own asyncio task, created via
    ``create_task`` — which snapshots the spawning context with
    ``copy_context()``. If message B's task is created from a context where a
    *concurrent* message A had ALREADY called ``set_session_vars``, B inherits
    A's **set** ContextVars. Between B's task start and B's own
    ``set_session_vars`` call, any subprocess B spawns reads A's
    ``HERMES_SESSION_*`` identity through the subprocess-env bridge. The bridge's
    strip-on-``_UNSET`` rule is no help: the inherited vars are set-to-A, not
    ``_UNSET``.

Verified in production 2026-06-21: a ``/bug`` turn ran ``bug_thread.py whoami``
and read a concurrent session's ticket (``cursor-captive-modals``) instead of
its own, because its task inherited that session's bound ContextVars.

The fix: ``gateway.session_context.reset_session_vars`` resets every session var
to ``_UNSET`` at the top of the per-message handler (``GatewayRunner._handle_message``),
*before* any work, so an inherited identity is dropped and the pre-bind window
strips safe instead of leaking the sibling's. The handler then binds its own
session a few steps later.
"""
import asyncio
from contextvars import copy_context
from typing import Any

import pytest

import gateway.session_context as sc
from gateway.session_context import (
    _SESSION_ASYNC_DELIVERY,
    _UNSET,
    _VAR_MAP,
    async_delivery_supported,
    reset_session_vars,
    set_session_vars,
)
from tools.environments.local import _make_run_env

SESSION_VARS = list(_VAR_MAP.keys())

MINE: dict[str, Any] = dict(
    session_key="agent:main:discord:thread:MINE:MINE",
    platform="discord",
    chat_id="MINE_CHAT",
    thread_id="MINE_THREAD",
    user_id="MINE_USER",
    chat_name="mine",
    message_id="MINE_MSG",
    coding_workflow="hybrid-v1",
)
FOREIGN: dict[str, Any] = dict(
    session_key="agent:main:discord:thread:FOREIGN:FOREIGN",
    platform="discord",
    chat_id="FOREIGN_CHAT",
    thread_id="FOREIGN_THREAD",
    user_id="FOREIGN_USER",
    chat_name="foreign",
    message_id="FOREIGN_MSG",
    coding_workflow="coupled-v1",
)


@pytest.fixture(autouse=True)
def _isolate_session_context():
    """Clean ContextVar + engaged-latch slate per test, restored afterwards."""
    import os

    saved_env = {k: os.environ.get(k) for k in SESSION_VARS}
    saved_ctx = {name: var.get() for name, var in _VAR_MAP.items()}
    saved_async = _SESSION_ASYNC_DELIVERY.get()
    saved_engaged = sc._session_context_engaged
    for var in _VAR_MAP.values():
        var.set(_UNSET)
    _SESSION_ASYNC_DELIVERY.set(_UNSET)
    sc._session_context_engaged = True  # a concurrent multi-session host is engaged
    try:
        yield
    finally:
        for var, val in zip(_VAR_MAP.values(), saved_ctx.values()):
            var.set(val)
        _SESSION_ASYNC_DELIVERY.set(saved_async)
        sc._session_context_engaged = saved_engaged
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _spawn_view():
    """What a subprocess spawned right now would see for the session vars."""
    env = _make_run_env({})
    return {
        "HERMES_CODING_WORKFLOW": env.get("HERMES_CODING_WORKFLOW"),
        "HERMES_SESSION_CHAT_ID": env.get("HERMES_SESSION_CHAT_ID"),
        "HERMES_SESSION_THREAD_ID": env.get("HERMES_SESSION_THREAD_ID"),
        "HERMES_SESSION_KEY": env.get("HERMES_SESSION_KEY"),
    }


async def _child_turn(reset_first: bool):
    """Simulate message B's processing task: created (copy_context) from a
    parent context where message A already bound its session.

    Returns the subprocess view from the *pre-bind window* — before B calls its
    own set_session_vars. With ``reset_first`` (the fix), B resets at entry.
    """
    captured = {}

    def _b_body():
        if reset_first:
            reset_session_vars()  # THE FIX: handler-entry reset
        captured["window"] = _spawn_view()  # pre-bind window
        set_session_vars(**FOREIGN)  # B binds its own session
        captured["bound"] = _spawn_view()

    # create_task snapshots the CURRENT (A-bound) context, exactly like the
    # gateway's per-message dispatch.
    await asyncio.create_task(_async_noop(_b_body))
    return captured


async def _async_noop(fn):
    fn()


def test_child_task_inherits_foreign_session_without_reset():
    """REPRODUCER: without the entry reset, B's pre-bind window leaks A's id.

    This is the production hijack. Asserting the leak EXISTS documents the bug
    the fix closes; the next test proves the fix.
    """
    set_session_vars(**MINE)  # parent A binds in the current context

    captured = asyncio.run(_child_turn(reset_first=False))

    # The pre-bind window inherited A's (MINE) identity — the leak.
    assert captured["window"]["HERMES_SESSION_CHAT_ID"] == "MINE_CHAT", (
        "Expected to reproduce the inheritance leak (window sees parent's "
        f"MINE_CHAT); got {captured['window']!r}"
    )


def test_reset_session_vars_closes_inheritance_leak():
    """THE FIX: resetting at handler entry strips the inherited identity.

    After reset_session_vars(), the pre-bind window must see NO session vars
    (stripped, because they are _UNSET in this context and the process is
    engaged) — NOT the parent's MINE_*. B's own bind then takes effect normally.
    """
    set_session_vars(**MINE)  # parent A binds in the current context

    captured = asyncio.run(_child_turn(reset_first=True))

    window = captured["window"]
    for var in ("HERMES_SESSION_CHAT_ID", "HERMES_SESSION_THREAD_ID", "HERMES_SESSION_KEY"):
        assert window[var] is None, (
            f"{var} leaked the parent session after reset: {window[var]!r}"
        )

    # B's own session still binds correctly after the reset window.
    assert captured["bound"]["HERMES_SESSION_CHAT_ID"] == "FOREIGN_CHAT"
    assert captured["bound"]["HERMES_SESSION_KEY"] == FOREIGN["session_key"]
    assert captured["bound"]["HERMES_CODING_WORKFLOW"] == "coupled-v1"


def test_coding_workflow_is_task_local_and_reset_strips_inherited_hybrid():
    set_session_vars(**MINE)

    leaked = asyncio.run(_child_turn(reset_first=False))
    assert leaked["window"]["HERMES_CODING_WORKFLOW"] == "hybrid-v1"

    isolated = asyncio.run(_child_turn(reset_first=True))
    assert isolated["window"]["HERMES_CODING_WORKFLOW"] is None
    assert isolated["bound"]["HERMES_CODING_WORKFLOW"] == "coupled-v1"


def test_reset_session_vars_restores_unset_not_empty():
    """reset_session_vars sets _UNSET (not "" like clear_session_vars).

    The distinction matters: "" is 'explicitly cleared' (suppresses os.environ
    fallback, used when a handler finishes); _UNSET is 'never bound here' (lets
    the bridge strip and a CLI fallback resolve). Entry-reset must use _UNSET.
    """
    set_session_vars(**MINE)
    reset_session_vars()
    for name, var in _VAR_MAP.items():
        assert var.get() is _UNSET, f"{name} is {var.get()!r}, expected _UNSET"


# ---------------------------------------------------------------------------
# Async-delivery capability inheritance (the sibling var outside _VAR_MAP)
# ---------------------------------------------------------------------------
#
# ``_SESSION_ASYNC_DELIVERY`` is NOT in ``_VAR_MAP`` — it is a bool capability
# flag read via ``async_delivery_supported()``, not a string ``HERMES_SESSION_*``
# var read via ``get_session_env``. So the ``for var in _VAR_MAP.values()`` loop
# in ``reset_session_vars`` does not touch it; it must be reset explicitly.
#
# Without that explicit reset, a task created (copy_context) from a context where
# a *concurrent* sibling A had bound ``async_delivery=False`` (the stateless API
# server) inherits A's ``False``. In B's pre-bind window
# ``async_delivery_supported()`` then wrongly reports B's channel as unable to
# route a background completion — even though B is e.g. a real gateway turn that
# CAN. Tools (terminal notify_on_complete / watch_patterns, delegate_task
# background=True) would refuse a promise the channel could actually keep.


async def _child_async_delivery(reset_first: bool):
    """Simulate message B's task created from a parent context where a stateless
    sibling A bound ``async_delivery=False``.

    Returns ``async_delivery_supported()`` as seen in B's pre-bind window.
    """
    captured = {}

    def _b_body():
        if reset_first:
            reset_session_vars()  # THE FIX: handler-entry reset
        captured["window"] = async_delivery_supported()  # pre-bind window

    await asyncio.create_task(_async_noop(_b_body))
    return captured


def test_child_task_inherits_foreign_async_delivery_without_reset():
    """REPRODUCER: without the entry reset, B inherits A's async_delivery=False.

    A stateless adapter (API server) opts out with async_delivery=False. A task
    spawned from that context sees the inherited False in its pre-bind window —
    the leak the explicit reset closes.
    """
    set_session_vars(**FOREIGN, async_delivery=False)  # stateless sibling A

    captured = asyncio.run(_child_async_delivery(reset_first=False))

    assert captured["window"] is False, (
        "Expected to reproduce the async-delivery inheritance leak (window "
        f"inherits A's async_delivery=False); got {captured['window']!r}"
    )


def test_reset_session_vars_closes_async_delivery_leak():
    """THE FIX: resetting at handler entry drops the inherited async_delivery.

    After reset_session_vars(), the pre-bind window must fall back to the
    default-supported behavior (True) — NOT the stateless sibling's False — so a
    real gateway turn isn't wrongly told its channel can't route async delivery.
    """
    set_session_vars(**FOREIGN, async_delivery=False)  # stateless sibling A

    captured = asyncio.run(_child_async_delivery(reset_first=True))

    assert captured["window"] is True, (
        "After reset, async delivery must default to supported; "
        f"got {captured['window']!r}"
    )


def test_reset_session_vars_restores_async_delivery_unset():
    """reset_session_vars restores _SESSION_ASYNC_DELIVERY to the _UNSET sentinel.

    The capability flag must read 'never bound here' (_UNSET), not a falsy value,
    so async_delivery_supported() resolves to the default-supported path rather
    than being mistaken for an opted-out stateless adapter.
    """
    set_session_vars(**FOREIGN, async_delivery=False)
    reset_session_vars()
    assert _SESSION_ASYNC_DELIVERY.get() is _UNSET, (
        f"_SESSION_ASYNC_DELIVERY is {_SESSION_ASYNC_DELIVERY.get()!r}, expected _UNSET"
    )
    assert async_delivery_supported() is True


def test_gateway_sessions_bind_profile_fallback_without_cross_contamination(
    monkeypatch, tmp_path
):
    from gateway import run as gateway_run
    from gateway.config import GatewayConfig, Platform
    from gateway.run import GatewayRunner
    from gateway.session import SessionContext, SessionSource, SessionStore

    config = GatewayConfig()
    store = SessionStore(sessions_dir=tmp_path, config=config)
    store._db = None
    hybrid_source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="hybrid-chat",
        user_id="hybrid-user",
    )
    coupled_source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="coupled-chat",
        user_id="coupled-user",
    )
    hybrid_entry = store.get_or_create_session(hybrid_source)
    coupled_entry = store.get_or_create_session(coupled_source)
    assert store.set_session_metadata(
        coupled_entry.session_key, "coding_workflow", "coupled-v1"
    )

    runner = object.__new__(GatewayRunner)
    runner.config = config
    runner.session_store = store
    runner.adapters = {}
    monkeypatch.setattr(
        gateway_run,
        "_load_gateway_config",
        lambda: {"coding_workflow": {"default": "hybrid-v1"}},
    )

    hybrid_context = SessionContext(
        source=hybrid_source,
        connected_platforms=[],
        home_channels={},
        session_key=hybrid_entry.session_key,
        session_id=hybrid_entry.session_id,
    )
    coupled_context = SessionContext(
        source=coupled_source,
        connected_platforms=[],
        home_channels={},
        session_key=coupled_entry.session_key,
        session_id=coupled_entry.session_id,
    )

    async def run_concurrently():
        hybrid_bound = asyncio.Event()
        coupled_bound = asyncio.Event()

        async def child(context, mine, sibling):
            tokens = runner._set_session_env(context)
            try:
                mine.set()
                await sibling.wait()
                await asyncio.sleep(0)
                return _spawn_view()
            finally:
                runner._clear_session_env(tokens)

        return await asyncio.gather(
            child(hybrid_context, hybrid_bound, coupled_bound),
            child(coupled_context, coupled_bound, hybrid_bound),
        )

    hybrid_env, coupled_env = asyncio.run(run_concurrently())

    assert hybrid_env["HERMES_CODING_WORKFLOW"] == "hybrid-v1"
    assert coupled_env["HERMES_CODING_WORKFLOW"] == "coupled-v1"
    assert (
        store.get_session_metadata(hybrid_entry.session_key, "coding_workflow")
        == "hybrid-v1"
    )
    assert (
        store.get_session_metadata(coupled_entry.session_key, "coding_workflow")
        == "coupled-v1"
    )

    reloaded = SessionStore(sessions_dir=tmp_path, config=config)
    reloaded._db = None
    resumed_entry = reloaded.get_or_create_session(hybrid_source)
    runner.session_store = reloaded
    monkeypatch.setattr(
        gateway_run,
        "_load_gateway_config",
        lambda: {"coding_workflow": {"default": "coupled-v1"}},
    )
    resumed_context = SessionContext(
        source=hybrid_source,
        connected_platforms=[],
        home_channels={},
        session_key=resumed_entry.session_key,
        session_id=resumed_entry.session_id,
    )
    tokens = runner._set_session_env(resumed_context)
    try:
        assert _spawn_view()["HERMES_CODING_WORKFLOW"] == "hybrid-v1"
    finally:
        runner._clear_session_env(tokens)
