"""Compression rotation hardening — state-loss fixes at the compaction boundary.

When auto-compression rotates ``agent.session_id`` to a continuation child,
three pieces of state used to be lost or corrupted:

  * #33618 — a persistent ``/goal`` did not follow the rotation (``load_goal``
    is a flat per-session lookup with no lineage walk), so it silently died.
  * #33906/#33907 — if the child ``create_session`` raised, the outer handler
    only warned and let the agent continue on the NEW (un-indexed) id,
    producing an orphan session missing from state.db.
  * #27633 — the compaction-boundary ``on_session_start`` notification omitted
    the ``platform`` kwarg, so context-engine plugins saw ``source=unknown``
    for every message after the boundary.

These tests drive the real ``compress_context`` path against a real SessionDB.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

from hermes_state import SessionDB


def _build_agent_with_db(db: SessionDB, session_id: str, platform: str = "telegram"):
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            platform=platform,
            quiet_mode=True,
            session_db=db,
            session_id=session_id,
            skip_context_files=True,
            skip_memory=True,
        )

    compressor = MagicMock()
    compressor.compress.return_value = [
        {"role": "user", "content": "[CONTEXT COMPACTION] summary"},
        {"role": "user", "content": "tail"},
    ]
    compressor.compression_count = 1
    compressor.last_prompt_tokens = 0
    compressor.last_completion_tokens = 0
    compressor._last_summary_error = None
    compressor._last_compress_aborted = False
    compressor._last_summary_auth_failure = False
    compressor._last_aux_model_failure_model = None
    compressor._last_aux_model_failure_error = None
    agent.context_compressor = compressor
    # ROTATION fallback path — pin in_place=False so these keep covering fork
    # rotation regardless of the global default (flipped to True in #38763).
    agent.compression_in_place = False
    return agent


def _msgs(n=20):
    return [{"role": "user", "content": f"m{i}"} for i in range(n)]


class TestGoalMigratesOnRotation:
    def test_goal_follows_compression_rotation(self, tmp_path: Path):
        db = SessionDB(db_path=tmp_path / "state.db")
        parent = "PARENT_GOAL_ROT"
        db.create_session(parent, source="cli")
        agent = _build_agent_with_db(db, parent)

        # Set a persistent goal on the parent via the real persistence path.
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path / ".hermes")}):
            (tmp_path / ".hermes").mkdir(exist_ok=True)
            import hermes_cli.goals as goals
            goals._DB_CACHE.clear()
            # Point the goal DB at the same state.db the agent uses.
            with patch.object(goals, "_get_session_db", return_value=db):
                goals.save_goal(parent, goals.GoalState(goal="finish the migration"))

                agent._compress_context(_msgs(), "sys", approx_tokens=120_000)
                child = agent.session_id
                assert child != parent  # rotation happened

                migrated = goals.load_goal(child)
                assert migrated is not None
                assert migrated.goal == "finish the migration"
            goals._DB_CACHE.clear()


class TestOrphanRollbackOnCreateFailure:
    def test_rolls_back_to_parent_when_child_create_fails(self, tmp_path: Path):
        db = SessionDB(db_path=tmp_path / "state.db")
        parent = "PARENT_ORPHAN_ROT"
        db.create_session(parent, source="cli")
        agent = _build_agent_with_db(db, parent)

        # Make the CHILD create_session raise, but let the initial parent
        # end_session/reopen work. We patch create_session to blow up.
        real_create = db.create_session

        def _boom(*a, **k):
            raise RuntimeError("FOREIGN KEY constraint failed")

        with patch.object(db, "create_session", side_effect=_boom):
            agent._compress_context(_msgs(), "sys", approx_tokens=120_000)

        # The live id must roll back to the still-indexed parent — NOT a
        # phantom child id that has no row in state.db.
        assert agent.session_id == parent
        assert db.get_session(parent) is not None
        _ = real_create  # silence unused


class TestPlatformForwardedAtBoundary:
    def test_on_session_start_receives_platform(self, tmp_path: Path):
        db = SessionDB(db_path=tmp_path / "state.db")
        parent = "PARENT_PLATFORM_ROT"
        db.create_session(parent, source="telegram")
        agent = _build_agent_with_db(db, parent, platform="telegram")

        agent._compress_context(_msgs(), "sys", approx_tokens=120_000)

        # The boundary notify must forward the platform so context-engine
        # plugins don't fall back to source=unknown (#27633).
        calls = [c for c in agent.context_compressor.on_session_start.call_args_list]
        assert calls, "on_session_start was not called at the boundary"
        kwargs = calls[-1].kwargs
        assert kwargs.get("platform") == "telegram"
        assert kwargs.get("boundary_reason") == "compression"


class TestTodoSnapshotMergedNotDuplicated:
    """When the compressed transcript already ends with a user message, the
    todo snapshot must be folded into it rather than appended as a second
    standalone user message (which would create consecutive user/user turns)."""

    def test_snapshot_merges_into_trailing_user(self, tmp_path: Path):
        db = SessionDB(db_path=tmp_path / "state.db")
        parent = "PARENT_TODO_MERGE"
        db.create_session(parent, source="cli")
        agent = _build_agent_with_db(db, parent, platform="cli")

        # Compressor returns a transcript ending with a user message — the
        # exact case where appending a todo snapshot would create user/user.
        agent.context_compressor.compress.return_value = [
            {"role": "user", "content": "[CONTEXT COMPACTION] summary"},
            {"role": "assistant", "content": "acknowledged"},
            {"role": "user", "content": "tail"},
        ]

        # Seed a non-empty todo store
        agent._todo_store._todos = [{"id": "t1", "content": "task A", "status": "pending"}]
        agent._todo_store.format_for_injection = lambda: "## Current Tasks\n- [ ] task A"

        result = agent._compress_context(_msgs(), "sys", approx_tokens=120_000)
        compressed = result[0] if isinstance(result, tuple) else result

        # The snapshot must be merged, not appended — so the transcript length
        # should be unchanged (3 messages, not 4).
        assert len(compressed) == 3

        # The trailing user message must carry both the original text and snapshot
        tail = compressed[-1]
        assert tail["role"] == "user"
        assert "tail" in tail["content"]
        assert "task A" in tail["content"]

        # No consecutive user/user pair
        for i in range(1, len(compressed)):
            assert not (compressed[i]["role"] == "user" and compressed[i - 1]["role"] == "user"), \
                f"Consecutive user/user at index {i}"

    def test_snapshot_merge_is_persisted_in_place(self, tmp_path: Path):
        db = SessionDB(db_path=tmp_path / "state.db")
        parent = "PARENT_TODO_INPLACE"
        db.create_session(parent, source="cli")
        agent = _build_agent_with_db(db, parent, platform="cli")
        agent.compression_in_place = True  # in-place mode

        agent.context_compressor.compress.return_value = [
            {"role": "user", "content": "[CONTEXT COMPACTION] summary"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "last user msg"},
        ]

        agent._todo_store._todos = [{"id": "t1", "content": "do thing", "status": "in_progress"}]
        agent._todo_store.format_for_injection = lambda: "## Current Tasks\n- [ ] do thing"

        agent._compress_context(_msgs(), "sys", approx_tokens=120_000)

        # Verify the live DB transcript has the merged tail, no user/user
        db_msgs = db.get_messages(agent.session_id)
        for i in range(1, len(db_msgs)):
            assert not (db_msgs[i]["role"] == "user" and db_msgs[i - 1]["role"] == "user"), \
                f"Consecutive user/user persisted at index {i}"

        # The last user message must contain both the original text and snapshot
        last_user = [m for m in db_msgs if m["role"] == "user"][-1]
        assert "last user msg" in last_user["content"]
        assert "do thing" in last_user["content"]
