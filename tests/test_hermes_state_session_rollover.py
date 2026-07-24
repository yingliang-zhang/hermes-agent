"""Behavioral contracts for atomic durable session rollover."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

import hermes_state
from agent.context_compressor import SUMMARY_PREFIX
from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path: Path) -> SessionDB:
    session_db = SessionDB(tmp_path / "state.db")
    yield session_db
    session_db.close()


def _seed_live_parent(
    db: SessionDB,
    *,
    session_id: str = "parent",
    holder: str = "holder",
) -> int:
    db.create_session(
        session_id,
        source="desktop",
        model="openai/gpt-5.6",
        model_config={
            "provider": "openai-codex",
            "reasoning_effort": "high",
            "service_tier": "priority",
        },
        system_prompt="stable system prompt",
        user_id="do-not-copy-user",
        session_key="do-not-copy-key",
        chat_id="do-not-copy-chat",
        chat_type="dm",
        thread_id="do-not-copy-thread",
        cwd="/workspace/project",
        profile_name="coding",
    )
    db.update_session_cwd(
        session_id,
        "/workspace/project",
        git_branch="feature/rollover",
        git_repo_root="/workspace/project",
    )
    db.set_session_title(session_id, "Durable work")
    db.append_message(
        session_id,
        role="user",
        content=(
            f"{SUMMARY_PREFIX}\n"
            "## Historical Task Snapshot\n"
            "Earlier durable decisions and constraints."
        ),
    )
    db.append_message(session_id, role="assistant", content="Compaction acknowledged.")
    db.append_message(session_id, role="user", content="Finish the atomic rollover slice.")
    final_id = db.append_message(
        session_id,
        role="assistant",
        content="The final durable assistant response.",
        finish_reason="stop",
    )
    assert db.try_acquire_compression_lock(session_id, holder) is True
    return final_id


def _complete(
    db: SessionDB,
    final_id: int,
    *,
    parent: str = "parent",
    child: str = "child",
    holder: str = "holder",
    expected_content: object = "The final durable assistant response.",
):
    return db.complete_session_rollover(
        parent_session_id=parent,
        child_session_id=child,
        holder=holder,
        expected_final_assistant_message_id=final_id,
        expected_final_assistant_content=expected_content,
    )


def _child_count(db: SessionDB, parent: str = "parent") -> int:
    return int(
        db._conn.execute(
            "SELECT COUNT(*) FROM sessions WHERE parent_session_id = ?",
            (parent,),
        ).fetchone()[0]
    )


def test_latest_active_message_identity_fences_exact_assistant_tail(db: SessionDB) -> None:
    final_id = _seed_live_parent(db)

    identity = db.get_latest_active_message_identity("parent")

    assert identity is not None
    assert identity["id"] == final_id
    assert identity["role"] == "assistant"
    assert identity["content"] == "The final durable assistant response."


def test_atomic_rollover_success_commits_parent_child_and_handoff(db: SessionDB) -> None:
    final_id = _seed_live_parent(db)

    successor_id = _complete(db, final_id)

    assert successor_id == "child"
    parent = db.get_session("parent")
    child = db.get_session("child")
    assert parent is not None and parent["ended_at"] is not None
    assert parent["end_reason"] == "rollover"
    assert child is not None
    assert child["parent_session_id"] == "parent"
    assert json.loads(child["model_config"])["_rollover_from"] == "parent"
    handoff = db.get_messages("child")
    assert [row["role"] for row in handoff] == ["user", "assistant"]
    assert child["message_count"] == 2


@pytest.mark.parametrize("failure", ["missing", "wrong-role", "stale", "mismatch"])
def test_final_assistant_fence_failure_leaves_parent_live(
    db: SessionDB,
    failure: str,
) -> None:
    db.create_session("parent", source="desktop")
    expected_id = 1
    if failure == "wrong-role":
        expected_id = db.append_message("parent", role="user", content="not complete")
    elif failure in {"stale", "mismatch"}:
        db.append_message("parent", role="user", content="question")
        expected_id = db.append_message("parent", role="assistant", content="answer")
        if failure == "stale":
            db.append_message("parent", role="user", content="new active turn")
        else:
            expected_id += 1000
    assert db.try_acquire_compression_lock("parent", "holder") is True

    assert _complete(db, expected_id) is None

    parent = db.get_session("parent")
    assert parent is not None
    assert parent["ended_at"] is None
    assert parent["end_reason"] is None
    assert db.get_session("child") is None
    assert _child_count(db) == 0

def test_final_assistant_content_change_after_snapshot_leaves_parent_live(
    db: SessionDB,
) -> None:
    final_id = _seed_live_parent(db)
    identity = db.get_latest_active_message_identity("parent")
    assert identity is not None
    db._conn.execute(
        "UPDATE messages SET content = ? WHERE id = ?",
        (json.dumps("Changed after the rollover offer."), final_id),
    )

    assert (
        _complete(db, final_id, expected_content=identity["content"])
        is None
    )

    parent = db.get_session("parent")
    assert parent is not None
    assert parent["ended_at"] is None
    assert parent["end_reason"] is None
    assert db.get_session("child") is None
    assert _child_count(db) == 0


def test_lost_lease_fails_closed_without_mutation(db: SessionDB) -> None:
    final_id = _seed_live_parent(db, holder="winner")

    assert _complete(db, final_id, holder="loser") is None

    assert db.get_session("parent")["ended_at"] is None
    assert db.get_session("child") is None


@pytest.mark.parametrize("case", ["missing-parent", "ended-parent", "conflicting-child"])
def test_conflicting_parent_state_fails_closed(db: SessionDB, case: str) -> None:
    if case == "missing-parent":
        assert db.try_acquire_compression_lock("parent", "holder") is True
        final_id = 1
    else:
        final_id = _seed_live_parent(db)
        if case == "ended-parent":
            db.end_session("parent", "session_reset")
        else:
            db.create_session(
                "rogue",
                source="desktop",
                parent_session_id="parent",
                model_config={"_rollover_from": "parent"},
            )

    assert _complete(db, final_id) is None

    if case != "missing-parent":
        parent = db.get_session("parent")
        assert parent is not None
        if case == "conflicting-child":
            assert parent["ended_at"] is None
            assert db.get_session("rogue") is not None
            assert _child_count(db) == 1
    assert db.get_session("child") is None


def test_idempotent_replay_returns_existing_successor_without_lease_or_sibling(
    db: SessionDB,
) -> None:
    final_id = _seed_live_parent(db)
    assert _complete(db, final_id) == "child"
    db.release_compression_lock("parent", "holder")

    replayed = _complete(
        db,
        final_id,
        child="different-proposed-child",
        holder="no-longer-held",
    )

    assert replayed == "child"
    assert db.get_session("different-proposed-child") is None
    assert _child_count(db) == 1

@pytest.mark.parametrize(
    ("replayed_final_id", "replayed_content"),
    [
        (None, "Wrong final assistant content."),
        (-1, "The final durable assistant response."),
    ],
    ids=["wrong-content", "wrong-id"],
)
def test_idempotent_replay_rejects_wrong_predecessor_without_sibling(
    db: SessionDB,
    replayed_final_id,
    replayed_content: str,
) -> None:
    final_id = _seed_live_parent(db)
    assert _complete(db, final_id) == "child"
    db.release_compression_lock("parent", "holder")

    replayed = _complete(
        db,
        final_id if replayed_final_id is None else final_id + replayed_final_id,
        child="different-proposed-child",
        holder="no-longer-held",
        expected_content=replayed_content,
    )

    assert replayed is None
    assert db.get_session("child") is not None
    assert db.get_session("different-proposed-child") is None
    assert _child_count(db) == 1


@pytest.mark.parametrize("fault_table", ["sessions", "messages"])
def test_rollover_write_fault_rolls_back_every_row(
    db: SessionDB,
    fault_table: str,
) -> None:
    final_id = _seed_live_parent(db)
    if fault_table == "sessions":
        trigger = """
            CREATE TEMP TRIGGER abort_rollover_child_insert
            BEFORE INSERT ON sessions
            WHEN NEW.parent_session_id = 'parent'
            BEGIN
                SELECT RAISE(ABORT, 'simulated rollover child failure');
            END
        """
        error = "simulated rollover child failure"
    else:
        trigger = """
            CREATE TEMP TRIGGER abort_rollover_handoff_insert
            BEFORE INSERT ON messages
            WHEN NEW.session_id = 'child'
            BEGIN
                SELECT RAISE(ABORT, 'simulated rollover handoff failure');
            END
        """
        error = "simulated rollover handoff failure"
    db._conn.execute(trigger)

    with pytest.raises(sqlite3.IntegrityError, match=error):
        _complete(db, final_id)

    parent = db.get_session("parent")
    assert parent is not None
    assert parent["ended_at"] is None
    assert parent["end_reason"] is None
    assert db.get_session("child") is None
    assert db.message_count(session_id="child") == 0

@pytest.mark.parametrize(
    ("reported_inserted", "reported_tool_calls"),
    [(1, 0), (2, 1)],
)
def test_handoff_insertion_count_mismatch_rolls_back(
    db: SessionDB,
    monkeypatch: pytest.MonkeyPatch,
    reported_inserted: int,
    reported_tool_calls: int,
) -> None:
    final_id = _seed_live_parent(db)
    insert_message_rows = db._insert_message_rows

    def misreport_handoff_counts(conn, session_id, messages):
        insert_message_rows(conn, session_id, messages)
        return reported_inserted, reported_tool_calls

    monkeypatch.setattr(db, "_insert_message_rows", misreport_handoff_counts)

    with pytest.raises(RuntimeError, match="exactly two messages"):
        _complete(db, final_id)

    parent = db.get_session("parent")
    assert parent is not None
    assert parent["ended_at"] is None
    assert parent["end_reason"] is None
    assert db.get_session("child") is None
    assert db.message_count(session_id="child") == 0


def test_successor_copies_only_rollover_allowlisted_metadata(db: SessionDB) -> None:
    final_id = _seed_live_parent(db)
    db._conn.execute(
        """UPDATE sessions SET
           display_name = 'do-not-copy-display',
           origin_json = '{"origin":"do-not-copy"}',
           expiry_finalized = 1,
           input_tokens = 101,
           output_tokens = 202,
           cache_read_tokens = 303,
           cache_write_tokens = 404,
           reasoning_tokens = 505,
           billing_provider = 'do-not-copy-provider',
           billing_base_url = 'https://billing.invalid',
           billing_mode = 'do-not-copy-mode',
           estimated_cost_usd = 1.25,
           actual_cost_usd = 1.5,
           cost_status = 'final',
           cost_source = 'provider',
           pricing_version = 'v1',
           api_call_count = 7,
           handoff_state = 'do-not-copy',
           handoff_platform = 'do-not-copy',
           handoff_error = 'do-not-copy',
           compression_failure_cooldown_until = 999,
           compression_failure_error = 'do-not-copy',
           compression_fallback_streak = 3,
           rewind_count = 2,
           archived = 1
           WHERE id = 'parent'"""
    )

    assert _complete(db, final_id) == "child"

    child = db.get_session("child")
    assert child is not None
    assert child["source"] == "desktop"
    assert child["profile_name"] == "coding"
    assert child["cwd"] == "/workspace/project"
    assert child["git_branch"] == "feature/rollover"
    assert child["git_repo_root"] == "/workspace/project"
    assert child["model"] == "openai/gpt-5.6"
    assert child["system_prompt"] == "stable system prompt"
    assert child["title"] == "Durable work #2"
    assert json.loads(child["model_config"]) == {
        "provider": "openai-codex",
        "reasoning_effort": "high",
        "service_tier": "priority",
        "_rollover_from": "parent",
    }
    assert child["parent_session_id"] == "parent"

    assert child["user_id"] is None
    assert child["session_key"] is None
    assert child["chat_id"] is None
    assert child["chat_type"] is None
    assert child["thread_id"] is None
    assert child["display_name"] is None
    assert child["origin_json"] is None
    assert child["expiry_finalized"] == 0
    assert child["tool_call_count"] == 0
    assert child["input_tokens"] == 0
    assert child["output_tokens"] == 0
    assert child["cache_read_tokens"] == 0
    assert child["cache_write_tokens"] == 0
    assert child["reasoning_tokens"] == 0
    assert child["billing_provider"] is None
    assert child["billing_base_url"] is None
    assert child["billing_mode"] is None
    assert child["estimated_cost_usd"] is None
    assert child["actual_cost_usd"] is None
    assert child["cost_status"] is None
    assert child["cost_source"] is None
    assert child["pricing_version"] is None
    assert child["api_call_count"] == 0
    assert child["handoff_state"] is None
    assert child["handoff_platform"] is None
    assert child["handoff_error"] is None
    assert child["compression_failure_cooldown_until"] is None
    assert child["compression_failure_error"] is None
    assert child["compression_fallback_streak"] == 0
    assert child["rewind_count"] == 0
    assert child["archived"] == 0


def test_bounded_handoff_is_deterministic_text_and_valid_alternation() -> None:
    summary = f"{SUMMARY_PREFIX}\n" + "\n".join(
        f"summary {index} " + ("界" * 1000) for index in range(30)
    )
    user = "\n".join(f"user {index} " + ("u" * 1000) for index in range(10))
    assistant = "\n".join(
        f"assistant {index} " + ("a" * 1000) for index in range(10)
    ) + "\ud800"

    first = hermes_state.build_session_rollover_handoff(summary, user, assistant)
    second = hermes_state.build_session_rollover_handoff(summary, user, assistant)

    assert first == second
    assert [message["role"] for message in first] == ["user", "assistant"]
    nonempty_lines = [
        line
        for message in first
        for line in message["content"].splitlines()
        if line.strip()
    ]
    assert len(nonempty_lines) <= 20
    assert sum(len(message["content"].encode("utf-8")) for message in first) <= 16 * 1024
    assert all("\ud800" not in message["content"] for message in first)
    assert "Earlier durable" not in first[0]["content"]
    assert "summary 0" in first[0]["content"]
    assert "user 0" in first[0]["content"]
    assert "assistant 0" in first[1]["content"]


@pytest.mark.parametrize(
    ("summary", "user", "assistant"),
    [
        (None, "valid user", {"image_url": "no textual assistant"}),
        (None, "", "valid assistant"),
        (None, "valid user", ""),
    ],
)
def test_bounded_handoff_rejects_missing_textual_exchange(
    summary,
    user,
    assistant,
) -> None:
    with pytest.raises(ValueError):
        hermes_state.build_session_rollover_handoff(summary, user, assistant)


def test_surrogate_successor_identity_is_rejected_without_sqlite_write(db: SessionDB) -> None:
    final_id = _seed_live_parent(db)

    result = _complete(db, final_id, child="bad\ud800child")

    assert result is None
    assert db.get_session("parent")["ended_at"] is None
    assert _child_count(db) == 0


def test_rollover_tip_projection_preserves_root_for_default_and_search(
    db: SessionDB,
) -> None:
    final_id = _seed_live_parent(db)
    successor_id = db.complete_session_rollover(
        parent_session_id="parent",
        child_session_id="child",
        holder="holder",
        expected_final_assistant_message_id=final_id,
        expected_final_assistant_content="The final durable assistant response.",
    )
    assert successor_id == "child"

    default_rows = db.list_sessions_rich(limit=100)
    search_rows = db.list_sessions_rich(search_query="parent", limit=100)

    for rows in (default_rows, search_rows):
        projected = next(row for row in rows if row["id"] == "child")
        assert projected["_lineage_root_id"] == "parent"


def test_rollover_tip_picker_archive_delete_and_resume_projections(
    db: SessionDB,
) -> None:
    final_id = _seed_live_parent(db)
    assert _complete(db, final_id) == "child"

    raw_picker_ids = {
        row["id"]
        for row in db.list_sessions_rich(project_compression_tips=False, limit=100)
    }
    assert "parent" in raw_picker_ids
    assert "child" not in raw_picker_ids

    projected = db.list_sessions_rich(limit=100)
    child_entry = next(row for row in projected if row["id"] == "child")
    assert child_entry["_lineage_root_id"] == "parent"
    assert db.get_compression_tip("parent") == "child"
    assert db.resolve_resume_session_id("parent") == "child"

    model_history, display_history = db.get_resume_conversations("child")
    assert model_history == db.get_messages_as_conversation(
        "child", repair_alternation=True
    )
    assert len(model_history) == 2
    assert len(display_history) == len(db.get_messages("parent")) + 2
    assert [message["role"] for message in model_history] == ["user", "assistant"]
    assert display_history[-2:] == db.get_messages_as_conversation("child")

    assert db.set_session_archived("child", True) is True
    assert db.get_session("parent")["archived"] == 1
    assert db.get_session("child")["archived"] == 1
    assert db.set_session_archived("parent", False) is True
    assert db.get_session("parent")["archived"] == 0
    assert db.get_session("child")["archived"] == 0

    assert db.delete_session("parent") is True
    child = db.get_session("child")
    assert child is not None
    assert child["parent_session_id"] is None
    assert "child" in {
        row["id"]
        for row in db.list_sessions_rich(project_compression_tips=False, limit=100)
    }


def test_rollover_child_is_not_reclassified_as_ephemeral_delegate(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "state.db"
    first = SessionDB(db_path)
    final_id = _seed_live_parent(first)
    assert _complete(first, final_id) == "child"
    first._conn.execute("UPDATE schema_version SET version = 15")
    first._conn.commit()
    first.close()

    reopened = SessionDB(db_path)
    try:
        child_config = json.loads(reopened.get_session("child")["model_config"])
        assert child_config["_rollover_from"] == "parent"
        assert "_delegate_from" not in child_config
        assert reopened.delete_session("parent") is True
        assert reopened.get_session("child") is not None
    finally:
        reopened.close()
