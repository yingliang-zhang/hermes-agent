"""Pure eligibility and quiescence snapshots for session rollover offers.

This module has no gateway, database, or agent imports.  The live gateway owns
all dependency queries and converts their outcomes into fail-closed snapshots.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass


@dataclass(frozen=True)
class RolloverFence:
    """Exact server-side identity for one rollover offer candidate."""

    runtime_id: str
    predecessor_stored_id: str
    settled_turn_generation: int
    history_version: int
    compression_count: int
    final_message_id: int
    final_content: str


@dataclass(frozen=True)
class RolloverOffer:
    """Opaque client token bound server-side to a complete immutable fence."""

    token: str
    fence: RolloverFence

    def event_payload(self) -> dict:
        """Return the public identity fence without repeating assistant content."""
        return {
            "token": self.token,
            "runtime_id": self.fence.runtime_id,
            "predecessor_stored_id": self.fence.predecessor_stored_id,
            "turn_generation": self.fence.settled_turn_generation,
            "history_version": self.fence.history_version,
            "compression_count": self.fence.compression_count,
            "final_message_id": self.fence.final_message_id,
        }


def new_offer(fence: RolloverFence) -> RolloverOffer:
    """Create a cryptographically opaque one-time handle for ``fence``."""
    return RolloverOffer(token=secrets.token_urlsafe(32), fence=fence)


@dataclass(frozen=True)
class EligibilitySnapshot:
    """Required preflight proofs before dependency queries may run."""

    config_enabled: bool
    source_desktop: bool
    local_capable: bool
    compute_host_clear: bool
    turn_clean: bool
    history_adopted: bool
    persistence_clean: bool


@dataclass(frozen=True)
class QuiescenceSnapshot:
    """Required positive proofs for an eligible rollover candidate.

    Indeterminate dependency results use the corresponding
    ``*_query_known=False`` proof, so evaluation always fails closed.
    """

    running_clear: bool
    inflight_clear: bool
    reservation_clear: bool
    steer_clear: bool
    queued_prompt_clear: bool
    queued_prompts_clear: bool
    tools_clear: bool
    background_tasks_clear: bool
    bridge_clear: bool
    approval_clear: bool
    bridge_query_known: bool
    processes_clear: bool
    process_query_known: bool
    delegations_clear: bool
    delegation_query_known: bool
    goal_active_clear: bool
    goal_waiting_clear: bool
    goal_followup_clear: bool
    goal_query_known: bool
    notification_texts_clear: bool
    notification_ids_clear: bool
    notification_adoption_clear: bool
    notification_latch_clear: bool
    addressed_completion_clear: bool
    notification_delivery_clear: bool
    notification_query_known: bool
    history_stable: bool
    history_tail_exact: bool
    agent_identity_current: bool
    db_tail_exact: bool
    db_query_known: bool
    compression_boundary_new: bool


_ELIGIBILITY_REASON_FIELDS = (
    ("config_enabled", "config_disabled"),
    ("source_desktop", "source_not_desktop"),
    ("local_capable", "local_capability_missing"),
    ("compute_host_clear", "compute_host_active"),
    ("turn_clean", "turn_not_clean"),
    ("history_adopted", "history_not_adopted"),
    ("persistence_clean", "persistence_cleanup_error"),
)

_QUIESCENCE_REASON_FIELDS = (
    ("running_clear", "running"),
    ("inflight_clear", "inflight_turn"),
    ("reservation_clear", "turn_reservation"),
    ("steer_clear", "pending_steer"),
    ("queued_prompt_clear", "queued_prompt"),
    ("queued_prompts_clear", "queued_prompts"),
    ("tools_clear", "tool_running"),
    ("background_tasks_clear", "background_prompt_running"),
    ("bridge_clear", "blocking_input"),
    ("approval_clear", "approval_pending"),
    ("bridge_query_known", "approval_query_unknown"),
    ("processes_clear", "background_process_running"),
    ("process_query_known", "process_query_unknown"),
    ("delegations_clear", "async_delegation_running"),
    ("delegation_query_known", "delegation_query_unknown"),
    ("goal_active_clear", "goal_active"),
    ("goal_waiting_clear", "goal_waiting"),
    ("goal_followup_clear", "goal_followup"),
    ("goal_query_known", "goal_query_unknown"),
    ("notification_texts_clear", "deferred_notification_texts"),
    ("notification_ids_clear", "deferred_notification_ids"),
    ("notification_adoption_clear", "deferred_notification_adoption"),
    ("notification_latch_clear", "notification_deferral_latch"),
    ("addressed_completion_clear", "addressed_completion_pending"),
    ("notification_delivery_clear", "notification_delivery_inflight"),
    ("notification_query_known", "notification_query_unknown"),
    ("history_stable", "history_changed"),
    ("history_tail_exact", "history_tail_mismatch"),
    ("agent_identity_current", "agent_identity_mismatch"),
    ("db_tail_exact", "db_tail_mismatch"),
    ("db_query_known", "db_query_unknown"),
    ("compression_boundary_new", "compression_boundary_missing"),
)


@dataclass(frozen=True)
class QuiescenceResult:
    allowed: bool
    reasons: tuple[str, ...]


def _evaluate_proofs(snapshot, reason_fields: tuple[tuple[str, str], ...]) -> QuiescenceResult:
    snapshot_fields = tuple(snapshot.__dataclass_fields__)
    mapped_fields = tuple(field for field, _reason in reason_fields)
    if snapshot_fields != mapped_fields:
        return QuiescenceResult(allowed=False, reasons=("proof_schema_mismatch",))
    reasons = tuple(
        reason for field, reason in reason_fields if getattr(snapshot, field) is not True
    )
    return QuiescenceResult(allowed=not reasons, reasons=reasons)


def evaluate_eligibility(snapshot: EligibilitySnapshot) -> QuiescenceResult:
    """Evaluate every preflight proof in stable order."""
    return _evaluate_proofs(snapshot, _ELIGIBILITY_REASON_FIELDS)


def evaluate_quiescence(snapshot: QuiescenceSnapshot) -> QuiescenceResult:
    """Evaluate every quiescence proof in stable order."""
    return _evaluate_proofs(snapshot, _QUIESCENCE_REASON_FIELDS)
