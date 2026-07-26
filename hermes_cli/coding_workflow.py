"""Coding-workflow contract for Hybrid v1 routing.

Pure helpers: an allowlist, fail-closed write validation, profile-default
resolution, and a workflow preset inventory. No I/O and no heavy imports —
safe to import from ``hermes_cli.config``, ``gateway.session_context``, and
``tui_gateway.server``.

Authority model (see docs/plans/2026-07-25-hybrid-routing-v1.md):

* The profile default lives at top-level ``coding_workflow.default`` in
  ``config.yaml``; the shipped default is ``coupled-v1``.
* A session snapshots its workflow on create and persists it in
  ``sessions.model_config.coding_workflow``; a manual session choice always
  wins over the profile default.
* ``coding_workflow`` is orthogonal to the primary provider/model. Selecting
  Hybrid (``hybrid-v1``) forces the controller route to the real Sol backend
  (``custom:sudo / gpt-5.6-sol``); plain model selection means ``coupled-v1``.
* Allowed values are exactly ``coupled-v1`` and ``hybrid-v1``. Malformed or
  unknown values fail closed at *write* boundaries and at *read* boundaries
  when explicitly present; a genuinely absent legacy section/key/value degrades
  to ``coupled-v1``.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

#: The two allowed workflow identifiers. Order is the canonical display order.
ALLOWED_WORKFLOWS: tuple[str, ...] = ("coupled-v1", "hybrid-v1")

#: Shipped default. The orchestrator profile is *not* activated to hybrid-v1
#: by shipping this module — that is a separate controlled config change.
DEFAULT_WORKFLOW: str = "coupled-v1"

# Hybrid v1 controller route — the real Sol backend that stays the Hermes
# controller/reviewer. Hard-coded: a session selecting Hybrid forces this
# route and the UI cannot expose a Hybrid/GLM or Hybrid/Terra partial state.
HYBRID_CONTROLLER_PROVIDER: str = "custom:sudo"
HYBRID_CONTROLLER_MODEL: str = "gpt-5.6-sol"
# The controller's thinking level. Sol runs at XHigh reasoning in Hybrid mode
# (the "Sol XHigh" controller from the plan). Plain model selection (coupled)
# never reads this — it is metadata for the Hybrid preset only.
HYBRID_CONTROLLER_THINKING: str = "xhigh"

# Hybrid v1 executor route — the real GLM backend that runs OMP implementation
# and the first two repair attempts (the "GLM Heavy Max" executor). Like the
# controller route it is hard-coded metadata: a Hybrid session maps implement
# and repair rounds 1–2 to this route. It is NOT a fake provider and is NOT
# user-selectable as a primary controller route.
HYBRID_EXECUTOR_PROVIDER: str = "custom:sudo"
HYBRID_EXECUTOR_MODEL: str = "glm-5.2-heavy"
HYBRID_EXECUTOR_THINKING: str = "max"


class InvalidCodingWorkflow(ValueError):
    """Raised at a *write* boundary for a malformed/unknown workflow value.

    Read boundaries (legacy/absent config) use :func:`normalize_coding_workflow`
    and degrade to :data:`DEFAULT_WORKFLOW` instead of raising.
    """


@dataclass(frozen=True, slots=True)
class CanonicalRoute:
    """One immutable controller route, including its reasoning contract."""

    coding_workflow: str
    provider: str | None
    model: str | None
    _reasoning_items: tuple[tuple[str, Any], ...] | None = None

    @property
    def reasoning_config(self) -> dict | None:
        """Return an isolated mutable copy for agent/session runtime fields."""
        if self._reasoning_items is None:
            return None
        return deepcopy(dict(self._reasoning_items))


def canonicalize_route(
    workflow: Any,
    provider: Any = None,
    model: Any = None,
    reasoning_config: Any = None,
) -> CanonicalRoute:
    """Validate and canonicalize workflow, controller, and reasoning together.

    Coupled sessions preserve the caller's ordinary route and reasoning.
    Hybrid sessions reject contradictory controller components and always
    return the fixed Sol XHigh identity.
    """
    canonical_workflow = validate_coding_workflow(workflow)
    requested_provider = str(provider or "").strip()
    requested_model = str(model or "").strip()
    if canonical_workflow == "hybrid-v1":
        fixed_provider, fixed_model = hybrid_controller_route()
        if (requested_provider and requested_provider != fixed_provider) or (
            requested_model and requested_model != fixed_model
        ):
            raise InvalidCodingWorkflow(
                "hybrid-v1 requires controller route custom:sudo / gpt-5.6-sol"
            )
        return CanonicalRoute(
            coding_workflow=canonical_workflow,
            provider=fixed_provider,
            model=fixed_model,
            _reasoning_items=(
                ("enabled", True),
                ("effort", HYBRID_CONTROLLER_THINKING),
            ),
        )

    reasoning_items = None
    if isinstance(reasoning_config, dict):
        reasoning_items = tuple(
            (str(key), deepcopy(value)) for key, value in reasoning_config.items()
        )
    return CanonicalRoute(
        coding_workflow=canonical_workflow,
        provider=requested_provider or None,
        model=requested_model or None,
        _reasoning_items=reasoning_items,
    )


def canonicalize_controller_route(
    workflow: Any,
    provider: Any = None,
    model: Any = None,
) -> tuple[str | None, str | None]:
    """Backward-compatible controller-only view of :func:`canonicalize_route`."""
    route = canonicalize_route(workflow, provider, model)
    return route.provider, route.model


def _clean(value: Any) -> str:
    return str(value or "").strip().lower()


def validate_coding_workflow(value: Any) -> str:
    """Normalize and validate a workflow value; raise on unknown/malformed.

    This is the fail-closed *write* boundary. Case-insensitive and
    whitespace-tolerant: ``"  HYBRID-V1  "`` → ``"hybrid-v1"``.
    """
    wf = _clean(value)
    if wf not in ALLOWED_WORKFLOWS:
        raise InvalidCodingWorkflow(
            f"unknown coding workflow: {value!r}; "
            f"expected one of {list(ALLOWED_WORKFLOWS)}"
        )
    return wf


def normalize_coding_workflow(value: Any) -> str:
    """Return a valid workflow for a *read* boundary.

    Genuinely absent values (``None``, empty/whitespace-only) degrade to
    :data:`DEFAULT_WORKFLOW` so legacy config or a partially-written row can
    never brick a session. An *explicitly present* but malformed/unknown value
    fails closed with :class:`InvalidCodingWorkflow` — a stored profile/session
    value that is present but wrong is a real error, not a legacy absence.
    """
    wf = _clean(value)
    if wf in ALLOWED_WORKFLOWS:
        return wf
    if not wf:
        return DEFAULT_WORKFLOW
    raise InvalidCodingWorkflow(
        f"unknown coding workflow: {value!r}; "
        f"expected one of {list(ALLOWED_WORKFLOWS)}"
    )


def is_hybrid(workflow: Any) -> bool:
    """True iff ``workflow`` normalizes to ``hybrid-v1``."""
    return _clean(workflow) == "hybrid-v1"


def resolve_profile_default(config: Any = None) -> str:
    """Read ``coding_workflow.default`` from a config dict (profile-local).

    A genuinely absent section/key/value (legacy config — missing section,
    missing ``default`` key, or ``None``/empty value) degrades to
    :data:`DEFAULT_WORKFLOW`. An explicitly present but malformed/unknown
    ``default`` value fails closed with :class:`InvalidCodingWorkflow`.
    """
    if not isinstance(config, dict):
        return DEFAULT_WORKFLOW
    section = config.get("coding_workflow")
    if not isinstance(section, dict):
        return DEFAULT_WORKFLOW
    return normalize_coding_workflow(section.get("default"))


def resolve_session_workflow(value: Any = None, config: Any = None) -> str:
    """Resolve a session's workflow: a manual choice wins over the profile default.

    A falsy/empty ``value`` means "no manual choice" → the profile default.
    A present ``value`` is validated at the write boundary (fail-closed).
    """
    raw = _clean(value)
    if raw:
        return validate_coding_workflow(raw)
    return resolve_profile_default(config)


def hybrid_controller_route() -> tuple[str, str]:
    """The fixed ``(provider, model)`` controller route for Hybrid v1."""
    return (HYBRID_CONTROLLER_PROVIDER, HYBRID_CONTROLLER_MODEL)





def hybrid_executor_route() -> tuple[str, str]:
    """The fixed ``(provider, model)`` executor route for Hybrid v1.

    The GLM Heavy Max executor runs OMP implementation and the first two
    repair attempts. It is hard-coded metadata, not a user-selectable primary
    route — a Hybrid session's controller route is always Sol (see
    :func:`hybrid_controller_route`).
    """
    return (HYBRID_EXECUTOR_PROVIDER, HYBRID_EXECUTOR_MODEL)


def workflow_presets() -> list[dict]:
    """Inventory of workflow presets for the model picker (``model.options``).

    Hybrid is represented as a preset carrying the real Sol controller route,
    NOT as a fake model provider. Coupled carries no forced controller — plain
    model selection means coupled mode. Both presets also carry the real
    executor/controller thinking levels (Sol XHigh / GLM Heavy Max) as
    inventory metadata; the executor route is NOT a primary controller route.
    """
    return [
        {
            "id": "coupled-v1",
            "label": "Coupled",
            "hybrid": False,
            "controller_provider": None,
            "controller_model": None,
            "controller_thinking": None,
            "executor_provider": None,
            "executor_model": None,
            "executor_thinking": None,
        },
        {
            "id": "hybrid-v1",
            "label": "Hybrid",
            "hybrid": True,
            "controller_provider": HYBRID_CONTROLLER_PROVIDER,
            "controller_model": HYBRID_CONTROLLER_MODEL,
            "controller_thinking": HYBRID_CONTROLLER_THINKING,
            "executor_provider": HYBRID_EXECUTOR_PROVIDER,
            "executor_model": HYBRID_EXECUTOR_MODEL,
            "executor_thinking": HYBRID_EXECUTOR_THINKING,
        },
    ]
