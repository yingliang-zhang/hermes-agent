"""Behavior tests for the pure coding-workflow contract (Hybrid v1).

These tests assert invariants about the allowlist, default resolution, and the
workflow preset inventory — they must NOT freeze transient values like model
lists; they assert how the data relates (the controller route is real and
fixed for Hybrid; coupled carries no forced controller).

Read-boundary semantics (frozen contract):
* a *genuinely absent* legacy section/key/value (``None``, empty/whitespace)
  degrades to ``coupled-v1`` so legacy config or a partially-written row can
  never brick a session;
* an *explicitly present* but malformed/unknown value fails closed with
  :class:`InvalidCodingWorkflow` — a stored profile/session value that is
  present but wrong is a real error, not a legacy absence.
"""

from __future__ import annotations

import pytest

from hermes_cli import coding_workflow as cw


# ---------------------------------------------------------------------------
# Allowlist + fail-closed validation
# ---------------------------------------------------------------------------


class TestAllowlist:
    def test_exact_allowed_values(self):
        assert set(cw.ALLOWED_WORKFLOWS) == {"coupled-v1", "hybrid-v1"}

    def test_shipped_default_is_coupled(self):
        assert cw.DEFAULT_WORKFLOW == "coupled-v1"

    @pytest.mark.parametrize("bad", ["", "hybrid", "coupled", "hybrid-v2", "v1", None, "sol", "coupled"])
    def test_validate_rejects_unknown(self, bad):
        with pytest.raises(cw.InvalidCodingWorkflow):
            cw.validate_coding_workflow(bad)

    @pytest.mark.parametrize("good", ["coupled-v1", "hybrid-v1"])
    def test_validate_normalizes_case_and_whitespace(self, good):
        assert cw.validate_coding_workflow(f"  {good.upper()}  ") == good

    def test_normalize_degrades_genuinely_absent_to_default(self):
        # READ boundary: genuinely absent (None / empty / whitespace) degrades
        # to coupled-v1 and never raises — legacy config / partially-written row.
        for absent in [None, "", "   ", "\t"]:
            assert cw.normalize_coding_workflow(absent) == "coupled-v1"

    @pytest.mark.parametrize("present_bad", ["legacy-mode", "hybrid", "weird", "hybrid-v2", "sol"])
    def test_normalize_fails_closed_on_explicitly_present_malformed(self, present_bad):
        # An explicitly present but malformed/unknown value is a real error,
        # not a legacy absence — fail closed at the read boundary too.
        with pytest.raises(cw.InvalidCodingWorkflow):
            cw.normalize_coding_workflow(present_bad)

    def test_normalize_keeps_valid_values(self):
        assert cw.normalize_coding_workflow("hybrid-v1") == "hybrid-v1"
        assert cw.normalize_coding_workflow("COUPLED-V1") == "coupled-v1"


# ---------------------------------------------------------------------------
# Profile-default resolution
# ---------------------------------------------------------------------------


class TestProfileDefault:
    def test_absent_config_yields_default(self):
        assert cw.resolve_profile_default(None) == "coupled-v1"
        assert cw.resolve_profile_default({}) == "coupled-v1"

    def test_absent_section_and_key_degrade(self):
        # Genuinely absent legacy shapes degrade to coupled-v1.
        assert cw.resolve_profile_default({"coding_workflow": {}}) == "coupled-v1"
        assert (
            cw.resolve_profile_default({"coding_workflow": {"default": None}})
            == "coupled-v1"
        )
        assert (
            cw.resolve_profile_default({"coding_workflow": {"default": ""}})
            == "coupled-v1"
        )

    def test_reads_coding_workflow_default_key(self):
        assert cw.resolve_profile_default({"coding_workflow": {"default": "hybrid-v1"}}) == "hybrid-v1"

    def test_explicitly_present_malformed_default_fails_closed(self):
        # A present but unknown ``default`` value is a real error, not legacy.
        with pytest.raises(cw.InvalidCodingWorkflow):
            cw.resolve_profile_default({"coding_workflow": {"default": "weird"}})
        with pytest.raises(cw.InvalidCodingWorkflow):
            cw.resolve_profile_default({"coding_workflow": {"default": "hybrid"}})

    def test_non_dict_section_degrades(self):
        # A non-dict section is a malformed *structure* (the ``default`` key is
        # not readable), so it degrades to the default rather than raising.
        assert cw.resolve_profile_default({"coding_workflow": "hybrid-v1"}) == "coupled-v1"


class TestResolveSessionWorkflow:
    def test_manual_choice_wins_over_profile_default(self):
        wf = cw.resolve_session_workflow("coupled-v1", {"coding_workflow": {"default": "hybrid-v1"}})
        assert wf == "coupled-v1"

    def test_empty_choice_falls_back_to_profile_default(self):
        wf = cw.resolve_session_workflow("", {"coding_workflow": {"default": "hybrid-v1"}})
        assert wf == "hybrid-v1"

    def test_empty_choice_falls_back_to_default_when_profile_absent(self):
        assert cw.resolve_session_workflow("", None) == "coupled-v1"
        assert cw.resolve_session_workflow("", {}) == "coupled-v1"

    def test_invalid_manual_choice_fails_closed(self):
        with pytest.raises(cw.InvalidCodingWorkflow):
            cw.resolve_session_workflow("nope", {"coding_workflow": {"default": "hybrid-v1"}})

    def test_profile_default_malformed_propagates_fail_closed(self):
        # An explicitly malformed profile default fails closed even when the
        # session provides no manual choice.
        with pytest.raises(cw.InvalidCodingWorkflow):
            cw.resolve_session_workflow("", {"coding_workflow": {"default": "bogus"}})


# ---------------------------------------------------------------------------
# Hybrid controller route + preset inventory
# ---------------------------------------------------------------------------


class TestHybridController:
    def test_is_hybrid(self):
        assert cw.is_hybrid("hybrid-v1") is True
        assert cw.is_hybrid("coupled-v1") is False
        assert cw.is_hybrid("") is False
        assert cw.is_hybrid(None) is False

    def test_controller_route_is_real_sudo_sol(self):
        provider, model = cw.hybrid_controller_route()
        # Real provider/model — not a virtual/fake provider entry.
        assert provider == "custom:sudo"
        assert model == "gpt-5.6-sol"

    def test_executor_route_is_real_sudo_glm(self):
        provider, model = cw.hybrid_executor_route()
        assert provider == "custom:sudo"
        assert model == "glm-5.2-heavy"


class TestWorkflowPresets:
    def test_inventory_ids_match_allowlist(self):
        ids = {p["id"] for p in cw.workflow_presets()}
        assert ids == set(cw.ALLOWED_WORKFLOWS)

    def test_hybrid_preset_carries_real_sol_controller(self):
        hybrid = next(p for p in cw.workflow_presets() if p["id"] == "hybrid-v1")
        assert hybrid["hybrid"] is True
        assert hybrid["controller_provider"] == "custom:sudo"
        assert hybrid["controller_model"] == "gpt-5.6-sol"
        assert hybrid["controller_thinking"] == "xhigh"
        assert hybrid["executor_provider"] == "custom:sudo"
        assert hybrid["executor_model"] == "glm-5.2-heavy"
        assert hybrid["executor_thinking"] == "max"

    def test_coupled_preset_has_no_forced_controller(self):
        coupled = next(p for p in cw.workflow_presets() if p["id"] == "coupled-v1")
        assert coupled["hybrid"] is False
        assert coupled["controller_provider"] is None
        assert coupled["controller_model"] is None
        assert coupled["controller_thinking"] is None
        assert coupled["executor_provider"] is None
