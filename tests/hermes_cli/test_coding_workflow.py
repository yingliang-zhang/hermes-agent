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
from dataclasses import FrozenInstanceError
from types import MappingProxyType

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


class TestCanonicalControllerRoute:
    def test_hybrid_route_atomically_binds_sol_and_xhigh_reasoning(self):
        route = cw.canonicalize_route(
            "hybrid-v1",
            provider="custom:sudo",
            model="gpt-5.6-sol",
            reasoning_config={"enabled": True, "effort": "low"},
        )

        assert route.coding_workflow == "hybrid-v1"
        assert route.provider == "custom:sudo"
        assert route.model == "gpt-5.6-sol"
        assert route.reasoning_config == {"enabled": True, "effort": "xhigh"}
        with pytest.raises(FrozenInstanceError):
            route.model = "other"

    def test_coupled_route_recursively_freezes_reasoning_state(self):
        caller_reasoning = {
            "nested": {
                "sequence": [{"effort": "high"}],
                "flags": {"trace", "cache"},
            }
        }
        route = cw.canonicalize_route(
            "coupled-v1",
            provider="custom:sudo",
            model="gpt-5.6-sol",
            reasoning_config=caller_reasoning,
        )

        assert route.requested_provider == "custom:sudo"
        caller_reasoning["nested"]["sequence"][0]["effort"] = "low"
        caller_reasoning["nested"]["flags"].add("caller-mutation")
        assert route.reasoning_config == {
            "nested": {
                "sequence": [{"effort": "high"}],
                "flags": {"trace", "cache"},
            }
        }

        stored_nested = dict(route._reasoning_items)["nested"]
        with pytest.raises(TypeError):
            stored_nested["sequence"][0]["effort"] = "private-mutation"
        with pytest.raises(AttributeError):
            stored_nested["sequence"].append("private-mutation")
        with pytest.raises(AttributeError):
            stored_nested["flags"].add("private-mutation")

        thawed = route.reasoning_config
        thawed["nested"]["sequence"][0]["effort"] = "thawed-mutation"
        assert route.reasoning_config["nested"]["sequence"][0]["effort"] == "high"

    def test_coupled_route_accepts_generic_mapping_root_without_exposing_nodes(self):
        reasoning = MappingProxyType(
            {
                "effort": "xhigh",
                "nested": MappingProxyType({"sequence": [{"enabled": True}]}),
            }
        )

        route = cw.canonicalize_route(
            "coupled-v1",
            provider="custom:sudo",
            model="gpt-5.6-sol",
            reasoning_config=reasoning,
        )

        assert route.reasoning_config == {
            "effort": "xhigh",
            "nested": {"sequence": [{"enabled": True}]},
        }
        stored_nested = dict(route._reasoning_items)["nested"]
        with pytest.raises(TypeError):
            stored_nested["sequence"][0]["enabled"] = False

    @pytest.mark.parametrize(
        "bad_root",
        [
            {1: "bad"},
            MappingProxyType({1: "bad"}),
            ["not", "a", "mapping"],
        ],
    )
    def test_coupled_route_rejects_invalid_reasoning_roots_and_keys(self, bad_root):
        with pytest.raises(TypeError):
            cw.canonicalize_route(
                "coupled-v1",
                provider="custom:sudo",
                model="gpt-5.6-sol",
                reasoning_config=bad_root,
            )

    def test_coupled_route_rejects_cyclic_reasoning_mapping(self):
        cyclic = {}
        cyclic["self"] = cyclic

        with pytest.raises(TypeError, match="cyclic"):
            cw.canonicalize_route(
                "coupled-v1",
                provider="custom:sudo",
                model="gpt-5.6-sol",
                reasoning_config=cyclic,
            )


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
