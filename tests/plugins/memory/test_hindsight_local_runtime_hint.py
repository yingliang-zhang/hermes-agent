"""NousResearch/hermes-agent#7718 — actionable message when the local_embedded
runtime probe fails.

Under the split runtime the agent venv holds only ``hindsight-client`` +
``hindsight-embed`` plus the lightweight ``embedded_runtime`` adapter (the ML
stack lives in the sealed server venv). When agent-side pieces are absent the
provider disables itself; the disable warning should point the user at the fix
rather than just echoing ``No module named 'hindsight_client'``.
"""

import plugins.memory.hindsight as hs
from plugins.memory.hindsight import HindsightMemoryProvider, _local_runtime_hint


def test_hint_for_missing_hindsight_client():
    hint = _local_runtime_hint("No module named 'hindsight_client'")
    assert "hermes memory setup" in hint
    assert "HINDSIGHT_EMBED_API_EXECUTABLE" in hint


def test_hint_for_missing_hindsight_embed():
    hint = _local_runtime_hint("No module named 'hindsight_embed.daemon_embed_manager'")
    assert "hermes memory setup" in hint


def test_hint_for_legacy_monolith_reason():
    # Pre-split configs surfaced the retired metapackage's module name; the
    # broadened predicate still matches (any "hindsight*" module name trips it).
    hint = _local_runtime_hint("No module named 'hindsight'")
    assert "hermes memory setup" in hint
    assert "hindsight-all" not in hint


def test_no_hint_for_unrelated_runtime_error():
    # e.g. the NumPy-on-old-CPU failure _check_local_runtime also guards against
    assert _local_runtime_hint("Illegal instruction (NumPy SIMD)") == ""
    assert _local_runtime_hint(None) == ""


# unavailable_reason() — surfaces the hint through the reachable path (#7718):
# is_available() gates initialize() out, so the hint must come from here.


def test_unavailable_reason_surfaces_hint_for_local_embedded(monkeypatch):
    monkeypatch.setattr(hs, "_load_config", lambda: {"mode": "local_embedded"})
    monkeypatch.setattr(hs, "_check_local_runtime", lambda: (False, "No module named 'hindsight_client'"))
    reason = HindsightMemoryProvider().unavailable_reason()
    assert "hermes memory setup" in reason
    assert reason == reason.strip()  # no leading/trailing whitespace


def test_unavailable_reason_empty_for_cloud(monkeypatch):
    monkeypatch.setattr(hs, "_load_config", lambda: {"mode": "cloud"})
    # Should not even probe the runtime for a cloud provider.
    monkeypatch.setattr(hs, "_check_local_runtime", lambda: (_ for _ in ()).throw(AssertionError("probed")))
    assert HindsightMemoryProvider().unavailable_reason() == ""


def test_unavailable_reason_empty_when_runtime_present(monkeypatch):
    monkeypatch.setattr(hs, "_load_config", lambda: {"mode": "local_embedded"})
    monkeypatch.setattr(hs, "_check_local_runtime", lambda: (True, None))
    assert HindsightMemoryProvider().unavailable_reason() == ""
