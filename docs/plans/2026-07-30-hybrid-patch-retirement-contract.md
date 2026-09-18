# Hybrid Patch Retirement Contract

Date: 2026-07-30
Run: `hybrid-patch-retirement-47e5ff56`
Operator decision: future orchestrator work remains on `coupled-v1`; the Hybrid core patch family is no longer a rollout candidate.

## Frozen pre-state

- Active and portable profile defaults: `coding_workflow.default=coupled-v1`
- Runtime target (must remain unchanged): `local/hermes-patched@1606b492ba9257b6f33a6a1fb936ad2752a31d1f`
- Shadow authority: `local/hermes-patch-stack@d8df618ca9164c63dbfc308052e01ca4360cf10e`
- Frozen base: `d71033a4077a6dfdcdb42c9e9eeab4c41e4a7012`
- PR layer: `a881225f181f9dd5636d07970460f905d49e7f3f`
- Current manifest: 32 ordered patches, final tree `70473bd4af9529429341f1b9fc81e6f5196098d5`

## Retirement set

Retire these routing/session-workflow patches from the authoritative ordered patch list:

1. `190-hybrid-role-based-routing-v1.patch`
2. `200-hybrid-routing-preactivation-repairs.patch`
3. `205-hybrid-routing-cwd-attribution-contract.patch`
4. `210-hybrid-routing-final-audit-repairs.patch`
5. `215-hybrid-routing-anti-drift-contract.patch`
6. `220-hybrid-routing-exact-authority.patch`
7. `240-coupled-workflow-first-row-persistence.patch`

Patch `240` is retired with the Hybrid family because it calls and tests the `coding_workflow` authority introduced by patch `190`; it is not independently applicable to the pre-Hybrid stack.

## Retained post-Hybrid patches

- `225-desktop-session-rotation-kind.patch`: retain. The pre-Hybrid source already defines `kind: 'compression' | 'rollover'`, while the cache emitter omitted the required `kind`; patch `225` closes that independent contract bug.
- `230-desktop-rebased-test-contracts.patch`: retain. It updates unrelated cron and prompt-action fixtures.
- `250-server-overload-retry-contract.patch`: retain. It is the accepted overload/retry contract and must be semantically rebased onto the pre-Hybrid tail without carrying Hybrid code.

Expected result: 25 patches (`32 - 7`) in the same order, with gaps in numeric prefixes allowed.

## Control-plane change

Extend `maintain_hermes_patch_stack.py` with a separate proof-aware intentional-retirement mode. It must not weaken the existing upstream-equivalent `--drop-patch-replacement` contract. Retirement requires explicit mode, explicit exact base, a target derived from the same PR layer, exact retained order/count, stable patch IDs or SHA-256-bound adaptation proofs, round-trip tree equality, thin-bundle verification, manifest-last publication, and shadow compare-and-swap.

## Execution boundaries

Allowed without another approval:

- Edit profile-local maintainer and focused tests.
- Build a detached candidate from the last pre-Hybrid semantic commit.
- Rebase/adapt retained patches `225`, `230`, and `250`.
- Atomically update only the shadow ref and patch-stack artifacts after all gates pass.
- Regenerate the credential-free portable archive and commit/push the feature branch.

Forbidden in this run:

- Move or rebuild `local/hermes-patched`.
- Run the canonical updater or promote/install a runtime.
- Install/restart Desktop or restart the gateway.
- Rewrite existing historical ledger evidence.
- Delete retired patch bytes; archive them outside the live `*.patch` set.

## Verification gates

1. Maintainer TDD: positive retirement, adapted-proof case, rejection/no-mutation matrix, legacy append/E10/E11 regression.
2. Candidate: exact 25-commit tail from the unchanged PR layer, zero merges, retained patches in order, no Hybrid files/symbols introduced by the retired family.
3. Focused runtime tests for `225`, `230`, and `250`; Python compile and Desktop type/test checks for touched files.
4. `validate_patch_stack()` exact-set, clean apply/round-trip tree, thin-bundle prerequisite, shadow-ref identity.
5. Exporter tests and deterministic double export; archive manifest has exactly 25 patches and no retired live patch member.
6. Direct Sol review of actual diff, candidate history, test output, and archive evidence; explicit `ACCEPT` required.

No quality or completion claim is permitted before these gates return real evidence.
