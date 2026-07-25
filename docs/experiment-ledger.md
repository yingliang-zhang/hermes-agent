# Experiment Ledger

## Project

- **Name:** Local Desktop session rollover MVP
- **Branch:** `feat/session-rollover-mvp`
- **Baseline:** `11e39ff5e066143a0b5ae787a7358c0c29653552`
- **Status:** MVP implementation and current-target semantic port accepted; ordered patch-stack packaging pending
- **Scope:** Opt-in, local Electron Desktop only

## Frozen acceptance contract

1. The feature is off by default and cannot activate for remote Desktop, TUI, CLI, gateway messaging, cron, or compute-host sessions.
2. A rollover may be offered only after a successful assistant final response is durably persisted and the turn has fully settled.
3. Unknown or active blockers fail closed: backend/client prompt queues, tool calls, processes, delegations, approvals or input requests, notifications, goals, another turn, stale identity, or changed history.
4. The successor contains a bounded textual handoff rather than replaying the full predecessor transcript.
5. Predecessor end, successor creation/linkage, and handoff persistence are atomic and idempotent.
6. The successor inherits workspace, profile, title lineage, and effective model/provider/reasoning/service-tier route.
7. Desktop follows only an identity-bound successor event for the exact foreground runtime and predecessor stored ID; stale/background events never steal focus.
8. A failure leaves the predecessor usable and emits no success event. No prompt/draft/notification/process/delegation migration is introduced.
9. Existing prompt-cache and message-role alternation invariants remain valid.
10. No runtime promotion or Desktop/gateway restart occurs without explicit approval.

## Decision log

| ID | Decision | State | Evidence |
|---|---|---|---|
| D1 | Build from the clean patched-runtime baseline, not the full Workstream/Epoch WIP | Accepted | Branch baseline above |
| D2 | Reuse compression concurrency/lineage lessons and Desktop stored-ID identity guards, but not compression's state-migration path | Accepted | Independent audit: architecture C rejected |
| D3 | Use a client-visible offer followed by a backend-revalidated commit so the renderer-owned composer queue can fail closed | Accepted | Independent audit: `READY TO PLAN` |
| D4 | Reuse the already-produced compaction summary for a deterministic handoff capped at 20 lines / 16 KiB; do not add a second provider call | Accepted | Frozen implementation plan |
| D5 | Use dedicated `rollover` lineage plus `_rollover_from` model metadata; do not masquerade as compression or add a schema column | Accepted | Independent audit architecture A |
| D6 | Build a fresh successor agent; ordinary fresh-session memory loading is allowed, but no explicit state/provider migration occurs | Accepted | Frozen implementation plan |
| D7 | Suppress a predecessor agent's delayed review-summary callback after the swap so it cannot appear in the successor transcript | Accepted | Frozen implementation plan |
| D8 | Treat process completion as a publication protocol: running, exited-unpublished, queued, or delivery-in-flight all block rollover | Accepted | Deterministic process marker and offer/commit interleaving tests |
| D9 | Use a conservative process-global TUI notification handoff latch from before dequeue through dispatch, requeue, consume, or intentional drop | Accepted | Hard re-review: D3 PASS, no queue-to-delivery visibility gap |
| D10 | Port the accepted net feature diff onto current patched target `a910b7170` rather than publish an obsolete-base mbox | Accepted | Current target advanced after development began; five semantic conflicts resolved with both current behavior and rollover contracts preserved |
| D11 | Keep capability fail closed for unknown connection state; only explicit local-primary state advertises `true` | Accepted | Nine production create/resume callsites classified; explicit local, remote, watch, and unknown tests retained |
| D12 | Restore the byte-consistent archived 21-patch artifact set before canonical rebased append | Accepted | Partial prior publication detected; restored set passed `validate_patch_stack` without modifying target/runtime refs |

## Verification runs

| Run | Commit | Scope | Command | Result | Notes |
|---|---|---|---|---|---|
| baseline-status | `11e39ff5e` | Worktree cleanliness | `git status --short; git diff --stat` | PASS | No output; clean tree before ledger creation |
| audit-1 | `11e39ff5e` | Read-only source audit | Route-aware OMP, 600 s | INCOMPLETE | Backend evidence collected; run compacted before synthesis |
| audit-1-resume | `11e39ff5e` | Remaining audit + verdict | Exact OMP UUID resume, 900 s | PASS | `READY TO PLAN`; compliant design is not a safe one-day implementation |
| slice-1-red | `11e39ff5e` + tests | SessionDB rollover contract | `scripts/run_tests.sh tests/test_hermes_state_session_rollover.py -q` | RED | 21 failures: rollover API/behavior absent |
| slice-1-focused | uncommitted Slice 1 | Rollover storage behavior | `scripts/run_tests.sh tests/test_hermes_state_session_rollover.py -q` | PASS | 26 passed, 0 failed |
| slice-1-combined | uncommitted Slice 1 | Rollover + compression locks + SessionDB regression | `scripts/run_tests.sh tests/test_hermes_state_session_rollover.py tests/test_hermes_state_compression_locks.py tests/test_hermes_state.py -q` | PASS | Orchestrator rerun: 436 passed, 0 failed |
| slice-1-static | uncommitted Slice 1 | Syntax and whitespace | `.venv/bin/python -m py_compile hermes_state.py tests/test_hermes_state_session_rollover.py && git diff --check` | PASS | Exit 0, no output |
| slice-2a-red | uncommitted Slice 2A tests | Offer/quiescence contracts | `scripts/run_tests.sh tests/tui_gateway/test_session_rollover.py -q` | RED | Initial 60 failures; hardening extension later produced 21 contract failures |
| slice-2a-focused | uncommitted Slice 2A | Offer/quiescence behavior | `scripts/run_tests.sh tests/tui_gateway/test_session_rollover.py -q` | PASS | 114 passed, 0 failed |
| slice-2a-adjacent | uncommitted Slice 2A | Offer + turn ordering + final persistence | `scripts/run_tests.sh tests/tui_gateway/test_session_rollover.py tests/tui_gateway/test_turn_origin_notifications.py tests/tui_gateway/test_finalize_session_persist.py -q` | PASS | Orchestrator rerun: 157 passed, 0 failed |
| slice-2a-static | uncommitted Slice 2A | Syntax and whitespace | `.venv/bin/python -m py_compile tui_gateway/session_rollover.py tui_gateway/server.py hermes_cli/config.py tests/tui_gateway/test_session_rollover.py && git diff --check` | PASS | Exit 0, no output |
| slice-2b-red | `4337ad2a9` + tests | Commit RPC and fresh-runtime swap | `scripts/run_tests.sh tests/tui_gateway/test_session_rollover_commit.py -q` | RED | 11 passed, 7 failed before commit implementation; lock-release fault later reproduced as 1 expected failure |
| slice-2b-focused | uncommitted Slice 2B | Commit/idempotency/failure behavior | `scripts/run_tests.sh tests/tui_gateway/test_session_rollover_commit.py -q` | PASS | Orchestrator-visible final: 22 passed, 0 failed |
| slice-2b-combined | uncommitted Slice 2B | Commit + offer + storage + turn/finalization/compression regressions | `scripts/run_tests.sh tests/tui_gateway/test_session_rollover_commit.py tests/tui_gateway/test_session_rollover.py tests/test_hermes_state_session_rollover.py tests/tui_gateway/test_turn_origin_notifications.py tests/tui_gateway/test_finalize_session_persist.py tests/test_hermes_state_compression_locks.py tests/agent/test_compression_rotation_state.py -q` | PASS | Orchestrator rerun: 246 passed, 0 failed |
| slice-2b-static | uncommitted Slice 2B | Syntax and whitespace | `.venv/bin/python -m py_compile tui_gateway/server.py tests/tui_gateway/test_session_rollover_commit.py && git diff --check` | PASS | Exit 0, no output |
| compute-host-red | `8873d2829` + tests | Compute-host exclusion proof | `scripts/run_tests.sh tests/tui_gateway/test_session_rollover.py -q -k compute_host` | RED | 4 expected failures before proof/server guard |
| compute-host-gate | `1cb1f3b74` | Offer/commit/adjacent regression | `scripts/run_tests.sh tests/tui_gateway/test_session_rollover.py tests/tui_gateway/test_turn_origin_notifications.py tests/tui_gateway/test_finalize_session_persist.py tests/tui_gateway/test_session_rollover_commit.py -q` | PASS | Orchestrator rerun: 183 passed, 0 failed |
| slice-3-red | `1cb1f3b74` + tests | Desktop capability, offer, completion, and rotation | Focused Vitest six-file gate | RED | Three missing-module suites plus 10 expected behavior failures; 78 tests already passed |
| slice-3-focused | uncommitted Slice 3 | Desktop rollover behavior | `npm exec vitest -- run --project ui` with six focused files | PASS | Orchestrator rerun after lifecycle hardening: 128 passed, 0 failed |
| slice-3-adjacent | uncommitted Slice 3 | Session state/actions/message stream/prompt/tile regressions | `npm exec vitest -- run --project ui` with 17 adjacent files | PASS | Orchestrator rerun: 226 passed, 0 failed |
| slice-3-types | uncommitted Slice 3 | Renderer, Electron, and E2E TypeScript contracts | `npm run typecheck` in `apps/desktop` | PASS | Exit 0 after concrete deferred-promise typing repair |
| slice-3-full | `40bed25f2` | Full Desktop Vitest regression | `npm test` in `apps/desktop` | PASS | 2293 passed, 2 skipped, 0 failed |
| slice-3-build | `40bed25f2` | Desktop production bundle | `npm run build` in `apps/desktop` | PASS | Main/preload/renderer build and `assert-dist-built` passed |
| review-fix-red | `40bed25f2` + tests | Predecessor env cleanup and process completion publication | Five focused Python files | RED | 5 expected failures before production repair |
| review-fix-focused | uncommitted review fixes | Storage/process/offer/commit lifecycle | Five focused Python files | PASS | 324 passed, 0 failed |
| review-fix-combined | uncommitted review fixes | Initial 15-file Python integration gate | `scripts/run_tests.sh` with state/process/gateway/provider/rotation files | PASS | 952 passed, 0 failed |
| d3-handoff-red | uncommitted tests | Process blocker propagation and queue-to-delivery handoff | Three focused Python files | RED | Offer 3 failures and commit 2 failures; notification tests blocked on absent handoff helper |
| d3-handoff-focused | uncommitted final fixes | Offer, commit, and notification delivery lifecycle | Three focused Python files | PASS | 127 + 26 + 36 passed, 0 failed |
| d3-handoff-lifecycle | uncommitted final fixes | Storage/process/notify/offer/commit lifecycle | Five focused Python files | PASS | 329 passed, 0 failed |
| d3-handoff-superset | uncommitted final fixes | Entire TUI gateway plus lifecycle files | `scripts/run_tests.sh tests/tui_gateway/ ... -q` | PASS | 44 files, 772 passed, 0 failed |
| final-python-gate | uncommitted final fixes | State/process/gateway/provider/rotation integration | 15-file `scripts/run_tests.sh ... -q` | PASS | Orchestrator rerun: 964 passed, 0 failed |
| final-static | uncommitted final fixes | Python syntax and whitespace | `.venv/bin/python -m py_compile ... && git diff --check` | PASS | Exit 0, no output |
| final-hard-review | uncommitted final fixes | Focused P0-P2 re-review after D3 repairs | Exact hard-tier OMP resume, 600 s | PASS | `ACCEPT`; six touched test files 365 passed; no P0/P1/material P2 |
| patch-authority-preflight | restored archived 21-patch set | Manifest/hash/bundle exactness | `validate_patch_stack(...)` | PASS | `RESTORED_VALID 21`; target `c98e5a39b`, stack tree `63618f1bd0`; runtime target untouched |
| current-port-focused-python | staged port on `a910b7170` | Storage/process/notification/coordinator/commit/turn-origin | Six focused files via `scripts/run_tests.sh` | PASS | 365 passed, 0 failed |
| current-port-python-final | staged port on `a910b7170` | Original final 15-file integration gate | `scripts/run_tests.sh` with rollover/state/compression/protocol/provider/process/delivery files | PASS | Orchestrator rerun: 1024 passed, 0 failed |
| current-port-desktop-focused | staged port on `a910b7170` | Capability, adoption, routing, current-target conflict paths | Six focused Vitest files plus adjacent command | PASS | 154 focused + 13 adjacent passed, 0 failed |
| current-port-desktop-full | staged port on `a910b7170` | Full Desktop regression | `npm test` in `apps/desktop` | PASS | Orchestrator rerun: 2836 passed, 3 skipped, 0 failed |
| current-port-types-build | staged port on `a910b7170` | Type contracts and production bundle | `npm run typecheck`; `npm run build` | PASS | TypeScript clean; renderer/main/preload build and `assert-dist-built` passed |
| current-port-lint-static | staged port on `a910b7170` | Changed TS/TSX lint; Python syntax; diff/unmerged state | exact changed-file `eslint`; `py_compile`; `git diff --check`; `git ls-files -u` | PASS | ESLint 0 errors/32 warnings; syntax/diff clean; zero unmerged/unstaged paths |
| current-port-hard-review | staged port on `a910b7170` | Five conflict areas, nine capability callsites, backend equivalence | Read-only hard OMP 900 s | PASS | `ACCEPT`; no P0/P1/material P2; backend equivalent except intentional config path |

## Implementation runs

| Slice | Commit | Tests | Result | Notes |
|---|---|---|---|---|
| Backend atomic successor | `4337ad2a9` | 26 focused; 436 combined | ACCEPT | P1 exact-content replay fence repaired and reverified |
| Quiescence/offer coordinator | `4337ad2a9` | 114 focused; 157 adjacent | ACCEPT | Required proof schema, exact history/DB tail, local capability, fail-closed blockers, and hidden-task descendant ownership verified |
| Commit + fresh runtime swap | `8873d2829` | 22 focused; 246 combined | ACCEPT | Token/fence revalidation, monotonic turn identity, route/workdir preservation, lease compensation, memory teardown, idempotency, and post-commit release fault verified |
| Compute-host exclusion guard | `1cb1f3b74` | 4 focused; 183 adjacent | ACCEPT | Required eligibility proof blocks active or unknown compute-host state |
| Desktop local/foreground adoption | `40bed25f2` | 128 focused; 226 adjacent; 2293 full; typecheck/build | ACCEPT | Local non-watch capability, exact active identity, queue/draft fail-closed checks, identity-bound route follow, no rollover migration, and bounded transient markers verified |
| Process/notification lifecycle hardening | uncommitted | 329 focused; 772 TUI superset; 964 combined | ACCEPT | Predecessor env cleanup, publication marker, complete process blocker propagation, and dequeue-to-delivery latch verified |
| Combined regression gate | uncommitted final fixes | 964 Python; 2293 Desktop; typecheck/build/static | ACCEPT | No failures in targeted full integration surfaces; changed Desktop lint has zero errors |
| Independent P0-P2 review | uncommitted final fixes | 365 touched-file tests | ACCEPT | Hard reviewer: D3/D6 PASS; no P0/P1/material P2 remains |
| Current-target semantic port | staged on `a910b7170` | 365 focused; 1024 final Python; 154 + 13 focused/adjacent Desktop; 2836 full Desktop; typecheck/build/static | ACCEPT | Profile routing, queue/stop/retry/drift guards, tile hydration, fresh-draft behavior, current config, and additive capability preserved |
| Current-target P0-P2 review | staged on `a910b7170` | Read-only hard port audit | ACCEPT | Nine create/resume callsites complete; no local-positive test weakened; no P0/P1/material P2 |

## Residual risks

- The global notification handoff latch intentionally delays rollover in all local Desktop sessions while any one session is transferring a notification.
- The live notification poller now checks the process-global queue every 100 ms while idle, adding small process-wide polling overhead.
- An exceptional `completion_queue.put()` failure blocks rollover until the finished process row is pruned; this is availability-conservative and fail closed.
- Terminal completion queues remain process-local across a full gateway crash/restart; this MVP does not add durable terminal-notification recovery.
- The current target has broad pre-existing lint debt; exact changed-file ESLint reports zero errors and 32 warnings.
- Patch-stack publication and runtime promotion remain external operations. The live runtime is unchanged until canonical `180-*` rebased append completes and restart is explicitly approved.

## Hybrid v1 routing shim — 2026-07-25

### Contract

- **Plan:** `docs/plans/2026-07-25-hybrid-routing-v1.md`
- **Branch:** `feat/hybrid-routing-v1`
- **Baseline:** `12198c191`
- **Status:** implementation in progress; active runtime/profile unchanged
- **Primary comparison:** existing `coupled-v1` route versus `hybrid-v1` (Sol XHigh controller/reviewer + GLM Heavy Max implement/repair)

### Runs

| Run | Commit | Scope | Command | Result | Notes |
|---|---|---|---|---|---|
| hybrid-baseline-python | `12198c191` | Existing session model/runtime persistence and subprocess context isolation | `scripts/run_tests.sh tests/tui_gateway/test_reasoning_session_scope.py tests/tui_gateway/test_custom_provider_session_persistence.py tests/tools/test_local_env_session_leak.py tests/gateway/test_session_context.py -q` | PASS | 36 passed, 0 failed; nonexistent fourth path was ignored by the runner |
| hybrid-baseline-desktop | `12198c191` | Existing model controls, session actions/cache, picker, pill, and Settings | six focused UI Vitest files | PASS | 99 passed, 0 failed |
| hybrid-plan-review | `12198c191` + plan | Independent Sol XHigh architecture review | route-aware hard OMP, 900 s + one exact-UUID 300 s resume, read-only | CHANGES REQUIRED | Both print-mode runs timed out; the exact-resume JSONL contained a partial final text with `VERDICT: CHANGES REQUIRED`, recovered verbatim to the profile run artifact. P0: no workflow markers, transactional route commit, controller instruction wiring, and writer ownership for every Hybrid invocation. This is not an ACCEPT verdict. |
| hybrid-policy-r0 | worktree staging | Initial profile-local route resolver | `python -m pytest .profile-staging/scripts/tests/test_hybrid_route_policy.py -q` | TEST PASS / REVIEW REJECT | 88 passed + 50 subtests; coupled passthrough, exact roles, and review-reason contract were wrong. Findings returned to GLM repair round 1. |
| hybrid-policy-r1 | worktree staging | Corrected resolver contract | `python -m pytest .profile-staging/scripts/tests/test_hybrid_route_policy.py -q` | TEST PASS / REVIEW ITERATE | 121 passed + 68 subtests; independent rerun passed, but omitted review facts still defaulted to false/zero. Final GLM repair round 2 requires complete explicit Hybrid review context. |
| hybrid-portable-junk-filter | control repo working tree | Portable exporter excludes patch rejection leftovers | exporter unittest + live-profile temporary export | PASS | Focused 1/1 and full exporter 12/12 passed; temp archive exported 1906 files with `JUNK_COUNT=0`. Live `.rej/.orig` files were not deleted. |
| hybrid-rest-profile-default | worktree | REST workflow preset/default API and existing schema integration | focused/helper/config suites + adjacent web-server file + exact failed selector rerun | PASS | Helper+REST 34/34, ContextVar leak 12/12, config 184/184. Adjacent web-server run reached 501 PASS with one new orphan-category failure; `coding_workflow` was merged into the existing agent category, then the failed selector passed 1/1 and focused REST remained 8/8. Full 502-test file was not redundantly rerun after the one-line category fix. |
| hybrid-policy-r2 | worktree staging | Complete explicit review context and stable Hybrid run identity | Python 3.11 unittest + compile + CLI canaries | ACCEPT (resolver only) | GLM round 2 made all review facts explicit and required run IDs for every Hybrid role. Independent run passed 183 tests; Sol direct review added one whitespace-only run-id invariant, then 184/184 passed. CLI canaries: missing run/fact exit 2; safe direct review, Sol audit, and coupled passthrough exit 0 with expected JSON. |
| hybrid-runtime-ui-focused | worktree | Session workflow persistence, atomic route RPC, ContextVar isolation, Desktop Workflow Presets/pill | 7-file Python adjacency; 9-file Desktop Vitest; Desktop typecheck; gateway inheritance | PASS | Python 280/280; Desktop 154/154; gateway inheritance 7/7; typecheck exit 0. Route transaction tests cover single session.info, marker-free commit, fixed Hybrid Sol controller, and rollback of model/provider/workflow on strict DB failure. |
| hybrid-wrapper-slice-a | worktree staging | Typed wrapper args, resolver dispatch, coupled parity, caller resume matrix, Hybrid normal-tier session isolation | 49 baseline wrapper tests plus 9 new Hybrid route tests; resolver suite separately 184/184 | PASS (Slice A only) | Real RED exposed a broken JSON parser (heredoc owned stdin); JSON is now passed as argv. Implement→GLM Max, repairs 1–2 exact GLM resume, repair 3 fresh Sol, independent review/audit Sol, direct review no spawn, trust mismatch, coupled parity, and session-dir checks pass. Slice B remains pending. |
| hybrid-profile-instructions | worktree staging | Workflow-generic SOUL/skill/routing references | `python -m unittest tests.test_hybrid_profile_instructions` | PASS | 8/8. Literal `--workflow "$HERMES_CODING_WORKFLOW"`, typed roles/run ID, exact repair UUID, direct-review no-OMP, explicit ACCEPT, wrapper-owned timeout, no personal paths/active session IDs, and non-overclaiming staged status are machine-checked. |
| hybrid-wrapper-slice-b1 | worktree staging | Writer ownership, exact JSONL attribution, effective-route verification, atomic credential-free sidecar | One 600 s implementation timeout + exact-UUID 600 s resume; independent staged suite | PASS | Exact session `019f9799-1d2f-7000-8ad5-4dd69de7263e`; resume exited 0. Independent rerun: 67 wrapper + 184 resolver + 8 instructions = 259/259. Ownership binds run/session/cwd/wrapper/child fingerprints; malformed/active/unverifiable cases fail closed; sidecar mode 0600 and parity/syntax pass. |
| hybrid-wrapper-slice-b2 | worktree staging | One exact timeout resume, bounded fresh-Sol fallback, no-overlap and ordered attempt evidence | Original + exact resume both 600 s timeout; JSONL/worktree recovery; local focused/full gates | PASS | Double-timeout policy stopped further resume. Production and 4 initial tests were on disk; local review added missing attribution-no-fallback, semantic-success, hard-slot retention, and cleanup contracts. B2 8/8; final total 75 wrapper + 184 resolver + 8 instructions = 267/267. No `--continue`; no recursive wrapper; outer hard slot retained across attempts. |
| hybrid-final-verification | worktree candidate vs clean `HEAD@12198c191` | Full Python attribution, Desktop full suite/build, staged shim, exporter, sealed authority, lint/syntax/diff | Canonical runners + clean detached baseline worktree | PASS (candidate-only deterministic failures: 0) | Initial concurrent Python full run: 40 failure files / 130 failed tests + 10 no-run. Two candidate-specific mock/rollover gaps (17 failures) were fixed; final changed-path adjacency is 473/473. Remaining baseline subset reproduced 37 failure files / 112 failed tests + 9 no-run on clean HEAD. The two residual candidate/base differences were isolated: `test_base_environment.py` passed on retry (known timing flake) and `test_context_compressor.py` passed 222/222. Desktop: 316 files passed, 1 skipped; 2,856 tests passed, 3 skipped; typecheck/build/postbuild/native staging PASS. Post-lint focused Desktop: 58/58, lint 0 errors (60 baseline warnings). Staged shim 267/267; exporter 12/12; current sealed authority valid at 22 patches ending `180-*`; `py_compile` and `git diff --check` PASS. |

### Metrics to collect after activation

| Metric | Coupled arm | Hybrid arm | Evidence source |
|---|---:|---:|---|
| First-pass acceptance rate |  |  | Sol review decisions bound to run IDs |
| Mean GLM repair rounds | n/a |  | Route-evidence sidecars + review records |
| Executor escalation rate | n/a |  | `repair_budget_exhausted` / `executor_unavailable` evidence |
| Independent-review finding rate |  |  | Fresh Sol review outputs |
| Direct-review escape rate |  |  | Deterministic 20% sampled reviews |
| End-to-end wall time |  |  | Wrapper evidence timestamps |
