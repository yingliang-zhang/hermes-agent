# Hybrid v1 routing shim implementation plan

**Goal:** Add a narrow, profile-local hybrid coding workflow in which GPT-5.6 Sol XHigh remains the Hermes controller/reviewer while OMP implementation and the first two repair attempts use GLM-5.2 Heavy Max. Preserve ordinary coupled model routing and keep durable workflow authority out of Hermes.

**Baseline:** `local/hermes-patched@12198c191`

**Branch:** `feat/hybrid-routing-v1`

**Status:** Frozen implementation contract; no runtime promotion or restart authorized.

## Frozen acceptance contract

1. `coding_workflow` is orthogonal to the primary provider/model. Allowed values are exactly `coupled-v1` and `hybrid-v1`. A genuinely absent legacy value degrades to `coupled-v1`; an explicitly stored, received, or requested malformed/unknown value fails closed instead of silently downgrading.
2. The profile default lives at `coding_workflow.default`. The shipped/default schema remains `coupled-v1`. Candidate installation enables explicit per-session Hybrid canaries but does not change that default; changing the orchestrator profile default to `hybrid-v1` is a later, separately approved decision after the first 20 opt-in runs meet the frozen gates.
3. A session snapshots its workflow on create, persists it in `sessions.model_config`, restores it on resume/rebuild/branch/rollover/compression continuation, and reports it through `session.info`. A manual session choice always wins over the profile default.
4. Plain model selection means `coupled-v1`. Selecting Hybrid means `hybrid-v1` and forces the primary controller route to `custom:sudo / gpt-5.6-sol`. A dedicated route RPC canonicalizes the complete target under a per-session route lock, snapshots old agent/session state, applies the switch, commits model plus workflow through a raising DB transaction, then emits exactly one `session.info`. Any failure restores the complete snapshot and emits nothing, so the UI cannot observe partial state.
5. The Desktop model picker has a dedicated **Workflow Presets** section; Hybrid is not represented as a fake model provider. The pill displays `Hybrid` while the backend continues reporting the real Sol controller provider/model.
6. Settings controls the future-session default. Hybrid settings keep Sol as the main model; coupled settings preserve the selected ordinary model. Auxiliary model assignments are never changed.
7. The session workflow is bridged to terminal subprocesses through a task-local `ContextVar` named `HERMES_CODING_WORKFLOW`. It is an internal bridge, not user-facing environment configuration. Concurrent sessions cannot inherit one another's workflow.
8. The system prompt remains byte-stable. Workflow switches never rebuild `_cached_system_prompt` and never append any conversation marker: Hermes converts non-initial internal system markers into provider-visible synthetic user content. Workflow state is metadata plus structured UI/route events only. Workflow-generic controller instructions may change only before a new session starts at the later approved activation boundary.
9. Profile OMP calls in Hybrid mode require explicit `--workflow hybrid-v1 --role ...`; the wrapper validates the explicit mode against `HERMES_CODING_WORKFLOW` when the trusted session variable is present. A mismatch fails closed.
10. Hybrid route mapping is fixed: `implement` and repair rounds 1–2 use `sudo/glm-5.2-heavy` with `max`; `review`/`audit` independent runs and repair after the two-round budget use `sudo/gpt-5.6-sol` with `xhigh`. Terra is not part of Hybrid v1.
11. Direct review remains in the Hermes Sol controller and must not spawn OMP. Independent review is mandatory for concurrency/lifecycle, security/auth, durability/transaction, public protocol/schema, cross-language/runtime, more than five production files, more than 300 changed lines, missing executable oracle, any substantive repair, operator request, or controller uncertainty. Every review-context field is explicit and validated; omitted/unknown facts cannot become false. Sol may upgrade but never downgrade this floor.
12. Direct-accepted runs are independently sampled when `sha256(run_id) % 5 == 0`. This is an explicit two-stage contract: the decision stage first returns direct review to Hermes, then a post-direct stage may launch the sampled independent review only after the caller records the direct outcome. Sampling is deterministic and computed by the profile-local Python resolver, never by a model.
13. OMP effective route evidence is read from the attributed JSONL `model_change` and `thinking_level_change` records. Requested flags alone are insufficient. Missing, ambiguous, or mismatched evidence fails closed and is recorded.
14. GLM infrastructure failure gets one bounded recovery. Every Hybrid invocation uses an isolated session directory plus a PID/fingerprint writer-ownership fence held across all attempts. A timeout resumes the exact attributed UUID once; it never uses `--continue` or opens a fresh GLM session. After recovery failure, Sol executor fallback is allowed only while the same wrapper still owns the fence, the GLM process is verifiably absent, and attribution/ownership is exact. Otherwise the wrapper fails closed. Fallback records `executor_unavailable`.
15. Semantic review rejection is not infrastructure failure. The same GLM UUID receives at most two targeted repair rounds. A third repair/takeover routes to Sol and requires fresh independent Sol review.
16. The wrapper writes machine-readable route evidence beside the requested output without recording credentials. Evidence includes workflow, role, attempt, requested/effective route, session UUID, recovery/fallback reason, and exit state.
17. Existing hard-tier admission, exact-session isolation, watchdog grace, Terra compaction overlay, explicit coupled routing, and portable-wrapper parity remain intact.
18. No Hermes Proposal/phase/approval/attempt/review database is introduced. Ananke remains the future durable workflow authority; Hermes Hybrid is a temporary stateless/dogfood shim.
19. No MoA integration is introduced in Hybrid v1.
20. No promotion, active-profile config mutation, Desktop restart, or gateway restart occurs without an explicit final approval after all gates and independent review pass.

## Data and authority model

| Fact | Authority | Persistence |
|---|---|---|
| Profile default workflow | `config.yaml` / `coding_workflow.default` | Profile config |
| Current session workflow | TUI gateway session record | `sessions.model_config.coding_workflow` |
| Desktop projection | `session.info.coding_workflow` | Renderer session cache/local draft preference only |
| OMP route decision | Profile-local `hybrid_route_policy.py` | Route-evidence sidecar |
| Semantic repair/review lifecycle | Hermes orchestration in v1; Ananke later | No new Hermes DB |
| Terminal subprocess mode | Session `ContextVar` bridge | Never process-global authority |

## Implementation slices

### Slice 1 — Pure workflow contract and backend persistence (TDD)

**Production files:**
- `hermes_cli/coding_workflow.py` (new)
- `hermes_cli/config.py`
- `gateway/session_context.py`
- `tui_gateway/server.py`

**Tests:**
- `tests/hermes_cli/test_coding_workflow.py` (new)
- `tests/gateway/test_session_context.py` or focused leak test extension
- `tests/tui_gateway/test_coding_workflow.py` (new)

**Steps:**
1. Write failing allowlist/default/config tests.
2. Implement normalization, default lookup, controller preset metadata, and fail-closed write validation.
3. Write failing create/persist/resume/rebuild/continuation tests.
4. Thread `coding_workflow` through session create, deferred/eager resume, runtime persistence, `session.info`, and task-local session context.
5. Add an atomic session route RPC that validates workflow + real provider/model before changing either.
6. Assert that workflow switching changes neither conversation messages nor the cached/persisted system prompt; emit only structured route/session metadata.

### Slice 2 — REST global default and Desktop projection (TDD)

**Production files:**
- `hermes_cli/web_server.py`
- `apps/desktop/src/types/hermes.ts`
- `apps/desktop/src/hermes.ts`
- `apps/desktop/src/app/types.ts`
- `apps/desktop/src/store/session.ts`
- `apps/desktop/src/lib/chat-runtime.ts`
- `apps/desktop/src/app/session/hooks/use-session-actions/index.ts`
- `apps/desktop/src/app/session/hooks/use-session-actions/utils.ts`
- `apps/desktop/src/app/session/hooks/use-session-state-cache.ts`
- `apps/desktop/src/app/session/hooks/use-model-controls.ts`
- `apps/desktop/src/app/session/hooks/use-message-stream/gateway-event.ts`
- `apps/desktop/src/app/shell/model-menu-panel.tsx`
- `apps/desktop/src/app/chat/composer/model-pill.tsx`
- `apps/desktop/src/app/settings/model-settings.tsx`
- touched locale/type files required by the existing i18n contract

**Tests:** focused backend REST tests plus colocated Desktop tests for model controls, session actions/runtime state, picker, pill, settings, and profile isolation.

**Steps:**
1. Extend model-info/options responses with real workflow metadata; do not add a virtual provider.
2. Add one validated profile-default route write path.
3. Add workflow atoms/local draft selection with the same generation/profile-race protection as model selection.
4. Include workflow in session.create and reconcile it from session.info per live session.
5. Render Workflow Presets and an honest Hybrid pill; plain model rows select coupled mode.
6. Add Settings default-mode control and prevent auxiliary drift.
7. Bump the Desktop/backend contract because the renderer depends on workflow RPC/info semantics.

### Slice 3 — Profile policy resolver and wrapper (TDD)

**Staged/live profile files:**
- `SOUL.md` (workflow-generic controller invocation rules; activated only for new sessions)
- `scripts/hybrid_route_policy.py` (new)
- `scripts/omp_with_timeout.sh`
- `scripts/tests/test_hybrid_route_policy.py` (new)
- `scripts/tests/test_omp_route_wrapper.py`
- identical script copies plus updated workflow guidance under the coding-agent-delegation skill

**Steps:**
1. Add pure resolver tests for role mapping, review floor, stable sampling, repair budget, invalid inputs, and coupled parity.
2. Stage controller/skill instructions that source the trusted session workflow, choose an explicit role, provide a stable run ID and complete review context, and preserve the GLM UUID for repair rounds 1–2. Do not change any live session prompt.
3. Integrate explicit workflow/role flags into the shell wrapper without weakening existing route validation/admission; require isolated Hybrid session directories and a writer-ownership fence for normal and hard tiers.
4. Attribute every OMP session and verify effective model/thinking JSONL records.
5. Implement one exact-UUID timeout recovery and safe Sol executor fallback for infrastructure failures.
6. Emit deterministic credential-free evidence sidecars and test timeout, ambiguity, mismatch, fallback, and no-double-writer paths with fake OMP.
7. Preserve compatibility only when no trusted Hybrid session mode is present; a Hybrid session cannot invoke the legacy untyped path.

### Slice 4 — Sealed patch-stack and portable profile

1. Build one semantic runtime/Desktop patch from the accepted feature commit onto the current sealed target.
2. Append it to the ordered patch authority without modifying prior patch bytes.
3. Update manifest/hash/shadow authority through the existing maintenance scripts.
4. Add the resolver/tests to portable required members and enforce live/skill/archive byte parity.
5. Verify rebuild/dry-run/archive determinism. Stop before promotion/restart.

## Verification matrix

| Gate | Required proof |
|---|---|
| Pure policy | allowlist, route matrix, review reasons, stable 20% sampling, repair cap |
| Session persistence | create, immediate info, DB row, deferred/eager resume, rebuild, branch/rollover/compression inheritance |
| Concurrency isolation | two sessions with different workflow ContextVars produce different child env values with no process-global leak |
| Session route switching | Hybrid forces Sol; plain model forces coupled; invalid/partial requests are no-ops |
| Desktop | pre-session selection, live primary, tile isolation, profile switch, resume, pill, Settings, rollback |
| Auxiliary | no auxiliary provider/model/config mutation for any route switch |
| OMP effective route | attributed JSONL proves GLM Max or Sol XHigh; ambiguity/mismatch rejected |
| Recovery | exact UUID only, one bounded attempt, writer fence held, no live/unknown GLM before Sol fallback, route evidence records escalation |
| Existing wrapper | route mapping, hard admission, watchdog, Terra overlay, explicit-mode compatibility all pass |
| Python regression | focused suites, adjacent gateway/session/model tests, then full relevant suite |
| Desktop regression | focused Vitest, full Vitest, typecheck, production build, changed-file lint |
| Packaging | patch replay/tree identity, candidate verification, portable archive parity/determinism |
| Independent review | fresh Sol hard-tier read-only review returns `ACCEPT` with no P0/P1/material P2 |

## Activation boundary

After all verification and review gates pass, prepare but do not execute:

1. promote the sealed candidate while keeping `coding_workflow.default` absent or `coupled-v1` and keeping `model.default: gpt-5.6-sol` / `model.provider: custom:sudo`;
2. install the repaired profile-local policy and wrapper;
3. restart gateway and Desktop;
4. run four-way UI/session/OMP canaries and collect the first 20 runs through explicit per-session `hybrid-v1` selection;
5. only after those preregistered gates pass, prepare a separate operator decision to set `coding_workflow.default: hybrid-v1`.

Promotion/profile installation and the later default change are separate approval boundaries. Neither is implied by candidate verification.
