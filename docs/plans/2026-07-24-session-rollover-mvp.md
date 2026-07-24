# Local Desktop Session Rollover MVP — Implementation Plan

> Status: READY TO IMPLEMENT after read-only audit at `11e39ff5e`.

## Goal

After an opted-in local Electron Desktop session has compacted in place and then completes a clean, durably persisted assistant turn, create one fresh durable successor at a quiescent boundary and follow it without user-visible focus theft or state migration.

## Non-goals

No Workstream/Epoch schema, prompt replay, provider exactly-once recovery, goal/notification/queue migration, remote Desktop, messaging gateway, TUI, dashboard chat, automatic successor turn, or runtime promotion.

## Frozen design

- Add `desktop.session_rollover: false` to `DEFAULT_CONFIG`.
- Trigger only after the current post-turn queue/goal/notification drains.
- Use a two-step protocol:
  1. backend emits `session.rollover.offer` after a candidate turn settles;
  2. local Desktop verifies foreground identity and an empty renderer queue, then sends `session.rollover.commit`.
- Commit revalidates every backend blocker under the same `history_lock` used by prompt admission.
- Keep the gateway runtime ID stable; change only the durable stored session ID and swap in a fresh `AIAgent`.
- Persist a dedicated rollover child using `parent_session_id`, `end_reason='rollover'`, and `model_config._rollover_from`.
- Handoff is deterministic: at most 20 non-empty lines and 16 KiB, drawn from the latest compaction summary plus the final completed user/assistant exchange. No auxiliary provider call.
- The successor receives a role-valid handoff pair and ordinary fresh-session memory/context initialization. No explicit MEMORY/USER/provider state transfer.
- Existing compression rotation behavior remains byte-for-byte unchanged. Rollover carries `kind: 'rollover'` and never migrates drafts or queued prompts.
- A delayed background-review result remains predecessor-owned; after the runtime swap its old callback is suppressed so it cannot appear in the successor transcript.

## Slice 1 — Atomic durable successor (TDD)

### Tests first

Create `tests/test_hermes_state_session_rollover.py` covering:

1. Parent live + exact latest assistant fence → one child, parent `end_reason='rollover'`, handoff rows committed.
2. Missing/mismatched latest assistant → no mutation.
3. Lost lease, ended non-rollover parent, missing parent, or existing rollover child → no mutation.
4. Replayed commit returns the existing successor identity without creating a sibling.
5. Any injected child/message write failure rolls back parent end and child rows.
6. Child copies only allowed metadata: source/profile/cwd/git/model/model_config/system prompt/title lineage.
7. Handoff obeys 20-line and 16-KiB ceilings and forms valid user→assistant alternation.
8. Rollover children are durable continuation tips: hidden from root pickers, not ephemeral cascade targets, resolved by resume, and included in display lineage while only child messages feed the model.

### Implementation

- Add rollover child predicates/classification beside existing compression predicates in `hermes_state.py`.
- Extend continuation traversal and ephemeral-child exclusion to include rollover children without changing branch/delegate/compression semantics.
- Add a narrow latest-active-message identity read.
- Add a lease-fenced `complete_session_rollover(...)` transaction using `_insert_message_rows` inside the same write transaction.
- Do not add a database column; store `_rollover_from` in child `model_config`.

### Gate

`scripts/run_tests.sh tests/test_hermes_state_session_rollover.py tests/test_hermes_state_compression_locks.py tests/test_hermes_state.py -q`

## Slice 2 — Quiescence coordinator and commit RPC (TDD)

### Tests first

Create `tests/tui_gateway/test_session_rollover.py` covering:

1. Default-off, non-desktop, and non-local-capable sessions never offer/commit.
2. Offer occurs only after clean completion, successful history adoption, final persistence, queue/goal/notification drains, and a new in-place compression boundary.
3. Each blocker independently suppresses offer/commit: running/inflight/reservation/steer/backend queues/tool/approval/input wait/process/delegation/goal/notification/history change/agent mismatch.
4. Import/query errors are blockers.
5. Prompt-submit vs commit race has one winner; a winning prompt leaves the parent live.
6. Fresh-agent build failure and DB failure leave the parent/runtime unchanged.
7. Successful commit swaps stored key/history/agent, transfers active-session lease and approval/slash registration, and emits one scoped successor event only after success.
8. No goal/notification/approval/queue migration occurs.
9. Replay returns the same successor and does not emit a second navigation edge.
10. Old-agent delayed review callback cannot emit into the successor.

### Implementation

- Add `desktop.session_rollover` default in `hermes_cli/config.py`.
- Put pure handoff/blocker helpers in `tui_gateway/session_rollover.py`; keep `server.py` orchestration thin.
- Record offer token `{runtime_id, predecessor_id, turn_generation, history_version, compression_count}` in the live session.
- Add `session.rollover.commit` to slow RPC dispatch.
- Build the fresh agent before DB mutation using the current effective model/provider/reasoning/service-tier/profile/cwd.
- Revalidate under `history_lock`, execute the DB transaction, then atomically replace live agent/history/key.
- Reuse active-slot and approval/slash reanchoring primitives but explicitly skip deferred-notification migration.
- Define post-commit swap recovery; if runtime reanchor cannot complete, fail closed before emitting success and restore a usable predecessor or successor deterministically.

### Gate

`scripts/run_tests.sh tests/tui_gateway/test_session_rollover.py tests/tui_gateway/test_turn_origin_notifications.py tests/tui_gateway/test_finalize_session_persist.py -q`

## Slice 3 — Desktop local acceptance and foreground follow (TDD)

### Tests first

Add/extend:

- `apps/desktop/src/app/session/hooks/use-message-stream/session-successor-event.test.tsx`
- `apps/desktop/src/app/session/hooks/use-session-actions.test.tsx`
- `apps/desktop/src/app/session/hooks/use-session-state-cache.test.tsx`
- config hook tests where the existing hook lives

Cover:

1. Only resolved `connection.mode === 'local'`, explicit opt-in, exact foreground runtime/predecessor, and empty lineage queue send commit.
2. Remote/malformed/stale/background/changed-route/queue-nonempty offers are ignored.
3. Commit edge is consumed once.
4. Success updates the runtime’s stored ID and follows with `replace:true` only under the existing foreground predicate.
5. A→B→C stale events never steal focus.
6. `kind:'rollover'` never calls draft/queue migration; `kind:'compression'` retains existing behavior.
7. Background successor events update only their own cache.

### Implementation

- Read opt-in through the existing config hook.
- Add offer/commit payload types in Desktop types only; shared gateway union already accepts named events.
- Extend stored-ID rotation provenance with `kind`.
- Handle `session.rollover.offer`/`session.successor` in the explicitly routed gateway event handler.
- Reuse the current foreground rotation effect and split migration behavior by `kind`.

### Gate

From `apps/desktop` with Hermes Node 22 in `PATH`:

`npx vitest run <focused rollover/config/session test files>`

Then `npm run typecheck` and `npm run lint` for touched Desktop code.

## Slice 4 — Combined verification and independent review

1. Run all focused Python and Desktop tests together.
2. Run unchanged compression/rotation/finalizer/queue regressions.
3. Run `git diff --check`, targeted Python compile/import checks, Desktop typecheck/lint.
4. Run an isolated temp-`HERMES_HOME` E2E contract test:
   - create predecessor;
   - persist final response;
   - commit rollover;
   - restart SessionDB/gateway fixture;
   - resume root and prove model history is successor-only while display lineage retains predecessor.
5. Independent P0–P2 review. Repair until ACCEPT.
6. Commit the accepted MVP, generate a tracked format patch, append it to the sealed local patch-stack manifest, and validate a clean replay in a disposable worktree.
7. Prepare promotion candidate but stop before live runtime/Desktop/gateway restart approval.

## Abort conditions

Pause and redesign if implementation requires changing the core conversation loop, compression implementation, remote/gateway surfaces, adding a DB schema column, migrating queued work, or relying on a renderer assertion without backend revalidation.
