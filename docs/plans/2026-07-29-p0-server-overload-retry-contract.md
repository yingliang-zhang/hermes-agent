# P0 `server_is_overloaded` Retry Contract

## Goal

Align Hermes runtime behavior with the documented `agent.api_max_retries` contract and make Sudo/OpenAI-compatible `server_is_overloaded` failures recover with provider-overload backoff instead of the generic short retry policy.

## Baseline and authority

- Runtime source: `/Users/yingliangzhang/.hermes/hermes-agent`
- Baseline branch/commit: `local/hermes-patched@1606b492b`
- Durable authority after acceptance: append-only `250-server-overload-retry-contract.patch`
- Hybrid run ID: `b2e61c80-f75e-4835-95af-117a3e073ea6`
- Active wrapper workflow: `$HERMES_CODING_WORKFLOW` (`hybrid-v1` at dispatch)

## Frozen behavioral contract

1. A structured error code exactly equal to `server_is_overloaded` classifies as `FailoverReason.overloaded` even when status and human-readable overload text are absent.
2. The classified error remains retryable and must not rotate credentials, compress context, or independently force provider fallback.
3. `server_is_overloaded` uses a dedicated bounded exponential overload policy: 10s, 20s, 40s, then 60s cap, with 20% positive jitter in production. A valid provider `Retry-After` value takes precedence for the generic policy.
4. Existing Z.AI Coding overload behavior remains unchanged and takes precedence over both the generic server-overload policy and generic `Retry-After`: its short tier uses the caller's effective default wait, while its long tier remains 30s, 60s, 90s, 120s.
5. Non-overload API errors retain the existing generic backoff path.
6. `agent.api_max_retries` means additional retries after the initial call. Default `3` therefore means exactly four total attempts per provider; `0` means exactly one attempt. Client/transport reconstruction must use a remaining already-budgeted attempt and must not reopen the ceiling. Negative values clamp to `0`; invalid values fall back to `3`.
7. Retry status, terminal status, logs, result text, and request-error hook metadata must not call total attempts “retries.” Terminal output should report both retry and attempt counts when useful.
8. Existing fallback selection and credential-pool policy are out of scope and must not be broadened.

## TDD acceptance matrix

| Contract | RED evidence required | GREEN assertion |
|---|---|---|
| Exact structured classification | Existing classifier returns `unknown` when only the code is present | reason=`overloaded`; retryable; no rotation/compression |
| Dedicated overload backoff | Helper/policy is absent or returns generic default | no-jitter sequence 10/20/40/60; policy label identifies server overload |
| Retry-After precedence | Overload currently bypasses Retry-After parsing | supplied Retry-After wins |
| Retry count semantics | Default 3 currently performs only three total calls | three failures followed by success succeeds on call 4; four failures terminate after call 4 |
| Transport reconstruction ceiling | Primary client rebuild resets the counter and reopens a full cycle | budgets 0/3 issue exactly 1/4 requests with monotonic hook attempt counts |
| Zero-retry boundary | Existing clamp converts 0 to 1 and loop semantics are ambiguous | 0 retries performs one call |
| Regression boundaries | Existing Z.AI and generic retry tests | unchanged Z.AI schedule; generic backoff unchanged |

## Verification

Iteration uses only focused files through `scripts/run_tests.sh` (whole file paths, not `path::node` selectors). Sol then reviews the real diff and adds adversarial tests hidden from the GLM implementation session. A fresh independent Sol audit is mandatory because this changes recovery behavior. The full suite runs once at the final promotion gate, not during each repair round.

## Activation boundary

Implementation, focused verification, patch generation, and build-only verification are authorized. Promotion may update the runtime branch through the canonical bootstrap only after an explicit Sol `ACCEPT`. Gateway/Desktop restart remains a separate operator-approval boundary.

## Accepted source artifact

- Candidate: `1660e5528fe6e135035536322e096dd807123034`
- Shadow authority: `d8df618ca9164c63dbfc308052e01ca4360cf10e`
- Candidate/shadow tree: `70473bd4af9529429341f1b9fc81e6f5196098d5`
- Durable patch: `250-server-overload-retry-contract.patch`, SHA-256 `a84bbaf25bfc2032382fd849ebf8f18483632d7b39eaf28b3b718fa738f85dd5`
- Focused gate: 8 files, 732 passed, 0 failed, 110.1s
- Independent Sol verdict: `ACCEPT`, no P0/P1/P2; the two documentation-only P3 findings were repaired and their affected tests rerun.
- Portable gate: 40/40 exporter tests; deterministic archive 2,058 files, 10,159,738 bytes, SHA-256 `59ec7dcb81d4f25337dfe7e3d6488555162f7a8d22e83c31fc9d590b0bd1b186`.

The raw candidate-only full-suite invocation is not a valid promotion verdict by itself: it reported 157 failures in 40 unrelated files and 11 no-run files. Candidate-only regression classification is delegated to the canonical baseline-versus-candidate promotion verifier.

## Activation approval

On 2026-07-30 the operator explicitly approved completion and any required restart. Activation must still use the canonical `hermes_promotion_bootstrap.sh update` route, followed by runtime, Desktop, gateway, and live-smoke verification.
