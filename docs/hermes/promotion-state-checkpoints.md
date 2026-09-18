# Hermes Promotion State Checkpoints

Memory-aide, not authority. Authority: `local/hermes-patch-stack` + promotion ledger + tracked PRRs.

## 2026-08-03 — base ee276a7982 integrity check

| Item | Value | Verified |
|---|---|---|
| frozen base | `ee276a7982` (upstream/main merge-base) | exists in runtime repo ✓ |
| runtime HEAD | `3c61b74dc5` on `local/hermes-patched` (base + 9 commits) | `git merge-base HEAD upstream/main` == frozen base ✓ |
| "remaining patch SHAs e20fc4b0ec..f799e04288" | **do not exist in any local repo** — stale artifact of compressed summary; actual patch pool is only 3 files (000/270/280) | `git cat-file -t` + reverse-apply ✓ |
| patch 270 false-stop (conversation_loop.py +49) | content present in HEAD | reverse-apply clean ✓ |
| patch 280 desktop update ancestry (main.ts +19/-1) | content present in HEAD | reverse-apply clean ✓ |
| 000 merged stack (53 files, +2316/-613) | represented in the 9 local 3-way-merge commits | merge-base arithmetic ✓ |
| py_compile | 9/9 changed files | clean ✓ |
| pytest 113 files (hermes follows model) | **352 passed, 2 xpassed** (35 hermes_cli-marked deselected) | clean ✓ / 3 entries in `eviction_check.md` ✓ |
| worktree `hermes-basecheck-ee276` | temp verification worktree | remove before next promotion ✓ |

**Not yet integrated**: upstream batch `ee276a7982..a4a91610b0` (274 commits) — triage in progress (delegate_task `deleg_97bf068c`, 4 domain subagents).
