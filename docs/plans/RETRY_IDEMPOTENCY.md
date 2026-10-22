# Durable failed-analysis retry requests

Master requirement 21 requires retries not to create duplicate jobs after an uncertain response. Failed-analysis retries now use a persistent request identity shared by the HTTP API and website. This is separate from native inference continuation, which resumes an already accepted job after a worker crash.

## HTTP contract

`POST /api/analyses/{source_analysis_id}/retry` requires exactly one `Idempotency-Key` header containing 16–128 ASCII letters, digits, underscores or hyphens. The owner is taken from the authenticated session. Another owner's source remains a 404 even if the key is absent or malformed; account-mode guests remain unauthorized.

A key is scoped to one owner and one source analysis. Reusing it for a different source returns 409. Another owner can independently use the same textual key without seeing or reusing the first owner's job. An OS-backed lock serializes requests with the same owner/key across HTTP processes. A request that cannot acquire it in 15 seconds returns a retryable 409 and must keep the same key.

The server atomically inserts its durable key-to-child mapping and reserves that child's quota before copying input. After acceptance, repeating that exact key returns the same child, even after a server restart, missing original source, a lower storage limit, or failure of the child itself. A confirmed failed child is a separate source for a fresh explicit retry. Distinct keys represent distinct explicit requests and can create distinct children subject to quota; this is not a global content-deduplication policy.

On a failure before queue acceptance, request cleanup releases capacity while retaining the key-to-child mapping and ownership. A later repeat reuses that child ID and atomically checks capacity when reactivating its reservation. If the process died before cleanup, the active reservation is retained and consumes one slot. Only that unaccepted child's exact temporary/original paths are replaced under its request lock; a complete input is renamed into place before queue insertion. Once a job exists, repeat requests do not copy or delete its input.

The retry's match metadata and notes are copied from its owner's source in the same SQLite transaction as the queue insertion. A lost acknowledgement cannot leave an accepted job without its initial metadata, and repeated acknowledgements do not overwrite edits made to the child after acceptance.

Retry admission binds the original analysis identity and expected byte length. It does not independently hash the source before/after copying, and a hard-link retry shares its inode with the retained original. No HTTP route rewrites an accepted original, but out-of-band same-length replacement or in-place mutation is a storage-integrity limitation. The native processing pipeline's separate input/checkpoint hashes remain required; retry admission is not a substitute for them.

## Website intent

The website saves a random retry key in account/source-scoped browser storage before submitting it. Lost network responses, HTTP failures and invalid child acknowledgements retain the key. Only a validated new child with matching `retry_of`, and a still-current owner session, acknowledges that intent. In-flight calls share one promise, and the history action is disabled while pending with explicit waiting/error feedback.

Private history/detail/batch reads are bracketed by session checks. A changed owner discards those delayed reads and invalidates pending actions. An observed 401 also clears the owner generation, jobs and batch, so a delayed cancel acknowledgement cannot restore the previous account's job. An authenticated server whose owner cannot be confirmed withholds private history. These are guards for observed account changes, not a guarantee that browser cookies cannot change between observations. Persisted tokens and acknowledged child IDs use exact character checks; a trailing newline cannot evade validation through a regular-expression end-anchor exception.

Blocked browser storage falls back to memory for that runtime. This cannot promise persistence across refresh while storage is disabled. Local storage is not a cross-tab mutex: two tabs starting distinct brand-new intents can obtain different keys. No global same-content deduplication is claimed.

## Evidence and remaining scope

The guarded backend scope passes **171 tests** with zero failures/errors/skips, stable source snapshots and no model imports or actual RAM/GPU probes: `artifacts/validation/yolo26_execution_2026-10-22/retry_intent_checks_v1/receipt.json`. Two separate child processes execute the real HTTP retry flow and exit with `os._exit` immediately before queue insertion (85) or after its durable commit (86). Each leaves a recorded original hash and selected child identity. A new service instance then acquires the released OS lock and repeated requests produce exactly that child and one queued row. This tests process loss; Windows power-loss durability is not established.

The current website passes full typecheck, lint and production build, plus **58 focused checks** across eight files, zero failures/errors/skips and **143 unchanged source/runner pins**: `retry_frontend_checks_v2/receipt.json` in the same validation directory. Tests cover persisted uncertain keys, invalid acknowledgements, waiting/double-click feedback, failed-child retries, delayed initial history across an owner change, observed 401 during cancellation and unavailable owner identity. The earlier frontend run's type/lint/build passed, but all seven API cases failed in cleanup because the Node environment's global `localStorage` lacked `clear`; that run and its source snapshot remain retained. The repaired fixture explicitly provides the browser storage interface, and further targeted owner/string-validation regressions were added. Actual browser visual inspection is separate from DOM/software testing and remains unavailable in the current Work browser settings.

This change closes failed-analysis retry requests. Resumable completion and correction command IDs retain their separate existing idempotence. Initial upload-session/batch creation after a lost creation acknowledgement, owner orphan-reservation recovery, controlled retention/deletion, public capacity, model training/qualification, long matches and live analysis remain separate master work. No model runtime or training is started by this change.
