# Owner work and storage admission

The single-host HTTP service admits each owner's uploads, retries and correction commands against a durable SQLite ledger. This implements the owner-admission part of master requirement 21. It does not establish public deployment capacity, a hard filesystem quota, an expiry/deletion policy, model accuracy, or long-match qualification.

## Limits and ownership

| Setting | Default | Meaning |
| --- | --- | --- |
| `TENNIS_MAX_OWNER_PENDING_JOBS` | `32` | Incoming uploads, queued/processing analyses and queued/processing corrections together |
| `TENNIS_MAX_OWNER_STORAGE_BYTES` | `21474836480` | 20 GiB of retained logical file bytes plus reserved incoming video bytes |
| `TENNIS_MAX_UPLOAD_BYTES` | `1073741824` | Existing one-file upload bound |
| `TENNIS_MAX_DURATION_SECONDS` | `7200` | Existing worker video-duration bound |

All four limits require positive integers. The server propagates owner limits to its separate worker. These are configurable engineering limits, not measured service capacity. The upload/duration bounds do not prove that a full match passes inference or qualification.

`GET /api/usage` returns only the current owner's counters and limits. Account-mode guests cannot read it; responses use the existing private/no-store policy. Health reports limits without private usage. Explicit local recovery mode retains the existing local-owner rules.

An incoming resumable session consumes one pending slot and reserves its declared video size before bytes arrive. A batch checks every file's combined size and job count in the same SQLite writer transaction as all session inserts: either all sessions are admitted or none are. Legacy streaming requests reserve their declared size, or the entire one-file limit when no Content-Length is supplied. Their body cannot exceed that header or the one-file limit. A retry reserves a new slot and the retained original's size before linking/copying it.

The queue, upload session and legacy/retry reservation share one analysis identity for accounting. A crash after queue insertion and before the completion acknowledgement does not create a second pending job or reserve its input twice. Existing resumable completion and correction command IDs remain idempotent. Processing cancellation frees its pending slot only after processing actually reaches a terminal status. Changing a pending-work limit does not stop already accepted work solely because that limit is now lower; its storage budget is still checked.

Failed-analysis retries additionally use a required owner/source-scoped `Idempotency-Key`; only a new explicit intent reserves a new child. A repeat after a pre-queue interruption reuses its saved ID, while reactivating a previously released reservation checks capacity atomically. Accepted repeats precede new-admission quota checks and never consume a second slot. See [the later retry delivery contract and evidence](RETRY_IDEMPOTENCY.md).

## Stored bytes and safe boundaries

Storage accounting streams owned paths without following symlinks, junctions or other reparse points. Invalid identifiers, redirected paths or unreadable storage fail closed with a sanitized 503. Hard links are charged once per retained path, making a retry's logical charge independent of physical block sharing.

The owner charge includes originals and interrupted uploads; resumable partial/chunk files; all analysis outputs, source and browser videos; all `.result` generations and failed publication copies; native inference journals and scratch; video-credit sidecars; and every correction attempt, including failed/cancelled attempts. Released reservations keep their ownership mapping so failed cleanup cannot create uncharged orphan media. Shared database pages, server logs, zero-byte coordination locks and shared runtime/model caches are outside these counters.

Resumable transfer scratch needs additional headroom beyond the reserved video payload. One OS-backed receive lock permits only one live incoming chunk per admitted session, at most 8 MiB, before the append lock serializes the durable offset update. Existing crash chunks remain charged. A storage check occurs before receiving a chunk and again before appending it; rejection preserves the last durable offset and cleans that request's temporary file. This is conservative admission: a file whose payload alone fills the owner's budget may need more headroom to complete its resumable transfer.

Correction source parsing occurs below the owned analysis directory. A per-owner OS-backed lock limits source expansion to one parser across review reads and command validation. Storage is checked before and after expansion; a new command also checks pending capacity before parsing and rechecks atomically before queue insertion. Retrying an existing command returns its retained acknowledgement before new capacity checks. Review reads use the storage check alone, so a full pending-work queue does not by itself deny review.

The analysis worker checks storage before model-asset admission, at native pipeline stage messages, before finalization and immediately before committing a complete native generation. The correction worker checks between rebuild stages and after generation copies immediately before publication. A correction's final check uses a WAL reader while its existing owner/cancellation writer guard is held, avoiding a second competing writer. Refusal retains the original input and previous committed generation. Detailed diagnostics remain private.

These checks are **not a per-write filesystem cap**. One normalization, inference, parser, codec, export or copy stage can grow storage before its next check; concurrent admitted transfers can also grow between snapshots. Failed work and publication scratch remain retained and charged. A filesystem or volume quota would be needed for a hard continuous disk bound.

## Failure and recovery limits

Legacy/retry cleanup reconciles the actual queue before removing its exact request-owned paths. Lost queue acknowledgements retain accepted originals. OS, SQLite and sanitized quota exceptions during reservation release are logged without replacing the accepted response or the original upload error; a committed job still deduplicates that reservation. If queue state cannot be read, cleanup leaves the bytes and reservation intact rather than assuming the job was unaccepted. Failed chunk-temp cleanup also preserves its durable offset acknowledgement or original request error, releases the receive lock and keeps the residual file charged.

### Owner-requested reservation recovery

`POST /api/usage/reconcile` recovers abandoned legacy/retry admission reservations for the authenticated owner. It accepts no owner or analysis selector; query/body values cannot select another account. It is available even when the owner is over quota. The private/no-store response contains `reconciled_reservations`, `busy_reservations`, `unverified_reservations`, and a fresh `usage` object with the same fields as `GET /api/usage`. Repeating the call is safe. It never queues work, revives a rejected upload intent, cancels accepted work, removes history, or deletes files.

New legacy requests acquire a per-analysis OS-backed lifetime lock **before** committing their reservation and hold it through body streaming, queue insertion and settlement. An additive SQLite `recovery_version` column marks participation as version 1. Existing legacy rows migrate to version 0: another still-running old HTTP process might own them without a lock, so recovery leaves them charged and reports them as unverified. Do not change that marker manually while a prior server may still run. Recovery of those historical rows remains an operator task requiring proof that all old request processes have stopped.

Retries already have a durable owner/key identity and lifetime lock. Recovery validates the intent's owner, source, child and size, then tries that same lock without waiting. The intent and selected child identity remain unchanged; the normal retry endpoint can reactivate its released reservation under the same key, subject to current admission checks. A missing or inconsistent mapping stays conservatively charged.

Recovery takes a request lock before opening each short `BEGIN IMMEDIATE` transaction and rechecks the current reservation and relationships inside it. Live requests are reported as busy. Any related resumable upload session or ambiguous accepted ownership stays unverified. A proven abandoned reservation becomes inactive; an accepted analysis continues consuming its own pending slot according to its status. All inactive ownership rows continue charging retained originals, partials, outputs and scratch. Only unwritten incoming bytes and abandoned pending capacity are released. No reservation age or HTTP heartbeat is used as proof of abandonment. Coordination lockfiles are retained to avoid splitting locks across different inodes.

Rows commit independently to bound memory and transaction lifetime. A later filesystem/database/snapshot error returns a sanitized 503; earlier safe releases may already have committed, and retrying completes the remaining recovery. Unreadable storage still prevents further admission and is never treated as empty. The response reports a point-in-time usage snapshot; concurrent requests can change it afterward.

No automatic deletion or expiry is introduced. Interrupted chunks/parser expansion and abandoned reservations can retain owned files indefinitely. Explicit owner deletion, retention scheduling, account recovery, global admission/rate controls, backups/restores and actual public capacity remain separate unfinished master requirements. Do not delete a reservation's ownership row while attributed files remain.

## Verification scope

The coupled guarded regression passed **227 tests**, zero failures/errors/skips, under `artifacts/validation/yolo26_execution_2026-10-22/owner_quota_checks_v2/`. It covers HTTP owner isolation, atomic concurrent admission, reservation and batch limits, lost queue acknowledgements, correction source attribution and final publication refusal, retained histories, and the existing separate-process native restart fixtures. Source snapshots stayed unchanged. Eleven resource snapshots were fixed test values; no actual RAM/GPU admission probe or model inference ran.

Two later production cleanup branches in `app.py` and `uploads.py` were then hardened. The current focused regression passes **110 tests**, zero failures/errors/skips, in **13.78 runner seconds**, under `owner_quota_cleanup_checks_v3/`. It binds unchanged current source/test snapshots and records zero inference imports/attempts, zero actual host/GPU probes and zero synthetic resource calls. These scopes overlap and are not summed. The earlier 225-pass run and the two failed cleanup-fixture attempts remain retained. The latter failures exposed test transport assumptions: synchronous TestClient coalesced body chunks, and the first asynchronous replacement correctly hit local-mode hostname protection. The final fixture uses an actual two-message ASGI body through loopback HTTP, writes the first message before overflow and verifies retained bytes, durable offset/error and lock release. No production behavior was weakened to pass the fixture.

The independent static review identified the unattributed correction-parser scratch and cleanup exception gaps before closure. Its source-bound receipt is retained under `owner_quota_review_v1/`; the final API test transport has a later pin than that receipt, while its seven reviewed production files match the final sources. Reviewer follow-up inspected the loopback/two-message delta; no independent test execution is claimed. The aggregate closure records exact evidence and revision differences.
