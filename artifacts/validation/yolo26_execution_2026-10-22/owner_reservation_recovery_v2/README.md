# Owner reservation recovery verification

The guarded coupled suite passed **228 checks**, zero failures/errors/skips, in **45.10 seconds**. All 430 source/test/runner pins stayed unchanged; all current files and copied snapshots were independently checked against those pins. No inference runtime import/attempt, synthetic resource snapshot, actual host/GPU probe or model inference occurred. These are synthetic software tests, not tennis accuracy or public-capacity evidence.

`receipt.json` contains exact test arguments, counts and before/after pins; `junit.xml` and `pytest.txt` retain the execution output. `closure.json` binds these files, the independent agent review and all five actual child-process observations by SHA256. `child_processes/` retains legacy exits after durable reservation and immediately before/after queue insertion, plus retry exits immediately before/after insertion. The children actually exited through `os._exit`; fresh services then reconciled their OS-released locks. No worker was enabled and no background process is required to survive this evidence.

The 32 focused new cases cover real process death, live body streaming and retry acceptance, repeated request cancellation and cancellation of all private asyncio Tasks, concurrent reconciliation, restart/idempotency, owner separation, retained-byte charging, corrupted retry maps, old-schema migration, safe-path refusal and database failures. The remaining cases exercise existing upload creation/rejection, retry, quota, auth, match, demo and correction admission APIs. Counts overlap the passing 218-check intermediate run under `owner_reservation_recovery_v1/`; they are not added.

During development, uncaptured standalone diagnostics exposed a test's buffered-write assumption and incomplete cleanup immediately after cancelled reservation commit. Both were corrected before the guarded v1 run. The final v2 implementation additionally replaces a separately cancellable operation Task with a directly tracked executor Future; all-task/repeated cancellation and five real process exits verify that lifetime locks cannot be released ahead of in-flight operations. Earlier receipts are retained at their own source revisions.

The production behavior is documented in [the quota contract](../../../../docs/plans/OWNER_RESOURCE_QUOTAS.md): authenticated `POST /api/usage/reconcile` retires abandoned pending/unwritten capacity, preserves retained ownership/files and durable request identities, and leaves uncertain historical legacy reservations charged. It does not delete data, restart jobs or infer liveness from age. Controlled deletion/retention remains the next task.

To reproduce, use a prepared model-free API Python with the recorded non-inference dependencies. Create a fresh sibling at the same directory depth; the runner refuses an existing output directory with receipts. From the repository root:

```bash
TENNIS_RECOVERY_RUN=artifacts/validation/yolo26_execution_2026-10-22/owner_reservation_recovery_reproduction_v1
mkdir "$TENNIS_RECOVERY_RUN" && cp artifacts/validation/yolo26_execution_2026-10-22/owner_reservation_recovery_v2/run_checks.py "$TENNIS_RECOVERY_RUN/" && /workspace/.venv-tennis-api/bin/python -B "$TENNIS_RECOVERY_RUN/run_checks.py"
```

Local ASGI/TestClient and child-process socket operations require this environment's execution permission. The runner supplies guarded synthetic resource seams, blocks model imports in parent/children, copies pinned sources and writes into its own fresh temporary database tree. Dependencies, temporary stores and source copies are intentionally ignored reproducible outputs; the receipts, runners, review and five child observations are tracked.
