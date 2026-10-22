# Result-storage latency runtime compatibility preparation

This separate verifier is prepared for root review and dispatch. Preparation
does not run the helper, inspect live processes or queue rows, hash the saved
analyses, contact a server, stop/start a process, or run models/inference. Earlier
runtime preflight/verification, benchmark and worker-smoke artifacts remain
unchanged. The original final-fixed verifier is bound to SHA256
`52ced1809e03381911aa397cfb7b85e15810496ad02b23aa6380a81054f8605a`.

The new helper uses the reviewed CPython 3.13 base interpreter. Root first runs
its `--preflight`, then independently uses the existing guarded idle-stop helper
for the exact recorded server/worker identities when present, then starts the
documented `scripts/run/serve_product.py` command. This verifier neither stops
nor starts a process. Root dispatches postverification only once the newly
started server and worker are available. Existing receipts are rejected through
exclusive creation; no automatic retry is implemented.

Preflight accepts exactly one owned server and one owned worker, or an entirely
absent runtime with no port-8000 listener. For an existing pair, the executable,
command arguments and workspace cwd must match, the worker must be the direct
server child, port 8000 must be owned by the loopback server, and the complete
server descendant set may contain only that worker and its direct `conhost.exe`
children. Duplicates, orphans, recognizable commands with wrong arguments/cwd/
interpreter, unknown listeners and other descendants are rejected. Identities
include PID and creation time, and are rechecked during collection.

The observed tool launch additionally has an enclosing `cmd /c` shell with a
direct Windows `conhost.exe` child. The corrected verifier distinguishes that
exact command string, workspace and verified base-Python server child from the
server/worker pair. This shell never owns the server listener or counts as a
server. Both preceding refused verifier sources and their diagnoses are
preserved in `result_storage_latency_runtime_verifier_repairs_2026-10-06.json`.

The original snapshot function is preserved: all columns of the ten ordered
saved analysis rows, plus the complete analysis-tree file-path/hash map including
hidden result generations and sibling derivative-credit files. `UPLOADED`
(queued accepted work) and `PROCESSING` rows are rejected. Two complete equal
snapshots and equal runtime/source observations are required before saving the
new preflight. Code, the original/new verifier, idle-stop/startup scripts, demo
manifest, complete frontend build and base interpreter are hash-bound. The
preflight output is
`result_storage_latency_runtime_preflight_2026-10-06.json`.

Postverification requires the previous exact server/worker/console identities
to have retired when preflight recorded them, and a newly owned pair to remain
unchanged throughout verification. It requires health 200 and the worker lock,
three curated demos and ten saved library rows. The original complete-byte
comparison remains: each demo's result JSON, downloadable JSON, CSV and readable
HTML equals the current presentation of its saved JSON and the current CSV/HTML
serializers, giving twelve complete exact exports with no excluded fields.
Catalog attribution must equal the current presented saved source credit, HTML
must contain that credit, and each catalog entry must remain nonqualification
evidence.

The helper verifies the existing `no-cache` declarations on downloadable JSON,
source/annotated media and the SPA index. The media checks request only byte
0 through a range GET and require 206 plus the exact single saved byte. It
records CSV/HTML/result cache headers without requiring a header those routes
do not currently declare. The served frontend index and every `/assets/` script,
preload or stylesheet referenced by that index must match the saved build bytes;
the complete build inventory is unchanged across the run.

Before and after full snapshots are retained in the distinct successful receipt
`result_storage_latency_runtime_verification_2026-10-06.json`, alongside original/
new helper bindings, process commands/identities/listeners/descendants, retired
prior identities, source bindings and exact response hashes. Failure preserves
the preflight and all original saved state and uses root's command log as the
failure evidence. No compatibility result qualifies scientific accuracy,
physical speed/3D or full-match capacity.
