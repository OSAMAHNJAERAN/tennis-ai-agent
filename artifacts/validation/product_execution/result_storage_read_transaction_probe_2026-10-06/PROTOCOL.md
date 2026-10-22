# One-shot read transaction probe

The authorized probe reads the retained synthetic 600-frame published database
from the completed latency trace. It creates no input or source-database writes
and runs no builder, models, footage, FFmpeg, network or live product action.

One clean child opens the production `ResultDocument` twice. The first uses its
ordinary behavior. The second issues only `BEGIN` on the readonly connection
after normal reader entry and before identical CSV/HTML exports. Export query
counts must stay equal, and every byte must match retained current exports.
The connection's transaction state is checked before/after export; close must
release its handle and closed reads must still fail. Source hashes/file names
must remain unchanged. Fixed SQL/span counters come from the sealed trace helper.

The fresh child has a 30-second wall guard, 256 MiB sampled aggregate owned-RSS
guard, 1 GB preflight available RAM, and a below-400-MiB available RAM guard for
five 0.25-second samples. Verified exact-process monitoring/handshake and owned
retirement checks are reused. Output reuse is refused, failures retained, and
no retry is authorized. Original trace, benchmark and production files stay
unchanged. Instrumentation and mode order limit timing interpretation.

The first static preparation correctly refused a changed `results.py` and is
retained in `static_binding_failure.json`. Root explicitly approved rebinding
that one file to the newly reviewed builder implementation; no builder runs in
this probe, both reader contexts import that same source, and its hash helpers
are unchanged. `binding_exception.json` pins both hashes. No other mismatch is
accepted. Root freezes further production edits until this probe is terminal.

After static binding/AST review, run once from base Python313:

```bat
C:\Users\ac-98\AppData\Local\Programs\Python\Python313\python.exe artifacts/validation/product_execution/result_storage_read_transaction_probe_2026-10-06/helper.py
```

Run receipts/counters are in `run/600_current/child_receipt.json` and the parent
resource/source/retirement receipt is `run/report.json`. This synthetic probe
does not test external mutation of generic v1 databases. Production snapshot
semantics and lock-release/failure tests remain required before any real change.
