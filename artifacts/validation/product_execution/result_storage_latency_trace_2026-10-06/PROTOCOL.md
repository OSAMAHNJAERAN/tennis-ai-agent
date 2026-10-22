# Prepared 600-frame result kernel diagnostic

Do not launch until root has reviewed the frozen helper/protocol and finished
other workloads. This is one new mechanism diagnostic, with no optimization or
whole benchmark repeat. It invokes production build, publication, reader and
JSON/CSV/HTML stream functions on the retained synthetic fixture generator.
Frozen results/reporting run on an untouched pre-publication raw view; current
code receives a byte-identical copied view. All five complete outputs must have
exact bytes. PTS and attribution remain explicitly synthetic; no video exists.

Instrumentation replaces four methods in this isolated interpreter only:
`ResultStorage._one`, `_execute`, `_encode`, and `_decode`. The SQL method
wrappers retain only fixed template categories and count/time aggregates.
They never record bound parameters, expanded SQL, arbitrary template sets or
evidence. Codec depth guards count nested calls but time only the outer call,
so recursive spans are not summed twice. Originals, arguments, return values
and exceptions are forwarded and restored; no production file is changed.

Ten stages separately record elapsed, SQL category counts/spans, encode/decode
counts/spans and unassigned elapsed. `_one` includes execute/fetch/cursor close;
`_execute` includes execute/cursor close. The remaining elapsed is not a waste
estimate: it includes grammar parsing, stdlib private JSON loads/dumps outside
codec calls, strict public serialization, validation, file writes, transaction
commits outside these methods, chart/CSV logic, and instrumentation overhead.
Per-call clocks and per-node depth wrappers change timing. Counts establish
query amplification; instrumented wall time is not benchmark throughput.

This exact fixture predicts CSV 9N+5 length reads and 5N point payload reads,
or 8,405 SELECTs at 600 frames. JSON has 12 dense array traversals, giving at
least 24N point/length SELECTs; HTML gives at least 16N before short intervals.
These are falsifiable source predictions; counter mismatch is saved as a
failed hypothesis. Larger throughput claims require a useful implementation
change and a separately frozen protocol, not repeated timing of this trace.

The parent uses the verified owned-process monitor, actual-interpreter atomic
sampling handshake, normal-exit retirement checks and deepest-first owned
cleanup. A child exceeding 60 seconds, 1.25 GB sampled owned RSS, or available
RAM below 400 MiB for five 0.25-second readings is stopped. Preflight requires
2.5 GB available RAM and 4 GB free disk. The one fresh run is never overwritten,
and the first failure is retained without retry. No live service or worker is
invoked; production saved results and original media are untouched.

After root review only, from the repository directory:

```bat
C:\Users\ac-98\AppData\Local\Programs\Python\Python313\python.exe artifacts/validation/product_execution/result_storage_latency_trace_2026-10-06/helper.py
```

Use `cmd.exe` with `login:false`. Preparation does not authorize execution.
The launcher creates `run` exclusively. Terminal resource/source records are
`run/report.json`; counters, output checks and original input hashes are in
`run/600_current/child_receipt.json`. Log/sample/partial files remain retained.
