# Prepared result storage measurement — do not launch during integration work

This one-shot protocol measures generated 600-frame and 18,000-frame artifacts.
It runs the frozen result normalizer/reporting code and the current production
storage, publication, reader and export functions in four separate clean Python
children, in that order. It runs no video pipeline, model, FFmpeg or decoder.

The baseline publication and read steps replay the exact preserved worker/API
statements rather than importing or executing a worker. Frozen result/report
hashes and the verified owned-process helper hash are fixed in `helper.py`.
Their shared dependencies and current code are captured in the reviewed
`prepared_source_bindings.json`, required to match at launch and checked
before/after every child. Any later edit refuses launch. Root must freeze source editing and finish integration
work before reviewing and launching this harness.

All growing input arrays are written from iterator factories. The fixtures
include calibrated valid/invalid intervals, camera cuts, source transition gaps,
unknown/empty/unavailable people, missing motion samples and ball observations.
Physical ball speed and xyz stay null, authoritative events stay false, and
measurement scopes stay withheld around gaps. A 65-frame pose prefix followed
by missing frames limits this measurement's scope. A generated source clock
uses index/60 with explicitly synthetic PTS strings; there is no footage or
independent camera clock. Generated attribution fields test escaping only.

The current child uses `build_result(..., _storage=storage)`, `publish_result`,
`ResultDocument`, and the same JSON/CSV/HTML chunk iterators as HTTP responses.
The frozen/current saved JSON, reader JSON, summary JSON, CSV and HTML must be
byte-for-byte equal, with no volatile exclusions. Inputs must have equal hashes.
Independent per-frame assertions cover time, camera/timing/position/speed gaps,
null physical quantities, roles and unknown people. Binary comparisons use
64 KiB reads. Verification costs are timed separately from production stages.

Each child has a 180-second wall guard, a 1.25 GB aggregate owned RSS guard,
and a host available-RAM guard below 400 MiB for five consecutive 0.25-second
samples. Before each child, at least 2.5 GB available RAM and 4 GB free disk are
required. The child waits for an exact sampled PID/createTime acknowledgment
before importing result code. The previously verified process helper monitors
the wrapper and all remembered descendants and checks every identity has
retired before another child can start. It stops only those owned processes.
No retries occur; any failure halts later cases and retains diagnostic files.

The receipts record child Windows peak working set and peak process commit,
all per-process samples, aggregate peaks, stage wall times and logical file
sizes. Reader-open time includes the production database's full hash read.
Each export records first-chunk availability, all-chunks close time and the
verification hash's full-file read time. Peaks include imports, generated inputs and independent checks. Host
background activity and shared pages limit interpretation. No separately reset
stage peak, physical disk blocks, full-match capacity or accuracy is claimed.

After root review, launch the parent from base Python313, with `cmd.exe` and
`login:false`, from the repository directory:

```bat
"C:\Users\ac-98\AppData\Local\Programs\Python\Python313\python.exe" "artifacts\validation\product_execution\result_storage_measurement_2026-10-06\helper.py"
```

The launcher exclusively creates `run`. Existing `run` means refuse reuse.
Final status and bindings are in `run/report.json`; each fresh case retains
`child_receipt.json`, `resource_receipt.json`, `samples.jsonl`, `stdout.txt`,
original generated inputs and exported files. Forced stops may have no child
receipt; the parent's failed resource receipt and partial files remain evidence.
