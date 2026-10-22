# Prepared disk-result latency comparison — not launched

This separate one-shot compares the sealed **preoptimization disk implementation**
with the current disk implementation at 600 and 18,000 generated frames. The old
resident-list versus disk experiment and its receipts remain historical evidence.
No worker, model, footage, FFmpeg or decoder is run here.

Both paths call their actual `build_result(_storage=storage)`, `publish_result`,
`ResultDocument`, presentation decoration and streamed JSON/CSV/HTML exports.
Old children load four frozen production modules under their actual `src.api.*`
aliases, storage first, before dependent production imports. Class/function identity
checks prevent mixing old and current storage. Common dependencies are final-bound
current code; the frozen 17-file baseline, preservation receipt and previous seal,
real worker smoke and resident/disk benchmark evidence are also bound.

Preparation does **not** create final `prepared_source_bindings.json` and does not
authorize execution. Root must finish edits/tests, review final code and fixtures,
then freeze that file. The launcher refuses missing or mismatched reviewed bindings.
The final candidate changes must be described from those frozen sources; this
protocol does not predeclare an improvement or allow automatic retry.

Inherited fixtures and all-frame semantic checks are unchanged. Synthetic index/60
timestamps and string PTS endpoints are explicitly generated, not decoded/native
camera timing. Null physical ball speed/xyz, disabled event authority, camera cuts,
timing/position gaps and unknown/null/empty people remain checked. Pose is a short
65-frame prefix, so this does not measure dense full-match pose.

Four clean children run sequentially: 600 old/new, then 18,000 old/new. Each has the
same predeclared 180-second wall guard, 1.25 GB aggregate owned-tree RSS guard, and
available RAM guard below 400 MiB for five consecutive 0.25-second samples.
Preflight requires 2.5 GB available RAM and 4 GB free disk before every child.
The child atomically publishes its exact PID/createTime identity and waits at most
10 seconds for sampled-identity acknowledgment before production imports or work.
The verified process monitor remembers exact owned identities and requires normal
or forced retirement before another child; only those owned processes are stopped.

Complete result JSON, reader JSON, summary JSON, CSV including CRLF and HTML must
match byte-for-byte, without runtime exclusions. Input hashes and independent
semantic checks must match. Closed generation manifest/hash, scratch and pending
export cleanup are verified for **both** disk paths. Any guard, source change,
child, retirement or output failure halts all later cases and preserves partials.

Child OS working-set and process-commit peaks include startup, imports, generated
inputs and verification. Sampled tree RSS can double-count shared pages and miss
short peaks; host RAM includes unrelated processes. Stage times, full child time
and parent-monitored wall time remain separate. Export stages include their
explicit full-file verification hash, with that cost separately recorded. Disk
figures are logical file bytes, not physical blocks or write amplification.

One observation per path/count does not establish stable performance, browser
capacity, representative long-video capacity or scientific accuracy. The browser
still holds full arrays. The 432,000 result case is outside this protocol.

After root review and final source freeze, launch from base Python313 with
`cmd.exe`, `login:false`, and the repository as working directory:

```bat
C:\Users\ac-98\AppData\Local\Programs\Python\Python313\python.exe artifacts\validation\product_execution\result_storage_latency_measurement_2026-10-06\helper.py
```

The launcher exclusively creates this unit's `run`. Existing output refuses reuse.
It preserves the exact executed helper, all input/identity/resource/child receipts,
samples, logs and exports. It never replaces old evidence or live product data.
