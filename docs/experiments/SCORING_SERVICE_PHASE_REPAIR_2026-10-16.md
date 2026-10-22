# Scoring service phase repair — 16 October 2026

Serve contact now exposes `SERVE_STARTED` and the appropriate first/second
serve-in-progress state. The public state becomes `LIVE`/`RALLY_LIVE` only
after the existing resolver accepts a verified legal service bounce. This
preserves the native caller's service-box context for that first bounce.

The change is confined to the `SERVE_IN_PLAY` branch in
`src/scoring/scoring_engine.py`. The resolver already uses that outcome for
both contact and a legal service bounce, while retaining their distinct
internal ball phases. The engine previously mapped both to public `LIVE`,
causing the exact Phase6 step 7 context block to request `RALLY` immediately
after contact. The fix copies the actual resolver phase and reports
`FIRST_SERVE_IN_PROGRESS` or `SECOND_SERVE_IN_PROGRESS` while the ball is
`SERVE_STARTED`. It does not parse outcome reason strings or add CV events.

Score, server rotation, serve attempts, faults, lets, point winners, review
pauses and duplicate-ID behavior retain their existing rule branches. After
point completion the public `POINT_COMPLETE` and resolver `DEAD` distinction
also remains unchanged; this is not a blanket synchronization of every state.

## Reproduction and evidence

Evidence is retained under
`artifacts/validation/yolo26_execution_2026-10-16/service_phase_repair_v1/`.

- `red/receipt.json`: actual pre-fix engine/resolver/rules plus exact native
  context-selection source nodes produced **11 failing, 1 passing** focused
  tests. The pre-fix engine bytes are retained as `baseline_scoring_engine.py`,
  SHA256 `46e3345062a7e756f440df0f76152dc93c9dca95fe8c07464493121316165d9d`.
- `minimized_pre_fix_trace.json`: actual contact outcome `SERVE_IN_PLAY`,
  resolver `SERVE_STARTED`, public `LIVE`/`RALLY_LIVE`, and actual native context
  `RALLY`. Changing only the isolated fixture's public ball phase to the
  resolver phase restored context `SERVE`. This distinguishes the incorrect
  engine mapping from premature resolver advancement or a caller that ignores
  `SERVE_STARTED`.
- `green/receipt.json`: **14 focused pytest cases passed in 0.38 s** after the
  fix. The original regression is covered, with additional `IN` compatibility
  and unstarted/review-required controls. Tests execute the real class bodies
  and real native context-selection nodes without transitive package imports.
- `legacy_and_trace/receipt.json`: all **12 existing Phase5 test definitions**
  passed through the same actual class loader in 0.0347 s. Root authorized
  replacing the obsolete immediate-after-contact `LIVE` assertion in
  `tests/test_phase5_scoring.py` with explicit `SERVE_STARTED` and
  `FIRST_SERVE_IN_PROGRESS` assertions. All other assertions were retained.
- `legacy_and_trace/scoped_source.diff` compares this fix with the retained
  pre-fix worktree bytes, excluding unrelated pre-existing Git changes.
  `fixed_scoring_engine.py` retains the validated source, SHA256
  `137d4f28a65e881c4820d31676c754939ac1685d3ca63bb1ccefc0728cee9d4e`.

The covered sequence is first contact, first fault, ignored dead-ball contact,
second contact, legal service bounce, legal return and terminal out bounce.
Actual native bounce contexts were `SERVE`, `SERVE`, `RALLY`. Controls cover
lets on both attempts, replay of the same attempt, double fault, subsequent
reset, duplicate IDs, unknown events, absent/unknown line decisions, premature
returns and consequential review gaps. A pre/post differential checks all
outcomes, score/attempt/winner fields, state versions and history fields except
the deliberately corrected public phase fields.

Both test runs used import tripwires and recorded no Torch, Ultralytics, NumPy,
CV2 or PIL imports. No model, media or GPU was used. Line-call decisions, lets
and event times are explicit synthetic rule fixtures; the tests establish
software semantics, not vision, geometry or real-match accuracy.

This repair does not establish correct point/rally membership analytics,
qualified service-end identity, missing-target abstention, bounded scoring
history or full-match/live behavior. Those remain separate work. Root released
the engine to the native point-ledger implementation after accepting this
phase correction; subsequent source revisions must preserve these semantics
and receive their own validation. The court175 trainer/diagnostic and five
historical frozen helpers were not modified or retried by this repair.
