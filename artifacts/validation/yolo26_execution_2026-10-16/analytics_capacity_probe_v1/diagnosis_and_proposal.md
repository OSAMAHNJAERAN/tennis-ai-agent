# Read-only analytics capacity diagnosis

The current native shot store is bounded, but its consumers reconstruct collections. This probe executed the exact current consumer class/function AST and the actual Phase6 step-10 statements on fresh typed, lazy shot snapshots. It supplied mixed shot types, alternating players, every seventeenth shot marked dead, and one synthetic landing per shot. It supplied no line-call or player-position records. These fixture frequencies are not match-frequency measurements.

The one executed command was:

```powershell
& 'C:\Users\ac-98\AppData\Local\Programs\Python\Python313\python.exe' 'artifacts\validation\yolo26_execution_2026-10-16\analytics_capacity_probe_v1\reproduce_current_analytics_materialization.py'
```

It intentionally exited **1** after saving the RED receipt, because all eighteen consumer/size cases violate the proposed eight-live-object-or-eight-retained-record bound. Its receipt SHA256 is `a2f76274333858916d6ec353115c202f61fdfcbe9502dbae2cf36b0a22e62633`; the script SHA256 is `4facfef9a07675fd6e208be31a0da004ddfee48f469a9250edbf99ed1563d1d1`. All eight bound production files were unchanged. The harness elapsed time was 1.23 seconds, process wall time 2.47 seconds and process peak working set 107,028,480 bytes. Neither Torch nor Ultralytics was imported, and no model, media, training or GPU operation occurred. Traced bytes include weak-reference instrumentation overhead.

| Consumer | Peak live shots at 257 / 1,025 / 4,097 inputs | Peak traced bytes at 4,097 | Retained output at 4,097 |
| --- | --- | --- | --- |
| Rally analyzer | 242 / 965 / 3,857 | 2,896,540 | One rally with 3,856 live shot IDs |
| Player 1 statistics | 122 / 484 / 1,930 | 1,500,140 | Nineteen fixed counter keys; 1,928 counted live shots |
| Serve statistics | 28 / 105 / 412 | 327,380 | Nine fixed top-level keys; 410 serves counted |
| Bounce landing heatmap | 2 / 2 / 2 | 1,470,864 | 4,097 landing dictionaries |
| Point analytics without supplied rally | 243 / 966 / 3,858 | 2,868,252 | One point with all 4,097 shot IDs |
| Actual Phase6 step 10 | 242 / 965 / 3,857 | 4,645,447 | 3,856 rally IDs + 4,097 point IDs + 4,097 landing dictionaries |

The combined step performed 2,057 / 8,201 / 32,777 source reads. Its retained traced bytes were 120,123 / 482,011 / 1,926,267. A peak live count can exceed the number ultimately selected because the current rejected row and a comprehension's retained rows coexist. Zero `ShotEventEvidence` objects remain after each measured consumer returns; retained output arrays still grow.

Ranked, falsifiable explanations are:

1. `RallyAnalyzer.live_shots` and the point fallback's live-shot comprehension dominate temporary shot retention. A scalar reduction retaining only first/last shots must reduce the live-shot peak to a small constant, while the current `shot_ids` result will still grow until separately streamed.
2. Rally/point shot-ID arrays and the per-shot landing heatmap dominate retained output. Caller-owned lazy membership/landing views plus incremental JSON serialization must remove that retained growth while preserving the loaded JSON values and row order. Replacing only the temporary shot lists cannot satisfy this prediction.
3. Player/serve subsets are unnecessary temporary lists because their results are fixed-size counters. One-pass reductions must preserve the exact counter schema and arithmetic while keeping at most the current record alive. They need no new result database by themselves.
4. The repeated scans add work when each shot is decoded from SQLite. Indexed selections or scalar reductions can reduce reads, but fewer scans alone cannot remove a materialized output array. No throughput or live-latency claim follows from this metadata probe.

The smallest ownership proposal reuses the already caller-owned `ShotEvidenceSequence` instead of copying all shots into another database. Add explicit read-only indexed selection/ID/landing views bound to that open owner. The source schema would index the existing dead flag, player, shot type and landing-presence selectors, while typed payloads retain the exact record fields. A sealed owner or explicit revision check must prevent views silently changing beneath a consumer. Phase6's existing export stack already supplies the necessary lifetime; no handles may escape in the returned pipeline summary.

An explicit native `result_owner=shot_output` option on the rally/point/match consumers would require the exact source alias and a stable open owner. Legacy default APIs would retain their existing lists. Native rally summaries would contain scalar fields and a lazy live-shot-ID view; point summaries would use a lazy all-shot-ID view; the match landing field would be a lazy ordered view over all shots with a landing, including dead shots because the current exporter includes them. Fixed player/serve statistics would reduce records directly. This proposal must not infer or fabricate point membership.

Export integration is required in the same change. `dataclasses.asdict()` on a dataclass containing a SQLite-backed view is unsuitable: it deep-copies arbitrary fields. Native rally/point exporters must write the scalar record and its `shot_ids` incrementally, retaining existing array names and loaded JSON values. The match exporter can stream the `ball_landing_heatmap` top-level array with the existing array-artifact writer. Publication must remain atomic per file, preserve existing destinations on failure and remove only the exact owned temporary file. This is one coherent consumer/output seam, not permission to change common JSON behavior throughout the repository.

Proposed ownership after review: extend `src/utils/shot_evidence_sequence.py`, change the four `src/analytics/{rally_analyzer,shot_statistics,serve_analyzer,match_analytics}.py` consumers and narrow Phase6 step-10/export calls; add a focused analytics JSON helper only if necessary. No detector, training, data, production configuration or scoring source changes are needed for this capacity seam. The current completed shot-store hashes remain frozen until root review authorizes changes.

Required verification would compare exact prechange and native outputs with nonempty mixed types/players, all-dead and empty groups, duplicate IDs, absent/zero/partial positions, unknown values, enum outcomes, tied and irregular times, source order and all current rounding. Native tests must exercise actual Phase6 step-10 and all three exports, not only a helper or authority-off branch. Re-run the three measured sizes; assert constant live-shot counts and no duration-sized Python result arrays. Exercise a serialization failure/cancellation while a membership iterator is suspended, then check owner lifetime, closed readers, exact scratch cleanup and preservation of an existing completed destination. Full-video memory testing remains separate.

Behavior findings stay separate from capacity. Phase6 calls the current one-group rally/point APIs once for its complete shot sequence and supplies one final scoring state. A fixture containing two serves separated by 600 seconds still returns one rally with `point_id=1`; the API receives no independent point-boundary input. This proves the lack of segmentation at this seam, not a measured missed-point rate. Match analytics also assigns `total_points_played=1` unconditionally before the authority-off gate clears it. Rally timing uses frame indices divided by the supplied FPS, including the terminal bounce, rather than actual event timestamps; a recorded irregular-timing fixture returns 0.48 seconds despite its last hit timestamp being 0.71 seconds. No bounce timestamp was supplied, so this is a method observation, not a qualified duration error. Serve statistics currently treat selected serves as first attempts and leave second-serve counters zero. Those semantics must remain visible and require separately specified fixes before authoritative full-match analytics can be qualified.

Player occupancy also builds `valid_pts`, `xs` and `ys` from all valid positions. That frame-count growth was identified in source but was not measured by this shot-only probe; current native ground positions remain unavailable. Scoring histories, line calls, landing overlays, optional activity arrays and serving adapters remain outside this proposal. No production source was edited, no fix is claimed and neither independent accuracy nor long-match capacity is established.
