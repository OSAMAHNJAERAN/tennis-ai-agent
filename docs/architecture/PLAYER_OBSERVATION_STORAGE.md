# Bounded player observations — 8 October 2026

`PlayerDetectionSequence` and `select_court_players_disk` provide indexed scratch storage and exact offline court-role selection for the existing detector observation domain. They are module seams; their deterministic fixtures do not establish independent player accuracy, whole-pipeline memory, complete-match capacity or live-video operation. The existing `CourtPlayerTracker.select` remains unchanged.

Create `PlayerDetectionSequence(output_directory)` inside the job's `ExitStack`. Append each actual detector frame, including empty frames. The sequence preserves proposal order, box coordinates, confidence, class and nullable local track ID. `BBox` and `PoseBBox` are explicit accepted types. Optional anatomical fields retain the seventeen keypoints, foot-point estimate, confidence, source and quality. Python float bits, signed zero, subnormal and diagnostic nonfinite values are represented in private typed JSON; NumPy scalar and point-array dtype, byte order and bytes are preserved. No pickle or arbitrary class loading is used. Unknown attributes and malformed anatomical shapes fail rather than disappearing silently. Actual detector IDs are integer/NumPy integer or null, including wide integers; fractional/string/boolean IDs are rejected.

Reads decode independent snapshots. An integer replacement explicitly persists an edited frame. Iteration retains the current decoded record; slices intentionally materialize their requested records. Default per-frame bound is 4,096 proposals and an oversize frame raises without accepting a truncated frame. A bounded SQLite page cache, file-backed temporary operations and 256-write commit cadence prevent a match-long Python frame list. Normal context exit removes only the exact exclusively created database and its known sidecars. A producer, serializer, database or consumer failure closes storage and retains accepted diagnostic rows; generator cleanup does not replace the original failure.

The selector runs the same legacy whole-segment court-role prior, using disk presence counts and two candidate scans. It preserves candidate occurrence counts, geometric/motion filters, score arithmetic, stable first ties, memory expiry, calibration-loss reset and mutual exclusion. It does not infer confirmed A/B identity or make the offline prior causal. Presence counts for arbitrarily many track fragments stay in a bounded-cache SQLite table; only current-frame candidates and two role-memory slots are resident. An open `PlayerDetectionSequence` is reused under its caller's lifetime. A one-shot input first spools to an owned sequence.

```python
raw_players = export_stack.enter_context(PlayerDetectionSequence(output_dir))
for frame in frames:
    raw_players.append(player_detector.detect_and_track(frame, persist=True))

selected = export_stack.enter_context(select_court_players_disk(
    CourtPlayerTracker(calibration), raw_players, fps, output_dir,
    calibrations=frame_calibrations,
))
p1_boxes, p2_boxes = selected[1], selected[2]
```

The output mapping supplies indexed read-only sequences, each returning an actual selected `BBox`/`PoseBBox` snapshot or null. `selected.index(role, frame_index)` gives the original detector proposal index; `selected.selected_detection_indices[frame_index]` returns both indices. Keep its context alive through the last rendering/export consumer. Disk snapshots cannot satisfy the current `image_people_records` object's `is` comparison. That caller must enumerate original proposals and compare their positions with these indices. Matching by local track ID or box equality would mislabel duplicate IDs or equal observations. This index-aware export and Phase6 orchestration remain caller integration work.

The worker already streams normalization frame mapping and uses disk result arrays, while Phase6 uses an eight-frame decode cache and owned candidate/registration/pose observation spools. This new seam targets the remaining `all_player_dets` and `CourtPlayerTracker` candidate-list/presence-Counter allocations. Full trajectory points, selected box compatibility lists, timestamps, positions and motion sample collections still scale with frame count in the currently integrated pipeline. The next invariant should make mutable temporal trajectory/motion observations disk indexed while proving identical forward/backward filtering, missing-state transitions, event inputs and publication semantics. Merely moving raw player proposals to disk does not prove complete-match capacity.

Synthetic 25,000-frame fixtures measure current/peak traced Python allocations, producer object lifetime and SQLite cache configuration. Tracemalloc does not include native SQLite page caches, decoder/model allocations or GPU memory. Real long-match qualification still needs peak process RSS/VRAM, disk usage, elapsed time, frame/event integrity and restart behavior on authorized recorded matches. Worker retries currently restart from the retained original; these temporary sequences do not implement frame-level checkpoints or resume.
