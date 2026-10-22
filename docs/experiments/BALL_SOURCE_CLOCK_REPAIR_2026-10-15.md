# Native ball association source-clock repair

The native YOLO26 Kalman interval adapter now explicitly supplies a lazy source-timing sequence. Exact source PTS differences drive image-space prediction, candidate association, gap budgets, interpolation and pixel velocity. Normalized timestamps still drive replay. This is a software timing repair, not vision, event or physical-speed qualification.

## Reproduced defect

The retained red regression uses the actual `src.api.media_timing.frame_mapping` and `track_candidate_intervals` functions. At a normalized 25 FPS, source PTS `0, 5` with time base `1/100` have an actual elapsed interval of 0.05 seconds. The existing one-source-tick cadence tolerance accepts that transition. For detections 10 pixels apart, the old adapter returned 250 px/s from normalized 0.04-second spacing; the source-clock result must be 200 px/s. The executed red run failed on that numerical difference, while normalized replay timestamps were correctly `[0, 0.04]`.

The cadence/qualification code is unchanged. Exact source timing does not make an otherwise ineligible frame transition eligible.

## Contract and policy

- `TemporalBallTracker.track_video_candidates(..., source_timing=sequence)` is optional. Omitting it preserves the existing fixed-FPS algorithm, list/sink contracts and backward pass. The adapter's `use_source_timing=False` default also preserves the old tracker keyword contract. The actual native YOLO26 Kalman call opts in explicitly.
- Each timing row retains `source_pts`, `source_time_base`, integer `source_frame_index`, nonempty `segment_id` and explicit boolean `measurement_transition_allowed`. The tracker reconstructs `Fraction(int(PTS)) * Fraction(time_base)`, subtracts exact times, then converts only the elapsed difference to float. Lossy absolute `source_timestamp_seconds` fields are not used for association.
- Missing or malformed timing, sentinel PTS, nonpositive time bases, duplicate/reversed time, nonconsecutive source-frame indices, changed/missing segments, false incoming eligibility, and unrepresentable elapsed float values break continuity. A valid current visual detection may remain visible as a raw observation, but unavailable timing cannot seed extrapolation. No nominal-FPS fallback supplies missing source time.
- The first row's false incoming transition is a segment boundary; it does not invalidate its own source timestamp or raw detection. Every output point keeps a normalized replay timestamp. Existing candidate-frame exports retain the original PTS/time-base/source-frame tokens separately.
- Predictions are limited by both missing-frame count and elapsed seconds (`max_prediction_gap / reference_fps`). Reconnection to a later observation is limited by missing-frame count and an anchor span of `(max_prediction_gap + 1) / reference_fps`. The extra interval reflects the final step into the later observation. Stale filters expire before that observation is incorporated.
- Interpolation is separately limited by missing-frame count and anchor span `(max_interpolation_gap + 1) / reference_fps`, with every intervening source transition eligible. Its fraction is elapsed source time divided by total source anchor time. It remains an explicitly labeled **offline** operation using a later anchor.
- Motion thresholds in pixels per reference frame are converted to pixels per source second. Both displacement from the last observed anchor and displacement from the immediately previous output are checked. A strong raw detection that fails association starts new continuity with null linked velocity. This prevents a stale predicted point followed by reacquisition from creating a velocity spike.
- The source-clock filter constructs its transition matrix from each actual elapsed step. Its existing reference-step process noise is scaled linearly by elapsed/reference time; this is an engineering image-space noise model, not a physically calibrated uncertainty model.
- Source-clock forward association never looks at future candidates. Static suppression uses only two eligible past neighbors. Future-supported low-confidence seeding and the legacy backward-confirmation pass are disabled in this opt-in policy. Disabling reacquisition still permits initial low-confidence initialization but prevents low-confidence reseeding after a previous track is lost. Point source strings distinguish source-clock forward results, resets and offline interpolation.
- Timing windows remain lazy, and stored point mutations use explicit writebacks. Existing sink prefixes are not reprocessed. This does not implement filter checkpoint/resume.

## Executed verification

Evidence is retained in `artifacts/validation/yolo26_execution_2026-10-15/ball_source_clock_v1/`:

- `red_junit.xml`: one numerical regression failure, expected 200 versus actual 250 px/s before implementation.
- `green_initial_junit.xml`: 12 passing tests, including the source-clock regression and unchanged legacy list/disk sink semantics.
- `focused_junit.xml`: 97 passing tests in 25.33 seconds: 53 source-clock cases, 11 legacy sink cases, 25 typed-storage cases and 8 media-timing cases. Three existing media-timing cases generated tiny synthetic FFmpeg inputs; no acquired match video or learned model was processed.
- `focused_sources_start.json` and the final receipt bind the executed source/test files. The initial red trace is retained, but a complete pre-edit source-file snapshot was not created by this task.

The source-clock cases cover irregular intervals, time-weighted interpolation, equal-displacement short/long gates, high-confidence jumps, elapsed/count gap boundaries, invalid/cut transitions, large positive and negative absolute PTS, changed time bases, float underflow, filter advancement/covariance, future-frame invariance, reacquisition behavior, disk-prefix parity and lazy 1,000-frame timing reads. The compatibility result and final SHA bindings are recorded in `verification_receipt.json`.

## Remaining limits

No model training, GPU inference, acquired-video replay, independent accuracy evaluation or full-match/live run was performed for this repair. The event detector and event/refinement clocks still require a separate source-time audit; normalized event timestamps must not be relabeled as native elapsed time. The tracker still lacks independently qualified impact transitions, trajectory accuracy and pixel-velocity accuracy. Pixel velocity is not meters/second or physical ball speed. Offline interpolation prevents this result from being used as evidence of a future-free live pipeline.
