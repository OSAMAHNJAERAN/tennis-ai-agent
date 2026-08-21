# PLAN.md — Autonomous Execution Plan for T88J709

## Purpose

This is the live implementation plan for **T88J709: Racket Sports Vision System**. The coding/research agent should update the checkboxes and evidence links/paths as work progresses.

The order is deliberate: audit and data validation come before expensive training; component validation comes before system integration; test-set evaluation comes only after model selection.

---

# Phase 0 — Bootstrap, Preserve Baseline, and Audit

## Goals
- Understand the existing codebase completely.
- Establish a reproducible baseline.
- Prevent accidental destruction of working code/data.

## Tasks
- [ ] Read `AGENT.md`, `PLAN.md`, `PROJECT_REPORT.md`, and `VIDEO_REFERENCE.md` completely.
- [ ] Inspect git status, branches, remotes, history, tags, and LFS configuration.
- [ ] Create a feature branch for implementation.
- [ ] Inventory all source, notebooks, model files, data, videos, tests, configs, UI, DB code, and docs.
- [ ] Detect accidentally committed secrets/API keys.
- [ ] Record current Python/framework versions.
- [ ] Attempt clean dependency installation.
- [ ] Run existing tests.
- [ ] Run existing example inference if assets exist.
- [ ] Capture baseline output and runtime metrics.
- [ ] Identify hard-coded paths/FPS/resolution/device assumptions.
- [ ] Write `docs/audit/REPOSITORY_AUDIT.md`.
- [ ] Build report-requirements-to-code gap matrix.

## Exit gate
Proceed only after the existing baseline can either be reproduced or its blockers are documented precisely.

---

# Phase 1 — Research Verification and Architecture Decisions

## Goals
- Verify the report/tutorial technical inputs.
- Choose candidate architectures for benchmarking, not final winners yet.

## Tasks
- [ ] Verify every dataset named by the project report that may be used.
- [ ] Verify every model/framework version to be used.
- [ ] Inspect supplied tutorial and companion GitHub repo.
- [ ] Record tutorial architecture and reproducible baseline config.
- [ ] Research current tennis ball tracking methods.
- [ ] Research TrackNetV3 and TrackNetV4 applicability to tennis.
- [ ] Research RacketVision tennis subset and license/access.
- [ ] Research robust tennis-court registration/keypoint methods.
- [ ] Research player tracking alternatives.
- [ ] Research bounce/contact/event detection methods.
- [ ] Research official tennis court dimensions/line semantics/scoring rules.
- [ ] Write/update `docs/research/RESEARCH_LOG.md`.
- [ ] Create ADRs for model family, storage architecture, and app architecture where decisions are sufficiently mature.

## Candidate set to benchmark
### Player
- [ ] YOLO11 baseline + ByteTrack/BoT-SORT.
- [ ] Optional alternative only if justified.

### Ball
- [ ] Tutorial-style dedicated YOLO baseline.
- [ ] YOLO11 high-resolution fine-tuned baseline.
- [ ] TrackNetV3/TrackNetV4 or another temporal candidate.

### Court
- [ ] Tutorial ResNet50 direct 28-output regression baseline.
- [ ] Heatmap keypoint model candidate.
- [ ] Court-line segmentation/geometric registration candidate if feasible.

## Exit gate
A short `BENCHMARK_MATRIX.md` identifies candidates, datasets, metrics, expected GPU cost, and rejection criteria.

---

# Phase 2 — Dataset Registry, Acquisition, and Quality Control

## Goals
- Build a legal, traceable, leakage-safe tennis dataset suite.

## Tasks
- [ ] Create `data/data_registry.yaml`.
- [ ] Register the tutorial Roboflow tennis-ball dataset with exact version/license.
- [ ] Check source match/video overlap across train/val/test.
- [ ] Register candidate court keypoint dataset with provenance/license.
- [ ] Register any RacketVision tennis subset used.
- [ ] Register local/owner-authorized video data.
- [ ] Refuse/reject sources with unclear rights.
- [ ] Create checksums/manifests.
- [ ] Normalize annotation formats.
- [ ] Create label validator scripts.
- [ ] Visualize randomized samples.
- [ ] Find duplicate/near-duplicate frames.
- [ ] Create grouped train/validation/test split by match/source video.
- [ ] Freeze untouched test manifests.
- [ ] Create hard-case tags: blur, occlusion, night, clay, grass, compression, tiny ball, etc.
- [ ] Document annotation conventions.
- [ ] Produce dataset cards.

## Suggested split policy
Use grouped 80/10/10 when dataset size allows, or a comparable grouped split with enough validation/test sequences. Do **not** random-split neighboring frames.

## Exit gate
No training beyond smoke tests until:
- license status is documented;
- leakage checks pass;
- label visualization passes;
- train/val/test manifests are frozen/versioned.

---

# Phase 3 — Reproduce Tutorial Baseline

## Goals
- Create a known baseline before improving it.
- Ensure tutorial concepts are understood rather than copied blindly.

## Tasks
- [ ] Reproduce tutorial player detection/tracking on one authorized clip.
- [ ] Reproduce tutorial-style dedicated ball detector.
- [ ] Reproduce 14-keypoint ResNet50 court estimator or run equivalent supplied weights if legally available.
- [ ] Reproduce mini-court projection.
- [ ] Reproduce short-gap ball interpolation.
- [ ] Reproduce simple shot/speed/stat output.
- [ ] Measure runtime and component metrics.
- [ ] Record discrepancies caused by newer dependencies.
- [ ] Save a baseline config/checkpoint/output.

## Required baseline report
`experiments/baseline_tutorial/RESULTS.md` must contain:
- exact code commit;
- data version;
- hardware;
- FPS/time base;
- detector metrics;
- court metrics;
- known limitations;
- sample output location.

## Exit gate
A reproducible reference baseline exists that can be compared against later models.

---

# Phase 4 — Player Detection and Tracking

## Goals
- Reliably identify the two court players and maintain identity.

## Tasks
- [ ] Fine-tune or configure YOLO11 candidate if needed.
- [ ] Benchmark detector sizes/resolutions.
- [ ] Configure/benchmark ByteTrack/BoT-SORT.
- [ ] Use court polygon/proximity to filter non-participants.
- [ ] Project player ground-contact point to court coordinates.
- [ ] Add identity recovery after short occlusion.
- [ ] Create tests/metrics for ID switching.
- [ ] Evaluate by court surface/lighting/camera subgroup where data exists.

## Acceptance direction
- Meaningful improvement or equal accuracy with lower compute versus baseline.
- Persistent two-player identity on target clips.
- Low enough projected position error for speed/distance analytics.

Do not set an arbitrary “99%” goal without a benchmark definition.

---

# Phase 5 — Tennis-Ball Detection and Temporal Tracking

## Goals
This is the highest-priority accuracy workstream.

## 5A. Detector baseline
- [ ] Train YOLO11 ball detector at 640.
- [ ] Benchmark 960/1280 or appropriate high resolution.
- [ ] Benchmark court crop/ROI.
- [ ] Add plausible motion-blur/compression augmentation.
- [ ] Tune confidence/NMS on validation data.

## 5B. Temporal candidate
- [ ] Prepare sequence-based tennis data.
- [ ] Reproduce TrackNetV3/TrackNetV4 or selected temporal model when permitted.
- [ ] Benchmark multi-frame length.
- [ ] Benchmark median/background cue if applicable.
- [ ] Evaluate long/short occlusions.

## 5C. Trajectory fusion/rectification
- [ ] Define ball point states (`DETECTED`, `PREDICTED`, etc.).
- [ ] Implement configurable max interpolation gap.
- [ ] Disallow blind initial backfill.
- [ ] Implement trajectory smoothing with validation.
- [ ] Optionally fuse detector + temporal tracker if it improves held-out metrics.

## Metrics
- [ ] detection P/R/F1/mAP;
- [ ] center localization error;
- [ ] trajectory completeness;
- [ ] max/mean missed gap;
- [ ] false-ball rate;
- [ ] bounce-point error;
- [ ] subgroup metrics for blur/occlusion.

## Model selection gate
Choose the ball pipeline from validation metrics and practical runtime. Final test results are recorded only after the choice is frozen.

---

# Phase 6 — Court Keypoints and Homography

## Goals
- Stable, metric court registration.

## Tasks
- [ ] Establish canonical 14-point indexing and diagram.
- [ ] Reproduce ResNet50 regression baseline.
- [ ] Train/evaluate heatmap keypoint alternative.
- [ ] Evaluate segmentation/geometric candidate if promising.
- [ ] Add keypoint visibility/confidence.
- [ ] Compute robust homography.
- [ ] Calculate reprojection error.
- [ ] Reject low-quality homographies.
- [ ] Detect camera drift/motion.
- [ ] Re-estimate homography after detected camera changes.
- [ ] Add synthetic geometry tests.

## Metrics
- [ ] normalized keypoint error;
- [ ] reprojection error;
- [ ] projected line/court geometry error;
- [ ] failure rate;
- [ ] per-court-surface results.

## Exit gate
Projected player/ball locations are stable enough to support metric analytics on held-out clips.

---

# Phase 7 — Canonical Mini-Court and Physical Analytics

## Goals
- One shared metric coordinate system for all calculations and drawings.

## Tasks
- [ ] Encode authoritative tennis-court dimensions in one tested module.
- [ ] Implement image→court and court→mini-court transforms.
- [ ] Use actual source timestamps/FPS.
- [ ] Support variable frame rate or deterministic normalization.
- [ ] Implement player position smoothing.
- [ ] Calculate distance covered.
- [ ] Calculate instantaneous/segment/average speed.
- [ ] Calculate ball speed with quality gating.
- [ ] Exclude/down-weight unreliable interpolated samples.
- [ ] Add uncertainty flags.
- [ ] Draw synchronized mini-court.

## Evaluation
On calibrated or manually validated clips:
- [ ] player position MAE;
- [ ] distance error;
- [ ] player speed MAE/RMSE;
- [ ] ball speed MAE/RMSE.

---

# Phase 8 — Bounce, Shot, Hitter, Rally Events

## Goals
- Convert trajectories into tennis events.

## Tasks
- [ ] Create event annotation schema.
- [ ] Label validation clips for bounce/contact/rally events.
- [ ] Implement trajectory-based bounce baseline.
- [ ] Evaluate temporal bounce model if enough labels exist.
- [ ] Implement contact/shot candidate from trajectory changes.
- [ ] Attribute hitter using player/racket/proximity/temporal evidence.
- [ ] Add confidence and `UNKNOWN` state.
- [ ] Segment rallies if required by downstream stats.
- [ ] Add event debounce and consistency rules.

## Metrics
- [ ] event precision/recall/F1;
- [ ] event timing error;
- [ ] hitter attribution accuracy;
- [ ] bounce localization error.

---

# Phase 9 — IN/OUT Decision Support

## Goals
- Accurate, uncertainty-aware assisted line calls from bounce positions.

## Tasks
- [ ] Use authoritative line semantics.
- [ ] Determine relevant court boundary by match mode.
- [ ] Compute signed distance from bounce point to boundary.
- [ ] Include ball radius/footprint if calibration supports it.
- [ ] Incorporate homography/ball localization uncertainty.
- [ ] Implement `IN`, `OUT`, `UNCERTAIN` states.
- [ ] Store confidence/evidence in `AlertHistory` equivalent.
- [ ] Create near-line validation subset.
- [ ] Calibrate confidence if feasible.

## Metrics
- [ ] in/out accuracy/F1;
- [ ] near-line accuracy;
- [ ] uncertain coverage;
- [ ] false IN/false OUT;
- [ ] spatial error.

## Safety/academic wording
The UI/report must call this **AI-assisted line-call/decision support**, not professional-certified officiating.

---

# Phase 10 — Tennis Scoring State Machine

## Goals
- Deterministic scoring independent of computer-vision implementation.

## Tasks
- [ ] Implement point/game/set states.
- [ ] Deuce/advantage.
- [ ] Tie-break configuration.
- [ ] Serve ownership.
- [ ] Fault/double-fault transitions where available.
- [ ] Manual event correction.
- [ ] Replay/recompute score from event log.
- [ ] Comprehensive unit tests.

## Exit gate
Given a known event sequence, score output is deterministic and correct without loading ML models.

---

# Phase 11 — Persistence Layer

## Goals
- Preserve analyses and support reports/history.

## Tasks
- [ ] Write storage ADR.
- [ ] Implement SQLite by default unless repository/product constraints justify another DB.
- [ ] Add migration framework.
- [ ] Implement report-derived entities: Match, Player, DetectionResult, CourtKeypoints, MatchEvent, AnalyticsData, AlertHistory.
- [ ] Decide whether frame-level detections belong in SQL, Parquet, or hybrid storage.
- [ ] Store large media/model assets outside DB; store paths/hash/metadata.
- [ ] Add database indexes and foreign keys.
- [ ] Add CRUD/repository tests.
- [ ] Persist model/dataset version with each analysis.

---

# Phase 12 — Analytics, Heatmaps, Dashboard, and Reports

## Goals
- Deliver the report's user-visible product.

## Tasks
- [ ] Generate player movement traces.
- [ ] Generate per-player occupancy heatmaps.
- [ ] Generate ball-bounce/placement heatmap.
- [ ] Aggregate shot/rally statistics.
- [ ] Add line-call timeline.
- [ ] Add player speed/distance cards.
- [ ] Add ball speed/stat cards.
- [ ] Add mini-court trajectory view.
- [ ] Preserve/implement project navigation: Dashboard, Matches, Officiating, Analytics, Heatmaps, Reports where consistent with existing UI.
- [ ] Add model version/status information.
- [ ] Implement exportable analysis report.
- [ ] Ensure displayed metrics explicitly distinguish measured vs inferred/low-confidence values.

---

# Phase 13 — End-to-End Pipeline and Performance Optimization

## Goals
- Stable integrated inference from video to final outputs.

## Tasks
- [ ] Stream/process video instead of loading all frames into RAM when practical.
- [ ] Build deterministic pipeline orchestration.
- [ ] Add progress/error reporting.
- [ ] Cache expensive static per-match computations appropriately.
- [ ] Batch model inference where useful.
- [ ] Profile CPU/GPU/memory bottlenecks.
- [ ] Benchmark mixed precision.
- [ ] Evaluate ONNX/TensorRT only after model accuracy is frozen.
- [ ] Preserve a PyTorch reference implementation for correctness.
- [ ] Measure processing FPS on named hardware.
- [ ] Never claim real-time without benchmark evidence.

---

# Phase 14 — Automated QA and Evaluation Freeze

## Goals
- Prove reproducibility and quantify limitations.

## Tasks
- [ ] Run unit tests.
- [ ] Run integration tests.
- [ ] Run end-to-end golden clip test.
- [ ] Run data-leakage checks.
- [ ] Run final component validation.
- [ ] Freeze model/config bundle.
- [ ] Run untouched test set once for final reporting.
- [ ] Produce per-subgroup metrics.
- [ ] Produce failure cases.
- [ ] Record GPU/CPU/memory/runtime.
- [ ] Write `docs/evaluation/EVALUATION_REPORT.md`.
- [ ] Write model cards.
- [ ] Write data cards.

## Final metrics table must include
- player detector metrics;
- player tracking metrics where labels exist;
- ball detector/tracker metrics;
- court keypoint/homography metrics;
- bounce/contact event metrics;
- speed/distance error where ground truth exists;
- IN/OUT metrics;
- end-to-end processing speed.

No fabricated or literature-only metric may appear in the “Our Results” column.

---

# Phase 15 — Reproducibility and Handoff

## Goals
- Another developer/supervisor can run the work.

## Tasks
- [ ] Clean README with exact setup.
- [ ] Environment lock/requirements.
- [ ] `.env.example` without secrets.
- [ ] Training commands/scripts.
- [ ] Evaluation commands.
- [ ] Inference command.
- [ ] Database migration/setup command.
- [ ] Sample config.
- [ ] Architecture diagram/document.
- [ ] Known limitations/future work.
- [ ] Final git clean check.
- [ ] Model/data license inventory.
- [ ] Artifact checksum manifest.
- [ ] Produce final verification summary with commit SHA and exact measured metrics.

---

# Cloud Training Runbook

For each model training job:

1. [ ] Confirm dataset/version/hash.
2. [ ] Confirm train/val grouping and untouched test isolation.
3. [ ] Confirm license.
4. [ ] Record git SHA.
5. [ ] Record cloud instance/GPU/VRAM.
6. [ ] Create isolated environment.
7. [ ] Run label/data loader smoke checks.
8. [ ] Overfit a tiny sample to validate the training pipeline.
9. [ ] Run 1–2 epoch smoke training.
10. [ ] Check loss/metrics/visual predictions.
11. [ ] Start full baseline.
12. [ ] Save periodic checkpoints outside ephemeral storage.
13. [ ] Early stop/tune only using validation.
14. [ ] Save `best` according to explicit validation criterion.
15. [ ] Export experiment metadata.
16. [ ] Compare candidates.
17. [ ] Freeze winner.
18. [ ] Evaluate final winner on untouched test split.
19. [ ] Save model card and SHA256.
20. [ ] Release unused cloud GPU resources.

---

# Definition of Success

The system is successful when it can process an authorized single-baseline tennis video from beginning to end and produce synchronized, persisted, and evaluated outputs—ball/player tracking, court mapping, mini-court, movement/speed analytics, bounce/event data, assisted IN/OUT, heatmaps, score/report data—while clearly reporting confidence and limitations.

The target is **the highest defensible accuracy achievable with available licensed data and compute**, not an invented percentage. Every improvement must be supported by held-out validation/test evidence.
