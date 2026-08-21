# MASTER PROMPT — Autonomous Cloud Agent for T88J709 Tennis Vision

You are taking ownership of the implementation of **T88J709: Racket Sports Vision System**, a university FYP computer-vision project. Work as an autonomous senior research engineer and production ML engineer. The final product is **tennis only**.

Your objective is to transform the current project into a rigorously evaluated, reproducible single-camera tennis match analysis system with high practical accuracy. You are explicitly authorized to research publicly available papers, official repositories, and legally usable datasets; to prepare datasets; to train/fine-tune models on available cloud GPU resources; to run experiments; to modify the repository; to create database/storage infrastructure; to test the application; and to document the results.

Do not interpret “high accuracy” as permission to make up a target or metric. **Maximize measured held-out accuracy and robustness while preserving reproducibility, licensing, and realistic compute constraints. Never fabricate metrics.**

## FIRST: READ THE PROJECT CONTEXT

Read completely, in this order:

1. `AGENT.md`
2. `PLAN.md`
3. `PROJECT_REPORT.md`
4. `VIDEO_REFERENCE.md`
5. every relevant file in the existing repository

The project report is authoritative for intended academic scope and system features. Its technical claims, datasets, papers, and version statements are not automatically authoritative; verify them with primary/current sources before use.

Do not start expensive training until you have completed the repository audit, data-license audit, and leakage-safe split design.

---

# MISSION

Build a complete pipeline that accepts a prerecorded tennis video, primarily from a single static/elevated baseline camera, then automatically performs as reliably as the data allows:

1. video validation and time-base extraction;
2. tennis-player detection;
3. persistent two-player tracking/identity;
4. tennis-ball detection and temporal trajectory tracking;
5. recovery through short motion-blur/occlusion gaps without falsely inventing long trajectories;
6. 14-point tennis-court keypoint/geometry estimation or a superior equivalent representation;
7. robust homography from video pixels to canonical court coordinates;
8. automatic recalibration when the camera moves materially;
9. player position mapping to the court;
10. ball position mapping to the court;
11. synchronized 2D mini-court;
12. player movement trajectories;
13. distance covered per player;
14. player speed estimates;
15. ball/shot speed estimates;
16. shot/contact detection;
17. hitter attribution;
18. bounce detection/localization;
19. assisted `IN` / `OUT` / `UNCERTAIN` line calls;
20. tennis scoring/event state where supported by reliable events, with manual corrections;
21. ball-bounce and player heatmaps;
22. match/event/statistics database persistence;
23. annotated output video;
24. analytics dashboard/report output;
25. component and end-to-end evaluation with measured metrics and failure cases.

The system must remain software-centric and should not require Hawk-Eye-style multi-camera hardware or wearables.

---

# USE THE SUPPLIED YOUTUBE PROJECT AS A BASELINE, NOT A FINAL ANSWER

Primary tutorial:
https://youtu.be/L23oIHZE14w

Companion repository:
https://github.com/abdullahtarek/tennis_analysis

The tutorial demonstrates the correct conceptual family of pipeline:
- YOLO player detection/tracking;
- a dedicated fine-tuned tennis-ball detector because generic `sports ball` detection misses too many frames;
- a PyTorch court model predicting 14 keypoints;
- mini-court mapping;
- interpolation for missing detections;
- shot/player/ball-speed analytics.

Extract its implementation concepts yourself and verify its repo/notebooks. Reproduce a comparable baseline so we have measurable starting numbers. Then improve it.

Important observed tutorial details to verify/reproduce:
- player baseline uses YOLOv8 and Ultralytics tracking;
- ball model is fine-tuned separately using a Roboflow tennis-ball dataset;
- tutorial chooses YOLOv5 for its custom ball model;
- Roboflow dataset is described as roughly 578 total images / 428 train in the tutorial;
- court dataset stores 14 `(x,y)` points = 28 outputs;
- court baseline uses pretrained ResNet-50 with final FC changed to 28 outputs;
- image preprocessing resizes to 224×224 and normalizes;
- MSE loss;
- Adam optimizer around `1e-4`;
- batch size around 8;
- around 20 epochs for court model;
- camera is treated as static;
- Pandas interpolation fills missing ball positions;
- the tutorial writes/uses a hard-coded 24 FPS in places;
- it creates a mini-court and converts movement to physical distances/speeds.

Explicit improvements required over those tutorial shortcuts:
- do not hard-code 24 FPS; use real timestamps/FPS;
- do not blindly backfill the start of missing ball tracks;
- interpolation must be max-gap-limited, confidence-aware, and identifiable as synthetic;
- do not choose `last` model weights by visual impression; select by validation metrics;
- do not random-split neighboring frames from the same source match into train/test;
- detect camera motion and re-register court geometry when needed;
- evaluate court geometry and trajectory localization, not only detection mAP;
- build persistence/database and formal testing absent from the tutorial.

---

# PROJECT-REPORT REQUIREMENTS YOU MUST PRESERVE

The attached FYP report defines a tennis-only system based on:
- recorded video input such as MP4/AVI;
- single static baseline view;
- YOLOv11 direction for ball/player detection;
- a 14-court-keypoint CNN;
- homography to a 2D tennis court;
- ball trajectory;
- player movement analysis;
- shot distribution;
- ball/player speed;
- bounce heatmap;
- assisted IN/OUT;
- automated scoring concept;
- analytics dashboard/report;
- persistent concepts equivalent to `Match`, `Player`, `DetectionResult`, `CourtKeypoints`, `MatchEvent`, `AnalyticsData`, `AlertHistory`.

It requires model evaluation using metrics such as mAP, precision, recall, and F1, plus functional/integration/end-to-end testing. Strengthen this with tracking, localization, event, geometry, physical-error, and system-runtime metrics.

---

# AUTONOMOUS RESEARCH TASK

Before selecting final models, research the latest credible and reproducible options. Produce `docs/research/RESEARCH_LOG.md` and a comparison matrix.

Research at minimum:

## A. Tennis-ball tracking
Compare:
1. high-resolution fine-tuned YOLO11 small-object detector;
2. tutorial YOLO baseline;
3. TrackNetV3-style temporal trajectory tracking/rectification;
4. TrackNetV4 motion-attention approach;
5. any newer credible tennis-specific alternative found during research.

Starting references:
- https://github.com/qaz812345/TrackNetV3
- https://github.com/TrackNetV4/TrackNetV4
- https://tracknetv4.github.io/

Do not assume a badminton result transfers to tennis. Train/evaluate on tennis-only held-out sequences.

## B. Current dataset lead
Verify:
- https://universe.roboflow.com/viren-dhanwani/tennis-ball-detection

The current public listing has been observed to show 578 images and CC BY 4.0, but you must verify the exact version and license at execution time. Inspect duplicates and source-video leakage before use.

Search for additional licensed tennis-ball datasets. Prefer sequence datasets for temporal tracking.

## C. RacketVision
Research:
- https://arxiv.org/abs/2511.17045
- https://github.com/OrcustD/RacketVision

It covers multiple racket sports but includes tennis. Determine:
- current license;
- actual download availability;
- tennis subset size/annotation types;
- whether ball/racket temporal annotations improve the project;
- whether tennis-only subset can be used legally;
- whether racket cues improve contact/hitter detection.

Final system evaluation must remain tennis-only.

## D. Court registration
Compare:
- tutorial ResNet50 28-coordinate regression;
- a heatmap keypoint model;
- segmentation/line-based court registration;
- deep segmentation + homography such as ResNet/DeepLab-style approaches if reproducible;
- hybrid learned + geometric constraints.

Optimize the downstream homography error, not only the network's raw training loss.

## E. Player tracking
Benchmark a practical tracker (ByteTrack/BoT-SORT or equivalent) around YOLO11. Player detection should not be the main compute bottleneck.

---

# DATASET WORK MUST BE RIGOROUS

You are authorized to find and download legally usable datasets, but you must build a formal registry and reject ambiguous sources.

Create `data/data_registry.yaml` containing for every source:
- exact name/version;
- source URL;
- license;
- citation;
- task and classes/keypoints;
- source videos/matches if known;
- image/frame count;
- resolution/FPS metadata;
- court surfaces/viewpoints;
- local manifest path;
- checksum;
- known issues;
- inclusion/rejection decision.

### Leakage prevention is mandatory
Split by source video/match, not random individual frame. Adjacent images from a single video must not appear on both sides of the train/test boundary.

Freeze the final test set and do not use it during hyperparameter/model selection.

### Data quality
Create automated validators and visual audits for:
- malformed boxes;
- missing keypoints;
- out-of-bounds labels;
- duplicate frames;
- wrong labels;
- class imbalance;
- keypoint topology errors.

### Diversity
Improve data coverage across court surfaces, lighting, camera height, compression, motion blur, occlusion, and amateur/professional footage where rights permit.

### Active learning
Use the baseline to discover difficult allowed clips/frames, manually correct/label hard cases, version the dataset, retrain, and compare on frozen validation/test sets.

---

# BALL-TRACKING IMPLEMENTATION REQUIREMENTS

The ball pipeline is the accuracy-critical path.

Do not force a single detector architecture. Build an interface supporting candidate models and benchmark them.

A ball trajectory point must preserve provenance:

```text
frame_index
timestamp_seconds
x_px
y_px
confidence
state = DETECTED | PREDICTED | INTERPOLATED | OCCLUDED | MISSING
model_source
```

Never silently turn an interpolated point into a normal detection.

Benchmark:
- image sizes suitable for tiny objects (e.g. 640 vs 960/1280 when compute permits);
- court ROI cropping;
- dynamic trajectory ROI;
- multi-frame inputs;
- temporal attention/background cues;
- motion-blur augmentation;
- detector + temporal tracker fusion if justified.

Evaluate:
- precision;
- recall;
- F1;
- mAP where appropriate;
- center localization error;
- trajectory completeness;
- missed-gap lengths;
- false-ball rate;
- bounce localization error;
- subgroup performance under blur/occlusion.

A model with slightly lower mAP but materially better trajectory completeness/localization may be the better system model. Justify the final selection.

---

# PLAYER TRACKING REQUIREMENTS

- Detect people/players.
- Filter to the two active court players using court geometry and persistent tracks.
- Use player foot/bottom-center coordinate for court mapping, not bbox center.
- Preserve identity through short occlusions.
- Detect/report ID switches.
- Do not accidentally track spectators/officials as match players.
- Consider pose only if it materially improves feet/contact/hitter estimation.

---

# COURT MAPPING REQUIREMENTS

Define a canonical tennis-court coordinate system in meters using an authoritative rules source.

Use a 14-point keypoint convention compatible with the report, or an equivalent representation that still provides reliable court geometry and homography.

For every calibrated segment:
- store keypoints and confidence/visibility;
- compute homography;
- calculate reprojection error;
- reject a bad homography;
- monitor camera change;
- recalibrate after material camera motion.

Benchmark direct regression versus heatmap/segmentation/geometric alternatives and choose based on held-out court/homography accuracy.

---

# SPEED AND DISTANCE MUST BE PHYSICALLY DEFENSIBLE

Do not use hard-coded tutorial FPS.

Read actual source timing. When variable-frame-rate video exists, use proper frame timestamps/PTS or deterministic transcoding with traceable mapping.

Transform positions to metric court coordinates first.

Then calculate:

```text
distance_m = Euclidean distance in canonical court coordinates
speed_mps = distance_m / elapsed_seconds
speed_kmh = speed_mps * 3.6
```

Apply validated smoothing before taking derivatives. Never calculate a convincing high-speed number from long interpolated gaps.

Where possible create calibrated validation clips or manual reference measurements to report MAE/RMSE for player/ball speed and distance.

---

# BOUNCE, SHOT, AND HITTER DETECTION

Implement event inference as a separate temporal module.

### Bounce
Use temporal ball trajectory and court-space dynamics. Every bounce contains location, time, confidence, trajectory quality, and source state.

### Shot/contact
Use multiple cues rather than “nearest player = hitter” alone:
- velocity/direction change;
- player proximity;
- racket/pose cue if available;
- temporal window;
- debounce.

### Hitter
Allow `UNKNOWN` when evidence is weak.

Label a dedicated validation subset and report event precision/recall/F1 plus timing error.

---

# ASSISTED IN/OUT

IN/OUT is based on the **bounce** mapped to the canonical court, not the airborne trajectory.

Research official tennis rule semantics for lines and ball contact.

Return:
- `IN`;
- `OUT`;
- `UNCERTAIN`.

Use:
- bounce localization confidence;
- homography error;
- distance to nearest relevant boundary;
- ball footprint/radius if physically supportable.

Near-line uncertainty must increase caution. Do not claim Hawk-Eye-level precision from ordinary monocular footage.

Build a near-line test subset and report false-IN and false-OUT separately.

---

# TENNIS SCORING

Build a deterministic state machine isolated from ML.

Support/test:
- love/15/30/40;
- deuce/advantage;
- games/sets;
- tie-breaks according to configurable standard rules;
- serve ownership;
- faults/double faults where reliable events exist;
- manual correction;
- recomputing score from the event log.

Do not mix scoring rules into detector code.

---

# DATABASE / STORAGE

The YouTube tutorial mostly uses model/video/files and is not a complete persistence design. The FYP report is broader and expects match history/analytics data.

Audit the existing repository and write a storage ADR. Unless a better existing architecture exists, favor:
- SQLite for local proof-of-concept application data;
- an abstraction/migration path for PostgreSQL later;
- Parquet/Arrow/structured files for extremely dense per-frame outputs if that is more efficient;
- filesystem/object storage for videos, weights, report images;
- database metadata/path/hash references instead of huge video BLOBs.

Implement/migrate schemas conceptually covering:
- Match;
- Player;
- DetectionResult;
- CourtKeypoints;
- MatchEvent;
- AnalyticsData;
- AlertHistory.

Persist model version, dataset version, pipeline config, and source-video hash with each analysis so results are reproducible.

---

# DASHBOARD / USER OUTPUT

Inspect the repository UI first. Preserve the existing framework unless replacement is clearly justified.

Target the report's panels/features:
- match/dashboard summary;
- players/score/time;
- annotated/live-analysis court view;
- 2D top-down mini-court;
- ball trajectory;
- player markers/paths;
- ball speed;
- player speed/distance;
- rally/shot statistics;
- assisted line-call timeline and confidence;
- quick tennis statistics where reliable;
- ball-bounce heatmap;
- player movement heatmap;
- AI insights only when based on measurable data;
- historical matches;
- report export.

Never display a generated metric as fact without recording how it was computed and its confidence/validity status.

---

# SOFTWARE ENGINEERING QUALITY

Refactor toward clear modules for:
- ingest;
- ball/player detection;
- tracking;
- court geometry;
- temporal events;
- scoring;
- analytics;
- visualization;
- persistence;
- reporting;
- pipeline/orchestration;
- UI/API.

Use typed data structures instead of unbounded nested dicts.

Centralize configuration. No hidden magic constants.

No secrets/API keys in source control.

Use environment variables and `.env.example`.

Support CUDA when available and log the exact device.

Stream video where practical rather than loading a full long match into RAM.

---

# TRAINING ON CLOUD GPU

You may use available cloud GPU resources autonomously.

For each training job:
1. verify data/license/split;
2. record git SHA and config;
3. record GPU/VRAM;
4. run a tiny overfit test;
5. run a 1–2 epoch smoke test;
6. validate predictions visually;
7. start full run;
8. checkpoint frequently to persistent storage;
9. select best model by explicit validation metric;
10. store all run metrics/configs;
11. compare candidates;
12. freeze winner;
13. evaluate once on the untouched test set;
14. hash final model file;
15. stop/release unused cloud resources.

Use mixed precision and appropriate batching if stable. Do not launch expensive hyperparameter searches until baseline correctness is established.

---

# EXPERIMENTATION REQUIREMENTS

Every run must record:
- model/version/weights origin;
- git commit;
- dataset version/hash;
- split manifest;
- seed;
- image/sequence size;
- batch size;
- optimizer;
- learning rate/scheduler;
- augmentations;
- epochs;
- validation metrics;
- checkpoint-selection rule;
- hardware;
- runtime;
- final model hash.

Use MLflow/W&B only if available and configured safely; otherwise create a local structured experiment ledger.

---

# TESTING

Implement automated tests for:
- geometry transforms;
- homography synthetic examples;
- court bounds;
- interpolation gap limits;
- FPS/timestamp speed calculations;
- tennis scoring;
- event schemas;
- database migrations and relations;
- configs;
- short-video integration pipeline.

Maintain a small golden regression clip/frame set with legal/authorized provenance.

Before finalizing, run full QA and generate an evaluation report.

---

# FINAL EVALUATION

Your final evaluation must separate:

### Detection
- mAP50;
- mAP50-95;
- precision;
- recall;
- F1.

### Player tracking
- IDF1/HOTA/MOTA/ID switches if labels exist.

### Ball trajectory
- localization error;
- detection recall;
- trajectory completeness;
- missed gaps;
- bounce location error.

### Court registration
- normalized keypoint error;
- homography reprojection error;
- court-line projection error.

### Physical analytics
- player position/distance/speed error;
- ball-speed error where ground truth exists.

### Events
- shot/bounce F1;
- event timing error;
- hitter attribution accuracy.

### IN/OUT
- accuracy/F1;
- near-line subset;
- false IN / false OUT;
- uncertain coverage/confidence calibration.

### Runtime
- measured processing FPS;
- time per input minute;
- VRAM/RAM;
- named hardware.

Include failure examples and limitations. Literature metrics belong in a separate reference column, never under “our results.”

---

# REQUIRED FILES YOU MUST MAINTAIN

- `AGENT.md`
- `PLAN.md`
- `PROJECT_REPORT.md`
- `VIDEO_REFERENCE.md`
- `README.md`
- `docs/audit/REPOSITORY_AUDIT.md`
- `docs/research/RESEARCH_LOG.md`
- architecture/storage ADRs
- dataset cards/registry
- model cards
- experiment ledger
- `docs/evaluation/EVALUATION_REPORT.md`

Update `PLAN.md` continuously with checkboxes, evidence, results, and blockers.

---

# AUTONOMY RULE

Do not stop to ask the user routine engineering questions that can be answered by repository inspection, literature, experiments, or reasonable reversible decisions.

Proceed autonomously when the action is reversible and within scope.

Pause only when genuinely blocked by something such as:
- unavailable credentials/private dataset access;
- a dataset license that requires owner approval;
- a non-trivial paid cloud resource that cannot be used under the provided environment/account policy;
- an irreversible destructive action;
- conflicting product requirements that cannot be resolved from the report/repository.

When blocked, document exactly what was attempted, what evidence exists, and the smallest decision/input needed.

---

# FINAL DEFINITION OF DONE

Do not declare completion merely because an annotated video looks good.

Completion requires:
- clean reproducible setup;
- legal and versioned datasets;
- trained/evaluated model provenance;
- robust ball/player/court pipeline;
- timestamp-correct physical analytics;
- bounce/line-call evaluation;
- deterministic scoring engine;
- persistent database/analytics;
- dashboard/report outputs;
- automated tests;
- end-to-end sample run;
- measured held-out metrics;
- runtime profile;
- documentation;
- clean git state;
- explicit remaining limitations.

Start now with Phase 0 from `PLAN.md`: audit the repository, preserve/reproduce the current baseline, and create the requirements gap matrix before spending GPU time.
