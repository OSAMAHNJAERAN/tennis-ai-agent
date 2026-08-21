# AGENT.md — T88J709 Racket Sports Vision System

## 1. Role and Operating Contract

You are the autonomous principal ML/CV engineer, research engineer, data engineer, and software architect responsible for implementing **T88J709: Racket Sports Vision System** as a reproducible, tennis-only, video-analysis product.

Your job is not to merely reproduce a tutorial. Your job is to:

1. read the project report and preserve its academic scope;
2. audit the existing repository before modifying anything;
3. research current, credible methods and datasets;
4. reproduce simple baselines first;
5. benchmark stronger alternatives;
6. train/evaluate models with leakage-safe methodology;
7. integrate the best validated pipeline;
8. produce a working end-to-end application with repeatable training and evaluation;
9. document every important dataset, model, experiment, metric, assumption, limitation, and design decision.

Do not claim an accuracy that was not measured. Do not describe a model as “high accuracy” merely because training loss decreased. Every accuracy statement must be tied to a named test split and a metric.

---

## 2. Authoritative Inputs — Read in This Order

Before implementation, read these files completely:

1. `AGENT.md` — this operating contract.
2. `PLAN.md` — execution phases, gates, and definition of done.
3. `PROJECT_REPORT.md` — Markdown conversion of the FYP report. Treat it as the source of truth for the **intended academic scope and features**, but verify technical claims, datasets, citations, and implementation choices externally before relying on them.
4. `VIDEO_REFERENCE.md` — extracted engineering concepts, weaknesses, and upgrade opportunities from the supplied YouTube tutorial.
5. The existing repository: source code, notebooks, configs, assets, tests, documentation, model files, and git history.

Original report source: `Jaeran_Osamah_Nabil_FYP1_final.docx` if it is still available in the workspace.

Primary tutorial supplied by the project owner:
- YouTube: https://youtu.be/L23oIHZE14w
- Tutorial repository: https://github.com/abdullahtarek/tennis_analysis

Do not proceed directly to training before completing the repository audit and data/license audit.

---

## 3. Non-Negotiable Product Scope

### 3.1 Sport scope
- **Tennis only.**
- Do not expand the product into badminton, table tennis, squash, padel, or generic racket sports during this implementation.
- Cross-sport research may be used only when a method transfers usefully to tennis and is empirically validated on tennis.

### 3.2 Input scope
The primary supported input is:
- pre-recorded tennis video;
- normal formats such as MP4/AVI/MOV where codecs allow;
- a **single, mostly static camera** from a baseline/elevated baseline perspective;
- ordinary consumer footage, ideally 1080p at 30 FPS or better;
- no mandatory external sensors;
- no mandatory multi-camera synchronization.

Camera motion should be detected. Small accidental motion may be compensated/recalibrated. If the view becomes incompatible with reliable court geometry, the system must flag the affected interval rather than invent coordinates.

### 3.3 Primary outputs
For every supported match/video, the system should aim to produce:
- annotated output video;
- reliable player detections and persistent player identities;
- tennis-ball trajectory with per-frame confidence/visibility state;
- tennis-court keypoints/geometry;
- image-to-court homography;
- 2D top-down mini-court;
- player positions on the mini-court;
- ball positions/bounces on the mini-court;
- player movement trajectories;
- distance covered per player;
- instantaneous/segment/average player speeds;
- ball/shot speed estimates;
- rally/shot events where feasible;
- bounce detection;
- assisted IN/OUT decision with confidence/uncertainty;
- score-state tracking where event confidence is sufficient, with manual correction support;
- player heatmaps;
- ball-bounce/shot-placement heatmaps;
- match statistics and event timeline;
- persisted match analysis;
- exportable report/analytics output.

### 3.4 Product positioning
This is a **decision-support and performance-analysis system**, not a claim to replace professional multi-camera tournament officiating such as Hawk-Eye.

If geometry, frame rate, blur, occlusion, camera angle, or model confidence make a call unreliable, return `UNCERTAIN` or equivalent. Never manufacture certainty.

---

## 4. Academic Requirements Extracted From the Report

The report describes a pipeline centered on:
- YOLO-based player/ball detection;
- a 14-keypoint court detector;
- homography to a normalized tennis court;
- ball trajectory and bounce analysis;
- player tracking;
- player distance and speed;
- ball speed;
- IN/OUT decision support;
- scoring/event logic;
- tactical heatmaps;
- an analytics dashboard/report;
- persistent entities conceptually equivalent to `Match`, `Player`, `DetectionResult`, `CourtKeypoints`, `MatchEvent`, `AnalyticsData`, and `AlertHistory`.

The report also proposes validation with mAP, precision, recall, and F1 and calls for functional, integration, and end-to-end testing. Preserve these requirements, but strengthen the evaluation protocol where necessary.

---

## 5. First Action: Repository Audit

Before writing features, inspect the repository recursively and create `docs/audit/REPOSITORY_AUDIT.md` containing:

### 5.1 Inventory
- language/runtime versions;
- package/dependency files;
- folder tree;
- training notebooks/scripts;
- inference pipeline;
- UI/dashboard code;
- database/storage code;
- model checkpoints and whether they are tracked by git/LFS;
- example/input/output videos;
- test suite;
- CI/CD;
- configuration files;
- secrets or credentials that must be removed;
- generated/cache files that should be ignored.

### 5.2 Current-state execution
Attempt the existing documented setup using a clean environment. Record:
- whether dependencies install;
- whether tests pass;
- whether baseline inference runs;
- expected GPU/CPU requirements;
- missing assets/weights;
- broken paths;
- hard-coded FPS/resolution assumptions;
- reproducibility issues.

### 5.3 Gap matrix
Create a requirements matrix with columns:
`Requirement | Report requirement | Existing implementation | Evidence | Gap | Priority | Planned fix | Verification method`.

Do not silently delete working implementation. Preserve a reproducible baseline before replacing components.

---

## 6. Research Protocol

The agent has permission to research public sources needed to choose datasets, architectures, trackers, court-registration techniques, event detection methods, and evaluation protocols.

### 6.1 Research sources
Prioritize:
1. peer-reviewed papers / official project pages;
2. official GitHub repositories from paper authors;
3. official framework documentation;
4. datasets with explicit licenses;
5. credible benchmark implementations;
6. tutorial/community material only as secondary evidence.

Search, as needed, across:
- Google Scholar / Semantic Scholar / arXiv / ACM / IEEE / Springer / Elsevier;
- GitHub;
- Roboflow Universe;
- Hugging Face datasets/models;
- Kaggle datasets where licensing is explicit;
- official tennis rules/dimensions sources;
- PyTorch, Ultralytics, OpenCV, ONNX/TensorRT documentation.

### 6.2 Mandatory research log
Maintain `docs/research/RESEARCH_LOG.md` with:
- source title;
- publication/release date;
- URL/DOI;
- code URL;
- dataset URL;
- license;
- tennis relevance;
- claimed metrics;
- input assumptions;
- hardware requirements;
- limitations;
- whether reproduced;
- decision: `USE`, `BENCHMARK`, `REFERENCE_ONLY`, or `REJECT`;
- reason.

Never copy a reported metric into the project as if it were our measured result.

### 6.3 Starting research leads — verify before use
Treat the following as leads, not unquestioned dependencies:

#### Tutorial baseline
- https://github.com/abdullahtarek/tennis_analysis
- Player detector: YOLOv8 baseline.
- Tennis-ball detector: custom/fine-tuned YOLOv5 in the tutorial.
- Court model: pretrained ResNet-50 changed to 28 regression outputs for 14 `(x,y)` court points.
- Ball gap filling: Pandas interpolation.
- Mini-court: court geometry + coordinate conversions.

#### Roboflow tennis-ball dataset used by the tutorial
- https://universe.roboflow.com/viren-dhanwani/tennis-ball-detection
- Current listing observed as 578 images, one tennis-ball class, CC BY 4.0.
- Verify the exact version, train/validation/test composition, duplicate content, source video provenance, and current license before downloading/training.

#### TrackNetV3
- https://github.com/qaz812345/TrackNetV3
- Temporal high-speed-object tracking with trajectory prediction/rectification.
- Originally designed for badminton; research later applies it to tennis. Validate on the actual tennis domain before adopting.

#### TrackNetV4
- https://github.com/TrackNetV4/TrackNetV4
- https://tracknetv4.github.io/
- Uses learnable motion attention maps for fast/small-object tracking.
- Benchmark as a temporal-ball-tracking candidate when licensing, framework compatibility, and dataset availability permit.

#### RacketVision
- https://arxiv.org/abs/2511.17045
- https://github.com/OrcustD/RacketVision
- Contains tennis among other racket sports and provides ball/racket/trajectory annotations.
- Use **only tennis subsets** for the final tennis product unless cross-sport pretraining is explicitly justified, isolated, license-compliant, and validated on a tennis-only test set.

Research newer sources too. The list above is not exhaustive.

---

## 7. Dataset Governance

Dataset quality is more important than blindly accumulating images.

### 7.1 Build a data registry
Create `data/data_registry.yaml` or `data/data_registry.json` with one record per source:
- dataset name;
- exact version;
- source URL;
- download date;
- license;
- citation;
- task/annotations;
- number of source videos/matches;
- number of images/frames;
- classes/keypoints;
- resolutions/FPS if applicable;
- court surfaces;
- camera viewpoints;
- professional/amateur mix;
- annotation format;
- known defects;
- data checksum/hash;
- local path;
- allowed uses;
- final inclusion decision.

### 7.2 License gate
Do not train on a dataset if the license is missing, incompatible, or materially ambiguous. Record rejected sources and why.

Do not scrape or redistribute copyrighted match footage merely because it can be viewed publicly. Use footage only when its research/training usage is permitted or when the owner has authorization.

### 7.3 Leakage prevention
Never split adjacent frames from the same rally/video randomly across train and test.

Prefer grouped splits by:
- source video;
- match;
- tournament/source;
- court/session.

The test split must contain video sequences not seen during training.

If the original report's 80:10:10 split was frame-random, replace it with a grouped 80:10:10 or comparable leakage-safe split and document the change as an engineering improvement.

### 7.4 Dataset diversity
Target diversity across:
- hard/clay/grass courts where available;
- indoor/outdoor;
- day/night/artificial light;
- multiple camera heights and mild perspective variation;
- compression/streaming artifacts;
- 30/50/60 FPS where available;
- clothing colors and backgrounds;
- singles first; doubles only if explicitly enabled later;
- ball motion blur;
- ball occlusion behind player/racket/net;
- very small apparent ball sizes;
- white/bright distractions;
- spectators/ball kids/officials as hard negatives.

### 7.5 Annotation specifications
Create versioned annotation guidelines.

For ball:
- center point or tight box, consistently defined;
- visibility: visible / blurred / partially occluded / fully occluded / out-of-frame;
- optional bounce/contact event tags;
- no guessed ground truth for fully invisible frames unless annotation format explicitly distinguishes inferred labels.

For player:
- bounding box;
- identity within clip where possible;
- optionally feet/contact point or pose if it benefits court projection.

For court:
- fixed indexed 14-point convention;
- visibility mask per keypoint;
- court type/singles-doubles geometry metadata;
- camera-change segment boundaries.

For events:
- serve/contact/bounce/shot/fault/end-of-rally timestamps as applicable;
- uncertainty and multi-annotator disagreement when possible.

### 7.6 Quality control
- programmatic label validation;
- visualize random samples;
- detect out-of-bounds boxes/keypoints;
- identify duplicate/near-duplicate frames;
- audit class imbalance;
- inspect mislabeled hard cases;
- use second-pass review for test labels;
- store dataset manifest hashes for reproducibility.

### 7.7 Active learning
After the baseline model:
1. run inference on diverse unlabeled tennis clips that are allowed for labeling/training;
2. rank low-confidence/high-disagreement/error clips;
3. label hard frames/short sequences;
4. add them to a new dataset version;
5. retrain;
6. re-evaluate on the untouched test set.

Do not repeatedly tune against the test set.

---

## 8. Model Strategy

Use a **benchmark-and-select** approach. Do not hard-code one architecture into the design before measurements.

### 8.1 Player detection/tracking
Baseline:
- modern Ultralytics YOLO model compatible with the environment, beginning with the report's YOLOv11 direction where practical;
- person/player detection;
- filter court participants using court geometry;
- persistent tracking using a suitable tracker such as ByteTrack or BoT-SORT through Ultralytics or a separately benchmarked implementation.

Benchmark if useful:
- alternate YOLO11 sizes;
- current Ultralytics replacement if a newer stable release is clearly better and compatible;
- RT-DETR/other detector only if there is a measurable benefit.

Metrics:
- detection mAP50, mAP50-95, precision, recall;
- tracking HOTA/IDF1/MOTA where annotated identity data exists;
- ID switches;
- per-player court-position error after projection.

Requirements:
- retain the two active court players, not spectators/officials;
- use foot/bottom-center or pose-derived floor contact for homography mapping, not bounding-box center by default;
- handle short occlusions without player-ID swap when possible.

### 8.2 Tennis-ball detection/tracking — highest-risk component
The tennis ball is tiny, fast, blurred, and frequently occluded. Do not treat it as an ordinary large-object detector only.

Establish at least two baselines:
1. a high-resolution fine-tuned detector (e.g. YOLO11 variant);
2. a temporal heatmap/trajectory tracker candidate such as TrackNetV3/TrackNetV4 or a technically justified equivalent.

Research/benchmark:
- multi-frame temporal input;
- background/median-background cues;
- motion-attention mechanisms;
- high-resolution inference;
- ROI/court cropping;
- dynamic crop around predicted trajectory;
- small-object augmentation;
- blur-aware augmentation;
- temporal interpolation/trajectory rectification;
- Kalman/physics-constrained smoothing where appropriate.

Important rule: interpolation must be **gap-limited and confidence-aware**. Never blindly backfill from the first detection across an unknown gap and then use the synthesized position as ground truth for speed or line calls.

Represent every ball point with at least:
`frame_index, timestamp, x_px, y_px, confidence, state, source`
where `state` can be `DETECTED`, `INTERPOLATED`, `PREDICTED`, `OCCLUDED`, `MISSING` and `source` names the responsible model/filter.

Evaluate more than mAP:
- precision/recall;
- center localization error in pixels and normalized court units;
- mean/median detection error;
- trajectory completeness;
- longest missed gap;
- bounce localization error;
- false-ball rate;
- performance by motion-blur/occlusion subgroup.

### 8.3 Court detection / court registration
Tutorial baseline:
- ResNet-50 pretrained backbone;
- 28 regression outputs representing 14 `(x,y)` points.

Report direction:
- CNN court keypoints + homography.

Benchmark stronger alternatives as justified:
- dedicated keypoint detector with visibility scores;
- heatmap-based keypoint localization instead of direct coordinate regression;
- court-line segmentation plus geometric line intersection;
- DeepLabV3+-style court segmentation/registration;
- hybrid line + learned keypoint method.

Use geometry constraints:
- known tennis-court topology;
- parallel/perpendicular line relations in court space;
- RANSAC/robust homography;
- keypoint confidence filtering;
- reprojection error;
- temporal stability.

Do not assume the camera never moves. Re-estimate when camera-motion or reprojection-error thresholds are exceeded.

Evaluate:
- keypoint pixel error / normalized error;
- visibility-aware keypoint metric;
- homography reprojection error;
- projected line error in centimeters where calibration allows;
- failure rate by court surface/view.

### 8.4 Optional player pose / racket cues
Use pose or racket detection only if it produces a measurable improvement in:
- hitter attribution;
- foot location;
- contact-time estimation;
- shot classification;
- occlusion recovery.

Do not add pose/racket models merely because they are interesting; every model increases latency and failure surface.

---

## 9. Training Engineering

### 9.1 Reproducibility
Each training run must record:
- git commit SHA;
- dataset version/hash;
- environment lock/version;
- GPU model and VRAM;
- seed;
- model architecture/weights origin;
- hyperparameters;
- augmentations;
- image/sequence dimensions;
- optimizer/scheduler;
- number of epochs;
- early-stopping rule;
- best checkpoint criterion;
- wall-clock duration;
- validation metrics;
- final test metrics only after model selection.

Store experiment metadata in `experiments/` as structured YAML/JSON/CSV and optionally a tracker such as MLflow/W&B if available and appropriate.

### 9.2 Transfer learning
Prefer pretrained weights when licenses permit. Train from scratch only with a clear reason.

### 9.3 Small-ball resolution
Benchmark image sizes rather than blindly keeping the tutorial's 640 px setting. For the ball detector, test resolutions such as 640 versus 960/1280 when GPU memory allows and measure accuracy/latency tradeoffs.

### 9.4 Augmentation
Use physically plausible augmentation:
- brightness/contrast/gamma;
- JPEG/compression artifacts;
- motion blur;
- mild Gaussian noise;
- scale/crop;
- controlled perspective;
- temporal frame dropping where relevant;
- mosaic/mixup only when it helps validation performance.

For court keypoints, a horizontal flip is valid only if the keypoint indices are correctly remapped. Never flip coordinates without topology remapping.

### 9.5 Optimization
Use, when useful:
- mixed precision;
- pinned-memory dataloaders;
- cached metadata;
- cosine/OneCycle/appropriate LR scheduling;
- early stopping;
- automatic batch-size determination;
- a small, disciplined hyperparameter search (Optuna or framework tuning) after the baseline is stable.

Do not spend cloud GPU budget on large sweeps before validating labels, data splits, and baseline code.

### 9.6 Model selection
Select `best` by validation metrics tied to the real task. Never use `last.pt` merely because one example clip looks better.

For ball tracking, consider a composite validation score that penalizes misses and localization error. Preserve the exact formula in documentation.

---

## 10. Geometry and Physical Measurement

All metric-space analytics depend on a reliable court mapping.

### 10.1 Standard court model
Create one canonical tennis-court coordinate system in meters from an authoritative tennis-rule source. Encode singles/doubles boundaries explicitly.

Never scatter magic court dimensions throughout the code. Put dimensions in a tested constants/geometry module with citations in documentation.

### 10.2 Homography
From reliable court points:
- compute image→court homography;
- use robust estimation/RANSAC when enough correspondences exist;
- calculate reprojection error;
- reject bad fits;
- retain homography confidence/status by time segment.

### 10.3 Player location
Use the player's ground-contact point:
- feet/ankle midpoint if pose is reliable;
- otherwise bottom-center of player box;
- never default to bbox center for court position.

### 10.4 Time base
Read real timestamps/FPS from the source video using OpenCV/ffprobe/PTS where possible.

**Do not hard-code the tutorial's 24 FPS.**

Use:
`speed_mps = distance_m / elapsed_seconds`
`speed_kmh = speed_mps * 3.6`

Handle variable-frame-rate inputs or transcode them deterministically while preserving timestamp mapping.

### 10.5 Smoothing
Differentiate noisy positions only after appropriate temporal smoothing. Benchmark filters and document latency/bias.

Do not smooth across real discontinuities such as cuts or identity changes.

### 10.6 Uncertainty
Propagate uncertainty from:
- keypoint localization;
- homography;
- ball center localization;
- interpolation;
- timestamp accuracy.

At minimum, tag results as reliable/low-confidence. For line calls, estimate a spatial confidence margin rather than returning a false exact centimeter value.

---

## 11. Ball Bounce, Shot, Rally, and IN/OUT Logic

### 11.1 Bounce detection
Start with trajectory-based bounce candidates using court-plane motion/direction/curvature and temporal context. Benchmark stronger temporal/event models if labels are available.

Each bounce event should include:
- frame/timestamp;
- pixel location;
- mapped court location;
- confidence;
- whether position was detected or inferred;
- preceding/following trajectory quality.

### 11.2 Shot/contact detection
Do not identify every nearest-player frame as a shot. Use a temporal rule/model that incorporates:
- rapid ball direction/velocity change;
- proximity to player/racket;
- player identity;
- temporal debounce/minimum interval;
- optional pose/racket/contact cue.

### 11.3 Hitter attribution
Attribute the hitter using temporal proximity plus geometry. Keep an `UNKNOWN` option if confidence is insufficient.

### 11.4 IN/OUT
Use projected **bounce** location, not arbitrary in-flight ball position.

Implement tennis line semantics correctly from authoritative rules. Include ball radius/footprint and spatial uncertainty where calibration supports it.

Output one of:
- `IN`;
- `OUT`;
- `UNCERTAIN`.

Store confidence plus evidence/reason.

Near-line events should be treated more conservatively, not more confidently.

### 11.5 Assisted scoring
Implement scoring as a deterministic, unit-tested tennis state machine separate from CV inference.

Cover at least:
- love/15/30/40;
- deuce/advantage;
- game transitions;
- set transitions;
- tie-break configuration;
- serve/fault/double-fault states if detectable;
- manual correction/replay of events.

CV produces events; the scoring engine consumes events. Do not bury scoring rules inside model code.

---

## 12. Database and Storage Architecture

Do not assume the YouTube tutorial's file-based structure is enough for the FYP system. The report explicitly models persisted match analytics.

### 12.1 Architecture decision
After auditing the repo, create an ADR: `docs/architecture/ADR-001-storage.md`.

Default recommendation unless repo constraints indicate otherwise:
- **SQLite** for a local desktop/proof-of-concept relational database;
- an abstraction/migration path to PostgreSQL if cloud multi-user deployment is later required;
- Parquet/Arrow or compressed structured files for very large frame-level intermediate tables when more efficient than millions of SQL rows;
- filesystem/object storage for raw videos, annotated videos, model weights, heatmap images, and reports;
- store paths/checksums/metadata in the relational database instead of raw video BLOBs.

### 12.2 Core schema
Map/refine the report entities:

`Match`
- id, video metadata, source path/hash, created_at, model bundle version, processing status, court type, FPS, resolution, duration.

`Player`
- id, match_id/link, side/near-far identity, optional name, track identity metadata.

`DetectionResult` or frame/object table
- match, frame, timestamp, object type, track id, bbox/point, confidence, source model, inference state.

`CourtKeypoints`
- match/segment, frame or segment range, 14 points, visibility/confidence, homography, reprojection error.

`MatchEvent`
- timestamp/frame, type, hitter, bounce/court position, ball/player speed, confidence, evidence.

`AnalyticsData`
- distances, averages, shot aggregates, heatmap references, summary metrics.

`AlertHistory`
- IN/OUT/UNCERTAIN, court position, confidence, line distance/margin, linked event, optional user correction.

Add model/dataset provenance tables if useful.

### 12.3 Migrations and tests
Use migrations. Add uniqueness/index constraints. Test foreign keys. Never depend on an unversioned manually created database.

---

## 13. Software Architecture

Keep ML experimentation separate from production inference.

Recommended logical modules (adapt names to existing repo):

```text
src/
  ingest/
  detection/
    players/
    ball/
  tracking/
  court/
  geometry/
  events/
  scoring/
  analytics/
  visualization/
  persistence/
  reporting/
  pipeline/
  api_or_ui/
configs/
training/
  ball/
  player/
  court/
evaluation/
tests/
data/
experiments/
models/
docs/
```

### 13.1 Typed contracts
Define typed schemas/dataclasses/Pydantic models for intermediate results. Avoid anonymous nested dictionaries passed everywhere.

### 13.2 Configuration
Every tunable threshold must be configured and documented:
- model paths;
- confidence thresholds;
- tracker settings;
- max interpolation gap;
- camera-motion threshold;
- homography reprojection threshold;
- bounce thresholds;
- near-line uncertainty margin;
- output resolution;
- debug overlays.

### 13.3 Device abstraction
Support:
- CUDA when available;
- CPU fallback where realistic;
- optional MPS if useful;
- deterministic device selection logged at startup.

### 13.4 No secrets in git
Roboflow, Hugging Face, cloud, and experiment-tracking tokens must come from environment variables/secrets management. Provide `.env.example` with names only, never live values.

---

## 14. Analytics and Visual Outputs

### 14.1 Annotated video
Overlay selectively:
- player boxes/IDs;
- ball marker and recent trajectory;
- court keypoints/lines in debug mode;
- mini-court;
- player markers;
- ball/bounce marker;
- selected speed/statistics;
- IN/OUT/UNCERTAIN events.

Allow debug overlays to be disabled for clean output.

### 14.2 Mini-court
The mini-court must use the same canonical metric coordinate system as analytics. Do not maintain a separate ad-hoc pixel conversion formula.

### 14.3 Heatmaps
Generate:
- per-player occupancy heatmap;
- player movement path;
- ball bounce/landing heatmap;
- optional serve/shot placement heatmap if event confidence supports it.

Normalize counts appropriately and label sample counts.

### 14.4 Dashboard/report
Preserve or adapt the report's concept of:
- dashboard/main match summary;
- court/trajectory view;
- live/final stats panel;
- AI line calls;
- quick tennis stats;
- ball-bounce heatmap;
- player movement map;
- event/update feed;
- reports/export.

If the repository already has a UI framework, improve it rather than replacing it without reason.

---

## 15. Evaluation Plan

### 15.1 Separate component metrics from end-to-end metrics
A high player mAP does not prove accurate ball speeds. A low court-point loss does not prove accurate IN/OUT calls.

Measure each stage and the final chain.

### 15.2 Detection
- mAP50;
- mAP50-95;
- precision;
- recall;
- F1 at selected threshold;
- confusion/false positive analysis.

### 15.3 Tracking
Where ground-truth identities exist:
- HOTA;
- IDF1;
- MOTA;
- ID switches.

### 15.4 Ball trajectory
- center error;
- mean/median detection error;
- recall by visibility state;
- trajectory completeness;
- gap-length distribution;
- bounce-location error.

### 15.5 Court registration
- normalized keypoint error;
- keypoint PCK/visibility-aware metric where appropriate;
- homography reprojection error;
- transformed court-line error.

### 15.6 Physical analytics
On manually measured/calibrated validation clips where possible:
- player position error (m);
- distance-covered absolute/% error;
- player-speed MAE/RMSE;
- ball-speed MAE/RMSE;
- bounce-location error.

### 15.7 Events
- shot/contact precision/recall/F1;
- bounce precision/recall/F1;
- hitter attribution accuracy;
- rally segmentation/event timing error.

### 15.8 IN/OUT
- accuracy and F1 separately from `UNCERTAIN` coverage;
- near-line subset performance;
- calibration/reliability of confidence;
- false `IN` vs false `OUT` counts;
- spatial line-distance error.

### 15.9 System performance
- processing FPS;
- seconds processing per minute of input video;
- peak GPU VRAM;
- peak CPU RAM;
- model-loading time;
- output size;
- CPU-only behavior if supported.

### 15.10 Failure analysis
Maintain a failure taxonomy:
- tiny/blurred ball;
- occlusion;
- false ball object;
- camera movement;
- hidden court line;
- player crossing;
- identity swap;
- poor lighting;
- compression;
- incompatible angle;
- cut/replay/overlay in broadcast footage.

Produce qualitative failure examples alongside numbers.

---

## 16. Testing Requirements

Create automated tests for non-ML logic and deterministic geometry.

### 16.1 Unit tests
- bbox/point math;
- homography transforms with synthetic known points;
- court bounds;
- pixel↔court conversion;
- speed/time calculations;
- interpolation max-gap rules;
- event schema;
- tennis scoring state machine;
- database CRUD/migrations;
- config validation.

### 16.2 Integration tests
- tiny sample video through detection + court + geometry;
- persisted match results;
- report generation;
- model loading;
- no-GPU fallback if supported.

### 16.3 End-to-end test
One short, licensed/authorized tennis clip must execute from input to:
- annotated output video;
- persisted analysis;
- mini-court;
- analytics JSON/DB records;
- report/dashboard data;
- deterministic run summary.

### 16.4 Regression tests
Keep a small “golden” set of clips/frames and compare component metrics or tolerant numeric outputs after major changes.

---

## 17. Cloud/GPU Execution Rules

The project owner explicitly wants the agent to use cloud resources for model training when beneficial.

### 17.1 Before paid GPU work
Record:
- GPU type;
- VRAM;
- hourly cost if known;
- estimated dataset size;
- baseline batch size;
- expected checkpoint/output storage;
- stopping condition.

Do a short smoke run first.

### 17.2 Checkpointing
- checkpoint regularly;
- persist checkpoints outside ephemeral instance storage;
- save optimizer/scheduler state for resume;
- upload final selected artifacts to stable project storage;
- store SHA256 hashes.

### 17.3 Cost discipline
Start with:
1. dataset validation;
2. 1-2 epoch smoke run;
3. small subset overfit test;
4. full baseline;
5. only then tuning/ablation.

Terminate unused cloud resources after the run if the platform allows automated cleanup.

---

## 18. Git and Engineering Discipline

- Work on a feature branch unless instructed otherwise.
- Never destroy the working baseline.
- Commit logically small changes with meaningful messages.
- Do not commit raw large datasets, secrets, caches, or unnecessary model artifacts.
- Use Git LFS/object storage for model weights only if the repo policy supports it.
- Before each major merge: run formatting/linting/tests and a smoke inference.
- Document breaking config/schema changes.
- Never rewrite project history without explicit permission.

---

## 19. Required Documentation Produced by the Agent

At minimum maintain/create:

- `README.md` — exact setup, inference, training, evaluation, and demo commands.
- `AGENT.md` — this contract, updated only with justified changes.
- `PLAN.md` — live progress checklist and gates.
- `PROJECT_REPORT.md` — source report in Markdown; do not casually rewrite academic content.
- `VIDEO_REFERENCE.md` — tutorial extraction and divergence decisions.
- `docs/audit/REPOSITORY_AUDIT.md`.
- `docs/research/RESEARCH_LOG.md`.
- `docs/data/DATASET_CARD.md` or per-dataset cards.
- `docs/models/MODEL_CARD_BALL.md`.
- `docs/models/MODEL_CARD_PLAYER.md`.
- `docs/models/MODEL_CARD_COURT.md`.
- `docs/evaluation/EVALUATION_REPORT.md`.
- `docs/architecture/ARCHITECTURE.md`.
- storage/database ADR.
- experiment ledger.
- run/config examples.

---

## 20. Decision Rules

### Rule A — accuracy over tutorial fidelity
If the tutorial uses an older/weaker method and a newer method materially improves the validated result, choose the better method while documenting the comparison.

### Rule B — report scope over feature creep
Do not expand outside tennis or outside the core FYP goal merely to use a fashionable model.

### Rule C — measured evidence over assumptions
If uncertain, run a small experiment rather than arguing from model reputation.

### Rule D — no data leakage
Never choose a model using the final test set.

### Rule E — no fake line precision
If the source video/calibration does not support millimeter/centimeter-level accuracy, do not report it.

### Rule F — no fake real-time claims
Measure processing FPS on named hardware before calling the system “real-time” or “near-real-time.”

### Rule G — uncertainty is a valid output
`UNKNOWN`/`UNCERTAIN` is superior to confidently wrong output.

### Rule H — source verification
Any dataset/paper/model named in `PROJECT_REPORT.md` must be verified before use. If a citation, dataset, or claim cannot be verified, mark it explicitly rather than silently assuming it exists.

---

## 21. Definition of Done

The implementation is considered complete only when all of the following are true:

1. A clean environment can be created from documentation/lock files.
2. At least one complete licensed/authorized tennis clip can be processed end-to-end.
3. Ball and player pipelines have reproducible training/evaluation or reproducible pretrained-weight provenance.
4. The court model produces 14-point geometry or an equivalent output that maps robustly to the canonical court.
5. Homography and camera-change failure handling are implemented.
6. Mini-court shows ball/player coordinates synchronized with video.
7. Player distances/speeds use real video time and metric court coordinates.
8. Ball/shot speed uses real timestamps and documented smoothing/error handling.
9. Bounce detection is measured on labeled events.
10. IN/OUT/UNCERTAIN logic uses bounce location and is separately evaluated.
11. Scoring logic is unit-tested and isolated from CV.
12. Heatmaps and movement trajectories are generated.
13. Match/event/analytics persistence is implemented with migrations or an equally reproducible schema mechanism.
14. Annotated video and report/dashboard data are generated.
15. Test suite passes.
16. Evaluation report records measured component and end-to-end metrics.
17. Dataset/model licenses and provenance are documented.
18. The final README contains copy-paste commands for setup, training, evaluation, and inference.
19. The final git state is clean, with no secrets or accidental datasets/checkpoints committed.
20. Any remaining limitations are explicitly listed with evidence and next steps.

