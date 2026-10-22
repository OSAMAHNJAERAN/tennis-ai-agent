# Master Execution Prompt: YOLO26 Tennis Video Analytics Platform

Prepared on 8 October 2026. This text is ready to be assigned to an AI agent later as an implementation goal. The numbers in the acceptance gates are proposed engineering targets, not results already achieved in the project. Preparing this prompt does not start training or publish the website.

**START OF PROMPT — Copy from here, including the references.**

**1. Your role and the outcome I want**

Act as the person responsible for the complete implementation, bringing together computer vision engineering, model training, data engineering, tennis analysis, frontend and backend development, and user experience. Develop the existing project into a professional website for analyzing **tennis matches only** from video, using **Ultralytics YOLO26** as the foundation of the vision models we will use in production.

I want a working experience: a user opens the website, sees analyzed tennis examples, uploads one or several videos, follows processing progress, then watches the video with player, ball, and court tracking and explores the analysis, score, and report. I also want live video support in a separate phase, after the system has demonstrated its ability to analyze long recorded matches.

The goal is analysis quality that can be verified. Do not promise 100% accuracy, and do not treat successful training or attractive bounding boxes as evidence that speeds or points are correct. Every result must have supporting evidence, a quality measurement, and a confidence level or an unavailable state when the footage does not support the inference.

Once this goal is activated, continue until the required deliverables have been implemented and verified. A plan, downloaded models, a visual interface, and processing a short video are all intermediate steps. Final success requires a connected upload, analysis, and report experience; independent accuracy tests; support for long durations; and evidence of actual live video operation under stated conditions.

**2. Build on the existing project and apply this new scope**

**Cloud continuation, 22 October 2026:** use the live `codex/yolo26-platform` branch of `OSAMAHNJAERAN/tennis-ai-agent` and the existing cloud checkout. The Windows paths below are historical. Verify GitHub and read the canonical handoff before continuing; Section 34 governs checkpointing and cross-account continuity.

The current project is at `C:\Semester 9\FYP2\tennis_ai_agent_bundle`. Begin by inspecting its state when implementation starts, because files may change after this document is written. Read `AGENT.md`, `README.md`, `PROJECT_REPORT.md`, and the relevant parts of `PLAN.md`, then:

- `docs/PRODUCT_WORKFLOW.md` and `docs/PRODUCT_DASHBOARD.md`.
- `docs/plans/TENNIS_EXECUTION_PROGRESS.md`.
- `docs/plans/TENNIS_WEBSITE_EXECUTION_PROMPT.md` to understand previous decisions.
- The two research notes, `docs/research/YOLO26_OFFICIAL_RESEARCH_2026-10-08.md` and `docs/research/TENNIS_ANALYTICS_EVIDENCE_2026-10-08.md`.
- The actual code and configuration, rather than relying only on descriptions in older documents.

These new instructions supersede any earlier request that makes comparing model families a project objective or leaves scoring, winner prediction, and live video outside the final scope. Retain the requirements for scientific integrity, data rights, reproducibility, and evaluation. Do not make comparisons or model names part of the user interface.

There is a working foundation to use: a React/TypeScript/Vite frontend, a FastAPI backend, video uploads, processing jobs persisted in SQLite, a separate processing worker, video playback and timing, and HTML/JSON/CSV reports. In particular, inspect `src/api/`, `src/pipeline/phase6_pipeline.py`, `src/scoring/`, `frontend/src/pages/Upload.tsx`, `frontend/src/components/VideoEvidencePlayer.tsx`, and the replay and court tools.

When this document was prepared, the historically active models were YOLO11 for players and body pose, WASB for the ball, and a different model for court landmarks. This is not an already qualified YOLO26 system. Do not rename files or change interface text and call that a model upgrade. Implement a documented, reversible migration to the selected system.

There are many local changes that have not been saved in a commit; preserve them. Record the current baseline and inspect dependencies, paths, and reports before replacing any component. Do not delete data, experiments, or models simply because they are old. Existing file size and duration limits are not evidence that complete matches have passed validation, and a manual scoring page is not live video analysis.

Initially, produce a concise table covering the required feature, what currently works, operating evidence, the gap, implementation, and verification method. Identify the single production pipeline that will serve the website.

**3. The model system: three clear roles within YOLO26**

Choose the following system as a concrete implementation starting point, without creating a project to compare YOLO26 with YOLO11, YOLOv8, TrackNet, or other families:

| Role | Initially selected model | Model responsibility |
| --- | --- | --- |
| Ball | YOLO26m Detect, starting from `yolo26m.pt` | Accurately detect the visible tennis ball and return its candidates, bounding box, and center in the original image. |
| Players | YOLO26s Pose, starting from `yolo26s-pose.pt` | Provide player boxes and human body keypoints, including both ankles, from one model. |
| Court | YOLO26s Pose with a custom head for the court landmark set | Detect one court and its fourteen landmarks in a fixed semantic order, using `kpt_shape: [14, 3]`. |

Use no more than these three specialized vision models; do not add a separate person detector if the pose model supplies the required boxes. Applying the same weights to an additional crop or tile does not make it a new model. Start from official weights and train the ball and court landmarks on tennis data. Improve the player model on annotated tennis data when evaluation shows a need; do not damage a usable model through training performed merely for its own sake on weak data.

Tracking, court calibration, distance calculations, event inference, scoring rules, region coloring, reporting, and statistical win prediction are software modules around the models. Do not assume YOLO provides them out of the box. Do not silently add a ReID model, an OCR model, or a fourth deep event model. The scoreboard can be read manually, or an optional classical OCR tool can be used with verification; do not make it a requirement for the platform to function.

These model scales are initial engineering choices. A scale may be adjusted within YOLO26 when a documented memory limitation arises or an accuracy gate is not passed, with one clear reason and a limited experiment. Do not run a search grid across every scale. If a requirement remains unattainable within the three models, first improve the data, processing, and tracking, then document the obstacle instead of silently changing the scope.

YOLO26 supports detection and pose estimation, and a Pose model provides boxes and body keypoints together. Custom court keypoints require appropriate data and a suitable head. Consult the [YOLO26 documentation](https://docs.ultralytics.com/models/yolo26/), [Pose task](https://docs.ultralytics.com/tasks/pose/), and [custom keypoint dataset format](https://docs.ultralytics.com/datasets/pose/).

**4. The camera view we start with**

The primary view is the **Elevated End-Court / Baseline Broadcast View**: an elevated camera behind one of the baselines that sees almost the entire court, the near player, and the far player, and remains stationary or nearly stationary during rallies. This is the familiar tennis broadcast view intended here; the camera does not have to be vertically overhead.

Start with singles matches from this view only. Vary courts, surfaces, lighting, clothing, and tournaments within the same view. Do not include side angles, arbitrary spectator-stand footage, shaky handheld cameras, or close-ups in the initial qualification scope. Doubles and familiar side angles are later extensions with separate data and acceptance tests.

For a complete file, detect camera cuts, replays, advertisements, close-ups, artificial slow motion, and intervals between points. Analyze valid intervals from the primary view and mark other intervals as outside the analysis. Do not reject the entire file because of a short unsuitable shot, and do not invent measurements during it. Recalibrate after a camera change or zoom.

Recommend 1080p and 50/60 FPS to users for fast ball motion when available. You may support 30 FPS for some analyses after measurement, but do not promise the same ball and event accuracy. The system must retain PTS and the actual frame rate; it must not treat every video as 30 FPS or interpret slow-motion footage as normal match timing.

**5. What we take from the two supplied references**

Use the two references to inform feature design and the presentation of evidence:

- The first post describes combining court geometry, identity tracking, motion constraints, occlusion handling, ball and bounce tracking, audio cues, and physical inference. Take from it the need to combine signals and validate them for tennis. [Computer Vision Store post](https://www.linkedin.com/posts/computervision-sportsanalytics-multiobjecttracking-ugcPost-7497560583752396801-CmPw/).
- The second post presents a timestamped event log, rally segmentation, player points on a mini-court, and a distinction between ground-plane measurements and airborne-ball estimates. It concerns pickleball; transfer only the presentation and verification ideas, and apply tennis rules. Its time and cost figures are not our website's performance. [Luís Condados post](https://www.linkedin.com/posts/luis-condados_computervision-sportsanalytics-pickleball-ugcPost-7505970975868534784-mhkG/).

Also add the proportion of actual playing time, the duration of each rally, and the counts of valid shots and bounces, excluding warm-up activity, pre-serve bounces, and balls after a point has ended. Do not turn a count of every sound or every bounce into a count of match points.

**6. Data to collect before training**

Build a tennis dataset from matches and footage authorized for training and evaluation. Select videos with the familiar broadcast style and specified view; a video's popularity or availability on a channel does not by itself grant training or republication rights.

Create a manifest for every source: URL or path, owner or publisher, usage rights, tournament/match/camera session, surface, view, resolution, frame rate, clip durations, checksum, and acquisition date. Record rights for weights, data, and published examples separately. Document the appropriate Ultralytics licensing route for the product according to the [official licensing page](https://www.ultralytics.com/license).

An initial collection plan, adjustable after the audit: approximately 24–36 independent matches or sessions from the primary view, covering at least 6 different courts or broadcast contexts. The aim is independent diversity; many hours from one match cannot replace it. Start with a small, well-annotated sample to confirm that the task is viable, then expand in response to errors in development evaluation.

Reserve splits by match and camera session, for example 60% for training, 20% for validation and threshold calibration, and 20% for a frozen final test. Separate statistical calibration sessions within the validation set when necessary. Overlapping clips, replays, or neighboring frames from the same match must not appear in different splits. Reserve a harder test from new courts or contexts within the supported view, and do not use it to select settings.

Do not reuse project examples you have already inspected and repeatedly optimized the system on as a new final test. Check duplicates, near-duplicates, and leakage, and record the split before training. Metrics published in the [original TrackNet paper](https://arxiv.org/abs/1907.03698) are not YOLO26 or project metrics; use the paper to understand the challenges of a tiny ball, occlusion, and generalization across matches.

Datasets such as CalTennis may be useful for research involving independent human-player tests if their rights, angles, and synchronization are suitable; do not assume they contain ball-speed ground truth or match the broadcast training scope. Review the [CalTennis project](https://ilonadem.github.io/caltennis-website/) and the terms of the specific dataset version before using it.

**7. Annotations and ground truth**

Write an annotation guide that precisely defines what must be labeled:

- Ball: its visible box and center; visible / occluded / absent / ambiguous states; blur severity; and near- or far-court location. Do not label an imagined location while the ball is hidden as a real observation.
- Players: the two actual players, their boxes, available body keypoints, and a stable player A/B identifier across changes of ends. Include bending, jumping, overlap, and partial occlusion.
- Court: a fixed order of 14 known intersections, each point's meaning and visibility state, axis directions, and court dimensions. Use the project's ordering if correct and publish a diagram explaining it.
- Events: point start, first and second serves, racket contact, bounce, rally end, and outcome, with timestamp, frame number, and uncertainty range.
- Score: the server, initial score, match format, point winner when independent evidence is available, and classification of the reason for losing the point when visible.
- Negative test examples: advertisements, logos, white lines, light-colored clothing, spare balls, ball kids, officials, rackets, and the ball before the point starts.

Initially plan for tens of thousands of selected ball frames, thousands of player and court frames, and continuous temporal clips for events and tracking. Record the actual quantity; do not make reaching an image count a substitute for quality and diversity. The test sample must also include intervals without a valid ball and events occurring close together.

Annotate ball and event clips at their original frame rate, especially around shots and bounces. Low-rate sampling may be suitable for collecting court keypoints, but it is insufficient for testing a fast ball. Use independent reviewers for events, line-call footage, and a sample of frames, and record their disagreements and the adjudication procedure. Do not treat model output as test ground truth; review semi-automated proposals before accepting them.

**8. The defined training protocol**

Pin suitable Ultralytics, PyTorch, and CUDA versions, the code revision and data configuration, official weight names, and a hash for each artifact. Check the GPU, memory, and disk capacity before starting a long training run. Do not replace the project's working environment without a recovery path.

Start with proposed operating settings: `imgsz=1280` for the ball, `imgsz=960` for players, and `imgsz=1280` for court calibration. Original-resolution crops may matter more than enlarging the entire image. Inspect the ball's pixel size after preprocessing and adjust the batch to fit memory using actual measurements. These are starting settings, not promises of accuracy or execution speed.

For each model, first run a small learning test on a limited annotated set to verify labels, loss, keypoints, and transforms. Once that succeeds, begin controlled fine-tuning, for example with a ceiling of 100–150 epochs, early stopping, monitoring of useful metrics, resume support, and checkpoint recording. If the signal is weak or the data is incorrect, stop the run and fix the cause. Do not repeat long training runs with random settings.

For the court model, adapt the head to 14 keypoints, transfer only compatible weights from the official model, and document what was transferred and what was randomly initialized. Do not claim that human-keypoint weights are trained on court landmarks. Configure `flip_idx` according to court landmark semantics, or disable reflection if it is incorrect. Apply geometric transformations to all boxes and keypoints without losing their correspondence.

For augmentation, use moderate variations in lighting, compression, blur, and scale that represent real video, and verify that the tiny ball does not disappear because of cropping, downscaling, or mosaic. Preserve the meaning of the ground plane, lines, and court keypoints. Inspect the quality of a sample of transformed images before training.

Use only training arguments supported by the official version. Details in the research recipe are not always valid arguments to `model.train()`. Do not invent ready-made weights such as `yolo26m-p2.pt`; if a P2 architecture is needed in the future, verify its support and that it is an architecture YAML requiring appropriate training, and record a limited decision within the selected model. Consult the [official training documentation](https://docs.ultralytics.com/modes/train/) and [YOLO26 training recipe](https://docs.ultralytics.com/guides/yolo26-training-recipe/).

Select a checkpoint based on independent validation on the custom data, rather than only the best loss or last epoch. Record the reason for selecting it. Fix errors through a limited cycle: error analysis, adding relevant training data, retraining, then validation. Do not tune thresholds on the final test; if used for selection, it becomes development data and a new test must be reserved.

**9. Player detection and identity tracking**

Use the player model's boxes and keypoints with an explicitly selected tracker such as BoT-SORT configured with `with_reid: false`, without loading an additional identity model. Pin the tracker YAML and its settings; do not rely on a default tracker that may change with the library version. [Tracking documentation](https://docs.ultralytics.com/modes/track/).

Separate the local `track_id` from the stable match `player_id`. Player A is not always the near player, and player B is not always the far player. Handle changes of ends, interruptions in footage, officials and ball kids entering, a player leaving the image, and overlap. Use the court, player location, continuity, and color as supporting evidence, and request correction when identity is ambiguous instead of confidently swapping names.

Every video, processing job, and live session must have independent state. Do not share tracker states across multiple files. At a cut, reset local state and reconnect match identity using sufficient evidence. Across chunk boundaries, preserve state, timing, and event identifiers so counts are not duplicated and identities do not switch.

Do not use the center of a body box as a ground point. Use the foot point in contact with the ground when visible; the bottom-center of the box may be used as a clearly identified estimate when needed, but do not treat jumping or an occluded foot as an accurate ground measurement. Keep ground-point and identity quality separate from person-detection confidence.

**10. Tracking the small, fast ball**

Run the ball model on the image or court area, using original-resolution crops/tiles when necessary and searching a wider area again when the ball is lost. The same weights may be reused on several crops. Document the exact transformation from tile coordinates to the original image and the removal of duplicate candidates. [Official sliced-inference guide](https://docs.ultralytics.com/guides/sahi-tiled-inference/).

Associate candidates over time using actual timing, motion limits, trajectory continuity, court location, and shot and bounce cues. Use a suitable temporal filter with explicit transitions at impacts; do not impose one smooth trajectory across a shot or bounce, and do not copy a person tracker for the ball without verification.

Distinguish every position: `observed` for an evidence-supported detection, `predicted` for a filter prediction, `interpolated` for filling a short replay gap, and `unknown` when evidence is absent. Define the maximum gap duration in seconds and qualify it through testing. Do not fill a long disappearance with an attractive line, count a prediction as a correct observation, or derive a confirmed line call from it.

When a spare ball or multiple candidates exist, preserve ambiguity until the evidence is sufficient. Do not automatically switch to the nearest white spot. Measure near- and far-court errors, blur, and occlusion separately, including false alarms per minute and periods of loss, rather than mAP alone.

Draw a short, subtle trail, distinguish estimated segments, and let users hide it. Do not allow smoothing to shift the bounce time or erase a change in ball direction; separate the display trajectory from the trajectory used for measurements and events.

**11. Court calibration and measurements in meters**

Extract court keypoints, refine them using line evidence when helpful, and estimate a homography from distributed, non-collinear points. Four points are an initial mathematical minimum, not proof of accuracy. Use additional points, RANSAC, checks against lines and the image, optical distortion analysis when its data is available, and verification points excluded from fitting. [OpenCV homography explanation](https://docs.opencv.org/4.13.0/d9/dab/tutorial_homography.html).

Apply an official dimension model: length 23.77 m, singles width 8.23 m, doubles width 10.97 m, and the service line 6.40 m from the net. Choose a documented origin and axes, and record the positions of the net, baselines, sidelines, service boxes, and the appropriate line widths. [ITF 2026 Rules of Tennis, Rule 1](https://www.itftennis.com/media/7221/2026-rules-of-tennis-english.pdf).

Regulation dimensions are measured to the outside edges of the lines. Court-intersection annotations may refer to line centers; document the definition used by the keypoints, and reconcile calibration coordinates with line thickness and legal boundaries before IN/OUT decisions. Do not confuse the center of a drawn line with its outside boundary and introduce a bias of centimeters.

An assisted calibration correction may be provided by clicking known intersections, with automatic verification and an overlay on the image. The automatic path must remain the default, and manual intervention and the resulting measurement type must be recorded. Clicking two points produces a correct measurement only when both are on the calibrated ground plane; YOLO does not turn any two airborne points into a real distance.

Save calibration for each stable camera interval. On pan/zoom/cut or poor alignment, stop ground-plane measurements for the affected interval and recalibrate. Do not use one homography for an entire long match without monitoring it. Show the proportion of time with valid calibration and test regions far from the fitting points.

**12. Player speed, distance, and movement**

Convert each ground-contact foot point to `(x, y)` coordinates in meters, preserving its actual timestamp. Calculate speed from position differences over a documented time window after appropriate filtering, and calculate distance over valid continuous intervals. Display both m/s and km/h, and define the average, maximum speed, and window duration; a single-frame spike caused by jitter is not a useful maximum speed.

Do not connect two positions separated by a camera cut or a long absence and then count the resulting distance. When coverage is partial, display “distance observed during valid intervals” with the coverage percentage; do not scale it up and present it as the distance covered during the entire match.

For each player, provide distance, average speed and validated peak speed, movement time, positioning zones, trajectory, frequency of approaches to the net, percentage of time behind the baseline, and a time-weighted position heatmap. Recovery time to an appropriate position after a shot may be added when there is a clear definition and measurement method.

Acceleration and detailed movement metrics are more sensitive to noise; display them only after specific validation. Reduced movement alone does not establish fatigue or injury. Validate player speed against a suitable independent source, such as a measured and calibrated human trajectory or independent reference video, accounting for the reference source’s own error.

**13. Ball speed: establish the correct definition before displaying any number**

Do not derive “true ball speed” by projecting every airborne ball position through the ground-plane homography and dividing distance by time. The ground transformation describes the ground plane, while the ball is outside that plane during flight.

Separate the following measurement types in the data, interface, and report:

- Ball image position in pixels: a visual observation, not a distance in meters.
- Bounce position on the ground: measurable on the calibrated plane, with uncertainty.
- Ground displacement between two bounces divided by elapsed time: an average horizontal displacement measure, not instantaneous speed or the speed of the ball as it leaves the racket.
- Estimated flight or shot speed: the result of physical reconstruction with explicit calibration and assumptions, requiring independent validation.
- Serve speed measured by a trusted reference: keep its source, timing, and definition separate from the website’s estimate.

Develop flight-segment reconstruction using camera calibration, court dimensions, collision constraints, gravity, and the available temporal evidence. Do not assume that an off-the-shelf depth estimate or knowledge of two bounce positions resolves every ambiguity in three-dimensional motion. If the solutions cannot be distinguished or there are insufficient frames, return `unavailable` or `estimated` with an uncertainty range.

Keep ball speed within the project’s objectives, and implement an actual validation process: record shots from the supported viewpoint with synchronized radar or independently calibrated cameras or high-speed footage that measure the same speed at the same instant or over the same segment. Broadcast speed displays can serve as supporting references after their meaning and synchronization have been verified; they are not sufficient on their own for every shot type.

Start with average speed over flight segments that can be validated, then add serve speed near racket contact when the data qualify it. Do not publish a physical speed value as a reliable measurement before it passes its acceptance gate. Document MAE, median error, P95, coverage, and the differences between serves and ordinary shots. [Where Is The Ball research on ambiguity in ball reconstruction](https://openaccess.thecvf.com/content/CVPR2025W/CVSPORTS/papers/Ponglertnapakorn_Where_Is_The_Ball_3D_Ball_Trajectory_Estimation_From_2D_CVPRW_2025_paper.pdf).

**14. Hits, bounces, and rally segmentation**

Build a timestamped event log that combines the ball trajectory, changes in direction, proximity to a player, body pose, ground-plane evidence, and audio when available. Use impact audio as a supporting signal; applause, commentary, and balls on neighboring courts are not automatically racket contacts. Handle audio/video synchronization differences and missing audio.

Associate events with the point state: waiting for a serve, first serve, first fault, second serve, rally, terminal event, and then the dead-ball interval. Distinguish a player’s hit from a ground bounce, a net collision, and an inactive ball. Do not count one hit twice because two signals indicate it.

For each event, store the original timestamp, supporting footage, event type, player if known, position if valid, and the event’s confidence and status. Contact between two frames may be estimated and reported as a temporal estimate with a range, not as an observation with greater precision than the video provides.

Use one-to-one matching during evaluation and an explicit time tolerance, initially such as ±80 ms, with stricter analysis around line calls. Test short and long rallies, volleys, lobs, serves, second faults, and negative cases. Do not promise forehand/backhand or ace classification when the evidence is insufficient; these classifications require separate qualification.

**15. Highlight the entire court zone when the ball lands**

Implement the requested visual feature clearly: when an active ball has a confirmed bounce, show a small marker at the contact position, then fill the entire polygon of the landing zone with a translucent overlay on the original video, while highlighting the same zone on the synchronized mini-court.

Provide two types of court partition: tennis rule zones, such as service boxes, and tactical analysis zones, such as a 3×3 or 4×3 grid on each side. Their type and boundaries must be fixed and explained. A tactical cell is not a legal line-calling zone.

Use the calibrated court transformation to project the zone into the video viewpoint. Its boundaries must align with the actual lines and remain correct through window resizing, letterboxing, and cropping. Do not draw a fixed rectangle on the screen. Associate the overlay with the same video timestamps, calibration, and revision used by the analysis.

Display a color and a text label for the status: landing zone inside the boundaries, outside the boundaries, or uncertain. Do not rely on color alone or obstruct the view. The highlight may last approximately 0.6–1.0 seconds before fading, with an option to hide it. If the bounce-position uncertainty range overlaps two zones, show the ambiguity instead of highlighting one as certain.

Do not highlight a zone merely because the ball passes above it in flight. Do not count every bounce as a point, or treat a highlighted service box as proof of a valid serve without applying the serve state and required side. Include these overlays in the annotated video export, and make clicking a bounce on the timeline take the user to its supporting evidence.

**16. IN/OUT, faults, and point decisions**

Apply tennis rules from the official source: touching a line counts as in; a serve must land in the correct diagonal service box; a first fault differs from a double fault; and a service let has a separate state. [ITF Rules: 12 and 17–24](https://www.itftennis.com/media/7221/2026-rules-of-tennis-english.pdf).

A line call depends on ball-ground contact, the line boundaries, and the uncertainty in position, calibration, and timing. Do not use the center of the ball’s bounding box alone, or treat a visual circle projected from an airborne ball as its actual contact footprint. If the footage cannot resolve a close line call, return `uncertain` with the supporting evidence. Do not claim centimeter accuracy merely because the average localization error is below half a meter.

Define separate outputs: `line_call`, `event_type`, `point_winner`, `loss_reason`, and `score_state`. A bounce may be known while the point winner remains unknown. The score may also be established from a confirmed source while the visual reason for it remains unresolved.

Handle service faults, double faults, balls landing out, balls hitting the net, failure to return a legal ball, and second bounces. Do not assume that a player stopping or the ball disappearing means that player lost the point. Foot faults, precise net touches, and double hits cannot be resolved from every wide broadcast view; leave them unsupported for automatic decisions until specific evidence and testing are available.

Provide a short review list of uncertain events that allows users to correct them and see the effect on the point, statistics, and report. Record manual corrections separately from automatic inferences.

**17. The score and the actual winner and loser**

Use a deterministic rules engine that updates points → games → sets → match. Start with singles, a choice of best-of-3 or best-of-5, an explicit tie-break rule, and an option for another explicitly supported format. Do not apply one ruleset to every match or tournament.

Cover 0/15/30/40, deuce/advantage, tie-break sets, service order, changes of ends, first and second faults, and lets. Use the project’s existing engine after auditing and testing it, instead of scattering scoring logic throughout the interface.

When a video starts midway through a match, provide simple optional input for player names, the initial score, server, and match format. Without sufficient initial state, show the clip analysis and whatever partial score can be established; do not invent the winner of the full match. The broadcast scoreboard may be read as a supporting source with provenance and conflict handling, rather than silently treating it as unquestionable truth.

Separate “point winner,” “set winner,” and “match winner.” Do not announce the match winner based on the last rally or the end of the file. Handle incomplete matches, retirement or suspension, and points missing because of camera cuts. When a consequential point is missing, stop advancing the confirmed score, optionally retain hypotheses, and resume after a correction or confirmed synchronization.

After a correction, recompute dependent events, the score, predictions, and the report from the point of change. Preserve result revisions and an audit trail, and distinguish results dependent on human corrections from purely automatic accuracy.

**18. Probabilistic prediction of the winner and loser**

Add a “win probability” panel separate from the “confirmed result.” Do not present the future as a fact. At each time `t`, use only information available through `t`: the current score, server, previous points, reliable previous movement and shot indicators, and historical information available before the match if appropriately licensed data exist.

To preserve the scope of three vision models, start with a probabilistic tennis engine: estimate each player’s probability of winning a point on their serve using a prior, shrinkage, and incremental updates, then calculate game, set, and match probabilities according to the selected format’s rules. This is a statistical component that can be calibrated; it does not require a fourth vision model or a new neural network merely to predict the result.

Prevent the final result, subsequent frames, commentary revealing the outcome, and full-match statistics from leaking into an earlier prediction. Even recorded-video analysis must show a prediction that could have been calculated at that moment, without using future information through the video trajectory or smoothing when constructing its features.

Calibrate probabilities on independent matches and test them with an appropriate chronological split. Measure Brier score, log loss, and a reliability curve, grouping uncertainty by match so that thousands of snapshots from one match do not appear to be thousands of independent trials. Verify their usefulness against a simple reference based on the current score; this checks prediction quality and is not a vision-model comparison project.

When point data are insufficient or the score is unstable, display “insufficient data for a reliable prediction,” or explicitly label a prediction based on the prior. Do not present 50% as information calculated from a player’s performance when it is simply a neutral prior. Once match completion is confirmed, switch the display to the actual result. [Probability calibration documentation](https://scikit-learn.org/stable/modules/calibration.html).

**19. Explain reasons for losing points and assess performance**

For each qualified point, write an explanation linked to evidence: “Player A lost the point with a double fault,” “Their final shot landed out at 00:12:34,” or “They did not return the legal ball before the second bounce.” If the cause is not established, write “The reason the point ended cannot be resolved from the footage.”

In the tactical report, use auditable facts: bounce distribution, depth, angles, frequency of short balls, player positions before shots, rally length, serve outcomes, and the percentage of points that passed validation. State sample counts, coverage, and the timestamps of examples.

Distinguish the observed terminal cause from a tactical hypothesis. You may say “They lost 6 of 8 qualified points following short balls in the specified zone,” but do not assert that they lost the match because of fatigue or a poor psychological decision. Forced/unforced error classification requires a criterion and review; it cannot be inferred merely from OUT.

A language generator may formulate the report using only a record of established facts. Do not allow it to calculate speeds, change the score, or invent events and causes. Do not make it a prerequisite for a correct report; the structured report must work on its own.

**20. Website and analysis dashboard design**

Develop a cohesive, professional interface focused on the video, score, and evidence. Use a restrained design with court-related colors, readable typography, clear spacing, consistent colors for players A/B, and appropriate contrast. Keep model names and experiment phases in administration tools and documentation.

Required routes:

1. **Home:** A clear value proposition, different genuine tennis examples with synchronized analyses, and a prominent “Upload video” button at the top of the page. Do not use one video with statistics from other matches, or unmeasured marketing accuracy claims. Respect example rights and display their sources when required.
2. **Upload:** Multiple-file drag-and-drop, a preview of file names and metadata, eligibility checks, a choice between independent analyses or clips from the same match, and then resumable uploading.
3. **Processing:** Actual upload progress, followed by queued / processing states and understandable stages, options to cancel or retry, and state preserved when the page is refreshed.
4. **Analysis:** A main video with controllable overlays, a mini-court, match score, two player cards, an event timeline, speed and distance charts and heatmaps, bounce zones, and probabilistic predictions.
5. **Review:** Events requiring corrections with evidence and timestamps, and straightforward calibration, identity, and score correction.
6. **Library:** Saved analyses, search, status, date, resuming work, and report export.
7. **Live:** An actual live-video connection, connection status, latency and coverage, and continuously updated score and analysis.
8. **Reports:** Downloading and viewing a readable report, statistics, annotated video, and structured data.

Keep the video, charts, timeline, and mini-court synchronized during play/pause/seek and playback-speed changes. Events have time ranges independent of the user’s playback speed. Display unknown as a missing value with a reason, not as 0 that a user could interpret as a measurement.

Prioritize a reliable 2D court. If you retain the existing 3D view, do not present fabricated ball height or detailed body motion as measurements. They may be shown as a separate labeled illustration or replaced with qualified data. The interface must work on desktop and adapt to mobile, with keyboard support, loading/empty/error states, Arabic RTL if Arabic is used, and English terminology where needed.

**21. Multiple-video uploads and user management**

Extend the current single-file upload to multiple files. Create a `batch_id` and an independent job for each file, with separate progress, status, errors, cancellation, and retries. One failed file must not fail the entire batch. Prevent duplicate jobs caused by retry through idempotency.

By default, each video is an independent analysis. If the user selects “clips from the same match,” require a reliable order or start time, preserve offsets, and prevent overlapping clips or replays from being counted twice. Do not automatically merge player identities and scores across unrelated videos. Clips from different viewpoints do not automatically provide multi-camera support or shared calibration.

The website is available to the public, but users’ videos and results are private to their accounts by default. Implement sign-in and owner authorization for every upload/job/result/media/report/download/live session. Test that changing an ID in a URL does not expose someone else’s file. Public examples are separate data explicitly selected for publication.

Apply quotas for duration, size, job count, and storage; verify the actual file type and whether it can be decoded; and process video files in isolation. Make large uploads resumable/multipart, show actual size and progress, and provide controlled expiry and deletion under a clear policy. Do not assume that the old 1 GiB limit is appropriate for every long match.

**22. Backend, storage, and data contract**

Retain FastAPI and the current frontend as long as they remain suitable. Process video outside the HTTP request lifecycle. Local storage and existing jobs may remain during development, but the public-service plan needs a database supporting multiple users, a durable queue, GPU workers, and private media storage; PostgreSQL, Redis, and S3-compatible storage may be adopted after reviewing migration and budget.

Jobs must include the owner, a clear state, fair priority, start and update times, heartbeat/lease, resource limits, a checkpoint, worker-failure information, and support for resume. Do not write partial results as though they were complete. Publish each report revision and its files atomically or through a valid manifest so that users cannot read a mixture of two revisions.

Define a versioned data contract for the match, video, calibration, identity, tracks, events, measurements, score, prediction, and report. For every measurement, store the name, value or null, unit, time interval, method, evidence source, observed/estimated/manual/unknown status, calibration and identity quality, an uncertainty range when valid, and the reason for unavailability.

For every event, preserve a stable `event_id`, `analysis_id`, `segment_id`, original timestamp, event type, `player_id` if known, a valid position when available, review status, and links to supporting footage. Save the revision when a correction is made. Associate every result with the model version, configuration, and source data.

Preserve source PTS and its mapping to the browser playback version. All overlays, reports, and corrections must use one documented mapping. Separate match time from live wall-clock time, processing time, and the timeline of a video containing replays. Test variable-frame-rate video, trimming, transcoding, and rotation metadata.

**23. Long matches and efficiency**

Design reading, processing, and publication around streaming and chunks, without loading every match frame or every track into memory. Persist detailed data incrementally to disk or storage, and fetch only the required range for the frontend, with downsampled data for long charts.

At chunk boundaries, retain enough ball, player, and event state, with documented overlap and deduplication; do not lose a bounce between two chunks or count a stroke twice. Long-video analysis must produce the same result, within a declared tolerance, whether processed continuously or in chunks. Resume safely after a worker crash without duplicating points.

Test progressively with a 10-minute video, then 30 minutes, then a complete two-hour or three-hour match within the testing budget, including camera cuts, changes of ends, replays, and missing audio. Do not rely on looping a short video to establish correctness on a long match, although a loop can be useful solely for memory testing.

Record RAM, VRAM, disk usage, decode/inference/tracking/encoding/export times, queue length, waiting time, and browser performance. Memory usage must remain tied to the size of the bounded buffers rather than match duration. Publish supported duration, file size, resolution, and FPS after testing, and make the limits configurable for the service.

Do not skip frames arbitrarily around the ball or bounces to improve the reported processing speed. You may reduce court-calibration or body-pose update frequency while monitoring quality, and use crops, but measure the effect on events. Do not promise a completion time unsupported by measurement; display a clearly identified estimated ETA if a reliable model for it is available.

**24. Exporting and accelerating YOLO26**

After qualifying the original models, export the system to ONNX or TensorRT for the selected hardware. Pin preprocessing, resolution, head, NMS, and precision type in the production configuration. The availability of an NMS-free head in YOLO26 does not mean every version uses it by default; verify the version and configure the option explicitly. [End-to-end guide](https://docs.ultralytics.com/guides/end2end-detection/).

Start with FP16 where appropriate for the hardware, and review the current export API instead of copying obsolete parameters. Do not use INT8 for the small ball until calibration data and validation establish that it has not degraded far-court ball detection or event timing. [Official export documentation](https://docs.ultralytics.com/modes/export/).

Compare source-model and exported-model outputs solely to verify conversion integrity, not to launch a comparison study of model families. Measure the complete application on the actual device with concurrent processing. COCO results or a developer's reference video are not this project's throughput. Retain the production model, its configuration and test, and a hash for every version.

**25. Genuine live video**

This feature is required in the final scope, but comes after qualification of the recorded-video pipeline. Start with one clearly defined live source, such as an authorized HLS/RTSP feed entering through a controlled ingest service, or sequential clip uploads with actual timestamps. Do not call recorded-video playback or manual score entry "live analysis."

Use the same three models, with state for each session, a bounded buffer, and incremental browser updates through WebSocket or SSE. Separate video-capture performance and source timing from analysis and display latency. Handle interruption, reconnection, camera changes, time gaps, and queue draining; under resource pressure, display the degradation rather than allowing silently increasing delay.

The live pipeline must not use future frames. If an event requires a short look-ahead to confirm a bounce, specify its size and include it in the declared latency. Events may move from provisional to confirmed to corrected, with update timestamps; do not freeze an early uncertain result as the final history.

Initial performance target: sustain the source rate without a growing backlog, with P95 latency from frame arrival to displayed analysis of no more than approximately two seconds under a declared configuration. This is a target, not a claim; publish it only if the system passes with all its features on the selected device. Document input FPS, coverage, drop rate, and event-confirmation delay separately. Test a long live session and multiple sessions within the service's capacity.

Restrict any URLs accepted by the service to prevent access to private-network addresses, metadata endpoints, or unauthorized redirects; ingest must not become an SSRF entry point. Keep stream keys and private links out of public logs and reports.

**26. Detailed reports for every video and match**

Produce a clear, self-contained HTML report that can be saved and printed, with a downloadable PDF when the pipeline supports it, alongside JSON/CSV and the annotated video. Reuse the existing report and add qualified data. Do not merely provide a JSON download and call it a report understandable to the user.

The report must contain:

- Video and match details, viewpoint, format, names when known, total duration, and the actual interval analyzed.
- The confirmed score, winner, and loser when evidence is sufficient, or a clear statement that the score is partial or incomplete.
- Each player's movement summary, distance, qualified speeds, heatmaps, and positioning, with window definitions, units, and coverage.
- Ball speed according to the supported measurement type, the number of measurable strokes, error ranges, and unavailable states instead of fabricated numbers.
- Rallies, their duration, stroke and bounce counts, effective playing time, first and second serves, and double faults where verified.
- A chart of landing zones, depth, and direction, with examples and timestamps, separating uncertain cases.
- Evidence-supported reasons for point endings, and probabilistic tactical observations with sample counts and coverage.
- The progression of win probability before the outcome, with its definition and limits, separate from the actual winner.
- A table of user reviews and corrections, the source of the score, and calibration, tracking, and event quality.
- Evidence links or images for important moments, the analysis version, report date, and metric definitions.

For multiple files, provide a report for each video and a batch summary. Provide a unified match report only after establishing that the clips belong together, are correctly ordered, and are not duplicates. Every report must match the revision of the video, dashboard, and mini-court; do not combine files from different versions.

**27. Defining accuracy and confidence**

Do not use one number called "system accuracy." Separate detection, localization, tracking, calibration, events, speeds, scoring, and prediction. Detector confidence is not the probability that a point decision is correct, and converting it to a percentage in the interface is insufficient.

Calculate and calibrate task-appropriate confidence on an independent validation set, and propagate calibration, timing, and identity uncertainty into measurements and events. When a reliable range cannot be established, use a clear quality state rather than inventing a confidence interval.

For every gate, declare the sample, denominator, scope, tolerance, coverage, and number of unknown cases. For the ball, do not test only frames in which the model detected it. For events and scoring, do not test only points the system successfully resolved while hiding the proportion it left unknown.

**28. Proposed quantitative acceptance gates**

Freeze metric definitions and thresholds before the final test. The targets below are proposed to guide the work; they may be adjusted based on reference measurements and scope, with the reason documented before final evaluation. They must not be lowered after viewing the test to relabel failure as success.

| Task | Initial target and acceptance method |
| --- | --- |
| Player identity | IDF1 ≥ 0.95 during supported play; publish HOTA, identity switches and interruptions per match, and near/far breakdowns. |
| Visible ball | Precision and Recall ≥ 0.95 under a frozen center/box definition appropriate for a small ball; report median and P95 localization error in original-image pixels, blur and far-court errors, and false alarms per minute. |
| Calibration | Agreement with independent verification points and lines excluded from the fit, error in meters across widely separated regions, and the proportion of valid time; internal reprojection alone is not sufficient for acceptance. |
| Player ground position | Median error ≤ 0.20 m and P95 ≤ 0.50 m against independent reference ground points within the qualified scope. |
| Strokes and bounces | Precision ≥ 0.95 and Recall ≥ 0.90 under an initial ±80 ms tolerance, with one-to-one matching, temporal-error analysis, and dead-ball controls. |
| Player speed | MAE ≤ 0.5 m/s against an independent reference, with bias, P95, coverage, and the measurement-window definition. |
| Physical ball speed | Median absolute error ≤ 5 km/h and P95 ≤ 10 km/h on strokes with an independent synchronized reference matching the speed definition; publish coverage for every stroke type. |
| Point winner | Precision ≥ 0.95 for automatic decisions, with initial coverage ≥ 80% of evaluable points within the supported viewpoint; also publish the outcome for every unresolved point. |
| Scoring rules | Pass all specified cases, with no duplicated points or score drift in complete reviewable matches; any consequential gap suspends trust in the score. |
| Line decisions | A separate gate for balls near lines, with independent contact and line references; the general position targets above do not qualify centimeter-level officiating. Publish precision, coverage, and uncertain cases. |
| Win prediction | Independent data without leakage, Brier/log loss and a calibration plot, and documented value against the simple baseline; no guarantee that the player with the higher probability will win. |
| Long and live video | Bounded memory, correct resume behavior, no lost or duplicated events, synchronized timestamps, and throughput and P50/P95 latency on named hardware. |
| Multiple users | Complete job, media, and report isolation; access by a non-owner must fail, and each file's state must persist after refresh/retry. |

Calculate confidence intervals for metrics where appropriate, grouping by match rather than treating correlated frames as independent. Review sample size: a small test does not establish a high percentage as an absolute fact. Visual quality and improvement on development data do not remove the independent-test requirement.

If a core requirement fails, continue a defined repair cycle. If it needs unavailable data, a reference, or hardware, keep it explicitly marked as unqualified, complete independent work, and identify the specific blocker and required solution. Withholding a number does not fulfill the requirement to measure it; withholding protects the product while qualification continues.

**29. Practical tests you must execute**

Run useful tests linked to plausible failures:

- Unit tests for geometric transformations, axes, crop/letterbox handling, source timing, and distance and speed calculations across gaps and cuts.
- Scoring rules for every supported format, first and second faults, lets, tie-breaks, changes of ends, clips beginning mid-match, and corrections.
- Identity tracking through changes of ends, interruptions, and people outside play entering the scene, with separate state for every stream.
- Chunk boundaries, overlap, worker resume, and prevention of duplicated events and results.
- Integration from a real file upload to a saved analysis and video playback, then a report and annotated video matching the source.
- Batch uploads, one-file failure, retry/cancel/resume, independent analyses, and overlapping clips from one match.
- Account isolation and every endpoint, media asset, and download; the test must not rely on the interface alone.
- Interface checks on desktop and mobile, video and chart controls, perspective-aligned polygons during resize/seek, and missing-data and error states.
- Short independently annotated clips for detailed verification, and complete matches for identity drift, score drift, and memory checks.
- Genuine live video when ingest is supported: interruption, reconnection, zoom, camera changes, and accumulating network and worker pressure.
- Blind human review of a final sample, retaining the video, data, and report versions that produced the metrics.

Label demonstration data when it is mock data, then replace it with real analysis examples before release. Successful API tests or the number of passing tests must not be treated as evidence of vision accuracy.

**30. Implementation phases and priorities**

Execute one practical plan, progressively developing the interface with real data:

1. **Audit and establish the baseline:** inspect the current project, rerun the upload-to-report workflow, preserve the baseline, and identify gaps, library versions, and scope.
2. **Data protocol:** authorized sources, independent splits, an annotation guide, and accuracy, timing, and calibration references. Prepare small samples before long runs.
3. **YOLO26 models:** ball, players/body pose, and court landmarks; perform only necessary training or fine-tuning, then produce reproducible artifacts.
4. **Tracking and geometry:** stable identity, an honest ball trajectory, valid calibration, a mini-court, and player-movement measurements with coverage.
5. **Events and overlays:** strokes, bounces, rallies, playing time, full-zone highlighting, synchronized evidence, and review of unknown cases.
6. **Scoring and analysis:** an event-linked rules engine, confirmed winner and loser, reasons for point outcomes, and traceable corrections.
7. **Ball speed:** reconstruction, measurement definition, reference data, and independent qualification; do not defer investigation of its reference until the end of the project.
8. **Win prediction:** a statistical engine, calibration, and evaluation without leakage, with presentation separated from the actual outcome.
9. **Complete product:** multiple uploads, accounts and ownership, a synchronized analysis interface, a library, comprehensive reports, exports, and annotated video.
10. **Capacity:** long matches, bounded-memory processing, queues and resume, quality-verified export optimization, and cost measurement.
11. **Live:** ingest, updates, event confirmation, timing, latency, and a real long-duration test; do not consider it complete before demonstrating live video.
12. **Deployment qualification:** independent tests, an operating guide and pinned versions, declared limits, a final acceptance report, and review of the public service.
13. **Later expansion:** common side views or doubles, with separate data and qualification that preserve the success of the primary viewpoint. Do not delay the initial release by trying to support every camera.

Data collection, contract design, and the upload and report interfaces may proceed in parallel, but do not base speeds or scoring on an unqualified component. Do not launch costly work or purchase resources without an available budget and authorization. When either is missing, complete the audit, permitted data work, configurations, interface, and small experiments locally within available resources, and identify the compute resources required for the costly step.

**31. Documentation and deliverables**

Deliver working code, models, configurations, and reference data permitted to be retained, together with:

- A final scope and requirements statement, an updated gap matrix, and the three-model decision.
- Data/source/license manifests, an annotation guide, and frozen training, validation, and test splits.
- Reproducible training, evaluation, and resume commands, library versions, seeds, and weight hashes.
- Descriptions of calibration, timing, speeds, identity, events, rules, prediction, and uncertainty.
- The API contract, data versions, corrections, and report, plus an operating guide for development, the worker, and the ingest service.
- An independent accuracy report with actual measurements, samples, coverage, and errors, separating the accuracy components.
- A long-video and live-capacity report: hardware, configurations, memory, storage, latency, and concurrent jobs.
- Different tennis examples with appropriate rights, their analyses, annotated videos, and reports, and clips demonstrating zone highlighting and review.
- A user guide and supported-input limits, a reviewable upload-to-report journey, and an operation, backup, and update plan.

Update the existing project documentation instead of duplicating it across dozens of files. Maintain a concise progress record distinguishing completed / in progress / unqualified / blocked, with evidence and the next step. Do not present a result that was never executed as successful.

**32. The official YOLO26 reference and scientific integrity**

Use the paper referenced by the official Ultralytics documentation: **Ultralytics YOLO26: Unified Real-Time End-to-End Vision Models**, by Glenn Jocher, Jing Qiu, Mengyu Liu, Shuai Lyu, Fatih Cagatay Akyon, and Muhammet Esat Kalfaoglu, published as an arXiv preprint on 2 June 2026, identifier **2606.03748**. Do not call an independent explanatory paper the official paper, or describe a preprint as peer-reviewed research without verification. [Official paper](https://arxiv.org/abs/2606.03748).

YOLO26's small-object and deployment features provide a useful foundation, but do not guarantee detection of an occluded ball or valid tennis measurements. Do not copy COCO, research-paper, or LinkedIn-post numbers into the project's results section. Every number published in our report must have a specific split, version, date, and measurement method.

In academic writing, separate the scientific source, engineering assumption, implementation method, measured result, and limitations. At execution start, verify the current official API, then pin the production version. Stay with the requested YOLO26 family even if the "latest" page changes; market changes do not justify automatically switching the training objective.

**33. Final definition of done**

Do not consider the objective complete until a real user, within the supported filming conditions, can:

1. Upload multiple videos, receive saved jobs and analyses for each, and return to them later.
2. View ball and player tracking, identity, court calibration, and a synchronized mini-court.
3. See the entire zone of a confirmed bounce highlighted in the video and mini-court, and access the evidence for every event.
4. Obtain player movement, speed, and distance with qualified measurements and declared coverage.
5. Obtain ball speed under the measurement definition that has actually been qualified, with estimates or unavailable states clearly identified in other cases.
6. Follow the score, winner, and loser under correct evidence and rules, review and correct unknown points, and read supported reasons for lost points.
7. See changing win probabilities as a calibrated prediction separate from the outcome.
8. Download an understandable report, an annotated video, and analysis data matching the current revision.
9. Analyze a complete long match without growing memory usage, duplicated events, or lost state, under limits that have actually been tested.
10. Use a genuine live-video pipeline with a declared source, hardware, capacity, and latency after it passes testing.
11. Access files and results only within their permissions, with a consistent website experience on desktop.

Returning unknown when footage does not support a conclusion is correct, but do not hide weak coverage behind it. Core features must pass their gates within a clear scope, and unqualified features must be identified as remaining work. Do not add comparisons or train other models to expand the project without a need.

After this text is assigned as an objective, begin with the audit and finalize the actual implementation plan, then complete it progressively with evidence. Make the website comfortable and beautiful, make the analysis auditable, and keep every claim within what the data have established.

**34. Mandatory Codex Cloud multi-account continuity, frequent GitHub saves, and pre-limit handoff**

**Purpose and non-negotiable workflow.** This complete master objective is intended to be executed progressively in **Codex Cloud on the ChatGPT website**, with **three or four independently logged-in ChatGPT accounts taking turns**. Each account may be a fresh session without access to the previous account's conversation or ephemeral workspace. Treat **GitHub as the persistent source of truth**, and ensure that the next account can continue from the **exact last completed and verified implementation step**, without restarting the project, repeating completed phases, or losing any code. These requirements apply to **every account, every session, and every handoff**, not merely at the end of a phase.

**A. Authoritative repository and starting point**

- Canonical repository: `https://github.com/OSAMAHNJAERAN/tennis-ai-agent.git` (`OSAMAHNJAERAN/tennis-ai-agent`).
- Current integration/development branch: `codex/yolo26-platform`. This is the working YOLO26 branch; **do not blindly start from the default `master` branch** or an older SHA. Re-evaluate the current branch head from GitHub at the start of every session because it may have advanced.
- A new Codex session must connect to the same repository with the necessary GitHub read/write authorization. If it cannot read or publish to the repository, report that limitation immediately; a task running in a cloud sandbox is **not** automatically backed up to GitHub.
- Read `AGENT.md`, the master execution prompt, `docs/plans/YOLO26_EXECUTION_PROGRESS.md`, `docs/plans/TENNIS_EXECUTION_PROGRESS.md` if it exists, and the latest recorded handoff. Also inspect `docs/deployment/SECOND_ACCOUNT_HANDOFF_AR.md` for the existing cross-account workspace/artifact restoration procedure; verify referenced releases/artifacts actually exist before depending on them.
- Determine which steps in Section 30 are genuinely complete, in progress, unqualified, or blocked based on the current code and evidence. The master prompt is the scope definition, **not** an instruction to redo completed work. Continue the oldest actionable unfinished step, while respecting the dependencies and priorities already defined.
- Use `git fetch origin` and inspect the remote branch, current commit SHA, pending PRs and handoff record before changing code. Confirm the actual checkout and working-tree status. Synchronize safely; do not discard another account's unmerged work. Do not automatically treat the newest commit on an unrelated branch as integrated.

**B. One continuous project across accounts, not four disconnected projects**

- The intended sequence is Account 1 → Account 2 → Account 3 → Account 4 → Account 1 again, after its usage allowance becomes available, repeating as needed. **The current user controls account rotation; you must not claim that you can switch accounts or reset usage limits yourself.**
- Each account must **resume** the current implementation rather than make a fresh plan and duplicate prior work. Preserve previously completed steps and successful tests; only revisit them when a current defect, regression, or dependency makes it necessary.
- Maintain **one canonical, concise, Git-tracked handoff log** at `docs/plans/CODEX_MULTI_ACCOUNT_HANDOFF.md` (create it if absent; update it on every checkpoint/handoff). Continue updating `docs/plans/YOLO26_EXECUTION_PROGRESS.md` as the evidence-based project progress record. Avoid creating a new scattered status file for every session.
- Prefer sequential changes to the same integration line. If GitHub/Codex permits a safe fast-forward push to the unprotected shared development branch and nobody else has advanced it, you may publish the checkpoint directly to `codex/yolo26-platform`. Otherwise, push a named task/continuation branch and open or update a PR targeting `codex/yolo26-platform`. **Never force-push or silently overwrite a collaborator's branch.**
- If the current account has a pushed but **unmerged PR**, the next account must inspect its exact head branch and commits; it must not silently restart from an older target branch and drop that work. Prefer a reviewed/authorized merge into the canonical integration branch before handoff. When a merge is unavailable or pending, record the precise PR URL, head branch/SHA, base branch/SHA, and a safe continuation procedure. Never claim the integration branch includes code until it actually does.
- If multiple accounts happen to run at once, isolate their tasks and feature branches; review and integrate changes safely rather than editing or pushing the same files blindly. The default operating mode for this handoff process is **sequential**.

**C. Mandatory proactive saves BEFORE any five-hour/usage-limit interruption**

- The user may rotate accounts when a roughly **five-hour usage window** or other Codex quota is exhausted. **Saving, committing, and publishing work to GitHub before an interruption is a mandatory priority.** Never leave hours of valuable work only in a temporary cloud workspace.
- Codex may not expose a reliable countdown or a guaranteed warning before its rate limit, task timeout, context limit, or account usage allowance is reached. Therefore **do not wait until the final minutes** or promise to detect the exact limit. Preserve and publish checkpoints **after every meaningful working increment**, and otherwise aim for a checkpoint approximately **every 20–30 minutes of active work when feasible**, plus immediately before risky operations, costly training, major refactors, or an anticipated session interruption.
- A checkpoint means: (1) stop at a safe boundary; (2) inspect and stage the intended changes; (3) run the practical subset of available verification; (4) commit the current code and documentation with a descriptive message; (5) **push those commits to a durable remote GitHub branch**; (6) verify the push actually succeeded by comparing the relevant remote branch SHA/commits; and (7) update/publish the handoff record. A local `git commit`, a Codex task diff, or an unpushed branch **does not count as a successful GitHub save**.
- If the quota or task seems close to ending, **stop starting new implementation work**, even if the current feature is incomplete. Immediately preserve a small, internally consistent checkpoint, explicitly mark its unfinished tests or limitations, push it, and document the exact next action for the next account. Prefer a safe recoverable partial feature behind the existing quality gates to disappearing uncommitted code. Never represent an untested partial feature as finished or qualified.
- You may save unfinished work on a dedicated WIP branch or draft PR, with accurate status notes; preserve the last passing integration state. Do not bypass validation, weaken evidence gates, or force untested partial changes into the production integration branch simply to achieve a checkpoint.
- Before every session ends, verify that every intended tracked change is either included in a **verified remote push** or clearly documented as intentionally excluded, with a durable way to recover the excluded state. Confirm `git status` and the remote SHA. If a push, PR creation, or merge fails, **explicitly report that the work is not saved to GitHub** and provide a precise recovery path. Do not state “uploaded,” “synced,” or “ready for the next account” without remote confirmation.
- Do not commit secrets, credentials, access tokens, `.env` files, unlicensed data, or giant model/training artifacts indiscriminately. Use the repository's established Git LFS, verified GitHub release/snapshot workflow, or other approved persistent artifact storage when appropriate and actually available. Record checksum, storage URL/identifier, and restoration instructions in Git-tracked manifests. An artifact omitted by `.gitignore`, a locally stored training checkpoint, or a file merely listed in a manifest is **not remotely preserved** until its availability has been verified. Never assume Codex Cloud has room or permission to download the entire large project snapshot.

**D. Exact last-step handoff: the next account must be able to act immediately**

At every meaningful checkpoint and definitely before an account reaches its limit or stops, update `docs/plans/CODEX_MULTI_ACCOUNT_HANDOFF.md` with a concise but executable snapshot containing:

1. **Checkpoint identity and UTC timestamp**, session/account label if useful (never account secrets), repository name, current working branch, integration base branch, latest **pushed** commit SHA, and remote-push verification status.
2. **Current position in Section 30:** exact phase number, completed substeps, the **last successfully implemented and verified step**, its evidence/tests, and the **single active unfinished step**. Explain whether the current step is `complete`, `in_progress`, `blocked`, or `unqualified`; distinguish code that exists from features that passed independent acceptance tests.
3. **Precisely what changed:** files/modules touched, important design decisions, migrations/config changes, actual tests and their pass/fail results, command invocations, and any regression or known failure.
4. **The next action to execute first:** an explicit next substep, target files/functions, relevant commands, dependencies/prerequisites and success criteria; then the next few prioritized steps. The incoming account must not need to guess what the previous agent intended.
5. **Handoff Git references:** shared branch and head SHA; if work is awaiting integration, include its feature-branch name, PR URL/number and head SHA, review/merge status, and the exact ref the next agent should start from.
6. **Assets and environment:** model/data/artifact hashes and durable locations for needed assets, runtime configuration, required permissions/resources, confirmed missing assets, training checkpoints, active background jobs and whether those jobs will survive a fresh Codex environment. Never claim a live process or local filesystem persists across accounts without verifying it.
7. **Blockers and unresolved risks:** missing GPU/RAM, rights, credentials, ground truth, failed checks, or permissions; a feasible next action and what would unblock it. Explicitly differentiate `blocked` from `not attempted`.

Use a stable **CURRENT HANDOFF** section that always reflects the latest verified state, optionally followed by a compact chronological checkpoint history. Update the existing evidence/progress documents without erasing prior results. **Commit and push the handoff file together with the work it describes**; if the handoff has to be amended after a push, commit and push the amendment and verify it too. Do not depend on the previous account's chat history or a manually copied summary as the sole state carrier.

**E. Required startup procedure for EVERY incoming account**

1. Open the repository from the new Codex Cloud account and verify that the intended GitHub installation has read/write access. Select the current YOLO26 integration branch or the precisely identified pending handoff branch/PR. Explicitly verify the actual branch/ref; cloud task environments may create their own working branches.
2. Fetch the newest remote state. Read the **CURRENT HANDOFF** record and the official execution progress; inspect the latest commits/PR status and compare the documented SHA to reality. If the remote head changed after the handoff, reconcile it before modifying files.
3. Verify that the preceding account's claimed changes are **actually available on the remote ref**. If an unmerged PR holds the latest work, choose a safe approved merge or continue from that exact branch, rather than silently losing the PR's work.
4. Confirm the last completed step and the immediately next unfinished substep by inspecting the relevant code/tests. Do not restart the audit, reread every reference in full, retrain completed experiments, or repeat work merely because you are a new account. Do only the context checks necessary for a safe continuation.
5. Continue implementing the next actionable substep. Keep the master requirements, technical gates, data integrity, and scientific limitations unchanged. Execute verification appropriate to each checkpoint and record factual outcomes.
6. Repeat the proactive **commit → push → verify → update handoff** cycle through the session. Before rotation, perform a final checkpoint and leave the next account one unambiguous executable starting point.

**F. Handling interruptions and final completion**

- Treat losing context, task timeout, a five-hour usage-window limit, a revoked GitHub permission, or a new account as a **normal resumable handoff**, never a reason to silently abandon work. Prioritize early persistence because a hard interruption may occur before the final checkpoint can run.
- If an external prerequisite prevents progress on the current phase, record the exact blocker, complete only safe independent tasks from the existing plan, and do not falsely declare the project finished. If expensive compute, qualified data, or permission is unavailable, never fabricate results to keep the progress moving.
- Once the acceptance criteria in Section 33 and all required qualified phases have genuinely passed, mark the work as complete with linked evidence and a final verified GitHub push/PR. A usage-limit boundary is **only a session boundary**; it never counts as project completion.
- Every session's final message must state: **last completed step; current unfinished step; GitHub branch and verified remote commit SHA; PR link if applicable; tests executed; what the next account should do first; any unsaved work or blocked artifacts.** The final message must never substitute for the Git-tracked handoff record.

**Priority rule:** At any sign that the Codex session or allowance may end, **persist the current state on GitHub first**, document the precise handoff second, and only then continue further implementation if enough time and capability remain. Apply this rule continuously across all four accounts until the entire original master objective is genuinely completed and verified.

**END OF PROMPT.**
