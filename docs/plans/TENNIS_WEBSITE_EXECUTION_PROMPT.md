# Execution prompt: accurate tennis video analysis website

Prepared on 30 September 2026 from the current repository and primary-source research. Use this entire document as the implementation brief for an autonomous coding and computer-vision agent. Repository paths below are relative to the project root, currently C:\Semester 9\FYP2\tennis_ai_agent_bundle. Recheck the working tree when execution starts because the repository may have changed since this audit.

## 1. The goal and your responsibility

Turn this existing project into a focused, polished tennis video analysis website. A visitor should immediately see several different tennis clips playing with real analysis, understand the product, upload their own video through a prominent upper-right button, follow processing, explore synchronized analysis, and download a useful report and annotated video.

The central engineering objective is the highest defensible accuracy achievable for the declared input conditions: detect and track the tennis ball, the two active players, their body pose, and their rackets; register the court accurately; analyze player movement and speed; and provide a convincing, evidence-based court replay in 2D and 3D. Develop ball flight and ball-speed estimation through actual calibration, reconstruction, and independent validation. Keep uncertainty visible where the footage cannot support a reliable measurement.

Act as the responsible CV/ML engineer, frontend engineer, backend engineer, and product designer. Implement, integrate, test, and verify the result. A plan, model download, training run, attractive screenshot, or collection of disconnected demos is an intermediate deliverable. Completion requires a working upload-to-report experience and measured accuracy evidence.

Take the time needed for useful work. Allocate effort to the weak components and the visitor experience. Use research, licensed datasets, suitable pretrained models, targeted fine-tuning, and authorized compute when these improve the final system. Keep the product and production architecture small. Pursue one selected production model system, which may contain a small set of complementary specialists.

This brief supersedes older project instructions that make model comparisons, manual scoring, automated officiating, predictions, or an AI coach central product features. Retain scientific integrity, provenance, reproducibility, and useful academic evidence from those documents. Keep engineering evaluations private and bounded; users should interact with matches and analysis, rather than model names or experiment phases.

## 2. Start from the actual implementation

Read AGENT.md, the relevant scope in PROJECT_REPORT.md, README.md, docs/PRODUCT_WORKFLOW.md, docs/PRODUCT_DASHBOARD.md, docs/evaluation/PRODUCT_INTEGRATION_RESULTS.md, and docs/research/TENNIS_PRODUCT_RESEARCH_2026-09-30.md. Then inspect the active code and configuration. Consult older plans and experiment reports when they explain a component or previous failure; avoid replaying the complete historical roadmap.

The following facts were verified during the repository audit:

| Area | Current implementation | What the execution must address |
| --- | --- | --- |
| Web stack | React, TypeScript, Vite, React Router, Radix, Three.js/react-three-fiber, Recharts, and Zod | Reuse the existing stack and visualization utilities. |
| Upload and processing | src/api/app.py, jobs.py, worker.py, results.py, and product.py implement actual uploads, persistent SQLite jobs, a separate worker, normalized video, inference, result loading, media ranges, and downloads | Preserve this working foundation; improve integration and capacity. |
| Active model recipe | configs/phase6_analytics/wasb_tiled_residual_pose_small_validation.yaml is the API default | Treat it as the starting candidate, then qualify and pin one production recipe. |
| Player detection | yolo11m.pt with person detections and ByteTrack | Improve active-player selection and identity stability. |
| Ball tracking | artifacts/models/ball/wasb_tennis_best.pth.tar with code under artifacts/research/WASB-SBDT, temporal heatmaps, spatial crops, and filters | Diagnose remaining false detections and misses before replacing it. |
| Court | models/keypoints_model.pth predicts 14 landmarks with a ResNet-based regressor | Improve real image alignment and generalization, rather than trusting internal fit residual alone. |
| Body pose | yolo11s-pose.pt supplies 17-joint crop pose; continuous pose is exported and skeletons are drawn in the annotated video | Expose pose through the result adapter and web contract, and validate temporal quality. |
| Racket | The shared detector uses COCO tennis-racket class 38 on player crops and exports racket_tracking.json | Expose the data to the website and improve specialist detection and ownership where needed. |
| Segmentation | No production player instance-segmentation model is connected | Introduce a useful, validated player-mask overlay without confusing it with rectangular exclusion masks. |
| Homepage | ProductHome.tsx shows match records and upload/manual-scoring journeys | Build the requested homepage with distinct playing analyzed clips. |
| Examples | Seven historical entries reuse the same 214-frame sample; only one carries usable video and full replay arrays | Replace the public phase showcase with genuinely different analyzed clips. |
| Featured catalog | featuredMatches.ts and featured/matches.json contain official replay links and published match scores, without inference artifacts | Do not present these as analyzed matches or attach another clip's telemetry. |
| Current 3D | TennisCourt3D.tsx has a dimensioned court and useful controls; body motion and ball height include procedural illustration | Upgrade the presentation and feed qualified measurements into the scene. |
| Reports | The API report download is normalized JSON; existing frontend exports include CSV and browser printing | Add a readable PDF or self-contained HTML report while retaining structured exports. |

The working tree contains substantial uncommitted and untracked changes, including the real product API and upload experience. Preserve them. Establish the baseline from the current working tree, rather than reverting to a commit that lacks this work.

Existing documentation records successful upload integration on a 214-frame, 30 FPS, 7.13-second clip, with about 2.29 processing FPS in that run. This supports an offline processing workflow. It does not establish real-time operation, full-match capacity, or physical measurement accuracy. Historical test counts and metrics are evidence from their recorded runs, not substitutes for checking your final changes.

## 3. Freeze the product scope

Build these main journeys:

1. Home: several real analyzed tennis examples playing immediately, with a persistent Upload video action.
2. Upload: choose a video, validate it, submit it, and follow persistent processing.
3. Analysis: synchronized video, ball/player/racket tracking, pose, optional masks, motion statistics, speed graphs, court replay, heatmaps, and data quality.
4. Saved analyses: reopen a processed clip and find its exports.
5. Export: download the readable report, annotated video, and useful JSON/CSV data.

Begin with singles tennis from a mostly static elevated baseline camera, with the full playing court substantially visible. Publish the actual supported resolution, frame-rate, duration, and file-size limits after testing. Support ordinary MP4, MOV, and AVI inputs where their codecs can be decoded. Handle other views and camera cuts explicitly; expand the supported camera envelope only after corresponding validation.

Remove model-selection controls and phase-oriented comparisons from the consumer experience. Defer manual live scoring, tournament management, ranking/profile comparison, win prediction, scouting, conversational coaching, and automated line-officiating claims. Preserve underlying records or utilities when another required feature uses them. Shot/contact and bounce events belong in scope when they improve ball reconstruction or useful analysis and are validated. Full score inference is not a prerequisite for this brief.

## 4. Audit and clean the repository safely

Prepare one compact cleanup manifest with: path, size, purpose, runtime/import/config references, whether it is reproducible, preservation requirement, planned action, and validation. Classify each candidate as keep, consolidate, archive, or delete. Then perform the safe actions within the authorized workspace.

The audit measured approximately 11.85 GiB under artifacts/, of which artifacts/training/ accounted for about 9.38 GiB and 138,676 files. outputs/ was about 1.69 GiB. These are useful investigation priorities, not proof that the contents are expendable.

Review these concrete candidates:

- Duplicate historical pipeline orchestrators and phase-specific configurations after identifying the one supported runtime chain.
- Unused legacy frontend pages, thin aliases, model/phase demo entries, and invented marketing statistics. Examine the legacy files under frontend/src/pages/ named Overview.tsx, Analytics.tsx, Players.tsx, Tournaments.tsx, Scouting.tsx, Matches.tsx, and System.tsx, and locate IntelligenceCarousel.tsx under frontend/src/; check live references before deletion.
- The manual-live, profile-comparison, prediction, and coach product journeys after required match storage is isolated.
- Repeated training exports, duplicate media, stale caches, rejected checkpoint copies, and temporary output directories such as outputs/product_regression_tmp*, outputs/product_test*, and outputs/product_review_tests.
- Generic root weights and models/yolo5_last.pt that are not required by the selected runtime, reproducible baseline, or retained training process.
- Broad dependency freezes with unrelated packages or conflicting OpenCV distributions. Derive compatible inference/web, training, and optional research dependency groups from actual imports.
- Repetitive historical plans and reports. Keep a compact index and the evidence needed to reproduce important decisions; reduce the active documentation surface.

Preserve user work, academic source documents, unique source data, final test labels, dataset/model license records, selected checkpoints, useful baseline evidence, and live uploaded analyses. outputs/product/ contains persistent database records and uploads, so it must not be treated as a generic disposable output directory. artifacts/research/WASB-SBDT is currently a runtime dependency despite its directory name; package or relocate that dependency before removing research snapshots.

Delete only after the replacement works, references are migrated, and the retained baseline is recoverable. For filesystem cleanup, resolve exact target paths, check they remain inside the intended workspace, and use a single safe native filesystem mechanism. Keep the cleanup manifest and before/after footprint. Finish with fewer active routes, dependencies, configurations, and redundant files while retaining required data and reproducibility.

## 5. Research to resolve specific accuracy problems

Read the accompanying research note before selecting new methods. Inspect the supplied [tennis_analysis reference](https://github.com/abdullahtarek/tennis_analysis) for transferable ideas and tutorial limitations. Use original papers, author repositories, official documentation, dataset cards, and actual source code. Verify repository revisions, available weights, preprocessing, licensing, hardware requirements, and input assumptions.

Use these leads selectively:

- [WASB-SBDT](https://github.com/nttcom/WASB-SBDT): inspect the existing temporal ball baseline and official tennis implementation before adding another tracker.
- [TrackNetV3](https://github.com/qaz812345/TrackNetV3): a temporal high-speed-object lead whose released application/checkpoints must be checked for sport/domain suitability.
- [RacketVision](https://github.com/OrcustD/RacketVision): investigate tennis ball/racket annotations, racket keypoints, training modules, and released model assets.
- [MMPose](https://github.com/open-mmlab/mmpose): investigate RTMPose/whole-body alternatives only if the current pose system fails important wrists, feet, or body-motion requirements.
- [SAM 2](https://github.com/facebookresearch/sam2): evaluate prompted, identity-linked player masks as an overlay and possible geometric cue. Check any newer alternative for practical benefit and its own license.
- [OpenCV calibration documentation](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html): camera calibration, projective geometry, and reconstruction foundations.
- [ITF rules and regulations](https://www.itftennis.com/en/about-us/governance/rules-and-regulations/): verify court dimensions and relevant definitions against the official current document.

For each accuracy problem, record the failure, the smallest useful intervention, expected improvement, required data, and experiment outcome. Start with the current system plus one or two justified alternatives, then expand only when an unresolved failure warrants it. Research should lead to labeled data, tested changes, or a rejected idea; it should not become an endless literature catalogue.

Previous experiments already document failed or incomplete court, racket, player, and ball improvements. Reuse those findings. Keep model-selection checks internal and deliver one production bundle. A popular repository, newer architecture, paper metric, or attractive demo is not sufficient evidence for promotion.

## 6. Build a credible tennis dataset and qualification set

Audit data/data_registry.yaml and existing dataset notes before more training. Some legacy entries describe tutorial footage or court data as usable despite missing or ambiguous licenses. Resolve each source's actual training, derivative, and public-display permissions. Code, weights, annotations, and underlying footage may have different rights. Prefer clearly permitted data and owner-authorized clips; treat additional LinkedIn/GitHub links from the owner as leads whose contents and permissions still need verification.

Record source URL, version/revision, license, allowed uses, local path, hash, annotation schema, source match/session, camera conditions, resolution/FPS, split assignment, and known label defects. Use public images and videos only when the intended use is allowed. Generated design imagery belongs to visual design, not the ground-truth evaluation set. Synthetic training material may be considered only as separately tracked supplementary data with demonstrated tennis benefit.

Use grouped train/validation/test splits by match/session/source. Prevent adjacent-frame leakage, duplicate media, resized copies, and shared rallies from crossing splits. Reused development clips and the historical cross_match_final_holdout are diagnostic material; they are not a pristine final test set.

Create a new untouched qualification set before final promotion. Aim initially for at least 12 clips from at least six independent matches or recording sessions, with diversity in court appearance, lighting, player scale, compression, and camera behavior within the declared support envelope. Record the actual coverage and confidence intervals; this minimum is a practical starting sample, not proof of universal generalization. Keep challenging unsupported views as a separate rejection/generalization check.

Annotations should cover:

- Visible ball centers/tight boxes, blurred or partially occluded observations, confirmed absent/out-of-frame periods, and uncertainty. Fully invisible positions must not be silently labeled as measured coordinates.
- Active-player boxes and identity sequences, including occlusion, re-entry, and camera changes.
- Visible body keypoints, especially shoulders, elbows, wrists, hips, knees, and ankles, with visibility labels.
- Racket boxes or racket keypoints and player ownership, including occlusion and actual negative intervals.
- Indexed court landmarks/lines, visibility, camera segment boundaries, and independent known court-plane locations.
- Contact/bounce timing when used for flight or speed reconstruction, with annotation uncertainty.
- Segmentation masks on a representative subset if the mask module is promoted.
- Independent speed/3D references when available, with their own calibration and timing errors.

Audit loose boxes, missing labels, annotation role errors, false absence labels, and disagreement before interpreting metrics. Include difficult fast serves, faint balls, net crossings, racket/player occlusions, distant players, white shoes/logos, spectators, camera cuts, and dead-ball sequences. Use targeted active learning to label the failures that constrain accuracy, rather than collecting large numbers of redundant easy frames.

## 7. Select a small, accurate production vision system

### Ball

Keep temporal information because the ball is small, fast, and often blurred. Analyze native-resolution and crop behavior, proposal coverage, false-ball distractors, frame-rate-aware temporal spacing, and selection/filter losses. Separate a model's failure to propose the ball from a downstream filter rejecting a valid proposal.

Use confidence-aware temporal association and bounded gap recovery. Store observation state and provenance for every frame. Preserve the distinction among detected, tracked, interpolated, predicted, occluded, and missing observations. Gap limits must use time as well as frame count. Avoid smoothing away genuine hits or bounces. Exclude unsupported gaps and extrapolated coordinates from authoritative speed/placement measurements.

Score direct observations and inferred gap recovery separately so interpolation cannot inflate the detector's claimed accuracy.

The current candidate has documented development improvements, but also poor difficult-clip performance. Later spacing/head/threshold candidates have their own no-promotion decisions. Compare on the intended validation data with identical labels and metric definitions; qualify the chosen revision before making it the API default.

### Players and identities

Detect the two actual participants rather than relying on successful court calibration as the only way to find them. Combine image evidence, appearance/motion continuity, court context, and plausible participant behavior. An incorrect court fit must not silently erase every player or admit spectators as participants.

Track identity consistently through short occlusions and end changes where evidence permits. Separate persistent tracked identity, near/far court role, display name, and handedness. Use generic labels until player names are supplied or confirmed. Use ankle/foot-ground evidence for court location when validated, with an explicit bbox-foot fallback. Preserve nullable ground positions through calibration loss and re-identification uncertainty.

### Pose and the requested skeleton

The requested lines on the player are a body skeleton overlay from pose estimation. Render torso, arms, and legs from actual per-frame joints, including shoulders, elbows, wrists, hips, knees, and ankles. If a torso center line is useful, derive it from documented shoulder/hip midpoints. Keep raw keypoints and their confidence separately from display smoothing.

Associate pose with player tracks, expand crops enough to include swing reach, and validate near/far-player wrist and ankle quality. Fade or hide unsupported joints and segments. The current 17-joint system does not describe individual fingers or exact racket orientation; introduce hand/whole-body/racket keypoints only when the footage resolution and labeled data support the required detail. Use 2D motion for 2D joint analysis; calibrated 3D biomechanics requires an independently validated reconstruction.

### Racket

Start from the existing crop detector and ownership logic. Improve it with permitted tennis-specific racket data and fine-tuning if recall or assignment quality remains weak. Preserve racket identity, confidence, visibility, and player association. Evaluate handle/head keypoints when useful for orientation/contact. Wrist distance alone is an insufficient ownership guarantee when two people or objects overlap.

Keep existing checkpoint-loading safety gates. Prefer trusted published tensor-only assets or training a compatible specialist when a checkpoint requires unsupported executable deserialization. This technical issue must not derail the whole product or justify bypassing an existing safeguard.

### Segmentation and augmentation

Segmentation is the visible filled silhouette around a player. Add a subtle, track-linked mask overlay with a toggle and appropriate confidence/occlusion behavior. Validate whether masks improve foot localization, participant filtering, pose visibility, or the presentation. Benchmark the actual memory/latency cost and mask flicker; compute masks in the background or cache them when that preserves required accuracy. Select one compatible mask method rather than stacking redundant models.

Augmentation is a training process: plausible blur, exposure/compression changes, scale, crop, contrast, and camera variation. Keep temporal transformations coherent across sequences and transform annotations consistently. Avoid physically implausible ball paths and court labels. Train only where failure evidence and suitable labels justify it.

## 8. Fix calibration, time, speed, and spatial meaning

Use one canonical metric court coordinate system shared by the backend, 2D view, 3D scene, heatmaps, reports, and tests. Define origin, axis directions, units, singles/doubles lines, net position, and every conversion explicitly. Verify court geometry against the official dimensions.

Evaluate court detection against visible lines and independently annotated landmarks. A low reprojection residual between a predicted template and its own correspondences proves fit consistency, not physical accuracy. Check projective topology, image-line support, visibility, camera motion, and segment-level validity. Detect cuts and incompatible views; recalibrate when evidence supports it and withhold world coordinates when it does not. A small assisted landmark-correction option may be useful for difficult clips, provided the user sees it as assisted calibration and automatic accuracy is evaluated separately.

Use original presentation timestamps when interpreting the source. The existing worker normalizes videos to a constant-frame-rate analysis timeline. Preserve source-to-normalized timestamp/frame mapping, duplicates/dropped frames, resolution transforms, and timing precision. Never derive physical speed from an assumed global 30 FPS or treat duplicated frames as new motion evidence.

For player speed and distance, compute from valid contiguous metric positions and real time deltas, with documented smoothing and uncertainty. Break intervals at unsupported geometry/identity gaps. Validate floor-contact choice and smoothing against independently annotated/reference motion. Report units, interval coverage, and estimate status; limit decimal precision to the evidence.

For ball speed and 3D flight, complete these steps:

1. Establish a calibrated camera model, including necessary intrinsics/extrinsics and lens-distortion treatment or a justified approximation.
2. Retain accurately timed image observations, their visibility/confidence, and validated contact/bounce anchors.
3. Fit temporally constrained flight segments with explicit physics/geometry assumptions, respecting contact discontinuities and camera changes.
4. Assess identifiability and sensitivity to camera, timing, label, and trajectory errors; propagate useful uncertainty.
5. Test reconstruction and speed against independent reference data that did not define the fitted output.
6. Publish only supported estimates, with a clear quality level and interval of validity.

An image-to-court homography maps court-plane points. For an airborne ball, the current projection represents the viewing ray's intersection with the ground plane; it is not the ball's actual ground footprint or height. Differentiating that projection does not establish actual ball flight speed. Never restore those calculations just to populate a dashboard number.

Single-camera airborne reconstruction is underconstrained without additional assumptions or evidence. Actively implement and validate an appropriate estimator; record unresolved speed/height accuracy as an incomplete milestone when necessary. Keep useful 2D observations and player movement available while that work proceeds. The final handoff must distinguish a completed product journey from any unmet scientific accuracy goal.

## 9. Upgrade the 3D court and replay

Reuse TennisCourt3D.tsx, TacticalCourt.tsx, TelemetryDashboard.tsx, SpatialHeatmapPanel.tsx, and the existing geometry helpers where sound. Build a polished court with correct scale, legible markings, a well-proportioned net, restrained materials, credible light/shadows, smooth camera controls, and clear player/ball contrast. Provide broadcast, overhead, follow, orbit, zoom, and reset views where useful.

Create an image-generation concept of the court analysis workspace if that capability is available and helps design decisions. Use it for composition, lighting, palette, and visual hierarchy, then implement the actual scene in Three.js/react-three-fiber. Create custom lightweight assets with Blender or another available 3D tool only when useful. Court dimensions and replay motion come from validated geometry/data, never from the generated reference image.

Use real tracked ground positions for player movement. Drive avatar limbs from qualified pose/reconstruction when available; label a generic body or procedural motion as an illustration. Support confidence-aware disappearance and tracking gaps. A visible enlarged ball marker may aid legibility, but distinguish marker scale from physical ball dimensions.

Provide separate spatial modes:

- Court projection: confirmed player ground movement and clearly described projected ball observations.
- Reconstructed flight: estimated ball xyz and, if supported, pose/racket reconstruction with its quality and uncertainty.

Replace the fixed decorative sine-wave ball arc with measured/validated reconstruction when available. Keep trails, bounce markers, heatmaps, and uncertainty graphics readable. At any selected timestamp, the 3D scene must show the same evidence interval as the video. Preserve lazy loading, reduced-motion behavior, responsive controls, WebGL fallback, and usable 2D analysis when 3D is unavailable.

## 10. Connect complete inference results to the website

Extend src/api/results.py and frontend/src/data/contracts.ts together with a versioned, validated contract. Maintain isolated analysis IDs and explicit capabilities. Include:

- Original and normalized media metadata and time mapping.
- Active production bundle version, checkpoint/config/dataset hashes, and processing provenance.
- Ball image observations, state/source/confidence, and optional qualified world coordinates and speed.
- Player track IDs, identity/role metadata, image boxes, valid court positions, speed/distance samples, and uncertainty/coverage.
- Pose keypoints, visibility/confidence, bone topology, and optional qualified 3D pose.
- Racket detections/keypoints, ownership, confidence, and missing intervals.
- Segmentation-mask references with transforms and alignment metadata; use compact assets rather than bloated per-frame JSON masks.
- Court/camera segments, calibration status, coordinate semantics, and quality.
- Validated optional events and their evidence/time uncertainty.
- Export locations and unavailable-section reasons.

Use null/absent values consistently for unavailable measurements. Preserve valid sections when another optional section fails. Never allow an uploaded analysis to inherit another match's names, speeds, scores, trajectories, or illustrative statistics.

Investigate two known runtime inefficiencies: Phase6Pipeline does not forward all configured player-detector confidence/tracker options, and pose may be loaded twice through separate consumers. Connect settings correctly and share inference where appropriate. Construct only modules required by the selected production path. Package court/ball assets so normal inference does not depend on incidental network downloads.

After deliberate asset acquisition, the standard processing workflow should be self-contained without automatic weight downloads. Document any intentionally selected external inference dependency explicitly.

## 11. Make playback and overlays genuinely synchronized

When video is visible, use decoded video presentation time as the authoritative replay clock. A suitable browser video-frame callback should drive frame-aligned telemetry, with a tested fallback. The current independent animation clock and roughly 120 ms drift correction are insufficient for precise analysis.

Synchronize play, pause, buffering, seek, frame-step, rate change, source/annotated switching, event selection, graphs, 2D, and 3D. If autoplay is blocked or the decoder stalls, hold the corresponding telemetry and present the correct transport state. In a pure 3D view, use an explicit simulation clock that maps back to the same analysis timestamps.

Compute overlays through the full source-to-display transform, including normalization, crops, letterboxing, resize, fullscreen, and device pixel ratio. Provide toggles for players, ball/trail, skeleton, racket, masks, court geometry, and validated events. Use calm stable colors per player, thin readable skeletons, subtle mask opacity, and unobtrusive confidence states. Preserve actual image evidence behind the overlays.

## 12. Deliver the requested homepage

The first screen should visibly demonstrate tennis analysis within a few seconds. Use one prominent playing analyzed clip and at least two additional distinct analyzed previews on desktop, with an appropriate mobile layout. Every example must originate from a different actual match/session/clip source and carry its own real inference artifacts, metadata, calibration, and export. Short curated excerpts are acceptable; identify them honestly as excerpts.

Use muted inline looping previews, browser-compatible encoding, optimized media, and a clear play fallback when autoplay is blocked. Lazy-load secondary previews and pause videos outside the viewport. Preview video can contain baked overlays; any adjacent dynamic telemetry must use that same clip's timestamps and results. Test actual playback, rather than placing thumbnails that resemble video players.

Show practical analysis: visible ball/player tracks, skeleton/racket overlays, movement readouts, qualified speed, and a compact court replay. Display uncertainty in plain language. If ball speed is not validated for an example, use real player speed and accurate tracking evidence while clearly indicating ball-speed availability. Select or acquire authorized clips suitable for proving the required ball-speed/reconstruction capability; unresolved values remain part of the engineering work.

Keep the Upload video button in the upper-right header on desktop and obvious on mobile. Use short copy explaining upload, analysis, replay, and report. A visitor should reach upload or a full example analysis in one click. Navigation should revolve around Home, Upload, and My analyses/Matches. Build a clear visual hierarchy around footage and evidence, with restrained tennis-inspired colors, readable typography, and accessible controls. Keep developer jargon, model names, experiment phases, invented accuracy badges, and unrelated dashboard widgets out of the main journey.

## 13. Finish upload, processing, and results

Keep drag/drop and file selection, validation, upload byte progress, upload cancellation, persistent status, isolated storage, and restart-safe history. The existing cancellation aborts uploading; it does not cancel an accepted inference job. Leaving a page or stopping polling must not imply processing stopped. If processing cancellation is added, require a persistent cancellation state, worker acknowledgement, and safe handling of partial artifacts. Distinguish upload progress from inference progress. Processing stages may include validation, normalization, court registration, tracking, pose/racket/masks, motion/reconstruction, overlay rendering, and report generation. Display actual progress when measurable; otherwise show the true stage rather than an invented percentage.

The results view should make the video and its synchronized 2D/3D replay central. Group movement/speed, pose/racket, placement/heatmaps, and quality into useful sections or tabs. Let users select a player, inspect a frame/time interval, jump to reliable events, and download evidence. Make units and estimate status readable without turning the interface into a diagnostics console.

Handle corrupt media, unsupported views, missing weights, out-of-memory failures, interrupted workers, empty detections, and partial optional analysis with actionable messages. Retain history and completed results across restarts. Measure memory, disk, and total latency on a meaningful long clip before advertising full-match support; the current configuration's two-hour limit is not a verified two-hour capacity claim. Bound in-memory arrays/caches, use chunking or streaming when necessary, and document operational storage/retention behavior.

Keep the local end-to-end website working. If public multi-user hosting is an authorized delivery target, add the necessary access isolation, non-public upload storage, quotas, and worker deployment for that target without creating unrelated account-management features. Use existing approved infrastructure where available. Seek missing credentials or paid-resource authorization only for a concrete blocked action, while continuing independent local work.

## 14. Produce a useful downloadable report

Generate a readable PDF or a self-contained HTML report that matches the selected analysis and displayed values. Include clip details, processing/input quality, player movement and distance, qualified speed summaries, heatmaps, ball/racket/pose coverage and quality, useful validated events, and selected timestamped evidence frames. Explain estimated and unavailable measurements in plain English and use sensible numerical precision.

Keep annotated video, JSON, and useful CSV exports alongside the readable report. Reports must include the analysis ID, production bundle version, units, valid measurement intervals, and a concise limitations statement. Keep machine throughput and model metadata in a technical appendix or structured export. Test that downloads open, media plays, report figures are legible, and all exports belong to the correct clip.

## 15. Training, measurement, and promotion gates

Use transferable pretrained weights where suitable and licensed. Before expensive training, validate labels, run a short inference/training smoke check, confirm output resolution and temporal preprocessing, and verify that a tiny subset can be learned. Record environment, hardware, memory, data version, seed, augmentations, hyperparameters, checkpoint hashes, validation criteria, duration, and stopping rule. Resume supported runs and retain the best validated checkpoint.

Spend compute on evidence-backed interventions. Reduce redundant pose passes, crop duplication, and inactive modules where measurements show a benefit without harming accuracy. Optimize only after determining whether the weak link is data, proposals, association, calibration, or reconstruction. Use local/authorized compute first; obtain an actual approved budget before creating paid resources if one is not already provided.

Freeze metric definitions and acceptance thresholds before the untouched test. The following are proposed goals for the declared supported input scope, not claims about the present project:

| Capability | Evaluation and initial goal |
| --- | --- |
| Ball | Aim for precision across all annotated visible and confirmed-absent frames, and visible-ball recall, of at least 95% at a predeclared scale-aware center tolerance. Report localization median/p95, a separate absent-frame false-positive rate, missed gaps, and per-clip/hard-case results. |
| Active players | Aim for at least 95% detection/selection precision and recall and IDF1 of at least 90% on labeled supported clips; also report ID switches and failure intervals. |
| Court and player ground positions | Declare a meter/pixel error budget after reference-label inspection and before final evaluation. Test independent landmark/position error and camera-failure rejection, separately from fit residual. |
| Body pose | Report visibility-aware PCK/OKS and temporal failures, with wrists/ankles and near/far players broken out. Freeze a useful joint-error threshold from the validated label protocol. |
| Racket | Report localization precision/recall/AP and correct player ownership on visible and absent/occluded intervals. Define the promotion threshold before the final test. |
| Masks | Report mask overlap/boundary quality, identity consistency, flicker, runtime, and whether the overlay/geometric cue is useful. |
| Player speed/distance | Measure absolute/relative error against independent references with uncertainty and interval coverage; set the acceptable budget before qualification. |
| Ball speed/3D | Require independent reconstruction/speed evidence, identifiable assumptions, and declared uncertainty. A plausible arc or projection derivative cannot pass this gate. |
| Synchronization | On qualified clips, verify overlays and spatial views match the presented video within one normalized frame through play, seek, step, rate changes, and buffering; document timing limits. |

For every accuracy claim, name the dataset/split, sample count, metric, tolerance, model version, conditions, and limitations. Report both aggregate and clip-level results. Estimate statistical uncertainty by match/session groups rather than treating correlated neighboring frames as independent samples. Investigate severe failures even when pooled metrics pass. Test synthetic perturbations as supplemental checks, while reserving real held-out footage for practical accuracy claims. Keep confidence values distinct from localization accuracy and tracking availability distinct from correctness.

Select the production bundle on validation evidence, freeze it, and then run final qualification once. If it fails, keep the best reproducible candidate, document the failure, and use new development/final data for the next attempt. Avoid repeatedly tuning on a supposedly untouched test or quietly relaxing its threshold after viewing the result.

## 16. Execution sequence and stage completion

Work in the following sequence, parallelizing independent frontend/research/annotation work where helpful:

1. Preserve and reproduce: capture the dirty baseline, identify active assets and dependencies, reproduce one real upload, and inventory missing capabilities. Done when the current system and its limitations are reproducible without losing user work.
2. Focus and connect: simplify the product navigation, version the full result contract, expose existing pose/racket evidence, and repair video-driven synchronization. Done when the current selected pipeline reaches coherent video/2D/3D/report views with honest missing states.
3. Establish evidence: resolve data permissions, audit labels and failure modes, group splits, and reserve the new qualification set. Done when each planned training/evaluation action has valid inputs and an explicit measurement protocol.
4. Improve vision and geometry: fix active-player/court failures and the limiting ball/racket/pose components through targeted data and experiments. Add useful masks; train only the justified specialists. Done when a bounded, reproducible selected stack improves the declared validation objectives without hiding serious regressions.
5. Implement motion reconstruction: validate player measurements, camera/time semantics, ball flight, speed, and uncertainty. Done when each published measurement has independent evidence or is explicitly recorded as an unmet milestone.
6. Finish the experience: build three distinct playing analyzed examples, upgrade the 3D scene, finish upload/results, and generate readable exports. Done when a new visitor can perform the complete intended journey and identify what the site does immediately.
7. Consolidate and qualify: finish safe cleanup, pin the production bundle, verify a clean setup, run final accuracy/product checks, and measure supported capacity. Done when the final deliverable and remaining scientific limits are supported by reproducible evidence.

Maintain one concise progress record containing completed gates, current work, selected production assets, measured outcomes, unresolved blockers, and the next meaningful action. Avoid creating another stack of overlapping plans. For future goal-driven execution, retain the objective and resume from this record rather than repeating completed experiments. Set the goal complete only when its required product and accuracy outcomes are actually satisfied.

## 17. Verify the delivered product

Use existing meaningful backend/frontend checks and extend them around changed behavior. Recheck the available scripts before execution; the current repository exposes a Python test suite, frontend pnpm check, and scripts/run/verify_product.py. Those establish software behavior and artifact integrity; separate evaluation establishes model accuracy.

Required final evidence:

- A real browser journey from home to upload, persistent job, completed analysis, synchronized replay, and downloaded report/video.
- Three different authorized analyzed demo clips with actual playing media and matching artifacts.
- Actual browser checks for autoplay fallback, buffering, resize/fullscreen, frame-step/seek/rate changes, overlay placement, and video/2D/3D agreement.
- Dense difficult-interval visual review, alongside quantitative component and end-to-end metrics.
- Correct missing/occluded/camera-invalid behavior, with no invented coordinates, speed, or player identity.
- Corrupt/unsupported media, interrupted jobs, missing assets, restart persistence, and analysis-ID isolation checks.
- Mobile usability, keyboard accessibility, reduced motion, and WebGL fallback.
- Readable correct reports and usable media/JSON/CSV exports.
- Declared long-clip processing limits supported by measured time, RAM/VRAM, and disk footprint.
- A fresh documented installation that restores the selected code, assets, dependencies, and demos through tracked files or reproducible acquisition. Review .gitignore so it does not accidentally exclude required frontend manifests, lockfiles, and demo packaging.
- Before/after cleanup evidence and no broken runtime references.

End with a concise handoff: how to run/open the website, the selected model system and why it was chosen, what was removed/consolidated, measured accuracy and runtime, example-analysis/report locations, and any remaining concrete limitation. Call out an unmet measurement target explicitly rather than declaring success because the interface looks polished.

The successful result is a focused tennis website where a visitor sees real analyzed play immediately, uploads a video easily, obtains trustworthy ball/player/racket/pose analysis and movement information, explores a polished synchronized court replay, and leaves with useful evidence and a report. Make that experience accurate and complete, and keep the implementation limited to what serves it.
