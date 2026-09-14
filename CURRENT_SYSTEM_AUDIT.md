# Current system audit

2026-09-14: opt-in original-view court recovery passes five controlled camera-cut cases, independent artifact verification and 35 focused tests. Both sustained returns resume at frame 96 (0.10 seconds after return); healthy matrices remain exact and unrelated/brief views remain withheld. A separate 150-frame real-model CUDA pipeline check reproduces 114 valid frames, preserves missing ground coordinates/speeds across the gap, and fully decodes the annotated video. Runtime including model setup is 48.8064 seconds. Existing configs/models remain unchanged; the new opt-in recipe is wasb_returning_view_validation.yaml. This does not establish natural-cut, identity, ball or physical-speed accuracy. Production readiness remains NO. See docs/experiments/COURT_RETURNING_VIEW_RESULTS.md.

2026-09-14: the controlled FPS-aligned head experiment completes and fails its frozen gate. At fixed .20 on 600 internal-selection labels, aligned and adjacent heads both score P91.1263% / R93.8489% / F1 92.4675%, while aligned correct-proposal coverage falls from 546 to 545 of 569 visible labels. All sample/view/output-slot choices match the seeded reference, all non-head tensors remain exact, and independent scores/provenance verify. The sole recovery over original weights was already achieved by the adjacent head and is visually reviewed. Twenty-two focused tests pass. No external run or runtime promotion follows; the verified aligned cache remains reusable. See `docs/experiments/WASB_SPACED_HEAD_PILOT05_RESULTS.md` and `WASB_SPACED_TRAINING_DATA_RESULTS.md`.

2026-09-14: head-only WASB adaptation completes training and both verified selection/development comparisons. On 900 development labels, selected-0.35 precision/recall rises from 95.4272%/95.6574% to 95.5475%/95.7780%, entirely from one blurred-streak localization crossing the tolerance boundary. The frozen raw gate passes; proposal coverage stays 798/829 and only eight of eighteen clips meet both targets. A separate candidate-preserving feasibility check caps recall below the current filtered pipeline (798 versus 803 possible/current correct balls), so further continuous compute is held pending better proposals. No runtime promotion or readiness change follows. All 900 reused points/candidates and independent scores verify, the sole changed outcome is visually reviewed, and sixteen focused tests pass. See `docs/experiments/WASB_HEAD_EXTERNAL_RESULTS.md` and `WASB_HEAD_ADAPTATION_RESULTS.md`.

Training data expansion on 2026-09-14 adds the next twenty fixed publisher-training clips, bringing the audited corpus to sixty clips and 2,999 labels. The previous split is preserved (48 training / 12 internal-selection clips), all 6,751 new frames decode, and independent source/count checks pass. All 35 preselected examples were inspected; contact clutter and uncertified absence labels remain explicit limitations. Eight new tests pass. This is training preparation, with no new model or runtime change. See `docs/experiments/BALL_TRAINING_EXPANSION60_RESULTS.md`.

A frozen external check of the original model's training-selected 0.35 threshold completes all 900 development labels. Sparse raw precision/F1 improve from 93.4884%/95.2043% to 95.4272%/95.5422%, while recall falls from 96.9843% to 95.6574% (eleven fewer true balls, eighteen fewer false detections). Both pooled targets pass, but individually passing clips fall from ten to eight and the frozen no-recall-loss gate fails. All 900 control coordinates reproduce exactly, independent scoring and seventeen focused tests pass, and five selected examples are visually reviewed. Preserve the tradeoff without runtime promotion. See `docs/experiments/WASB_SELECTED_THRESHOLD_RESULTS.md`.

The previously completed training-threshold inference now has an independently verified completion review. Across 400 labels on eight reserved training matches, the original checkpoint at its selected 0.35 threshold achieves F1 94.5260%, versus 94.3700% for spatial epoch 03 at 0.50. All ten pooled scores replay exactly, sixteen inference reference checks are verified, five selected examples are visually inspected, and eleven focused tests pass. Keep the adapted checkpoint rejected under the frozen protocol; no external promotion run or runtime threshold change follows. See `docs/experiments/WASB_TRAINING_THRESHOLD_RESULTS.md` and `MODEL_PROGRESS.md`.

The current spaced-ball error audit now includes candidate evidence for all 900 labels. All 250 newly replayed 25 FPS predictions match the saved stream exactly. The 26 misses divide into nineteen proposal failures, five surviving-candidate selection failures, one merge suppression and one filtering loss; twenty-nine of 43 false detections occur on absent labels. Nine preselected examples and the single merge case were visually inspected. Final P94.9173%/R96.8637% is unchanged; this diagnosis directs further proposal/training work. See `docs/experiments/BALL_SPACED_ERROR_AUDIT_RESULTS.md`.

The frozen calibration-ratio comparison completes 183 reused cases. Reducing required support from eleven to ten of fourteen landmarks adds two Madrid fits, which refine to 28/28 label matches, but also accepts a visibly false court over spectators in an amateur video. Every existing accepted fit stays identical, and independent metrics/matrix checks pass. All three new fits were visually reviewed. Reject the global ratio reduction; runtime calibration remains unchanged. See `docs/experiments/COURT_CALIBRATION_SUPPORT_RESULTS.md`.

The frozen combined court refinement completes all 125 remaining pilot images: 1,439 to 1,519 correct matches versus original refinement (101 gains, 21 losses). Label-agreement precision/recall improves 83.81%/82.46% to 88.47%/87.05%. Seventeen regression images, six selected gains and both rejected initializations were visually reviewed; independent metrics and geometric invariants pass. Eighteen initially correct landmarks are still lost, and two plausible court views fail calibration. These are reused training/development images, with no runtime promotion. See `docs/experiments/COURT_REMAINING_PILOT_RESULTS.md`.

Combining the frozen no-white line evidence with boundary-preserving refinement completes all 64 reused court cases. Selection agreement reaches 387 matches, versus 386 for no-white alone and 370 for the original refinement; losses of initially correct points fall from seven to two. Across 40 labeled images the combination gains 20 matches and loses one versus the original refinement. All five changed images and the two remaining initial-point regressions were visually inspected, with independent metric and geometric-invariant verification. This remains a research candidate. See `docs/experiments/COURT_RIDGE_BOUNDARY_RESULTS.md`.

Three frozen photometric ablations complete 192 evaluations across 64 reused court cases. Removing only the whiteness filter raises selection matches from 370 to 386 and external-source matches from 65 to 67, with 22 gains and four losses relative to prior refinement. All 21 changed cases and four losses for this candidate were visually inspected; all three metric replays and 37 focused tests pass. Local regressions and annotation limitations remain, so this is a leading research variant with no runtime promotion. See `docs/experiments/COURT_RIDGE_ABLATION_RESULTS.md`.

A boundary-preserving court refinement completes 64 reused case/model evaluations. It repairs the four diagnosed landmark regressions and loses no initially correct matches across 40 labeled images, but selection agreement is 369 versus 370 matches for the previous refinement (eight gains, nine losses relative to that refinement). All twelve changed corrections and nine lost previous matches were visually inspected; 33 focused tests pass. Hard constraints can preserve an incorrectly placed boundary, so this remains a research tradeoff with no runtime integration. See `docs/experiments/COURT_BOUNDARY_REFINEMENT_RESULTS.md`.

The unchanged court refinement completes all 32 preselected internal controls: 293 to 370 correct landmark matches among 447 eligible labels, with 81 gained and four lost matches across two images. All 20 accepted corrections and four native-resolution regressions were visually inspected; 18 focused tests pass. Missing baseline constraints can permit local regressions despite increased overall line support. Keep this correction research only pending revision and independent validation. See `docs/experiments/COURT_REFINEMENT_SELECTION_RESULTS.md`.

The frozen court refinement improves unchanged publisher-label agreement from42 to65 of112 landmarks across eight images, with23 gained matches and none lost relative to projected geometry. All eight image panels were visually reviewed and18 targeted tests pass. Known annotation errors remain in the score, so this is not independent court-accuracy qualification. See `docs/experiments/COURT_REFINEMENT_RAW_LABEL_RESULTS.md`.

A frozen projective court-line refinement corrects the observed baseline-placement error in six sampled frames from two clips. Propagating only each first-frame correction through saved camera motion preserves improved sampled line support across850 rendered frames; both videos fully decode and15 targeted tests pass. Eighteen earlier controls accept no correction, but15 fail before refinement, so broad court accuracy remains unproven. See `docs/experiments/COURT_PROJECTIVE_LINE_REFINEMENT_RESULTS.md`.

The temporal-spacing research recipe is now integrated as a separate opt-in pipeline configuration. Real-model runs on 60 FPS and 25 FPS footage reproduce every benchmark ball coordinate and missing state exactly across 850 frames; both videos fully decode. All 63 targeted tests pass. Processing is 2.81-3.11 FPS including setup, and visual review still exposes court-placement errors. See `docs/experiments/BALL_SPACING_PIPELINE_INTEGRATION.md`.

Completed temporal-spacing validation covers all 6,082 frames and 600 labels in the twelve-clip expansion. With identical final filters, ball precision improves 94.35% to 94.64% and recall 95.40% to 97.61% (518 to 530 TP; 31 to 30 FP). All 25 changed outcomes were visually reviewed, and a 600-frame, 60 FPS hard-clip comparison fully decodes. Across both development sets, P94.92%/R96.86% still misses the precision target; only 9 of 18 clips individually clear both targets. This remains a research candidate. See `docs/experiments/BALL_TEMPORAL_SPACING.md`.

Temporal-spacing candidate completes all2,893 frames of six development clips (2,394 newly inferred,499 verified cached). Final P95.45%/R95.45%, versus adjacent-frame P95.77%/R95.10%: three gained true balls, two lost, one additional FP. Sparse/continuous predictions agree exactly. A600-frame60FPS comparison video fully decodes and sampled frames were inspected. This six-clip tradeoff remains part of the completed expansion comparison above. See `docs/experiments/BALL_TEMPORAL_SPACING.md`.

TOTNet tennis pilot completes 300 labeled predictions across six existing validation clips (2,561 decoded frames). Raw P87.00%/R88.78%; exact-origin absence P90.00%/R88.78%. Same-clip WASB overlap control reaches P97.18%/R93.88%; TOTNet adds 5 true detections but loses 20. Exact tensor architecture, author preprocessing, unchanged paired labels and metric replay are verified; six source-frame errors were visually inspected. No production promotion. See `docs/experiments/TOTNET_BALL_PILOT.md`.

Consolidated measured player comparison: `docs/experiments/PLAYER_DETECTOR_TRACKER_COMPARISON.md` records all five detector/tracker/racket combinations with their common scope and limitations.

RT-DETR-l comparison completes all 2230 UVY frames. Tracked person coverage gains 322 labeled boxes, but downstream original-racket player selection regresses from P91.71%/R27.10% to P71.76%/R25.93%, with 326 additional false selections. Reviewed false supports follow handheld flags and a courtside nonplayer; V02 still has no racket evidence. Checkpoint/replay hashes and the fully decoded 968-frame comparison video are verified. RT-DETR remains rejected for production; see `docs/experiments/RTDETR_PLAYER_PROPOSALS.md` and `RTDETR_RACKET_PLAYER_SELECTION.md`.

Paired BoT-SORT comparison with the original racket model gains 148 net matched player boxes with the same 103 false selections across all 2230 UVY frames. Precision improves 90.59% to 91.71% and recall 23.58% to 27.10%; V02 remains empty and V01 IDF1 regresses. The clipped ByteTrack control reproduces every prior selected frame exactly. The trained racket model adds a spectator track, so it is not preferred. Both paired videos fully decode and targeted gains were visually reviewed. Research-only; see `docs/experiments/TRACKER_ORIGINAL_RACKET_SELECTION.md` and `TRACKER_RACKET_SELECTION.md`.

Trained-racket downstream evaluation completes all2230 UVY frames:301 additional matched player boxes, but170 additional unmatched selections and V01 ID switches1 to3. Aggregate precision falls90.59% to82.56%, recall rises23.58% to30.74%. Reviewed V02 examples expose loose publisher boxes, without changing labels or scores. Both comparison videos fully decode;38 targeted tests pass. Research-only; see `docs/experiments/TRAINED_RACKET_PLAYER_SELECTION.md`.

Brief duplicate-ID reconciliation recovers59 additional actual player boxes in V01 with no added false selections; its IDF1 improves37.57% to42.14%. Combined with image-flow linking, the candidate gains119 matches across two recordings, but aggregate recall is only23.58% and V02 remains empty. All42 targeted tests pass; the shared-ID boundary and819-frame comparison video were verified. Research-only; see `docs/experiments/PLAYER_DUPLICATE_HANDOFF.md`.

Image-flow fragment linking recovers60 additional actual player boxes on the third UVY recording without adding false selections; IDF1 rises63.18% to69.68% and HOTA53.37% to61.13%. All2230 frames complete, all four links were visually reviewed, and36 targeted tests pass. Aggregate player recall remains22.17%, so this is an experimental recovery rather than production qualification. A443-frame paired video and exact observation-integrity audit are saved. See `docs/experiments/PLAYER_FLOW_FRAGMENT_LINKING.md`.

Player-recovery diagnosis completed: all102 labeled broadcast rackets fit entirely inside existing crops, yet36 lack a matching raw proposal. Static and bidirectional-motion fragment-linking pilots complete all2230 UVY frames but recover no additional selected players. The target near-player ID boundary has inconsistent local box motion;17 targeted tests pass and every accepted boundary was visually reviewed. Both candidates remain research-only. See `docs/experiments/PLAYER_FRAGMENT_LINKING.md` and `PLAYER_MOTION_FRAGMENT_LINKING.md`.

Court-independent player selection now has a measured ownership fix: nearest-person crop assignment removes445 false selected boxes across2,230 UVY frames while preserving872 matched boxes on every frame. Precision rises61.41% to89.44%, but recall remains20.75% and one clip emits no players. COCO racket precision improves82.74% to91.58% with two lost matches; sparse match results improve slightly. All18 targeted tests pass and a443-frame paired video is fully decoded. The candidate remains experimental; see `docs/experiments/UVY_RACKET_SUPPORTED_PLAYERS.md`.

Pretrained-context verifier pilots completed and rejected. Global visual-score ranking gives P84.43%/R82.87%; preserving detector order among visually accepted proposals improves this to P89.31%/R87.66% and reduces wrong visible localizations61 to35. Both remain below the strongest tiled configuration. Complete feature/checkpoint hashes and paired decisions were verified;19 feature/cache tests and9 ordering/metric tests passed. See `docs/experiments/VERIFIER_PRETRAINED_CONTEXT_PILOT05.md` and `VERIFIER_ACCEPTED_ORDER_PILOT06.md`.

Training-negative quality ablation completed: excluding nine questionable training negatives raises verifier recall86.74% to88.95% but lowers precision94.39% to93.24% against the matched control. Both eight-epoch runs and all checkpoint hashes were verified;12 targeted tests pass. Labels and validation data are unchanged, and the model is rejected for production. See `docs/experiments/VERIFIER_NEGATIVE_QUALITY_PILOT04.md`.

Complete ball absence-error review:14 of26 remaining predictions visibly follow non-ball objects;5 appear to be moving balls despite publisher absence labels;7 remain unresolved. This is an assistant diagnostic on a selected error subset, not independent ground truth. Labels and measured scores are unchanged. Further training must address both coherent player/racket distractors and negative-label quality. See `docs/experiments/BALL_ABSENCE_LABEL_AUDIT.md`.

Ball temporal detour replay removes two labeled false detections across18 reused development clips without losing a labeled true ball. Expansion P94.35%/R95.40% and additional-set P95.77%/R95.10% remain insufficient for broad qualification. All16 rejected frames were visually reviewed; one unlabeled case remains uncertain. The rule remains a research replay, and a possible publisher absence-label conflict is recorded without changing scoring. See `docs/experiments/BALL_TEMPORAL_DETOUR.md`.

Mixed-data racket pilot04 reaches match recall83.33% but precision69.67%; COCO P75.46%/R72.44% still trails the original80.89%/80.89%. It retains15 of19 large-racket matches and improves over earlier trained candidates on COCO, but is rejected for production. Training interruptions and restricted checkpoint verification are documented in `docs/experiments/RACKET_REHEARSAL_PILOT04.md`.

Racket scale pilot03 recovers16 of19 large COCO racket matches lost by pilot02, supporting a scale-coverage diagnosis. Overall COCO P70.31%/R71.56% and match P67.29%/R70.59% still fail to improve the original detector. The candidate is rejected; no production defaults changed. See `docs/experiments/RACKET_SCALE_PILOT03.md`.

Racket training pilot02 improves match-frame recall from63.73% to80.39% but lowers precision to71.93%. COCO recall falls from80.89% to64.44%, including loss of all19 large-racket matches. The checkpoint is rejected for production. A verified fused-weight initialization failure in pilot01 was corrected before this run; see `docs/experiments/RACKET_TRAINING_PILOT02.md`.

Shared racket assignment recovers five additional COCO rackets (P82.74%/R83.11%) but leaves the92-frame match benchmark unchanged (P77.38%/R63.73%). Raw crop proposals cover only66 of102 labeled match-frame rackets even with label-assisted matching, identifying a detector limitation. The assignment remains experimental; see `docs/experiments/RACKET_GLOBAL_ASSIGNMENT.md`.

Racket comparison: three fixed-confidence modes completed on167 COCO tennis images with225 racket annotations. Full-frame1024 reaches P81.90%/R84.44%; existing person crops reach80.89%/80.89%. A fixed global overlap filter improves crop precision to85.85% with unchanged IoU0.5 recall, but removes no boxes on the92-frame RacketVision replay. Duplicate boxes and unresolved player-racket ownership remain. No production promotion; see `docs/experiments/COCO_RACKET_COMPARISON.md`.

Court image-evidence diagnostic: fixed bright-line support rejects all12 visibly misplaced first-frame court templates and accepts5 of6 broadcast controls. The rejected broadcast control also shows displaced left-side landmarks on visual review. Direct lines recover useful markings on a ground-level frame, but surface segmentation loses distant regions and fails one fan view. This is development evidence, not a production gate or complete court detector. See `docs/experiments/COURT_LINE_EVIDENCE.md`.

Court training pilot01 completed three epochs on125 images from64 source groups, with32 internal-selection images and8 external-source images. Raw landmark matches rise75 to78 of112, but visual review finds serious publisher label errors, so this is not verified accuracy improvement. Side/ground-view detection remains insufficient. The pilot is not promoted; see `docs/experiments/COURT_HEATMAP_TRAINING_PILOT01.md`.

Court architecture update: a separately inspected fifteen-channel heatmap checkpoint detects14/14 landmarks on all six broadcast control frames, but only0-8 points on side-view first frames and0-2 on nine ground-level samples. The strongest side-view output has incorrect landmark roles. It is not promoted. CalTennis camera projection hypotheses broadly align visually, but exact coordinate conventions and image scaling remain unverified. See `docs/experiments/COURT_HEATMAP_CANDIDATE.md` and `docs/research/CALTENNIS_CAMERA_CONVENTIONS.md`.

Camera failure diagnosis: all12 quarter-rotated court predictions remain visibly wrong. A new complete person-proposal audit measures IoU0.5 coverage before court selection: tracked640 is91.97%/32.59%/63.14% across three fan recordings; raw1024 is96.19%/32.95%/77.37%. These are oracle localization diagnostics against imperfect original labels, not role-tracking accuracy or precision. V02 also has visibly loose publisher boxes. Rotation and higher resolution do not solve camera-general player tracking; defaults remain unchanged. Evidence: `docs/experiments/UVY_PLAYER_TRACKING_BASELINE.md`.

Court candidate check: a geometrically augmented research checkpoint restores298 player detections on one side-camera clip but still draws the wrong court and fails both other clips. It is not promoted. Improved role IDF1 alone cannot validate physical court geometry; the annotated overlay and paired results are recorded in `docs/experiments/UVY_PLAYER_TRACKING_BASELINE.md`.


Side-camera validation update: all three acquired fan-recorded Wimbledon clips complete, but the upgraded court-dependent player selector emits **zero tracks across2,230 frames**. Court landmarks are visibly predicted on spectator stands; one incorrect fit nevertheless passes geometric consistency checks. Legacy role selection also performs poorly on two clips. This is a confirmed camera-generalization failure, separate from known publisher-label errors. Full evidence: `docs/experiments/UVY_PLAYER_TRACKING_BASELINE.md`. Universal reliability remains **NO**; court localization and fallback player association require further work.


2026-09-10 update: spatial training pilot03 completed three epochs on 32 training match IDs with 8 internal-selection IDs. Its frozen twelve-clip external result worsens precision to **88.61%**, with recall **95.95%**, after the same residual filter. It is rejected; the original candidate remains 94.18% precision / 95.40% recall on that development set. New fan-recorded tennis videos are acquired for temporal tracking diagnosis, but visual checks found publisher GT omissions/misclassification; they cannot establish production qualification. The full regression suite passes **535 tests**, with one existing warning. See `docs/experiments/WASB_SPATIAL_TRAINING_PILOT03.md` and `docs/experiments/UVY_PLAYER_TRACKING_BASELINE.md`.


Latest integrated candidate (2026-09-09): tiled WASB plus pixel motion and a guarded patch-residual filter scores **94.18% precision / 95.40% recall** on twelve development clips and **95.44% / 95.10%** on six added development clips. It remains experimental. The complete difficult match148 run scores **77.50% / 81.58%**, processes 600 frames in 181.53 s (3.31 FPS), and matches the saved replay exactly on all 600 exported positions/missing states. The fully decoded broadcast review includes the independently benchmarked YOLO11s-pose candidate. **503 regression tests pass**, with one existing warning. Full protocol, failed similarity-only gate and artifact paths: `docs/experiments/BALL_PATCH_SIMILARITY.md`. Universal reliability remains **NO**.


Previous measured checkpoint: the five-view ball candidate reaches 95.40% recall on the twelve-clip expansion but only 93.67% precision; it is not promoted. Independent COCO tennis-context validation now covers 167 images. Under identical detected-crop settings, changing YOLO11n-pose to YOLO11s-pose improves OKS AP from 45.21% to 54.49% (+9.28 percentage points), and average recall from 53.11% to 61.55%. This supports a separate full-video pose candidate, not reliable biomechanics or temporal identities. The unchanged existing person detector scores 70.32% box AP; a 1024-resolution replacement worsens overall box AP. The full regression suite passes 494 tests with one existing warning before the thin crop-benchmark addition, which was then exercised by two completed real-data runs. See docs/experiments/COCO_TENNIS_PEOPLE_BASELINE.md and docs/experiments/BALL_TILED_PROPOSALS.md.


Audit started 2026-09-07, before implementation changes. Status: **baseline measured in part; production qualification fails**.

Architecture: [component map and code findings](docs/architecture/VISION_SYSTEM_ARCHITECTURE.md). This audit preserves the user's pre-existing changes and historical artifacts. New measurements belong under `outputs/vision_upgrade_audit/` and `artifacts/validation/vision_upgrade/`.

## Fresh baseline evidence

| Check | Result | Meaning |
|---|---|---|
| Main/reference clones | Complete source snapshots `644808f` / `0949a31` | Local working baseline remains `cee65bb` plus existing changes |
| Runtime | Python 3.13.5, torch 2.13.0+cu126, OpenCV 4.13.0, Ultralytics 8.4.36 | Existing Python installation; default MSYS Python lacks torch |
| GPU | NVIDIA RTX 4050 Laptop, 6,141 MiB total VRAM; CUDA available | No new environment installed |
| Existing tests | **352 passed**, 1 deprecation warning, **9.40 s** | Software/artifact checks, not CV accuracy |
| Phase 6 real-video smoke | **FAILED**, `NameError: width is not defined`, line 213 | Fails after model inference, before tracking/export |
| Smoke video | Existing sample, **214 frames**, **1920Ãƒâ€”1080**, **30 FPS** | 7.13 s development clip; not generalization evidence |
| Player presence on smoke | P1 **214/214**, P2 **214/214** | Coverage only; no independent box/ID annotations |
| Smoke player pass | **7.97 s**, including first-inference overhead | Observed wall time, not full pipeline FPS |
| Smoke ball proposal pass | **2.91 s** | Proposal extraction only, no precision/recall established |
| Court fit smoke output | **0.0220**, printed as px by old code | Actually mean destination-space residual in **m** |

Commands run unchanged before repairs:

```powershell
& 'C:\Users\ac-98\AppData\Local\Programs\Python\Python313\python.exe' -m pytest tests -q -p no:cacheprovider --tb=short
& 'C:\Users\ac-98\AppData\Local\Programs\Python\Python313\python.exe' -c "from src.pipeline.phase6_pipeline import Phase6Pipeline; Phase6Pipeline().run('data/sample_videos/input_video.mp4','outputs/vision_upgrade_audit/baseline_phase6')"
```

## Requested accuracy measurements

| Measurement | Current evidence | Required next evidence |
|---|---|---|
| Ball precision, recall, F1, FP, FN | Fresh 100-image validation comparison below; all models fail the >95% target | Independent source-grouped videos with actual negatives and point/visibility labels |
| Ball localization and false-object errors | 214-frame boxes/categories appear detector-derived; independence unverified | Independent point/visibility labels plus absent-ball negatives |
| Player detection/box accuracy | No independent player-box benchmark identified | Labeled near/far player boxes, IoU/AP and miss counts |
| ID switches/re-identification/occlusion recovery | Historical diagnostics and synthetic tests; not independent identity accuracy | Persistent identity labels and explicit occlusion intervals |
| Track consistency and lost duration | Phase 6 crash repaired; fresh trajectories available | Label-conditioned gap/recovery scoring; coverage alone is not accuracy |
| Speed error, direction error | No independent radar/3D or calibrated motion ground truth identified | Calibrated positions/timestamps and reference speed; true airborne speed withheld |
| Trajectory smoothness | Can measure image acceleration/jerk, but smoother may be wrong | Report diagnostics alongside localization and event preservation |
| Cross-match events | Existing 08Ã¢â‚¬â€œ10 set is diagnostic, not untouched | Frozen independent matches after validation-only model selection |

Unavailable metrics are **not zero**. A literature score, static test pass or coverage percentage cannot satisfy the >95% ball precision/recall target.

## Priority findings

1. **P0 execution, repaired:** Phase 6 passed undefined dimensions into temporal tracking. New orchestration tests reproduce the old failure and verify native dimensions and aligned exports.
2. **P0 measurement, Phase 6 repaired:** legacy homography errors/thresholds are in meters but labeled pixels. Phase 6 now fits canonical-to-image with a 3-pixel RANSAC threshold, then inverts; it exports both units and inlier support. Invalid calibration produces null player measurements and no line calls. Older phase APIs retain their legacy behavior pending migration.
3. **P0 validity:** airborne homography displacement is not true ball speed. Confidence tiers do not measure speed uncertainty.
4. **P0 evaluation:** diagnostic clips have been repeatedly tuned against. The historical ball benchmark contains model-derived candidate coordinates and confidence categories. Generalization remains unproved.
5. **P1 integration:** older callers omit actual frame size; default 1280Ãƒâ€”720 bounds can discard valid 1080p observations. Phase 6 omits the standalone proposal filter; declared camera compensation does not estimate camera motion.
6. **P1 players:** near/far fallback can select the same person for both identities and does not expire old positions. Pose is hit-window-only; racket detection and continuous player motion modules are absent.
7. **P1 geometry:** first-frame calibration is reused across cuts/motion; no keypoint visibility or per-segment fit quality. Invalid transforms can return image coordinates/zeros.
8. **P1 output:** final score/all bounces appear before their timestamps; trails bridge gaps; fixed overlays obscure evidence. JSON primarily provides persistence, without the planned relational store.
9. **P1 training:** no enforced group-disjoint splits, motion-blur training, data/license checks or complete checkpoint provenance in runners. Old court registry approval conflicts with its own missing-license entry; do not retrain from that entry without resolving provenance.
10. **P2 reproducibility:** requirements is an entire environment dump, includes three competing OpenCV packages and CUDA wheel variants; clean recreation has not been validated. Only about 9.46 GB disk was free at initial inspection, so avoid a blind duplicate GPU install or large dataset download.

## Upgrade gates

First restore real inference and reproducible evaluation; then benchmark ball detectors/temporal models using validation-only selection. Add continuous pose/motion and real racket observations, camera-aware geometry and physically valid measurement contracts. Event expansion remains behind independently measured tracking quality. Generate new annotated outputs and failure examples for every accepted change. Keep the final generalization test untouched until the selected model/config is frozen.

## Fresh detector comparison (2026-09-07)

100 existing validation images, 101 ground-truth boxes, **zero negative images**. Fixed confidence 0.25, matching IoU 0.50, NMS IoU 0.70. Maximum-cardinality one-to-one matching counts wrong locations as FP+FN and duplicate detections as FP. These are fixed operating-point metrics, not COCO AP. Original source-match separation is unverified; these results cannot qualify generalization.

| Model | Input size | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Current tennis YOLO11s | 640 | 82 | 21 | 19 | 79.61% | 81.19% | 80.39% |
| Current tennis YOLO11s | 1024 | 88 | 28 | 13 | 75.86% | 87.13% | 81.11% |
| Current tennis YOLO11s | 1280 | 79 | 87 | 22 | 47.59% | 78.22% | 59.18% |
| Legacy tennis YOLOv5 | 640 | 35 | 7 | 66 | 83.33% | 34.65% | 48.95% |
| Legacy tennis YOLOv5 | 1024 | 56 | 18 | 45 | 75.68% | 55.45% | 64.00% |
| Generic YOLO26n, sports ball | 640 | 0 | 2 | 101 | 0.00% | 0.00% | 0.00% |
| Generic YOLO26n, sports ball | 1024 | 12 | 6 | 89 | 66.67% | 11.88% | 20.17% |

Current YOLO11s warm prediction throughput: **121.54 / 80.46 / 56.84 FPS** at 640/1024/1280. This includes prediction preprocessing/forward/postprocessing and excludes model loading, image decoding, tracking and rendering. GPU timings are synchronized. Larger resolution alone worsened false positives; the latest generic detector was not promoted. This comparison does not establish that the YOLO26 architecture is intrinsically inferior: size, task training and checkpoint differ.

Evidence: `artifacts/validation/vision_upgrade/{yolo11s,legacy_yolov5,yolo26n}_val_baseline.json`. Reports include weights/data/evaluator hashes, per-image predictions, counts, latency, peak GPU allocation and environment. Evaluator: `scripts/evaluate/benchmark_ball_detection.py`.

## Repairs and real video verification

The original six-frame integration test failed at both 320Ãƒâ€”240 and 1920Ãƒâ€”1080 before the dimension repair. It then passed. The expanded suite verifies valid/invalid calibration at both resolutions, nullable measurements, actual 25 FPS contact refinement and six-frame output alignment. Seven independent calibration tests cover projective recovery, a wrong landmark, degenerate geometry and nonfinite inputs. All **11 calibration/integration checks passed** in 4.91 seconds.

| Sample run | Result | Processing time | Processing FPS |
|---|---|---:|---:|
| Original Phase 6 | Crash before tracking/export | Not complete | Unavailable |
| Dimension repair | 214 frames exported | 13.72 s | 15.59 |
| Unit-aware calibration | 214 frames exported | 12.32 s | 17.37 |

Times exclude model construction. Single-run timing variation is **not** evidence of a performance optimization. New calibration: 14/14 inliers, mean residual **0.9258 px / 0.03094 m**. These are training-point fit residuals; independent court/line-call accuracy is still unmeasured. Seven inferred shots and one inferred rally are output counts, not verified correctness.

Videos and logs: `outputs/vision_upgrade_audit/repaired_phase6/` and `outputs/vision_upgrade_audit/calibrated_phase6/`.

## Broader validation in progress

Acquired the first six entries of the publisher's RacketVision tennis validation split before inference, pinned to dataset revision `85157ca21faa2abca96d837dd2b963738029bcc8`. Acquisition verifies video SHA-256 against publisher LFS hashes and preserves the MIT-declared dataset card, raw ball CSV labels and racket COCO validation annotations. No publisher test media/labels were downloaded. Match IDs do not overlap the publisher training split; broadcast/source independence and overlap with the older YOLO training images remain unverified.

Ball CSV labels are sparse. The video evaluator scores **only explicitly annotated frames**, including visibility=0 negatives; it does not convert unannotated frames into negatives or interpolate ground truth. Point metrics are reported separately from box IoU metrics. Current YOLO+tracker and the authors' tennis WASB temporal model are being compared on identical video and annotation rows.

## Six-video temporal ball results

All 2,561 frames were processed; **300 explicit labels** were scored: 294 visible and six absent. Point matching uses a four-pixel radius after scaling coordinates to 512Ãƒâ€”288 (15 pixels at 1920Ãƒâ€”1080). These point results must not be compared numerically with the earlier box-IoU table. Threshold comparisons use validation only and do not create new independent evidence.

| Candidate | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Current YOLO11s raw top-one, configured 1024/.01 | 149 | 151 | 145 | 49.67% | 50.68% | 50.17% |
| Current YOLO11s + existing tracker | 201 | 96 | 93 | 67.68% | 68.37% | 68.02% |
| WASB step 3, threshold .50, raw top-one | 263 | 8 | 31 | 97.05% | 89.46% | 93.10% |
| WASB .50 + existing tracker | 268 | 23 | 26 | 92.10% | 91.16% | 91.62% |
| WASB step 3, threshold .35 | 269 | 9 | 25 | 96.76% | 91.50% | 94.06% |
| WASB step 3, threshold .25 | 270 | 11 | 24 | 96.09% | 91.84% | 93.91% |
| WASB overlapping step 1, threshold .25 | 276 | 8 | 18 | 97.18% | 93.88% | 95.50% |

**Measured improvement:** WASB improves ball localization substantially on this subset. The legacy tracker degrades WASB precision, so the separate candidate configuration uses model top-one observations with no fabricated gap filling. The single-model validation configuration now uses overlapping windows and threshold .20. This single-model operating point still fails the recall target. Only six negatives are available; negative-class generalization remains weakly tested.

WASB comes from the authors' MIT source and tennis checkpoint linked in their model zoo. The adapter preserves RGB normalization, affine transforms, chronological three-frame input and per-frame heatmap alignment. Added overlap inference averages real overlapping heatmaps with bounded memory; it is undergoing validation. Offline look-ahead is explicit. Reports include `racketvision_yolo11.json`, `racketvision_wasb.json`, `racketvision_wasb_threshold035.json`, and `racketvision_wasb_threshold025.json` under `artifacts/validation/vision_upgrade/`.

## Player, pose, racket and runtime verification

Visual inspection of frame 100 revealed that the old nearest-court-keypoint player selector chose a staff member and omitted the near player despite 100% coverage. The replacement uses calibrated foot-proxy position, separate near/far roles, identity presence, bounded movement and 0.5-second association memory. It abstains on invalid court calibration, never fills missing detections, and does not establish identity across changeovers/camera cuts. The corrected output at the same frame shows both actual players and excludes the staff member. This is a demonstrated failure repair, **not a measured player-detection accuracy rate**.

Continuous `player_motion_tracking` exports 17 COCO joints with per-joint confidence and null low-confidence positions. Calibrated bbox-foot motion estimates reject impossible jumps and do not bridge missing positions or long timing gaps. They are foot proxies, not measured foot contacts. The `racket_tracking` candidate uses enlarged player crops, real COCO racket detections, short association memory and image-plane motion. It never invents racket contact or fills a missing observation from a wrist.

The racket audit found a labeling trap: the mixed COCO export contains ball-only images with no racket annotation. The original 300-image racket report therefore has an invalid precision denominator and **must not be used as accuracy evidence**. Native racket file listings confirm sparse coverage. Replaying cached detections only on the **92 racket-positive frames / 102 boxes** gives TP=70, FP=44, FN=32, P=61.40%, R=68.63%, F1=64.81%. Absent-racket performance remains unmeasured. Corrected report: `racketvision_racket_yolo11m_positive_frames.json`; the original is preserved only to audit the correction.

Actual player-crop evaluation now measures **TP=65, FP=19, FN=37, P=77.38%, R=63.73%, F1=69.89%, mean matched IoU=.8143** on the same 92 positive frames. Player selection processes each full clip using first-frame calibration and ByteTrack; racket association memory is reset per sparse annotated image so nonadjacent labels are not treated as consecutive frames. Crops use 640 input size versus 1024 for the full-frame baseline. Crop inference takes 4.88 s for 92 labeled frames, plus 94.08 s for the full-video player/court pass; this is not end-to-end 18.86 FPS. Precision/F1 improve, but recall regresses and continuous racket identity/contact accuracy remains unmeasured. Report: `racketvision_racket_yolo11m_player_crops.json`.

Latest sample output: `outputs/vision_upgrade_audit/wasb_pose_racket/annotated.mp4`. Full decode verifies 214 frames at 30 FPS. Four extracted screenshots are alongside the video. Observed run time is **23.70 s excluding setup**, **28.20 s including model import/setup** (7.59 full-run FPS). Sampled peak RSS is **1,834,704,896 bytes**, about **1.71 GiB**, sampled every 100 ms. This is not a precise allocator high-water mark. The eight-frame decode cache bounds decoded-image storage; metadata/pose/detection outputs still grow with video length.

All **428 tests passed**, one pre-existing Starlette deprecation warning, 13.36 s, after ensemble, stationary filtering, chronological review and motion-export compatibility fixes. Motion summaries use observed contiguous intervals only, without accumulating distance across missing frames; disabled scoring exports no inferred point record. The summary-plus-samples player-motion format is now explicitly version 2.0; the review reader supports both legacy arrays and the summary structure. The default pipeline remains YOLO-based; `configs/phase6_analytics/wasb_validation.yaml` selects the original WASB temporal candidate with measured stationary suppression, pose/racket observations and authoritative scoring/events disabled.

Phase 6 now exports null ball speeds and a reason instead of interpreting airborne ray-plane displacement as physical shot speed. The old standalone speed estimator is retained for legacy consumers; it is no longer used by Phase 6. Calibration fit consistency does not validate ground truth. True ball speed still requires adequate geometric evidence and independent reference measurements.

## Bounded training pilot

Acquired the first 20 publisher-training clips, with no publisher validation match-ID overlap, using the same pinned RacketVision dataset revision. The training script verifies all acquired file hashes and rejects a validation-only manifest. Visual source review shows multiple broadcast court surfaces and lighting conditions; amateur footage and independently verified broadcast grouping remain absent. Contact sheet: `outputs/vision_upgrade_audit/data_camera_contact_sheet.jpg`.

Trained WASB's last HRNet stage and output layers (**982,259 trainable parameters**) for three epochs using **1,000 explicit frame labels**, including **74 absent-ball labels**. Context frames carry no invented labels. Randomized output-slot supervision is covered by six alignment tests. Augmentation uses shared temporal brightness/contrast changes, horizontal flips and 3/5-pixel horizontal/vertical motion blur. Batch size=1, AdamW learning rate=1e-4, frozen pretrained batch-normalization statistics, seed=7, mixed precision and clipped gradients. The trainer does not load validation or final-test labels.

Epoch mean focal training losses: 0.00009805, 0.00007786, 0.00007663; training epoch times: 88.31, 68.17, 65.83 seconds, excluding data preparation. **These losses are not accuracy evidence.** The final checkpoint at step 3/.25 gives P=93.86%, R=93.54%, F1=93.70%, versus the original's P=96.09%, R=91.84%, F1=93.91% at the same settings. Overlapping trained inference at .20 gives P=94.92%, R=95.24%, F1=95.08%. No tested trained-only threshold passes both targets or improves the original's best F1, so it was not accepted as a replacement. Weights, hashes, data lineage and augmentation settings are recorded under `artifacts/training/vision_upgrade/wasb_pilot01/`; log: `outputs/vision_upgrade_audit/wasb_pilot01.log`. Trainer: `scripts/train/finetune_wasb.py`.

The original model's cached overlap threshold replay confirms a precision/recall tradeoff on the same 300 labels: threshold .05 gives P=94.93%, R=95.58%; threshold .10 gives P=95.86%, R=94.56%; threshold .20/.25 gives P=97.18%, R=93.88%. No tested single-model threshold clears both requested rates. Failure crops show several misses near player/racket contact, wrong localizations and explicit absent-ball false detections. Labels were not altered after viewing errors.

## Frozen ensemble and additional footage

Cached heatmap averaging tested 18 predefined combinations of checkpoint weights and decoding thresholds on the same selection set. A mixture of 25% original / 75% epoch-three trained heatmaps, threshold .15 and overlapping step one, gives **TP=282, FP=14, FN=12; P=95.27%, R=95.92%, F1=95.59%**. All six explicit absent-ball labels are false positives. Crossing the aggregate target on this repeatedly used set therefore does not establish reliable absence handling or generalization. Deployed inference requires two model passes per temporal window; replay timing is not deployed throughput.

The candidate was frozen in `configs/phase6_analytics/wasb_ensemble_validation.yaml`. Next, publisher validation entries 6Ã¢â‚¬â€œ17 were acquired at the same pinned revision, without selecting by observed model performance. Their 12 clips are disjoint by publisher clip ID from both the initial six validation clips and the 20 pilot-training clips. Source broadcast independence remains unverified. This expansion is validation, not the untouched publisher test split. It contains **6,082 source frames, 600 labels, 543 visible-ball labels and 57 absent-ball labels**, all 1920Ãƒâ€”1080 at native 25 or 60 FPS.

| Expansion candidate | TP | FP | FN | Precision | Recall | F1 | Correct absent / 57 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Existing YOLO11s + tracker | 428 | 163 | 115 | 72.42% | 78.82% | 75.49% | 6 |
| Frozen ensemble, step 1/.15 | 504 | 89 | 39 | 84.99% | 92.82% | 88.73% | 5 |
| Ensemble + stationary filter, .50 s | 511 | 75 | 32 | 87.20% | 94.11% | 90.52% | 11 |
| Original WASB, step 1/.20 | 499 | 63 | 44 | 88.79% | 91.90% | 90.32% | 27 |
| Original + stationary filter, .25 s | 505 | 41 | 38 | 92.49% | 93.00% | 92.75% | 36 |

The original model takes 268.59 s of streaming decode/prediction versus 507.22 s for the ensemble on identical 6,082-frame input, excluding setup/tracking. Their average rates are 22.64 and 11.99 FPS respectively; this comparison does not include the additional full player/pose/racket/render pipeline. The original plus stationary filter is selected for further validation because it has the best expansion F1 and lower inference cost. Relative to the existing YOLO tracker on these identical labels, it adds 77 correct localizations, removes 122 false positives and raises F1 by 17.26 percentage points. It still fails both requested rates. The ensemble is preserved as an unsuccessful generalization experiment, not promoted.

Failure inspection confirms repeated selection of the scoreboard's stationary service dot in `match148_000`. The new causal filter rejects persistently recurring, nearly fixed image locations after sufficient real candidate observations. It does not use labels, hardcoded scoreboard coordinates, future frames, or inferred missing observations. **A real stationary ball can also be suppressed**; this remains a measured experimental heuristic. Three durations were compared on validation, which is further selection on the same data. On the initial six original-model clips, the tested filter leaves all point counts unchanged. Expansion failure crops are in `outputs/vision_upgrade_audit/ensemble_expansion_match148_failures/`.

Replay evidence: `artifacts/validation/vision_upgrade/racketvision_wasb_ensemble_pilot01.json`. Both source report/cache hashes and the complete parameter grid are retained. `WASBEnsembleDetector` shares the single-model frame-alignment implementation and averages before thresholding; tests compare its streamed output against the corresponding independently streamed heatmaps and reject mismatched cached ground truth.

Provenance correction: early long-running video/racket evaluators hashed their Python source when writing the report. The ensemble-expansion and crop-racket evaluator files were edited during execution, so those file hashes are not exact snapshots of loaded code. Saved data/weight identities and predictions remain available, but the source hash must not be presented as a fully immutable execution snapshot. New evaluator runs capture code hashes before inference. Original result files are preserved rather than silently rewritten.

## Chronological visual review

`src/visualization/vision_review_renderer.py` and `scripts/run/render_vision_review.py` create a broadcast-style review from recorded observations without rerunning models. Source-video hashes and frame/FPS alignment are checked. The match image is kept separate from the court panel; body poses, racket boxes and ball trails remain visible. Player colors match their court trajectories. The map includes court run-off and cumulative observed foot-proxy occupancy; it never plots an airborne ball as a grounded court position or uses future occupancy. Missing observations break trails. Physical ball speed and events remain explicitly unverified.

`outputs/vision_upgrade_audit/broadcast_review_v1/tracking_review.mp4` decodes to 214 frames at 30 FPS, with reviewed encoded screenshots including frame 87's racket observation. This first review uses the preserved earlier .50/step-three sample exports; it is a visualization demonstration, not evidence of the newly selected detector's accuracy. No bounce target was fabricated to satisfy the requested visual.

Fresh full-pipeline verification of the selected original/.20/overlap/stationary-filter configuration:

- `outputs/vision_upgrade_audit/wasb_selected_match148/`: all 600 frames decode at native 60 FPS. Full pipeline including construction takes **85.77 s / 7.00 FPS**, sampled peak process RSS **1,874,989,056 bytes**. This hard clip remains weak: **TP=24, FP=11, FN=14, TN=8, P=68.57%, R=63.16%, F1=65.75%** on 50 explicit labels. The actual pipeline export matches the filtered replay counts exactly. It is retained as a regression challenge rather than hidden by aggregate results. Evidence: `phase6_selected_match148_ball.json` and `broadcast_review_match148_v2/`.
- `outputs/vision_upgrade_audit/wasb_selected_sample/`: all 214 frames decode at native 30 FPS, with current versioned motion summaries and no inferred score/point outcomes. Recorded full run 35.49 s includes concurrent regression-test CPU activity, so it is an execution check rather than a controlled timing comparison. Fresh review: `outputs/vision_upgrade_audit/broadcast_review_selected_sample/`.

The first attempt to review the current motion export correctly rejected a length mismatch because the renderer expected legacy arrays. The reader was repaired to accept both layouts, version 2.0 identifies future summary exports, and regression coverage now exercises both. Failed and successful artifact directories remain separate.

## Second training pilot and evidence accounting (2026-09-08)

The second pilot targets stationary scoreboard markers using synthetic training-only graphics and fourfold sampling weight for explicit absent-ball labels. Three epochs completed from the untouched official checkpoint, with all checkpoints preserved. Synthetic panels are identical across temporal context, vary in corner position and size, and avoid the supervised ball plus an eight-pixel margin. The augmentation preview was visually checked. No validation pixels or coordinates were used in training. Actual absent draws were 234, 244 and 253 per 1,000-draw epoch.

At predeclared overlapping/.20 inference on the twelve expansion clips, the third epoch gives **P=86.50%, R=93.19%, F1=89.72%**. With the same .25-second stationary filter as the selected original, it gives **TP=512, FP=65, FN=31, P=88.73%, R=94.29%, F1=91.43%**. That is seven more correct localizations but 24 more false positives than the selected original, with correct absent frames falling from 36/57 to 15/57. The pilot is not promoted. This is a failed improvement hypothesis at the tested operating point, not proof that all synthetic-graphic training is ineffective. Full protocol and outcomes: `docs/experiments/WASB_DISTRACTOR_PILOT02.md`.

Point reports now distinguish wrong visible localization, visible abstention and false detection on absent frames. The eighteen-clip diagnostic combines 900 labels: 837 visible, 63 absent; only 39 absent frames are correctly left empty and only eight clips clear both rates. **This diagnostic combines initial threshold .25 and expansion threshold .20 and is not one fixed-configuration qualification score.** `selected_original_18clip_strata_v2.json` explicitly records the two operating points and marks their difference. The tool rejects duplicate clip/frame labels rather than double-counting evidence.

Export inspection also found that disabled event authority left an apparent longest rally of one and an unlabeled configured 0Ã¢â‚¬â€œ0 state. Future exports now withhold average/longest rally measurements in that mode and label match state as configured initial state rather than observed score. Detector source track IDs and confidence are retained beside selected player boxes, with explicit segment-role and detector-ID semantics; these fields support future identity audits without claiming ground-truth identity.

All **433 regression tests pass**, one existing Starlette warning, 20.51 seconds after these changes. Real video revalidation at `outputs/vision_upgrade_audit/wasb_selected_export_audit/` decodes all 214 frames at 30 FPS and completes in 36.44 seconds including setup. Inspection confirms disabled authority, null longest-rally measurement, explicitly configured initial match state and real detector IDs/confidence in player boxes. Prior immutable video reports retain their original schema/content. The original model remains selected, and events, physical ball speed and production authority remain withheld.

## Camera registration and motion measurement (2026-09-08)

The selected validation pipeline now registers distributed court features from the first calibrated frame to each current frame, excluding detected people. Forward/backward optical-flow checks and RANSAC support gate the transform. Per-frame calibration feeds player association, ground projection and court overlays. Invalid registration clears role memory and withholds metric positions, speed/map output and map trails. Failure stays latched until explicit reinitialization; automatic camera-segment recovery and resetting the ball model's temporal context at a cut remain unfinished. These checks cannot certify every camera cut or perspective deformation.

On saved detections from the two real videos, all 214 sample frames and all 600 `match148_000` frames retain registration. Maximum landmark displacement is 0.46 and 3.19 source pixels; maximum player-position correction relative to a static homography is 0.009 and 0.179 m respectively. **Correction relative to the old estimate is not ground-truth accuracy.** Audits, source hashes and per-frame evidence are retained in `outputs/vision_upgrade_audit/camera_registration_sample/` and `camera_registration_match148/`.

A controlled challenge applies known pan/zoom to one frozen real frame for 90 frames, inserts another real match for ten frames and restores the anchor for five. All 90 warped frames remain valid; all 15 frames from the cut onward remain invalid. Maximum player-position error relative to the known warp falls from **1.749 m with static geometry to 0.049 m with registration**; mean registered error is 0.0056 m. Maximum landmark warp error is 2.014 source pixels. This is synthetic transformation evidence using the estimated anchor court, not independent court ground truth or broad cut detection accuracy. Report: `outputs/vision_upgrade_audit/camera_perturbation_challenge/camera_perturbation_benchmark.json`.

The old distance summary accumulated every accepted detector displacement. A stationary five-second trajectory at 60 FPS with declared independent 0.03 m position noise incorrectly accumulated 14.88 m. Motion now fits velocity across the backward .15-second window and integrates accepted speed estimates; raw polyline distance remains available as a diagnostic. Acceleration uses each fitted velocity's effective timestamp, including startup. Missing positions, long gaps and impossible speeds interrupt measurements.

| Synthetic five-second case, 60 FPS, noise sigma 0.03 m | Known distance | Previous estimate | Window-speed integral |
|---|---:|---:|---:|
| Stationary | 0 m | 14.881 m | 1.236 m |
| Constant 3 m/s run | 15 m | 20.085 m | 15.029 m |
| Smooth circular motion | 16 m | 21.132 m | 16.001 m |

The reproducible challenge covers 27 combinations of 25/30/60 FPS, three trajectories and noise sigma 0/0.03/0.08 m. Noise is a declared stress condition, not measured detector uncertainty. **Stationary residual movement remains: 1.24 m over five seconds in the displayed case.** A fit residual is not independent sensor uncertainty. Before/after reports: `artifacts/validation/vision_upgrade/player_motion_noise_baseline.json` and `player_motion_noise_window_velocity.json`. The latter predates only the effective-time acceleration correction; its measured distance/speed quantities are unchanged. Regression tests verify the corrected acceleration on a known uniformly accelerated trajectory.

Combined real-video validation at `outputs/vision_upgrade_audit/wasb_registered_motion_sample/` completes and decodes all **214 frames at 30 FPS**, taking **39.21 s including construction, 5.46 full-pipeline FPS**, with sampled process peak RSS **1,772,613,632 bytes**. Exported observed player distances are 16.92 and 19.31 m; these have no independent real-world reference and are not accuracy claims. The fresh `broadcast_review_registered_motion_sample/tracking_review.mp4` also decodes all 214 frames at 30 FPS. Its frame-87 review shows both player poses, the near player's racket observation, the ball trail and aligned court map; physical ball speed and event authority remain unverified. Console reporting was subsequently clarified to avoid presenting configured initial score as an observed score in disabled-authority mode.

All **447 regression tests pass**, with one existing dependency warning, in 12.73 seconds after the camera, motion and console-reporting changes. Log: `outputs/vision_upgrade_audit/vision_upgrade_camera_motion_regression.log`. These tests establish software behavior, not independent computer-vision accuracy.

## Missing-candidate diagnosis and rejected recovery experiments

The selected .20 detector has a correct candidate anywhere in its output on 508/543 visible expansion labels, a **93.55% label-assisted recall ceiling**. Top-one selection already recovers 505. Thirty-five frames have no correct candidate, including eleven with no candidate at all; thirteen misses occur in the hard `match148_000` clip. Only one of the 35 has a candidate within eight reference pixels. Ranking alone cannot meet the target at this operating point. The diagnostic uses labels to classify errors, never to produce pipeline predictions.

A new full twelve-clip run at threshold .05 increases candidate coverage to **526/543 (96.87%)**, but raw candidate availability is insufficient. With the same .25-second stationary filter, selected output worsens to **TP=500, FP=96, FN=43; P=83.89%, R=92.08%, F1=87.80%**. Only 3/57 absent labels are correctly left empty. This threshold is rejected. Full predictions and labeled heatmaps are preserved in `racketvision_expansion12_original005.json` and its companion cache; filter and oracle reports are in `expansion12_original005_stationary.json` and `expansion12_original005_candidate_coverage.json`, under `artifacts/validation/vision_upgrade/`.

A second predeclared replay preserves the selected .20 observations and considers only real .05 candidates inside short gaps bounded by strong observations. It never exports a linear interpolation or extrapolation. At the fixed conservative settings it recovers zero observations, so all selected-baseline counts remain unchanged. It is not integrated or promoted. Ten targeted diagnostic/recovery tests pass. Full protocol, settings, limitations and result: `docs/experiments/BALL_CANDIDATE_COVERAGE.md`. These reused-validation experiments do not supply independent qualification.

## Pixel-motion evidence and additional six-clip validation

A causal appearance feature tests whether video pixels near a candidate actually change. It compares a 7x7 grayscale patch at 960x540 against an image at least .10 seconds earlier, averaging the three largest absolute differences. Candidates with a score below 12 are removed after the existing coordinate-stationary filter. Unknown startup history retains candidates. This is image-change evidence, not object recognition; a real stationary/low-contrast ball can be rejected and camera motion can preserve distractors.

| Validation subset, original .20 detector | Coordinate filter P / R / F1 | Added pixel filter P / R / F1 | Correct absent before / after |
|---|---|---|---|
| Initial six, rerun at exact .20 | 97.18% / 93.88% / 95.50% | 97.18% / 93.88% / 95.50% | 3 / 3 of 6 |
| Twelve-clip selection expansion | 92.49% / 93.00% / 92.75% | **94.92% / 93.00% / 93.95%** | 36 / 42 of 57 |
| Six added clips, settings frozen before acquisition | 94.93% / 91.61% / 93.24% | **95.27% / 91.61% / 93.40%** | 8 / 9 of 14 |

The added clips are publisher validation entries 18Ã¢â‚¬â€œ23 (`match154_000` through `match159_000`) at the pinned revision, with 300 labels, 286 visible and 14 absent. They were not used to choose the pixel settings, but source-broadcast independence and overlap with existing model training remain unverified. No final-test qualification is implied. The .05 detector still performs poorly with this feature and is not selected.

Visual review confirms one true ball recovered when a scoreboard marker is removed and one true detection lost near a court line in the expansion. A wider radius-eight patch was tested as a controlled follow-up and performed worse (F1=93.25%); radius three is retained. Full protocol and all outcomes: `docs/experiments/BALL_PIXEL_MOTION.md`. Reviewed original pixels: `outputs/vision_upgrade_audit/pixel_motion_decision_changes_v2/`.

The optional candidate is implemented in `wasb_pixel_motion_validation.yaml`, with per-frame `pixel_motion_audit.json` evidence. All **463 full regression tests pass**, one existing warning, 11.95 seconds; the later nested-provenance summary change passes its three targeted tests. A complete real run on the hard `match148_000` clip produces **TP=25, FP=5, FN=13, TN=9; P=83.33%, R=65.79%, F1=73.53%**, matching replay and improving on 65.75% F1 before the pixel filter. All 600 output frames decode at 60 FPS. Execution takes 86.76 seconds including setup, with sampled process RSS 1,875,304,448 bytes; brief concurrent CPU provenance checks mean this is not a controlled overhead comparison. Evidence: `outputs/vision_upgrade_audit/wasb_pixel_motion_match148/` and `artifacts/validation/vision_upgrade/phase6_pixel_motion_match148_ball.json`.

The pooled 24-clip diagnostic has TP=1043, FP=48, FN=80, TN=54 over 1,200 labels: P=95.60%, R=92.88%, F1=94.22%. Only nine clips individually exceed both requested rates. `pixel_motion_24clip_summary.json` preserves detector settings, feature configuration and differing feature source hashes; it explicitly flags that the pooled sources are not one identical implementation snapshot. It is diagnostic, not independent qualification. Neither aggregate precision nor software tests remove the remaining recall and generalization gaps.

## Learned candidate verifier: three rejected training pilots

To test recovery from the .05 candidate stream, a small 89,809-parameter current/past RGB crop classifier was trained on real proposals from twenty publisher training clips. The cache contains 5,379 proposals across 1,000 explicit labeled frames: 875 positive, 4,465 negative and 39 ambiguous proposals. No ground-truth proposals or unlabeled-frame negatives were created. Candidates within four reference pixels were positive, those at least eight away negative, and intermediate candidates excluded only from training loss. Validation scoring retains every explicit label and the original four-pixel tolerance. The last four training clips were reserved for checkpoint/threshold selection, separate from external validation.

The first eight-epoch pilot, selected internally at epoch 8/.9, gives **P=94.04%, R=87.11%, F1=90.44%** on the twelve-clip expansion. A controlled shared-motion-blur pilot selects epoch 7/.9 and gives **P=95.33%, R=86.56%, F1=90.73%**. Both lose substantial recall relative to the existing .20/pixel-filter candidate. The blur verifier accepts just 1/22 correct proposals below base confidence .20 versus 452/461 above .50; this is candidate-level diagnosis, not frame recall. Rejected-crop inspection shows blurred, low-contrast and player/net/line-adjacent balls.

A third pilot removes confidence/rank scalar inputs while keeping the blur settings, data, architecture, seeds and internal selection rule. Its internally selected epoch 8/.3 gives **P=77.13%, R=80.11%, F1=78.59%** externally. Removing the scalars does not solve weak-candidate recovery. All three pilots are rejected; none is connected to the pipeline and the optional pixel-motion configuration remains unchanged. Each checkpoint, selection history, cache and external result is retained. Full protocol and comparison: `docs/experiments/BALL_CANDIDATE_VERIFIER_PILOT.md`.

The full regression suite passed **470 tests**, one existing warning, in 12.66 seconds after the second pilot's implementation. The final appearance-only ablation was exercised by completed training and evaluation. These experiments improve the evidence about training limitations, not deployed ball accuracy. More representative supervised weak-ball examples and a stronger temporal representation remain unresolved; no claim that either would guarantee the target is made.

## Controlled resolution validation

The six added validation clips were evaluated at native 1920x1080 and area-downsampled 1280x720 / 853x480, with identical .20/overlap detector and stationary/pixel-filter settings. Native FPS and all 300 explicit labels were preserved; matching remains four pixels at reference 512x288. The native control matches all baseline labeled predictions within .001 source pixel, including missing states.

| Resolution | Precision | Recall | F1 | TP / FP / FN / TN |
|---|---:|---:|---:|---|
| 1920x1080 | 95.27% | 91.61% | 93.40% | 262 / 13 / 24 / 9 |
| 1280x720 | 95.24% | 90.91% | 93.02% | 260 / 13 / 26 / 9 |
| 853x480 | 95.24% | 90.91% | 93.02% | 260 / 13 / 26 / 7 |

Paired analysis shows two correct visible detections lost at 720p, with no gains. At 480p three are gained and five lost, and two correct absent decisions become false detections. Similar aggregate metrics conceal different failures. The candidate remains below the recall target at every resolution. No downsampling change is promoted.

This is controlled degradation of the same broadcasts, not independently captured lower-resolution video or extra independent labels. Source compression, amateur cameras, lighting, angles and longer footage remain unqualified. Full protocol, result identities and timing limits: `docs/experiments/BALL_RESOLUTION_STRESS.md`; artifacts: `additional6_ball_resolution.json` and `additional6_ball_resolution_comparison.json` under `artifacts/validation/vision_upgrade/`. Five resolution/paired-comparison tests pass. The experiment changes the evidence about input sensitivity, not the selected model.

## Higher-detail crop proposals

Four overlapping 60%-size image crops were processed using the original .20-threshold WASB temporal windows, then combined with the full-frame candidates by peak confidence. On the same six added clips (300 sparse labels), raw combined selection improves from **262 TP / 14 FP / 24 FN / 8 TN** to **272 TP / 15 FP / 14 FN / 6 TN**: precision **94.77%**, recall **95.10%**, F1 **94.94%**. This is actual selected-point scoring, separate from the label-assisted union coverage of 273/286 visible labels. The latter is an oracle ceiling, not deployed recall.

The completed continuous five-view run preserves all 300 sparse raw predictions exactly. Applying the frozen stationary and pixel-motion filters gives **272 TP / 13 FP / 14 FN / 8 TN: precision 95.44%, recall 95.10%, F1 95.27%**. Compared with the filtered full-frame control, ten visible detections are recovered, none are lost, and one additional absent-frame error appears. Only three of six clips individually pass both thresholds; six of fourteen absent frames still produce detections. This is a passing aggregate on reused development data, not generalization or production qualification. Frozen-setting validation on the twelve-clip expansion is still required.

The run resumed only its final unsaved clip after the original process ended, preserving five completed clips and their provenance. Summed completed-clip processing takes 998.04 seconds, including an unexplained wall-time stall; it excludes the interrupted attempt's unsaved work and is not a controlled full-pipeline benchmark. All eleven changed outcomes were rendered, three visually inspected, and **486 regression tests pass** with one existing warning. Protocol, per-clip counts, recovery/timing limits and artifacts: `docs/experiments/BALL_TILED_PROPOSALS.md`. Independent player/pose label-source research is recorded in `docs/research/TENNIS_PLAYER_POSE_LABEL_SOURCES.md`; no new person/pose accuracy claim is established by source research alone.

**Can the system reliably analyze any tennis match video now? NO.** Ball validation improved, runtime failures were repaired, and pose/racket evidence paths were added. Ball accuracy, independently measured player/pose/racket accuracy, automatic camera-segment recovery, true speed, stronger training and broad production qualification remain unfinished. This is an interim audit, not completion of the requested goal.

## Frozen twelve-clip expansion result

All 6,082 frames and 600 explicit labels completed using the same five-view/filter settings. Results:

| Variant | TP / FP / FN / TN | Precision | Recall | F1 |
|---|---|---:|---:|---:|
| Full-frame stationary/pixel control | 505 / 27 / 38 / 42 | 94.92% | 93.00% | 93.95% |
| Five views, raw | 514 / 64 / 29 / 17 | 88.93% | 94.66% | 91.70% |
| Five views, stationary | 518 / 49 / 25 / 24 | 91.36% | 95.40% | 93.33% |
| Five views, stationary/pixel | 518 / 35 / 25 / 33 | 93.67% | 95.40% | 94.53% |

Paired scoring gains fifteen correct visible detections and loses two, but introduces nine absent-frame errors. Seven of twelve clips individually pass both thresholds. The difficult match148_000 scores 31 TP / 9 FP / 7 FN / 6 TN: precision 77.50%, recall 81.58%, F1 79.49%, versus the full-frame pixel-filter F1 of 73.53%. More visible balls are recovered, but false detections prevent promotion. The passing six-clip aggregate does not establish generalization. No default detector or event-authority change is made.

Results: artifacts/validation/vision_upgrade/expansion12_tiled_ball_stream.json. Paired labels and source hashes: expansion12_tiled_ball_paired_changes.json in the same directory. The process completed without resume. COCO acquisition and dependency preparation overlapped this accuracy run; timing is diagnostic only, as recorded in outputs/vision_upgrade_audit/expansion12_tiled_execution_notes.txt.

## Independent person and body-pose measurements

A complete official COCO validation tennis-context slice now supplies independent human annotations: 167 images, 791 person instances including 30 crowds, and 515 eligible poses across 164 images. All selected images and crowd/visibility semantics are retained. Existing YOLO11m person-box AP is **70.32%**, with small-person AP **45.01%**. Existing YOLO11n full-frame pose OKS AP is **54.56%**, medium **45.11%**, large **71.41%**. These are standard COCO AP sweeps, not fixed-threshold detection precision or player identity accuracy.

A controlled 1024-resolution ablation changes person AP to **67.91%** and pose AP to **55.06%**. Small/medium people improve, but large-person performance worsens substantially. No resolution change is promoted. The next pose comparison must address model capacity and the real detected-crop path, not assume resolution alone fixes joint localization. Full protocol, source provenance, counts, results and limitations: docs/experiments/COCO_TENNIS_PEOPLE_BASELINE.md. The current pipeline's cropped-pose adapter is not measured by these full-frame results. Player identities, motion, real speed and production generalization remain unqualified.

The stronger-pose candidate now completes the 600-frame hard-clip pipeline and verified broadcast review at 60 FPS. Including model setup it takes 88.81 seconds, sampled RSS 1,951,121,408 bytes. All 600 ball observations remain exactly equal to the prior run; its known 65.79% ball recall failure remains. Independent detected-crop OKS AP improves from 45.21% to 54.49%, and the review screenshot confirms integration, not video joint accuracy. Artifacts and limits: docs/experiments/COCO_TENNIS_PEOPLE_BASELINE.md. Production readiness remains NO.
