# RT-DETR persons with the original racket selector

Protocol frozen before downstream inference. Compare all 2230 UVY frames against `tracker_original_racket_selection_clipped01`, using its BoT-SORT plus original YOLO11m racket detector as the reference. Change only the upstream person detector recipe to the completed RT-DETR-l proposal benchmark. The RT-DETR comparison uses native square stretching and no NMS; the YOLO reference uses its native rectangular preprocessing and NMS, so this is a detector-recipe comparison.

Use the same original YOLO11m racket checkpoint SHA256 `d5ffc1a674953a08e11a8d21e022781b1b23a19b730afc309290bd9fb5305b95`, restricted loading, confidence 0.25, crop size 640, nearest-person ownership and samples every 0.2 seconds at native FPS. Recompute ownership, racket detections and source-ID links from RT-DETR/BoT-SORT person observations. Do not transfer original source IDs or use annotations to select identities.

Measure unchanged raw-source and image-flow-plus-handoff variants, both requiring at least two racket support frames and 0.5 seconds of observed person boxes, using the existing ranking and maximum two selections. Every selected box/confidence must exactly match an actual source observation. Evaluate unchanged class-1 publisher annotations with the pinned official HOTA, CLEAR and Identity implementations. Report all per-clip TP/FP/FN and association scores and aggregate detection counts; do not choose a different detector per clip.

Verify input/cache/protocol/code hashes, save all sampled racket candidates and assignments, mappings, linking diagnostics and selected frames. Review first/middle/last paired scenes and unexpected new selections. Known label defects remain unchanged. Reused professional fan recordings and low recall cannot establish broad or amateur qualification. No production default changes without further evidence.

## Completed results

All 2230 frames, sampled racket passes and both selector variants complete. Every selected RT-DETR box and confidence is verified against the actual source observation; baseline, candidate, source, dataset and frozen-protocol hashes are verified.

| Clip | Person recipe + BoT-SORT + original racket | TP | FP | FN | IDSW | IDF1 | HOTA |
|---|---|---:|---:|---:|---:|---:|---:|
| V01 | YOLO11m | 641 | 102 | 804 | 1 | 40.13% | 38.32% |
| V01 | RT-DETR-l | 647 | 87 | 798 | 1 | 40.75% | 38.57% |
| V02 | YOLO11m | 0 | 0 | 1936 | 0 | 0.00% | 0.00% |
| V02 | RT-DETR-l | 0 | 0 | 1936 | 0 | 0.00% | 0.00% |
| V03 | YOLO11m | 498 | 1 | 324 | 0 | 75.40% | 62.77% |
| V03 | RT-DETR-l | 443 | 342 | 379 | 0 | 55.13% | 53.07% |

Aggregate YOLO: 1139 TP / 103 FP / 3064 FN, P91.71%, R27.10%. RT-DETR: 1090 / 429 / 3113, P71.76%, R25.93%. Both raw-ID and flow/handoff variants have identical selected outputs for each detector in this experiment. RT-DETR is rejected as the downstream replacement despite its upstream tracked coverage gain of 322 labeled boxes.

V01 source 3 is observed for 638 frames and has 107 sampled opportunities but zero racket supports; source 861 similarly has 132 frames and 22 opportunities without support. In V02, sources 4 and 16 last 967 and 960 observed frames with 162 and 161 opportunities, respectively, but the entire sequence has no assigned racket support. Improved person association alone does not qualify either active player for this selector.

### Visual failure diagnosis

RT-DETR V01 selects 87 unmatched observations from foreground spectator source 1449. All three supporting racket crops at one-based frames 738, 768 and 786 were inspected: the proposals follow handheld flags and the supporting hand. V03 selects 342 unmatched observations from source 227, a uniformed courtside person. Its four supporting proposals at frames 325, 337, 373 and 421 cover the region beside the person's legs, with no visible tennis racket in the reviewed crops. These remain diagnostic visual observations, not independent ground truth or a source-ID exclusion rule. The original labels are unchanged.

The V03 near player contributes 443 true positives, while its improved far-player track fails racket selection. All first/middle/last paired scenes were inspected, as were source 227's first/middle/last selected scenes and all seven false-support crops. Full identity triptychs for the other selected tracks are saved but were not separately visually reviewed; aggregate scenes and targeted failure crops are the inspected evidence.

### Validation, runtime and disposition

The V02 paired output fully decodes 968 frames at 29.904 FPS; decoded frame 741 was visually inspected. SHA256 `c736a6ca1f32cf715a2c8d2206fc077d3ffe1b5b718c0e193691179f31e26594`. Artifacts are in `outputs/vision_upgrade_audit/rtdetr_racket_selection_pilot01`: report, final review, false-support review, contact/identity images and video.

Decode plus racket inference takes 104.92 / 130.28 / 67.28 seconds; linking 13.96 / 2.50 / 20.31 seconds. The many RT-DETR person proposals create many crops. These are component timings, not end-to-end throughput, and much slower than the prior YOLO-person recipe. No production default changes. Preserve the original YOLO11m/BoT-SORT/original-racket candidate as the better measured option while investigating a racket-specialist model and better active-player evidence.

Focused post-run regression verification: 21 tests passed in 1.44 seconds across nearest racket ownership, repeated-support selection and official tracking-metric adaptation. The complete rendered artifacts and observation checks are recorded separately in the final review.
