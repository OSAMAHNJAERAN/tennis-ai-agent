# Ball temporal-spacing pilot

Frozen before stride-two inference on 2026-09-12. The current five-view WASB detector uses adjacent frames regardless of source FPS. On the completed 18-clip best-stream reports, 34 of 39 visible misses occur on 60 FPS footage. Across all 39, five-view raw selection was wrong on 26 and absent on 12; only one raw correct point was subsequently lost. These counts motivate a detector-input experiment, not another rejection filter. The FPS association is descriptive and is confounded by scene difficulty.

Hypothesis: expanding the temporal spacing on 60 FPS inputs may make motion cues more useful. This does not assert that the released model was trained at exactly 30 FPS or that resampling is universally better.

Use the original WASB checkpoint, threshold .20, the same five views (full frame and four overlapping 60% crops), peak-confidence merge radius four reference pixels, and unchanged affine preprocessing. For each labeled target frame, use all real three-frame windows containing it at stride `max(1, floor(fps / 30 + .5))`. This gives stride two on these 60 FPS clips and stride one on the 25 FPS clips. Average only the output heatmaps aligned to the target. All source indices retain native frame numbering and timing. No images or coordinates are interpolated. Maximum future context is four native frames at stride two. Require at least one real triplet; fail rather than silently pad.

First evaluate the six additional development clips (300 labels) against their saved continuous raw five-view outputs. Stride-one clips reuse those exact cached rows, with explicit cache attribution. Run all 200 labeled frames in the four 60 FPS clips, including absences. Label coordinates are used only for scoring. Save every candidate with view/window provenance, not just the selected point. Report correct-candidate coverage as an oracle diagnostic, never as deployed recall.

Report paired gained and lost correct detections, visible abstentions, wrong locations, absent false detections, and the existing 4-pixel/512x288 metric. No stationary, pixel-motion, patch-similarity or detour filter can be inferred from sparse labels. A favorable raw result requires a subsequent complete continuous-stream experiment at identical settings before any filtered or production claim. Do not tune temporal stride or threshold based on this outcome. The same data have already informed development; no independent test claim is made.


## Sparse result and continuous gate

All 300 labels complete. Raw baseline 272 TP / 15 FP / 14 FN / 6 TN; spacing 273 / 15 / 13 / 7, precision 94.79%, recall 95.45%. The candidate gains three true detections and loses two. Same FP totals hide two additional wrong localizations from previous abstentions and one corrected absence, alongside the changed true detections. Eight scored outcome transitions were rendered and all three boards visually inspected. Source hashes, exact paired labels and independent metric replay are recorded in `outputs/vision_upgrade_audit/ball_temporal_spacing_pilot01/review.json`. The 39-error stage audit is `outputs/vision_upgrade_audit/ball_visible_miss_stage_audit.json`.

This small gain justifies a continuous verification, not promotion. Partition the native source by index modulo stride, run the existing five-view overlapping detector independently for each phase, and interleave predictions back into native frame order. Apply the existing stationary and pixel-motion filters at native timestamps and FPS. No temporal state is shared across clips. All four 60 FPS clips are inferred on every frame; the two stride-one clips reuse exact verified continuous baseline objects and are explicitly attributed as cached. Require every sparse raw coordinate and missing state to agree with the continuous output within 0.0001 source pixel. Preserve maximum drift, frame counts, queue size and code hashes. No patch-persistence or detour output is claimed without a separate verified replay.

36 targeted tests pass in 0.65 seconds, covering empty/short/odd-length phase emission, every native index, spaced triplets, output-slot alignment and point scoring. Short phase streams inherit the existing model's end-padding policy; none of the evaluation clips need that fallback. The continuous result is pending.


## Continuous six-clip result

The stream completes all 2,893 frames: 2,394 newly inferred frames from four 60 FPS clips and 499 explicitly cached baseline frames from two 25 FPS clips. Every sparse raw coordinate and missing state agrees exactly (maximum difference 0.0 pixels). The shared decode queue peaks at five native frames. New-clip processing sums to 456.64 seconds, excluding setup and cached-clip time; small source-inspection work overlapped, so this is not isolated full-pipeline throughput.

| Spacing stage | TP / FP / FN / TN | Precision | Recall |
| --- | --- | ---: | ---: |
| Raw | 273 / 15 / 13 / 7 | 94.79% | 95.45% |
| Stationary | 273 / 15 / 13 / 7 | 94.79% | 95.45% |
| Pixel motion | 273 / 14 / 13 / 8 | 95.12% | 95.45% |
| Guarded patch persistence | 273 / 13 / 13 / 8 | 95.45% | 95.45% |
| Existing temporal detour replay | 273 / 13 / 13 / 8 | 95.45% | 95.45% |
| Prior adjacent-frame complete recipe | 272 / 12 / 14 / 8 | 95.77% | 95.10% |

The new final recipe gains three correct balls and loses two, with one additional false detection overall. No absence labels change final correctness. This is a precision/recall tradeoff, not a clear winner. The final metric replay and paired frame identities are verified in `outputs/vision_upgrade_audit/ball_temporal_spacing_pilot01/final_comparison.json`. The frozen patch and detour code was reused unchanged; the detour pass changes no scored outcome for the spacing candidate.

Raw proposal analysis on the four 60 FPS clips shows four newly available true candidates and no lost candidate coverage: match156 frames32/495 and match157 frames56/424. The two lost selected true balls in match155 remain available among candidates, but another candidate ranks higher. This ground-truth-assisted diagnosis does not implement a deployable selection rule.

A paired match156 video fully decodes to600 frames at60 FPS, SHA-256 `2758f4901a8e8670370688c1c8e0f18df57773511f4638469d2d4f07ca742319`. First/middle/last frames and frame246 were visually inspected. The overlay exposes both real observations and incorrect jumps; it makes no event or speed claims. `paired_match156.mp4`, contact images and verified source hashes are saved in the same output folder.

Next gate, fixed before expansion inference: use the identical sparse temporal-spacing rule on all600 labels of the existing twelve-clip expansion, with its completed raw five-view baseline. Reuse only exact stride-one rows and infer all stride-two labels. Keep every threshold/view/merge setting fixed. Do not promote from the six-clip result. A larger raw regression should stop this candidate; if it remains promising, continuous expansion verification is still required. All datasets remain reused development evidence.

Per-clip limitation: only two of the six clips individually clear both strict >95% gates after the full spacing recipe (match154 and match158), versus three for the prior recipe. Match155 recall falls to92%; match156 recall93.88%; match157 P/R94%; match159 precision89.80%. Pooled success must not be described as consistent multi-match qualification.


## Completed twelve-clip sparse expansion

All600 labels complete. Identical raw baseline:514TP/64FP/29FN/17TN, P88.93%/R94.66%. Temporal spacing:531TP/41FP/12FN/23TN, P92.83%/R97.79%, F195.25%. It gains18 correct visible detections and loses1 (match147 frame151), for net17; FP falls23. The hard match148 changes27TP/23FP/11FN to35TP/10FP/3FN. Neither result includes continuous filtering on the new expansion candidate yet.

All31 scored outcome transitions were rendered on11 boards and visually inspected. Blur/net-crossing gains and removed scoreboard dots are visible. The single lost true detection is a clearly visible elongated ball. Match150 frame578 adds a shoe-region absence error. Faint-object absence cases remain uncertain and labels are unchanged. Exact paired labels, metric replay, images and diagnostic comments are preserved in `outputs/vision_upgrade_audit/ball_temporal_spacing_expansion12/review.json`.

This favorable raw expansion result advances to the already defined every-frame gate. Use `benchmark_ball_spacing_stream.py` with the expansion dataset, raw baseline and completed sparse report; no algorithm change. All nine60FPS clips must be processed continuously, with three25FPS clips explicitly reused from their verified baseline. Require exact sparse agreement, then run the unchanged patch-residual and detour replays. Until those complete, do not claim that the expansion clears both targets or change production defaults.


## Interrupted expansion recovery

The original continuous expansion stopped after eight saved clips. Recovery preserves that report and its exact original driver under `artifacts/research/temporal_spacing_recovery/`. The resumed report validates model, dataset, configuration and code hashes, clip order, complete frame arrays, label identities and exact sparse coordinates before reusing a saved prefix. The per-clip inference body is AST-identical to the original; only recovery and provenance handling changed. All43 targeted recovery/alignment/scoring tests pass.

The ninth clip, match150, completed with exact sparse agreement. Its measured wall time was2691.37 seconds, much higher than the inherited roughly107-111-second clips. Host interruptions or resource contention were not isolated. Preserve the raw timing, but do not use this run for a stable throughput claim. Three remaining clips and the unchanged filter replays are pending.


## Completed continuous expansion and final comparison

The resumed report completes all 6,082 native frames: 5,334 newly inferred frames from nine 60 FPS clips and 748 verified cached frames from three 25 FPS clips. All 600 sparse raw predictions and missing states agree exactly. Eight inherited clip objects are byte-equivalent after JSON parsing; the original partial report is preserved. Resumed report SHA-256: `7bc173139a44cbd65de226905c6a7100289758722e9200cfa9fe0b38e8667ec8`. Recorded new-inference wall times sum to 4,436.24 seconds across original and resumed sessions, with large interruptions or contention unresolved. This is not a stable throughput benchmark.

| Recipe or stage | TP / FP / FN / TN | Precision | Recall |
| --- | --- | ---: | ---: |
| Spaced raw | 531 / 41 / 12 / 23 | 92.83% | 97.79% |
| Spaced stationary | 531 / 39 / 12 / 25 | 93.16% | 97.79% |
| Spaced pixel motion | 530 / 33 / 13 / 31 | 94.14% | 97.61% |
| Spaced guarded patch persistence | 530 / 30 / 13 / 34 | 94.64% | 97.61% |
| Spaced final detour replay | 530 / 30 / 13 / 34 | 94.64% | 97.61% |
| Prior adjacent-frame final recipe | 518 / 31 / 25 / 37 | 94.35% | 95.40% |

The unchanged detour pass changes no scored outcome. The final candidate gains 13 correct visible balls and loses one, for net 12, while FP decreases one. Absence handling worsens: three absent frames become correct and six become false detections, so absence specificity falls 64.91% to 59.65%. Fewer wrong visible localizations (11 to 7) explain the lower total FP despite more absence errors. Seven of twelve clips individually clear both strict >95% gates, unchanged from the prior recipe.

Hard match148 improves to 35 TP / 6 FP / 3 FN, P85.37%/R92.11%; it still fails both targets. Other weak clips include match143 P86.11%/R93.94%, match144 recall94%, match149 precision84.78%, and match150 precision87.76%. No threshold was adjusted after observing these outcomes.

Combined with the completed six-clip result, the same fixed spacing rule produces 803 TP / 43 FP / 26 FN / 42 TN on 900 labels: P94.92%, R96.86%, F195.88%. Only nine of eighteen clips individually pass both strict targets, versus ten for the prior recipe. All evidence remains reused development data with unverified source-broadcast independence. This result does not meet the requested precision target or qualify broad match reliability.

`outputs/vision_upgrade_audit/ball_temporal_spacing_expansion12_final/` contains independent metric replay, exact paired labels, final-stream reconstruction, all 25 changed outcomes on nine visually inspected boards, and the pooled 18-clip summary. The hard-clip video fully decodes to 600 frames at 60 FPS and 1280x400; SHA-256 `2eaaed911478b8d68421bf6196eda32226fe42261ad659d19cec2fedc0bd449f`. First/middle/last rendered video snapshots were inspected, separately from all source-context outcome boards. Native observation trails expose jumps and missing states; no interpolation, speed or event authority is added.

Visual review confirms net/blur/near-player recoveries and the sole lost visible elongated ball at match147 frame151. Some absence-labeled frames have ambiguous moving-ball context, including match148103/127/507; the labels and scores remain unchanged. Other added detections land on a court line, shoe, racket, shirt region or net advertisement. These observations guide further work but do not establish independent ground truth.

Next gate: preserve this stronger-recall recipe as a measured research option, verify any pipeline integration against these exact saved streams before use, and address the remaining absent-frame distractors with training/proposal evidence. Do not tune a threshold to remove the one extra pooled FP needed to cross 95%. Broader player, racket, court and physical-speed validation remains unresolved, and production defaults remain unchanged.


Pipeline integration is now verified on850 native frames from two real-model runs at60 and25FPS, with zero coordinate difference, exact missing states and identical detour masks. See `BALL_SPACING_PIPELINE_INTEGRATION.md` for the opt-in configuration, tests, videos and runtime limits. This does not change the eighteen-clip accuracy result or production qualification status.
