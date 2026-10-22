# Motion-only participant selection rejected — 7 October 2026

The unchanged image-motion selector **must not be promoted to active-player selection**. On three reused UVY videos it retains only 710 of 2,479 cached proposal matches against the original publisher player boxes. Native image review shows foreground spectators selected while court players are missed. Production sources and the selected configuration remain unchanged.

Evidence: `artifacts/validation/product_execution/participant_motion_uvy_2026-10-07/`. The original single run completed in 40.703013 seconds; all 2,230 selection frames and all 2,230 review-decode frames completed. This was a cached-observation development experiment, with no model inference or training.

## Fixed comparison

The source was the existing tracked640 ByteTrack cache, with its temporal fragment IDs, exact boxes and confidences. Raw640/raw1024 IDs restart by frame and were explicitly excluded. The unchanged selector requires at least 0.5 seconds of observed frames, four camera-supported samples spanning 0.5 seconds, and a compensated extent of 0.6 median body heights. It retains actual eligible observations, assigns no court role, merges no identity, fills no gaps and abstains when more than two candidates coexist.

The camera method preserves the prior Dimitrov diagnostic thresholds: masked ORB background features, descriptor ratio 0.75, partial affine RANSAC at two reference pixels, at least 20 inliers, at least 0.5 inlier fraction, mean residual at most two pixels, and distributed support spanning 0.4 image width and 0.3 height. Samples are every 12th frame plus the last frame, independent of labels. Native transforms use the full scale conjugation. The first identity anchor remains an explicit exception to distributed-match acceptance; no interpolation, reanchoring or verified cut handling is introduced.

All three selections were saved before semantic publisher-label parsing. Every historical proposal count was reproduced on every frame using the unchanged maximum-cardinality IoU matcher at 0.5. All source, cache, protocol and ordinary imported-module bindings remain retained.

| Clip | Original class1 boxes | Cached proposal matches | Motion-selected matches | Selected publisher-unmatched | Cached matches lost | Accepted/sample registrations, including anchor |
|---|---:|---:|---:|---:|---:|---:|
| V01 | 1,445 | 1,329 | 0 | 698 | 1,329 | 21/70 |
| V02 | 1,936 | 631 | 330 | 512 | 301 | 81/82 |
| V03 | 822 | 519 | 380 | 410 | 139 | 38/38 |
| Total | 4,203 | 2,479 | 710 | 1,620 | 1,769 | 140/190 |

These are frame-weighted agreements with reused, flawed publisher boxes. **Publisher-unmatched is not validated participant false positives or detector precision.** Some selected unmatched boxes visibly cover an actual participant with imperfect localization. No selected box overlaps a publisher referee box at IoU 0.5, but those labels do not exhaust spectators or officials. Zero referee overlap does not establish specificity.

The report records 2,330 selected observations, 340 frames without selection and zero more-than-two ambiguity frames. Lack of ambiguity does not mean the selected people are the two participants.

## Source review and limits

Root inspected every saved original and overlay: 13 native frames / 26 PNGs, with exact bindings in `visual_review.json`.

- V01 frames 0, 409 and 604 select a foreground bald spectator (source fragment3), while visible court players are unselected. The final zoomed frame selects nobody. Original frame604 player omissions remain unchanged.
- V02 frames 0, 99 and 484 follow the nearer player (fragment4), including a lunge. The distant participant remains unselected. Frame99's publisher-unmatched selection visibly covers a participant, illustrating why unmatched cannot be relabeled spectator false positives. The final downward pan selects nobody.
- V03 frames 0, 140, 221 and 230 select the near player (fragment4) alongside the extreme-left spectator (fragment5), leaving the far player unselected. Original frame140 class1 identity9 still marks the chair umpire. The final frame selects nobody despite both players being visible.

V01 and V02 are potential clips from the same Federer–Nishikori Wimbledon 2019 quarterfinal; V03 is the Federer–Djokovic final. These previously exposed spectator views add at most two match groups, outside the primary elevated-baseline support envelope. They are useful negative/generalization diagnostics and are not an untouched qualification set. Sparse background registration gates establish internal consensus, not correct camera compensation or physical geometry. The native reviews are AI observations, not independent human labels or full-clip participant precision.

The exact owned workload was sampled before heavy imports. Across 151 samples, peak aggregate owned RSS was 347,734,016 bytes and minimum available RAM was 2,980,417,536 bytes. Every remembered owned process retired; all ten live jobs and 319 files remain unchanged. The unchanged guards were 750 MB launch headroom, 1 GB sampled owned RSS, below 400 MB available RAM for five consecutive 0.25-second samples, and 300 seconds total. This bounded whole-clip prototype supplies no full-match storage or inference evidence.

The separate `audit/receipt.json` passes independent scalar IoU/bipartite matching for every 2,230 row, pure-arithmetic source-fragment eligibility and exact selection reconstruction, native PNG decoding/pixel hashes, camera-gate/scale arithmetic, 160 frozen input bindings, 645 loaded-module bindings and all 151 resource samples. The audit confirms exact owned retirement and unchanged original terminal evidence/product files. It invokes no selector, ORB, model or source-video decode. It checks the recorded evidence; it does not independently validate background feature correspondences, camera accuracy, video alignment on unsampled frames or participant labels. Semantic read order is checked from code and staged receipts, without a separate instrumented read trace.

## Decision

Retain the prototype and its failed generalization evidence as research. Do not insert its selected fragments into near/far role arrays, activate it as a participant default, or present its motion as physical speed. A court-independent path needs stronger participation evidence and representative negative controls, while keeping tracked fragment identity, court role and world position distinct. Existing proposal misses and selector losses must remain separately measured. No threshold was retuned on these three clips and no run was repeated.

The current real-worker storage smoke and website checks pass separately; actual pose/racket observations in that short input are still absent. Accurate participant recovery, downstream pose/racket/mask quality, independent speed/3D and full-match capacity remain open milestones of the original execution prompt.
