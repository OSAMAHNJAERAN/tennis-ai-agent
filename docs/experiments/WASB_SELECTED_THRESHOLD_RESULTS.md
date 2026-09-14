# Training-selected original WASB threshold: external development result

Completed 2026-09-13. The original checkpoint's training-selected 0.35 threshold raises sparse development precision and F1 but reduces recall and the number of individually passing clips. It meets both pooled 95% targets on this reused set; it does not meet the frozen no-recall-loss advancement rule. Preserve this measured tradeoff without changing runtime settings or claiming production qualification.

All eighteen clips and 900 original labels complete: 829 visible, 71 absent. The only changed setting is the decoding threshold, selected previously on eight reserved publisher training matches. Both arms use identical heatmaps from the original checkpoint, five image views, FPS-adjusted spacing, aligned real triplets and peak-confidence ranking. No training, new weights, external threshold search, interpolation or continuous filters are involved.

| Sparse raw arm | TP | FP | FN | TN | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original threshold 0.20 | 804 | 56 | 25 | 30 | 93.4884% | 96.9843% | 95.2043% |
| Training-selected threshold 0.35 | 793 | 38 | 36 | 43 | 95.4272% | 95.6574% | 95.5422% |

The candidate removes eighteen false detections but loses eleven true balls. Paired correctness changes are thirteen gained absences and eleven lost visible balls, with no gained visible balls or lost correct absences. Wrong visible localizations drop from fifteen to ten and visible abstentions rise from ten to twenty-six. These categories explain why the total false-positive reduction differs from the number of corrected absent frames: wrong visible localization counts both FP and FN.

Correct proposal coverage falls from 810 to 798 of 829 visible labels. This is ground-truth-assisted proposal availability, not deployed recall. The number of clips individually exceeding both 95% targets falls from ten to eight, so the pooled improvement does not establish uniform reliability.

| Clip | Control TP/FP/FN | Selected TP/FP/FN | Selected precision | Selected recall |
| --- | ---: | ---: | ---: | ---: |
| match143_000 | 31/11/2 | 31/8/2 | 79.49% | 93.94% |
| match144_000 | 48/1/2 | 48/0/2 | 100.00% | 96.00% |
| match145_000 | 46/1/0 | 45/1/1 | 97.83% | 97.83% |
| match146_000 | 48/0/0 | 48/0/0 | 100.00% | 100.00% |
| match147_000 | 48/1/1 | 45/1/4 | 97.83% | 91.84% |
| match148_000 | 35/10/3 | 33/6/5 | 84.62% | 86.84% |
| match149_000 | 39/8/1 | 39/6/1 | 86.67% | 97.50% |
| match150_000 | 43/6/2 | 42/3/3 | 93.33% | 93.33% |
| match151_000 | 47/2/0 | 46/1/1 | 97.87% | 97.87% |
| match152_000 | 48/0/0 | 48/0/0 | 100.00% | 100.00% |
| match153_000 | 50/0/0 | 50/0/0 | 100.00% | 100.00% |
| match154_000 | 43/1/0 | 43/0/0 | 100.00% | 100.00% |
| match155_000 | 46/3/4 | 46/2/4 | 95.83% | 92.00% |
| match156_000 | 46/2/3 | 45/2/4 | 95.74% | 91.84% |
| match157_000 | 47/3/3 | 46/3/4 | 93.88% | 92.00% |
| match158_000 | 47/1/2 | 46/1/3 | 97.87% | 93.88% |
| match159_000 | 44/5/1 | 44/4/1 | 91.67% | 97.78% |
| match15_000 | 48/1/1 | 48/0/1 | 100.00% | 97.96% |

## Verification and visual review

All 900 newly inferred control coordinates and missing states reproduce the saved spaced raw stream exactly: maximum difference 0.0 native pixels. Eighteen first-label comparisons also reproduce every control proposal and temporal window from the existing inference function. The independent reviewer rechecks original CSV targets, source/checkpoint/protocol/code hashes, every selected peak, all paired counts, every per-clip metric and the frozen advancement decision. Scores match exactly. Publisher training and validation match IDs are disjoint, but original pretraining exposure and source-broadcast independence remain unverified.

Seventeen focused tests pass in 3.40 seconds. The six new tests compare shared-forward decoding with separate existing inference at both thresholds, cover temporal boundaries and stride two, preserve the detector threshold after errors, and enforce replay absence/tolerance semantics. Existing temporal-alignment, threshold-selection and point-metric tests also pass.

The fixed review sample contains five examples on two boards. All five were inspected:

- `match143_000:117` and `:178`: the control follows static PERTH floor lettering; 0.35 correctly abstains under the unchanged absence labels.
- `match145_000:388`: the unmarked source context shows a moving yellow ball near the near player's head/serve action; 0.35 loses the detection.
- `match147_000:222`: the labeled location is in the far player's racket/hand contact clutter. The small three-frame crop does not isolate the ball conclusively; the original visible label and scored loss remain unchanged.
- `match157_000:329`: both thresholds follow a motion streak. The center shifts from 3.7072 to 4.4119 reference pixels away from the publisher label, crossing the unchanged four-pixel scoring boundary. This is a localization loss, not disappearance of the proposal. Do not move the tolerance to erase it.

The visual sample is diagnostic and does not cover every changed frame or replace independent annotation. No labels were edited.

The RTX 4050 Laptop GPU run took 814.43 seconds summed across clips, excluding setup (about 13.6 minutes). Peak PyTorch-allocated VRAM was 275,631,104 bytes; this is not total GPU memory usage. Runtime: Python 3.13.5, torch 2.13.0+cu126, CUDA 12.6, NumPy 2.3.1. Both thresholds share model passes, so these timings cannot compare their individual throughput. No all-frame pipeline FPS is established by sparse inference.

## Decision and remaining work

The protocol required increased F1 with neither precision nor recall decreasing before automatically advancing to continuous evaluation. Recall decreases, so the saved decision remains `REJECT_THRESHOLD_PROMOTION`. The pooled >95% result must also remain visible: it shows a potentially useful operating tradeoff, not a failed run or evidence of uniform professional reliability. A later decision to evaluate this tradeoff continuously would require a separate declared protocol; it must not be presented as passing this experiment's frozen gate.

The existing final filtered 0.20 result remains P94.9173% / R96.8637%, from a different processing stage. It is not a paired comparison against raw 0.35. Without a complete 0.35 stream, the effect of stationary, motion, residual and detour filtering is unknown. No default threshold, checkpoint, event authority or speed claim changed.

Further work should recover weak genuine-ball proposals and improve absence discrimination rather than run another externally chosen threshold sweep. The checked publisher metadata contains 350 training and 38 validation clips with disjoint publisher match IDs; only forty training clips have been used by the spatial pilot. This permits a bounded training-data expansion, subject to the existing provenance and label-quality checks, without using the 900 development outcomes as training targets. No additional data was acquired in this run. Independent court/player/ball-event/physical-speed qualification remains open. Production readiness remains **NO**.

Artifacts: `outputs/vision_upgrade_audit/wasb_selected_threshold01/` contains `report.json`, `review.json`, `visual_review.json` and two comparison boards. Report SHA-256: `03e885a6f7fd34b621f67084a7b2b69c29f7ae3fec29540cc2f6b04d0bf23593`.

Implementation: `scripts/evaluate/benchmark_wasb_selected_threshold.py`, `review_wasb_selected_threshold.py`, `render_wasb_selected_threshold.py`; protocol: `WASB_SELECTED_THRESHOLD_PROTOCOL.md`. Use the established Python 3.13 interpreter explicitly. Completed output directories are protected from accidental overwrite; `--resume` is for an incomplete benchmark after its original process has been confirmed stopped.
