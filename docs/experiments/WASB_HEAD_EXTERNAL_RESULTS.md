# Head-only WASB external development comparison

Completed and independently verified 2026-09-14 on all 900 unchanged labels from eighteen clips. The head-only candidate improves the frozen original-0.35 comparison by **one correctly localized ball and one fewer false detection**. It passes the predeclared raw comparison gate, but does not improve proposal coverage and is not promoted to runtime.

| Sparse raw model | TP | FP | FN | TN | Precision | Recall | F1 | Clips meeting both 95% targets |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original, selected 0.35 | 793 | 38 | 36 | 43 | 95.4272% | 95.6574% | 95.5422% | 8/18 |
| Head epoch 01, selected 0.35 | 794 | 37 | 35 | 43 | 95.5475% | 95.7780% | 95.6627% | 8/18 |
| Original 0.20, contextual control | 804 | 56 | 25 | 30 | 93.4884% | 96.9843% | 95.2043% | 10/18 |

There are 829 visible and 71 explicitly absent labels. Both selected-0.35 models produce 28 absent false detections. Adapted wrong visible localizations fall from ten to nine, with 26 visible abstentions unchanged. F1 increases 0.120482 percentage points. This comparison changes weights at the same training-selected threshold. The 0.20 control is contextual; it is not a new threshold-selection trial.

## What changed

Only `match149_000:18` changes outcome. In the three-frame native crop, a long yellow-green motion streak crosses the near side of a grass court. Original and adapted predictions lie along this streak. Error against the unchanged publisher point falls from 6.887948 to 3.926455 pixels at 512x288 reference scale, just 0.073545 pixels inside the four-pixel tolerance. This is a localization-boundary gain, not recovery of a previously unseen ball. Ball-center annotation on the streak remains uncertain; no source label is changed, and no significance claim follows from this one frame.

Correct raw proposal coverage is unchanged at 798/829 for both 0.35 models, versus 810/829 for original 0.20. The gain occurs in the 25 FPS group: precision/recall changes from 93.2432%/97.1831% to 93.6937%/97.6526%. All 60 FPS group counts remain unchanged (P96.2233%/R95.1299%). Ten clips still fail at least one pooled-per-clip target, and grass-clip `match149_000` still has only 88.8889% precision because of five absent false detections.

## Why another continuous filtering run is held

The raw comparison gate remains **passed**. A separate, explicitly subsequent feasibility check compares these saved proposals with the current continuous filtered 0.20 pipeline on exactly the same 900 label identities. That current result contains 803 correct balls. Arbitrary selection/rejection from the adapted 0.35 candidate coordinates can produce at most 798 correct balls (recall ceiling 96.2606%, below current 96.8637%). Eleven currently correct frames have no correct adapted proposal; six current misses have a potentially correct adapted proposal. Even an oracle selecting all six cannot offset the eleven losses.

Source inspection confirms that the configured `model_top1` path retains candidate coordinates and that stationary, pixel-motion, patch and detour filters reject candidates without synthesizing new coordinates. This ceiling is conditional on the saved candidate set and unchanged inference/selection semantics. A different model, threshold, coordinate correction or predictive tracker is outside its scope. A complete continuous run of this candidate has **not** been executed, so no filtered accuracy or full-video speed is claimed.

The feasibility screen changes the next engineering action without rewriting the frozen raw outcome: hold additional continuous compute for a recall-preserving replacement and address proposal coverage first. Do not promote this pair, and do not tune its threshold against the reused development labels. Its small raw gain remains useful recorded evidence of conservative transfer, not professional reliability.

## Verification and cost

Original predictions and all candidates are reused from the prior independently reviewed threshold comparison after verifying model, report, source, code, data, video and label hashes. All 900 original predictions/candidates, label targets and temporal windows verify exactly. On the first label of each of eighteen clips, the adapted inference path exactly matches the internal-selection helper, including all candidate coordinates and window identities. Independent scoring reproduces pooled/per-clip metrics, threshold choices, proposal coverage and the raw gate. All 16 focused comparison/temporal-inference tests pass (2.73 seconds). The sole outcome change is visually inspected.

Adapted sparse inference totals 782.60 seconds across clips on the RTX 4050 Laptop GPU, excluding setup. Peak allocated VRAM is 275,631,104 bytes (262.862 MiB). The historical original two-threshold run totals 814.43 seconds, but it performs additional decoding and was measured separately; this is not a controlled speed comparison. No new training or download occurs during this external check.

These clips are external to the current sixty-clip training/selection split, with disjoint publisher match identifiers. They have been used repeatedly for development; original checkpoint exposure, source-broadcast independence and complete annotation quality remain unknown. Results are not independent qualification. Overall production readiness remains **NO**, including unresolved court/player generalization and physical ball-speed validation.

Evidence is under `outputs/vision_upgrade_audit/wasb_head_external01`: `report.json`, `review.json`, `visual_review.json`, `comparison_00.jpg`, `filter_feasibility.json` and `completion_verification.json`. Report SHA256: `d0d88e69e52503edc2d941c313bb9e090a3872d851b5d3f222ce23ae861053fe`. The model, threshold and raw gate are specified in the unchanged `WASB_HEAD_EXTERNAL_PROTOCOL.md`; training is documented in `WASB_HEAD_ADAPTATION_RESULTS.md`.

## Per-clip comparison

| Clip | Original TP/FP/FN | Adapted TP/FP/FN | Original P/R | Adapted P/R |
| --- | --- | --- | --- | --- |
| match143_000 | 31/8/2 | 31/8/2 | 79.49% / 93.94% | 79.49% / 93.94% |
| match144_000 | 48/0/2 | 48/0/2 | 100.00% / 96.00% | 100.00% / 96.00% |
| match145_000 | 45/1/1 | 45/1/1 | 97.83% / 97.83% | 97.83% / 97.83% |
| match146_000 | 48/0/0 | 48/0/0 | 100.00% / 100.00% | 100.00% / 100.00% |
| match147_000 | 45/1/4 | 45/1/4 | 97.83% / 91.84% | 97.83% / 91.84% |
| match148_000 | 33/6/5 | 33/6/5 | 84.62% / 86.84% | 84.62% / 86.84% |
| match149_000 | 39/6/1 | 40/5/0 | 86.67% / 97.50% | 88.89% / 100.00% |
| match150_000 | 42/3/3 | 42/3/3 | 93.33% / 93.33% | 93.33% / 93.33% |
| match151_000 | 46/1/1 | 46/1/1 | 97.87% / 97.87% | 97.87% / 97.87% |
| match152_000 | 48/0/0 | 48/0/0 | 100.00% / 100.00% | 100.00% / 100.00% |
| match153_000 | 50/0/0 | 50/0/0 | 100.00% / 100.00% | 100.00% / 100.00% |
| match154_000 | 43/0/0 | 43/0/0 | 100.00% / 100.00% | 100.00% / 100.00% |
| match155_000 | 46/2/4 | 46/2/4 | 95.83% / 92.00% | 95.83% / 92.00% |
| match156_000 | 45/2/4 | 45/2/4 | 95.74% / 91.84% | 95.74% / 91.84% |
| match157_000 | 46/3/4 | 46/3/4 | 93.88% / 92.00% | 93.88% / 92.00% |
| match158_000 | 46/1/3 | 46/1/3 | 97.87% / 93.88% | 97.87% / 93.88% |
| match159_000 | 44/4/1 | 44/4/1 | 91.67% / 97.78% | 91.67% / 97.78% |
| match15_000 | 48/0/1 | 48/0/1 | 100.00% / 97.96% | 100.00% / 97.96% |
