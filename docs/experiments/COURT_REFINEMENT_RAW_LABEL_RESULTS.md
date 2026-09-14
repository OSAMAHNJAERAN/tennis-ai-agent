# Court refinement: unchanged publisher-label agreement

All eight planned source-separated images complete. The frozen local refinement increases agreement with unchanged publisher labels from42 to65 of112 landmarks within seven native pixels. Three refinements are accepted. All eight raw/projected/refined image panels were visually inspected, including rejected cases. This is not independently verified court accuracy: known label errors remain, and independence from the current ResNet checkpoint's training data is unknown.

| Stage | TP / FP / FN | Precision | Recall | Median available native error |
| --- | --- | ---: | ---: | ---: |
| Raw ResNet landmarks | 42 / 70 / 70 | 37.50% | 37.50% | 10.18 px (112 points) |
| Geometrically calibrated projection | 42 / 56 / 70 | 42.86% | 37.50% | 9.11 px (98 points) |
| Frozen line refinement | 65 / 33 / 47 | 66.33% | 58.04% | 3.78 px (98 points) |

The projected and refined stages withhold all fourteen points for one geometrically invalid initialization. Consequently, their precision and available-error statistics have different coverage from the raw detector. The paired projected-to-refined comparison has identical coverage:23 new correct landmark matches and zero lost correct matches. Mean available error changes19.04 to16.81 native pixels; the remaining annotation/model failures dominate the mean. A wrong localization counts both FP and FN. Labels outside the image would be excluded, but all112 supplied landmarks are eligible here.

| Image | Projected matches | Refined matches | Decision |
| --- | ---: | ---: | --- |
| PuAPCalPLM4_1700 | 0 | 0 | Insufficient compatible lines |
| aSfOC_BE9ac_1450 | 2 | 2 | Unstable or excessive proposed correction |
| 8PMR9mWhIzQ_1350 | 1 | 14 | Accepted |
| phGwJafjk3M_50 | 7 | 14 | Accepted |
| eOSRHcLforU_50 | 10 | 13 | Accepted |
| uXkAqALS0AA_100 | 12 | 12 | Insufficient compatible lines |
| ANPF-QPUe70_1500 | 0 | 0 | Invalid geometric initialization |
| VDy_CN9nNtY_50 | 10 | 10 | Insufficient compatible lines |

Visual review supports improved visible-paint alignment on the two accepted grass images and accepted indoor image. In PuAPCalPLM4_1700, the supplied labels visibly form a shrunken/shifted court even though predictions largely follow the real court lines; all zero matches remain in the score. aSfOC_BE9ac_1450 combines residual prediction misalignment with imperfect far-side labels. uXkAqALS0AA_100 is a close-alignment clay control left unchanged. The right side remains imperfect in VDy_CN9nNtY_50. ANPF-QPUe70_1500 is a replay/graphic court view with a gross prediction error that fails geometric initialization. None of these labels were removed or corrected.

The protocol in `COURT_REFINEMENT_RAW_LABEL_PROTOCOL.md` was saved before model inference. The current ResNet model, preprocessing, canonical geometry and all refinement thresholds are unchanged. Labels enter only after every detector, calibration and refinement decision. The complete manifest and all images were hash-checked. The dataset's external-source grouping is relative to the publisher heatmap training split; it must not be presented as proof of independence from the ResNet model.

`outputs/vision_upgrade_audit/court_refinement_raw_labels01/report.json` retains all native coordinates, label identities, distances, counts, source/code hashes and per-image refinement decisions. Its separate `review.json` records exact metric replay, all-eight visual inspection, unchanged labels, and the23-gain/zero-loss comparison. Rejected projected points undergo a reference-scale round trip; maximum numerical difference is1.14e-13 native pixels, with identical missing and scored correctness states. The original benchmark remains intact.

All18 targeted tests pass, covering landmark identity, inclusive seven-pixel tolerance, wrong-location FP+FN, missing predictions, ineligible labels, count mismatches, known-geometry refinement and calibration rejection. This is a targeted verification result, not a full repository regression run.

This result provides additional measured support for local refinement when the court initialization is close and enough visible line evidence exists. It does not resolve absent or misidentified court landmarks, graphic replays, gross viewpoint failures, annotation quality, ground-speed error or camera-general court mapping. No production court behavior changes. Further evaluation needs reliable labels and broader well-initialized controls before connecting the correction to physical analytics.
