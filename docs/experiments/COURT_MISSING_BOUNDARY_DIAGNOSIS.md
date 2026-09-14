# Missing baseline evidence: four-case diagnosis

The frozen line detector and geometric matcher find compatible baseline candidates in every one of the four diagnosed images. All are discarded by the subsequent photometric conditions. This localizes the loss of evidence before homography fitting. No model, label or threshold changes were made in this diagnostic.

| Image | Boundary | Geometrically compatible ridge proposals | Retained | Observed rejection |
| --- | --- | ---: | ---: | --- |
| 5QObSWGBQB8_1200 | Far baseline | 7 | 0 | Low-saturation condition: at most 2.5% white samples |
| ktiDOhLZIVs_3500 | Near baseline | 6 | 0 | Similar-side condition: at most 16.25% passing, despite 98.75-100% bright and 100% white samples |
| 6CRu9DY7KII_800 | Far baseline | 5 | 0 | Low-saturation condition: at most 5% white samples |
| 6CRu9DY7KII_850 | Far baseline | 3 | 0 | Low-saturation condition: at most 5% white samples |

The raw LSD compatibility counts are also 7, 6, 5 and 3. Ridge-center adjustment does not remove the candidates from geometric compatibility. All four source-context proposal overlays were visually inspected. The rejected segments follow visible baseline paint. The hard court has visibly different inner and outer surfaces; its line is brighter than both sides despite their brightness difference. On the clay examples, line appearance does not satisfy the fixed low-saturation filter. The evidence does not establish which physical or imaging factor causes that saturation.

These are deliberately selected diagnosis cases from reused development data. The observations motivate a controlled factorial ablation of whiteness and similar-side conditions. They do not establish that either condition can be removed safely on other images: extra nets, advertising and other bright structures can become candidates. The separate frozen protocol `COURT_RIDGE_ABLATION_PROTOCOL.md` requires all 64 saved case/model evaluations for each variant, unchanged geometry and scoring, and preservation of all results.

Artifacts: `outputs/vision_upgrade_audit/court_missing_boundary_diagnosis01/report.json`, its separate `review.json`, and four image overlays. Runner: `scripts/evaluate/diagnose_missing_court_boundaries.py`. The report records source/code/image hashes, raw compatible segments, shifted ridge candidates, each photometric fraction and initial/final assignments.
