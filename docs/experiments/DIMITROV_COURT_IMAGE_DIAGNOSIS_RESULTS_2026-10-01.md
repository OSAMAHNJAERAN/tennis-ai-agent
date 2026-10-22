# Dimitrov court image diagnosis

The saved fourteen-landmark regressor describes the wrong court orientation and position on the existing diagonal Dimitrov–Tiafoe demo. The single automatic surface initialization finds the court's general footprint but remains rejected by unchanged image-evidence checks. No model inference, quality-threshold change or production integration occurred.

The exact sequentially decoded original frame-zero PNG shows a far-baseline angle about −11.27 degrees; the saved prediction's far baseline is about −0.30 degrees. The rejected calibration has eight inliers among fourteen, below the unchanged 0.75 ratio. Raw predicted-line mean support is 0.1200 at the existing reference width 960. This is gross view/localization mismatch before refinement, not a reason to lower the inlier threshold. The earlier [calibration-support experiment](COURT_CALIBRATION_SUPPORT_RESULTS.md) already rejected that relaxation after a false court fit over spectators.

A source-only review marked thirteen visible landmarks with approximate ±8-native-pixel uncertainty and left the umpire-occluded far-left service intersection null. On the exact saved original frame, all thirteen raw landmarks differ by more than twenty pixels; approximate mean separation is 371.21 pixels, ranging from 50.80 to 604.01. These marks are a single AI visual review, not adjudicated ground truth or independent accuracy evidence. The original review used a native video tool's rounded time label and accidentally marked a different camera-framing instant. That first comparison remains in `report.json` and is explicitly invalidated by `review.json`; corrected scoring happened after the sole candidate run and is not blinded. The failure conclusion is supported by the original-pixel overlays and very large displacement, not by a fine accuracy claim.

The [frozen protocol](DIMITROV_COURT_IMAGE_DIAGNOSIS_PROTOCOL_2026-10-01.md) allowed one automatic hypothesis: the largest existing chromatic/smooth surface component whose hull simplifies to four corners at a fixed 1% perimeter epsilon, with shorter opposite edges assigned as baselines. No prediction coordinates or manual marks enter this candidate. Existing proposal photometry and support/refinement thresholds are unchanged.

| Diagnostic stage | Mean visible line support | Lines with support ≥0.5 | Outcome |
| --- | ---: | ---: | --- |
| Saved raw regression | 0.1200 | 0 | Invalid initialization preserved |
| Sole surface-hull seed | 0.5267 | 5 / 9 | Reject: fewer than six supported semantic lines |
| Refinement's unreturned proposal | 0.6078 | 5 / 9 | Reject: insufficient image-support improvement gate |
| Returned refinement | 0.5267 | 5 / 9 | Exactly unchanged seed |

The seed's approximate mean separation from corrected visible source marks is 18.78 native pixels, maximum 36.06, with five of thirteen above twenty pixels. It establishes a useful orientation lead on this exposed development image; it does not establish accurate physical mapping. The four outer-boundary support fractions remain 0.10, 0.07, 0.00 and 0.02. Mean support above 0.5 is insufficient when outer boundaries are wrong. The existing boundary-preserving refinement locks absent near-baseline and left-outer-sideline constraints, preserving their erroneous seed positions. That behavior matches the known tradeoff in [boundary refinement results](COURT_BOUNDARY_REFINEMENT_RESULTS.md).

The saved proposals further localize missing outer-line evidence. Comparing the approximate exact-frame source boundaries against unchanged six-degree/20-reference-pixel compatibility conditions gives:

| Source boundary | Geometrically compatible ridge proposals | Retained by frozen photometry | Finding |
| --- | ---: | ---: | --- |
| Far baseline | 5 | 1 | Long paint fragments mostly rejected by similar-side condition; retained fraction is only 0.525 |
| Near baseline | 7 | 0 | Every fragment is fully bright and white; similar-side support ranges 0–0.25 |
| Left outer sideline | 4 | 0 | Every fragment is fully bright and white; similar-side support ranges 0.0125–0.2375 |
| Right outer sideline | 4 | 2 | Bright/white paint retained only in some fragments; maximum support 0.8 |

Blue playing surface and green runoff differ in brightness on opposite sides of the boundary. The original similar-side filter discards actual outer paint even though its ridge is brighter than both sides. This reproduces the mechanism identified in [missing-boundary diagnosis](COURT_MISSING_BOUNDARY_DIAGNOSIS.md). It does not prove that removing the filter is safe: nets, advertisements, fences and other bright transitions remain distractors. No second candidate or photometric ablation was run in this task.

Artifacts are preserved under `outputs/vision_upgrade_audit/dimitrov_court_image01/`: exact original pixels, initial source marks, raw prediction overlay, frozen line proposals, surface/hull, sole seed, unchanged returned refinement, corrected exact-frame source marks, original `report.json`, correction/verification `review.json`, saved-proposal `boundary_evidence.json`, and original frozen protocol bytes. Input excerpt, geometry, helper, script and image hashes are retained. Review verified every original image hash, exactly one rejected candidate, and byte-identical numeric seed/returned coordinates. Runners are `scripts/evaluate/diagnose_dimitrov_court_image.py`, `review_dimitrov_court_image.py` and `inspect_dimitrov_boundary_evidence.py`; the first completed in about 5.7 seconds using pure OpenCV/SciPy and one decoded frame.

Runtime calibration remains rejected. A future development protocol could test image-supported initialization and boundary paint evidence on the existing failure controls before any integration. A fresh independently labeled supported-view set is still required for qualification.
