# Court line evidence: localization proposals and calibration diagnostic

## Direct line proposals

The failed regressors and heatmap pilot motivated a check of visible image evidence. `scripts/evaluate/probe_court_line_segments.py` processes the first frame of all three acquired UVY and all three CalTennis clips at width960. It uses OpenCV LSD, rejects segments shorter than30 pixels, then samples grayscale brightness and HSV saturation along each segment and on both sides. Small normal offsets account for LSD locating a line edge rather than its center. A supported sample requires ridge brightness above100, contrast above10 against both side strips, saturation below110, and side-brightness difference below45. Segments with at least0.5 supported samples are retained. Thresholds were fixed before the run.

The completed probe retains48/99/58 segments on UVY and79/168/86 on CalTennis. These are proposal counts, not accurate court-line counts. The first CalTennis overlay was visually inspected: near-side court markings are recovered where the learned models failed, but fences, net edges, poles and other bright structures also pass. Full candidates and support components are retained in `outputs/vision_upgrade_audit/court_line_segments/report.json`. No semantic court assignment or homography is inferred from these segments alone.

## Surface-support experiment

`scripts/evaluate/probe_court_surface_support.py` tests an additional fixed heuristic: a large connected region with moderate saturation/value and relatively smooth local brightness. It searches hue bins, closes small holes, requires a minimum area/solidity, and selects the largest eligible component below the top30% of the image. A line must have surface support at offsets on both sides. This is not a trained court segmentation model; grayscale courts, occlusion and similarly colored distractors can defeat it.

Retained line counts become35/49/0 on UVY and18/50/15 on CalTennis. The first CalTennis result was visually inspected: many fence/pole proposals disappear while useful near-court markings remain. However, the selected component omits distant court regions separated by the net and rejects boundary lines when adjacent runoff differs in color. UVY V03 has no eligible surface. This heuristic therefore does not establish a complete camera-general court detector. Artifact: `outputs/vision_upgrade_audit/court_surface_support/report.json`.

## Frozen court-support comparison

The broader direct-line proposals were also used as a diagnostic check of proposed court geometry, without the surface filter. `scripts/evaluate/audit_court_line_support.py` samples each of the nine expected court segments between supplied landmarks. At width960, a sample is supported if it lies within3 pixels of a detected bright-line segment. A fixed check requires at least six sufficiently visible lines and mean visible-line support of0.5. This protocol was fixed before its18-case comparison; it is not a calibrated probability or independent court-accuracy metric.

All twelve first-frame ResNet predictions from the three UVY and three CalTennis clips, using both baseline and geoaug checkpoints, fail this check. Their mean support ranges0.00-0.0933. Earlier visual inspection established that these courts were misplaced, including cases whose supplied-landmark RANSAC fit passed. Thus image support can expose a failure that geometric self-consistency misses.

The six broadcast heatmap controls have support0.6356/0.8644/0.9489/0.7933/0.4533/0.5378 in match138/139/14/140/141/142 order. Five pass. The rejected match141 control was then visually inspected: several left-side landmark predictions are displaced from the actual court lines. The rejection is therefore not automatically a false negative. Conversely, passing other controls is not proof of every landmark being correct; only match138 and match141 control overlays have been visually reviewed so far. Report: `outputs/vision_upgrade_audit/court_line_support_comparison.json`.

## Limits and next implementation work

The result supports developing an image-evidence check in addition to landmark-fit residuals. It does not yet justify a production gate: the small case set is known development footage, the supported-line metric can match unrelated bright structures, missing/short/off-screen segments complicate its denominator, and complete independent court annotations are absent. No threshold was relaxed to make all controls pass. No production defaults were changed.

Direct lines offer better visible localization in at least one difficult ground-level frame, but still need reliable line-family assignment and partial-court geometric fitting. The failed surface experiment and its artifacts remain available rather than being represented as a successful segmentation system. This work does not enable physical speed, bounce, or event authority, and does not resolve the user's overall tracking accuracy requirements.


## Follow-up on integrated temporal-spacing videos

The unchanged diagnostic was applied to projected canonical court lines from the actual pipeline camera-calibration reports at first/middle/last frames of match143 and match148. Both geometric calibrations were accepted by the pipeline. Match143 line support is0.4178/0.4267/0.3911 and all three fail the frozen0.5 mean-support gate. Match148 support is0.8100/0.8189/0.7144 and all three pass. The fixed width960, distance3px, minimum-six-lines and original bright-ridge proposal thresholds were not changed.

The first-frame diagnostic overlays from both clips were visually inspected. Match143 has a clearly displaced far baseline and left sidelines relative to visible paint; match148 matches the visible court more closely. This demonstrates an additional observed failure that image evidence can catch despite a low homography fit residual. It does not establish false-acceptance rates or justify a broad production gate. All six sample calculations, segment proposals, transformed canonical points, input/camera/code hashes and inspection scope are saved in `outputs/vision_upgrade_audit/spaced_pipeline_court_line_probe/report.json`.

Further court work should evaluate rejection or refinement against this failure alongside the prior broadcast, UVY and CalTennis controls. Preserve line identities and perspective constraints; do not independently snap landmarks onto whichever bright structure is nearest. No new runtime court decision is enabled by this diagnostic.


A frozen local projective-refinement pilot and850-frame first-anchor replay are complete. The visible match143 baseline error is corrected in the inspected frames, with image-support improvements retained under saved camera motion. No runtime gate or physical-accuracy claim is enabled. See `COURT_PROJECTIVE_LINE_REFINEMENT_RESULTS.md` for all24 case outcomes, initialization limitations, visual review and paired videos.
