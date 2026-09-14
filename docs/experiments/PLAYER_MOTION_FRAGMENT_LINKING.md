# Bidirectional motion fragment linking

Protocol frozen before replay. Static boundary-IoU pilot01 produces no selected-player improvement. V03 source4 ends at frame380 and source230 starts at383 (zero-based): elapsed0.1001s, boundary IoU0.25365, approximately11 pixels horizontal displacement. Static overlap is an inadequate model for a moving player.

Keep the0.20s maximum boundary interval, strict non-overlapping source lifetimes, unique mutual-best edge selection, ambiguity tolerance1e-9, offline chain identities and original racket-based selection. Replace static boundary overlap with bidirectional motion agreement. Fit least-squares x/y center velocity independently to actual observations within0.10s of each endpoint, requiring at least3 observations on each side. Reject either fit when maximum center residual exceeds one quarter of the median person-box width. Translate the last box forward and the first box backward by the fitted velocities and actual elapsed time, preserving each box size. Require IoU at least0.50 in BOTH directions; rank by the smaller IoU. Predicted boxes are for association only and must never be emitted as observed detections.

No threshold sweep or label-informed link selection. Evaluate all2230 UVY frames against unchanged labels, verify previous selections, preserve source IDs, and review every accepted edge. Direction reversals, camera motion, nearby similar people, overlapping/reused IDs and longer gaps remain limitations. This is reused development evidence, not verified re-identification or physical velocity.


## Completed result

All2230 frames completed, with exact reproduction of the original nearest-racket selections. Bidirectional motion links no sources in V01/V02. V03 links199 to208 (the same apparent distant white-clad person on boundary review) and5 to266 (a foreground spectator). Neither chain accumulates enough racket evidence to become selected. Selected-player metrics remain identical to the static-link and nearest-racket baselines:872 TP,103 FP,3331 FN; precision89.44%, recall20.75%. No metric improvement, no production promotion.

The intended near-player link4 to230 still fails. Its local preceding velocity is[47.63,9.45] pixels/s and following velocity[20.72,1.46] pixels/s. Over0.1001s the forward/backward predicted horizontal displacements are about4.77/2.07 pixels, while the observed boundary center displacement is about11.79 pixels. Forward IoU0.47006 and backward IoU0.33993 both fail the fixed0.50 requirement. These are image-space diagnostic slopes, not physical player speed. The images support the same near player across the gap, but the box motion is inconsistent with the chosen local model.

Both accepted edges and the rejected near-player boundary were visually reviewed. The next distinct hypothesis is that image motion across the missing observations can separate true player motion from detector/tracker box jumps. Do not lower the IoU threshold solely to pass this reused example, and do not fill the missing frames with fitted boxes.

Artifacts: `outputs/vision_upgrade_audit/uvy_motion_fragment_linking_pilot01/report.json`, `frozen_protocol.md`, `boundary_review.jpg` and `boundary_review.json`.17 motion/static/selection tests passed before replay, including moving-player recovery, reverse-direction rejection, overlapping-ID rejection, ambiguous edges and no missing-box synthesis. These tests validate implementation behavior, not re-identification accuracy. The full regression suite was not rerun for these isolated research modules.
