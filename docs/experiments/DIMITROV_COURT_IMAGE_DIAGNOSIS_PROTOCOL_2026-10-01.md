# Dimitrov diagnostic court image protocol

Review correction: the original coordinate table below was marked on a native video-tool preview whose rounded `00:00:00` label corresponded to a different camera-framing instant. It is invalid for exact frame-zero scoring and is retained as provenance. The original frozen protocol bytes are preserved at `outputs/vision_upgrade_audit/dimitrov_court_image01/frozen_protocol.md`. Corrected source-only coordinates on the exact saved decoded frame-zero PNG, their uncertainty and the post-run review limitation are recorded separately in `review.json`. The sole automatic candidate and all thresholds remain unchanged; see [diagnosis results](DIMITROV_COURT_IMAGE_DIAGNOSIS_RESULTS_2026-10-01.md).

This bounded development diagnosis uses only the existing ten-second, 600-frame display-permitted Dimitrov–Tiafoe rally excerpt and its saved rejected court prediction. No qualification candidates, new model inference or production threshold changes are permitted. The original source pixels were reviewed through native video-perception images before any prediction overlay or candidate selection.

The original court is an elevated diagonal view with a strongly oblique far baseline, substantial blue playing surface and green runoff. Most painted corners, singles sidelines and service intersections are visible. The umpire chair occludes the far-left service intersection; do not score that point as independently visible. Camera framing drifts during play. Saved raw predictions describe a different, almost horizontal far baseline. Existing evidence rejects a global inlier-ratio reduction, because it accepted a spectator court in a previous control. Local refinements allow only 20 reference-pixel displacement and six-degree line compatibility, so they are not designed to fix gross view orientation.

Before executing the candidate, preserve these approximate source-only frame-zero annotations in native 1920×1080 pixels. They are a single AI review with ±8-pixel coordinate uncertainty, not independently adjudicated accuracy labels. Hidden intersection 8 has no coordinate. A disagreement far beyond this uncertainty can diagnose gross placement; these marks cannot certify subpixel or seven-pixel accuracy.

| Landmark identity | Approximate x,y | Visibility |
| --- | --- | --- |
| 0, far-left outer corner | 17,378 | Visible |
| 1, far-right outer corner | 656,251 | Visible |
| 2, near-left outer corner | 1064,955 | Visible |
| 3, near-right outer corner | 1797,703 | Visible |
| 4, far-left singles corner | 108,360 | Visible |
| 5, near-left singles corner | 1160,921 | Visible |
| 6, far-right singles corner | 587,265 | Visible |
| 7, near-right singles corner | 1707,733 | Visible |
| 8, far-left service intersection | Null | Occluded by umpire chair |
| 9, far-right service intersection | 779,349 | Visible |
| 10, near-left service intersection | 852,753 | Visible |
| 11, near-right service intersection | 1378,592 | Visible |
| 12, far center T | 553,405 | Visible |
| 13, near center T | 1100,677 | Visible |

One automatic candidate is allowed: use the existing frozen chromatic/smooth-surface proposal (`probe_court_surface_support.surface_candidates`), choose its largest component whose convex hull simplifies to exactly four corners with fixed `approxPolyDP` epsilon 1% of perimeter, and initialize the standard court from those four corners. The pair of opposite shorter edges supplies baseline semantics; the upper midpoint supplies the far baseline. If no such component exists, record rejection without alternatives. This is a surface-derived initialization hypothesis, not a semantic court mask.

Evaluate the single candidate against the existing frozen bright-ridge proposals, line-support distance three pixels at reference width 960, at least six visible lines, mean support at least 0.5 and at least six individually supported lines. Retain original photometric filters. Report support of both baselines and outer sidelines separately. The candidate may undergo the existing boundary-preserving refinement once, with its exact unchanged conditions. Never accept an alternate initialization solely from a four-corner fit or lowered inlier ratio; record all failure reasons and leave runtime untouched.

Compare raw saved landmarks, automatic seed and returned refinement with the prewritten visible source marks and report all displacements/withheld points. Render original pixels, source-only marks, saved predictions, selected surface/hull, line proposals and the candidate separately. Preserve hashes for input excerpt, saved court geometry, protocol and helper implementations. This one reused clip can motivate development, but cannot establish qualification or the supported-view envelope.
