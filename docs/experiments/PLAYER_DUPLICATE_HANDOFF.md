# Brief duplicate-ID handoff

Pre-replay protocol. An all-track diagnostic finds one short overlapping handoff with IoU at least0.50: V01 source2 ends when source150 starts at frame382 (zero-based), overlap1 frame, IoU0.68879. No corresponding pair exists in V02/V03. Gap-only association excludes such simultaneous observations by design.

Detect candidates from original person observations only. Require predecessor.start < successor.start <= predecessor.end < successor.end. The inclusive overlap interval must last at most0.10 seconds; both IDs must have actual boxes on EVERY frame of that interval; all corresponding box IoUs must be at least0.50. Pick uniquely best outgoing/incoming edges by minimum overlap IoU, ties within1e-9 ambiguous. Before merging a chain, require every pair of simultaneously present source IDs in the full chain to meet the same box-overlap requirement. A conflict rejects that chain. No GT, court or racket labels choose links.

Combine these links with the already frozen image-flow pilot mapping. On concurrent same-chain observations retain the highest actual person confidence, then lowest original source ID. All retained geometry/confidence comes directly from an original observation. Preserve every original source ID in provenance, including the kept and suppressed box. Racket supports are mapped to the combined chain; keep at most one support per frame, selected by actual racket confidence then lowest original source ID. The same two-support,0.50-second observed-person,prior5 and maximum-two-player rules apply. No interpolation or generated coordinates.

Evaluate all2230 frames against unchanged publisher labels, compare to the verified flow baseline, inspect each accepted shared frame and the added player interval, and record exact additions/removals and ID metrics. Short heavy occlusions or overlapping people may mimic duplicates, so this is a development experiment, not proven re-identification. No production default change without wider validation.


## Completed replay and visual review

All2230 frames complete. The original nearest-racket and frozen image-flow selections reproduce exactly before applying duplicate reconciliation. Only V01 source2-to150 qualifies. Shared zero-based frame382 contains both IDs around the same near player on visual review; the adjacent frames show the continuous player. Source150 has higher person confidence on that shared frame, so its unchanged box replaces source2's box. The remaining59 source150 observations are newly selected. No other recording changes.

| Recording | Flow baseline TP / FP / FN | After duplicate reconciliation | IDF1 before / after | HOTA before / after |
|---|---|---|---|---|
|V01|492 /102 /953|551 /102 /894|37.57% /42.14%|30.33% /34.36%|
|V02|0 /0 /1936|0 /0 /1936|0% /0%|0% /0%|
|V03|440 /1 /382|440 /1 /382|69.68% /69.68%|61.13% /61.13%|

Aggregate991 TP/103 FP/3212 FN yields precision90.59% and recall23.58%, versus90.05%/22.17% for the flow baseline. Against the nearest-racket selector before both recovery steps, this is119 additional matched boxes across two recordings with unchanged103 false selections. Known publisher-label issues remain; these reused recordings cannot establish production accuracy.

The observation-integrity check records60 source150 additions and one source2 removal, comprising a shared-frame box replacement plus59 newly covered frames. Every retained box and confidence equals an actual detection on the same source frame. Nothing is interpolated. Source mappings, accepted edges, and suppressed duplicate identities are preserved in the report. The full paired V01 video decodes819/819 frames at29.97 FPS; both the shared boundary and a decoded frame inside the recovered interval were visually inspected.

42 targeted tests pass across duplicate reconciliation, optical flow, local box-motion/static linking, racket-supported selection, nearest ownership and official tracking metrics. Coverage includes a plausible local duplicate rejected because merging its full existing chain would combine distinct concurrent boxes. This is not a newly rerun full regression suite.

## Decision and artifacts

Keep this as a measured research candidate. V02 is still empty; V01 and V03 miss many distant-player boxes, and longer gaps, overlapping people and unsupported identities remain unresolved. A single successful duplicate handoff is not a general re-identification benchmark. Production defaults, court roles and physical analytics are unchanged.

Artifacts under `outputs/vision_upgrade_audit/uvy_duplicate_handoff_pilot01`: `report.json`, `frozen_protocol.md`, `observation_integrity.json`, `shared_boundary_review.jpg`, `handoff_review_V01.mp4`, `handoff_review_V01.json`, `handoff_review_frame401.jpg` and `final_review.json`. Input/model provenance inherits the verified flow and nearest-racket reports; the replay records their hashes and current source hashes. No new model inference or detector threshold changes were required.
