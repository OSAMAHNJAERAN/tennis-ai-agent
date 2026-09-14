# Conservative player-fragment linking

Protocol declared before evaluating the linking rule. Previous nearest-racket ownership removes foreground spectator errors but still misses actual person boxes when the source tracker changes ID. A label-assisted diagnosis finds 60 matched near-player observations in V03 on ID230 following the supported ID4, and 276 matched near-player observations in V01 on unsupported IDs. Labels identify the failure; they must never choose links.

For every actual source ID, retain its first and last observed boxes and frame indices. A link is eligible only when the successor starts strictly after the predecessor ends, the elapsed boundary time is at most 0.20 seconds, and the boundary boxes have IoU at least 0.50. Do not extrapolate positions, fill missing frames or use ground truth. Among eligible edges, retain only a uniquely highest-IoU outgoing edge that is also the uniquely highest-IoU incoming edge. Numerical ties within 1e-9 are ambiguous and produce no link. These time-ordered edges form disjoint chains; use the earliest source ID as each chain's canonical ID.

Replay the exact existing racket observations from the nearest-owner pilot, remapping their actual source IDs to the canonical IDs. Apply the unchanged two-racket-frame, 0.50-second observed-person, opportunity-prior5 and maximum-two-person selection rules. Remap every observed person box but never change coordinates or confidence. Preserve original source IDs in each exported observation for review. This is a whole-sequence offline linking experiment, not validated re-identification.

Evaluate all three UVY recordings and unchanged publisher labels with official TrackEval. Report links, coverage, FP/FN, ID switches, IDF1 and HOTA; review first/middle/last frames and every accepted boundary. These recordings are reused development data with known label errors. Similar nearby people, tracker ID reuse, abrupt camera changes and the absence of appearance evidence can still cause identity mistakes. Production defaults are unchanged unless wider evidence supports promotion.

The preceding crop diagnostic is `outputs/vision_upgrade_audit/player_racket_evidence_gap_audit.json`: all102 sparse broadcast racket targets are fully inside at least one existing crop, while36 lack any raw IoU0.5 proposal. This does not establish UVY racket recall, and larger crops are not justified by this sample. Missing detections and source-track fragmentation require separate treatment.


## Completed result

All2230 frames completed against the unchanged labels, with exact reproduction of the nearest-racket baseline selections. Static overlap links no sources in V01 or V02. V03 links5 to266 and265 to286, both foreground spectators on boundary-image review; neither chain has sufficient racket evidence to become a selected player. Selected-player metrics remain exactly unchanged: V01 TP492/FP102/FN953; V02 TP0/FP0/FN1936; V03 TP380/FP1/FN442. No production promotion.

The intended V03 near-player link4 to230 is rejected: source4 ends at zero-based380, source230 begins at383, elapsed0.1001s, boundary IoU0.25365. The center shifts about11.8 pixels horizontally. V01 IDs2 and150 overlap at frame382; later near-player fragments have gaps of23 and24 frames, and the248-to311 boundary has insufficient static overlap. Relaxing all constraints would risk merging distinct people, so the next experiment used motion agreement with the same maximum gap and overlap threshold.

The pre-run protocol is preserved in `outputs/vision_upgrade_audit/uvy_player_fragment_linking_pilot01/frozen_protocol.md`. Full results are in `report.json`; every accepted edge and the rejected near-player edge were visually reviewed in `boundary_review.jpg`.13 static-link and selection tests passed before replay. Boundary similarity alone is insufficient for recovery of the measured source-ID failure.
