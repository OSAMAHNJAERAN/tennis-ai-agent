# T88J709 — Phase 6: Shot Classification & Analytics Failure Mode Analysis

## 1. Primary Failure Modes & Mitigations

| Failure Mode ID | Category | Physical / Visual Root Cause | State / Edge Case | Architectural Mitigation |
| :--- | :--- | :--- | :--- | :--- |
| **FM-601** | Handedness Ambiguity | Player handedness unknown or left-handed player treated as right-handed. | Wrist vector reversed relative to body center. | Handedness configuration parameter (`PlayerHandedness`) + fallback to `UNKNOWN` when ambiguous. |
| **FM-602** | Wrist Occlusion & Motion Blur | Fast racket swing blurs wrist/elbow keypoints during impact frame. | Keypoint confidence $< 0.35$. | Multi-frame temporal window $[t-4 \dots t+4]$ + safe abstention to `UNKNOWN`. |
| **FM-603** | Far-Court Low Resolution | Player 2 far court bounding box is $< 80\text{ px}$ high, degrading joint localization. | Keypoints collapse to torso. | Court-side normalization factor + geometry relative offset fallback. |
| **FM-604** | Dead-Ball Continuation | Post-fault or post-point practice hit (e.g. Frame 84 return). | Ball is physically dead. | Strict **dead-ball suppression**: `PointOutcomeResolver` filters dead-ball hits, excluding them from shot statistics. |
| **FM-605** | Point Review Ambiguity | Unresolved line call (`POINT_REVIEW_PENDING`). | Point winner uncertain. | Point marked as `PENDING_REVIEW` in analytics without attributing unverified winner. |
| **FM-606** | Perspective Direction Distortion | Camera perspective distorts screen left/right angles. | Down-the-line shot appears diagonal on screen. | All direction classifications computed strictly in canonical metric court coordinates $(X, Y) \in [0, 10.97] \times [0, 23.77]$. |
