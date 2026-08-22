# Phase 3: Tennis Match Events & Speed Estimation Failure Analysis

## 1. Scope & Objective
This document provides an in-depth failure analysis of edge cases, physical ambiguities, and failure modes encountered during tennis match event detection, bounce localization, and monocular speed estimation.

---

## 2. Identified Failure Modes & Edge Cases

### 2.1 Monocular Court Elevation Parallax
- **Description:** Homography $\mathbf{H}$ assumes all detected points reside strictly on the flat 2D ground plane ($Z = 0$). When the tennis ball is high in the air (e.g. serve toss at $Z \approx 3.0$ m or high topspin lob), projecting its image coordinate $(x_{px}, y_{px})$ onto the court plane projects the ball artificially far beyond its true $(X, Y)$ ground footprint.
- **Impact on Speed:** Near the serve toss (frames 0--20) and high clearance shots, court-projected 2D instantaneous speeds exhibit spikes ($\sim 300$ km/h) because vertical ascent in image space is mapped into rapid ground displacement.
- **Mitigation Implemented:**
  1. Clearly designated all speed outputs as **"2D Court-Projected Ball Speed Estimate (km/h)"** in accordance with monocular vision physics.
  2. Applied piecewise flight segmentation so that high toss speeds do not corrupt in-flight rally speed metrics.
  3. Added an explicit `scientific_disclaimer` in `ball_metrics.json`.

### 2.2 Volleys and Mid-Air Racket Hits
- **Description:** A volley occurs before the ball bounces on the court surface. If an event detector assumes a strict alternating pattern of $\text{Hit} \to \text{Bounce} \to \text{Hit}$, a volley would be misclassified as a bounce or missed entirely.
- **Kinematic Signature:** Volleys exhibit sharp angular deflection ($\Delta\theta > 40^\circ$) and acceleration spike within a player reach zone ($d_P \le 160$ px) at a higher image altitude ($y < y_{court\_baseline}$).
- **Mitigation Implemented:**
  Player reach-zone proximity takes priority over vertical apex detection. If the ball is within $r = 160$ px of a player bounding box when a sharp direction change occurs, it is classified as `PLAYER_HIT` regardless of whether a bounce occurred.

### 2.3 Close Half-Volleys & Short Hops
- **Description:** When a player strikes the ball immediately after it bounces off the ground (within 3--5 frames), the bounce inflection and racket strike merge into a single temporal window ($\Delta t < 150$ ms).
- **Failure Mode:** A single-scale NMS window can suppress the bounce candidate in favor of the larger racket acceleration peak.
- **Mitigation Implemented:**
  Multi-scale temporal analysis: checks for dual inflection signatures (downward-to-upward ground reversal followed immediately by forward acceleration).

### 2.4 Deep Baseline Occlusions
- **Description:** When the ball lands deep on the far baseline, the ball's pixel diameter drops to $2\text{--}4$ pixels, and motion blur can cause temporary detector dropouts ($1\text{--}2$ frames).
- **Mitigation Implemented:**
  Temporal Kalman filter state propagation (`PREDICTED` / `INTERPOLATED`) preserves trajectory continuity, allowing numerical derivatives to identify the bounce inflection despite missing raw detections.

---

## 3. Robustness Matrix

| Challenge / Edge Case | Risk Level | Detection Behavior | Failure Mitigation |
| :--- | :---: | :--- | :--- |
| **Serve Toss Parallax** | Medium | Apparent high ground displacement | Flight segmentation restricts serve speed window |
| **Topspin Curvature** | Low | Parabolic arc mistaken for bounce | Curvature thresholding rejects gentle parabolic arcs |
| **Player Racket Overlap** | High | Player body bounding box occludes ball | Kalman velocity extrapolation maintains trajectory |
| **Net Cord Deflection** | Medium | Minor direction change at court center | Net-plane boundary check classifies deflection event |
| **Camera Shake / Vibration** | Low | High-frequency jitter in image coordinates | Savitzky-Golay / Central difference derivative smoothing |
