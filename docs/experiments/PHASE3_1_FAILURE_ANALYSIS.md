# Phase 3.1: Independent Event & Localization Failure Analysis

## 1. Scope and Objective
This document details the failure modes, edge cases, and physical limitations discovered during the Phase 3.1 independent evaluation of event detection, tracker state accuracy, and metric court localization.

---

## 2. Categorized Failure Modes & Root Causes

### 2.1 Far-Court Perspective Homography Sensitivity
- **Observation:** While image-space bounce localization error is exceptionally low ($0.05$ px median, $0.21$ px maximum), metric court localization error exhibits a long tail ($425.7$ cm maximum at Frame 138).
- **Physical Root Cause:** At the far baseline ($y \approx 250$ px in image space), the court surface is viewed at a grazing angle ($\sim 15^\circ\text{--}20^\circ$). A $1$-pixel vertical deviation along the optical axis maps to $1\text{--}2\text{ meters}$ of ground displacement in metric space:
  $$\frac{\partial Y_m}{\partial y_{px}} \gg \frac{\partial X_m}{\partial x_{px}}$$
- **Implications for Line Calling:** Line calls on the near baseline achieve millimeter/centimeter accuracy ($<1\text{ cm}$), whereas far-court line calls require either multi-camera setups or line-orthogonal local coordinate refinement.

### 2.2 Blind Kinematic Extrapolation Drift
- **Observation:** When the ball detector drops measurements for $\ge 3$ frames, the Kalman filter's `PREDICTED` state produces errors up to $455.29$ px (median $182.38$ px).
- **Physical Root Cause:** Constant acceleration or ballistic motion models in 2D image coordinates diverge rapidly when spin, air drag, and perspective foreshortening interact without measurement updates.
- **Remediation:** Enforce a strict max prediction threshold ($N \le 2$ frames). Beyond 2 frames without measurement, mark state as `MISSING` rather than propagating high-drift predictions.

### 2.3 Close Half-Volley Event Fusion
- **Observation:** In rapid half-volleys where the ball bounces and is struck by a racket within $\le 3$ frames ($< 100$ ms), the bounce and hit kinematics merge.
- **Physical Signature:** Ground impact deceleration immediately combines with forward racket acceleration.
- **Remediation:** Multi-scale temporal NMS detects the dual peak signature (downward-to-upward ground reversal followed immediately by forward impulse).

### 2.4 Pre-Serve Toss Parallax
- **Observation:** Projecting airborne serve toss points ($Z \approx 3.0$ m) onto the ground plane produces instantaneous speed spikes ($\sim 310$ km/h) and artificial flight distances.
- **Remediation:** Piecewise flight segmentation isolates the serve toss arc from in-rally ground strokes, preserving the physical integrity of rally statistics.

---

## 3. Failure Mode Severity & Readiness Assessment

| Failure Mode | Severity | Impact on Event F1 | Impact on Line Calls | Current Mitigation Status |
| :--- | :---: | :---: | :---: | :--- |
| **Far-Court Optical Axis Sensitivity** | High | Low (F1 unaffected) | High (Requires tolerance window) | Documented; near-court validated ($<1$ cm) |
| **Kalman Prediction Drift** | High | Low (Gating prevents false hits) | High (Avoid predicting bounce) | Prediction limited to $\le 3$ frames |
| **Half-Volley Temporal Fusion** | Medium | Medium (Requires fine scale NMS) | Low | Multi-scale candidate extraction |
| **Airborne Projection Parallax** | Medium | None | Low | Labeled "2D Court-Projected Speed Estimate" |
| **Corrupted Ground Truth Provenance** | Critical | High (Artificially inflated metrics) | Critical | **Resolved in Phase 3.1 via raw video GT** |
