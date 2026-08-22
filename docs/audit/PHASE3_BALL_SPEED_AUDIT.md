# Phase 3 Audit: Ball Speed Estimation & Monocular Physics Limitations

> **Audit ID:** `AUDIT_BALL_SPEED_20260822`
> **Date:** 2026-08-22
> **Auditor:** AI/ML Engineering Agent
> **Status:** AUDITED & CORRECTION SPECIFIED

---

## 1. Audit of Existing Ball Speed Implementation

The existing speed calculation in `src/analytics/ball_analytics.py` and `src/pipeline/phase2_pipeline.py` operates as follows:

```python
d_m = math.hypot(p_curr.court_x_m - p_prev.court_x_m, p_curr.court_y_m - p_prev.court_y_m)
dt_frame = timestamps[i] - timestamps[i-1]
if 0 < dt_frame and d_m <= 4.0:
    p_curr.speed_kmh = float((d_m / dt_frame) * 3.6)
```

### Findings & Strengths:
1. **Time Base:** Correctly utilizes native frame timestamps (`dt = timestamps[i] - timestamps[i-1]`), avoiding hardcoded 30 FPS assumptions.
2. **Metric Conversion:** Converts planar court displacement in meters into kilometers per hour ($\times 3.6$).
3. **Gating Limit:** Discards impossible frame displacements ($>4.0$ m/frame, corresponding to $>432$ km/h).

---

## 2. Critical Physical Limitations in Monocular Computer Vision

### The Projection Parallax Effect
The existing pipeline computes speed by projecting 2D pixel coordinates $(x_{px}, y_{px})$ onto the court ground plane $(X_m, Y_m, 0)$ using the $3 \times 3$ planar homography matrix $H$:

$$\begin{bmatrix} X_{proj} \\ Y_{proj} \\ 1 \end{bmatrix} \sim H \begin{bmatrix} x_{px} \\ y_{px} \\ 1 \end{bmatrix}$$

**Physical Reality:**
- A planar homography $H$ is valid **only for points that physically lie on the ground plane ($Z = 0$)**.
- When a tennis ball is in flight at elevation $Z > 0$ (e.g. $1.5\text{--}3.0$ meters above the court during serves and overhead rallies), the camera ray passes through the ball and strikes the court plane behind it.
- As the ball rises or falls, the ground-projected point $(X_{proj}, Y_{proj})$ moves faster or slower than the true 3D spatial velocity vector $\mathbf{v} = (v_x, v_y, v_z)$.

### Crucial Scientific Distinction:
To maintain rigorous scientific and academic integrity in project T88J709:
- **Never claim that simple 2D court-plane homography projection yields true 3D ball speed.**
- **Accurate Terminology:** The metric must be explicitly designated as **"2D Court-Projected Ball Speed Estimate"**.

---

## 3. Required Phase 3 Scientific Improvements

1. **Segment-Level Physics Smoothing:**
   - Rather than calculating instantaneous noisy frame-to-frame derivatives across individual frames, calculate piecewise segment speeds between physical event boundaries:
     - Segment 1: `SERVE_CONTACT` $\to$ `BOUNCE_1`
     - Segment 2: `BOUNCE_1` $\to$ `PLAYER_2_HIT`
     - Segment 3: `PLAYER_2_HIT` $\to$ `BOUNCE_2`
2. **Event Boundary Discontinuity Isolation:**
   - Do NOT calculate speed across hit or bounce frames where velocity vectors abruptly invert.
3. **State-Based Uncertainty Propagation:**
   - Segments composed of $\ge 80\%$ `DETECTED` / `TRACKED` points carry `HIGH` confidence.
   - Segments relying on `PREDICTED` or `INTERPOLATED` points carry `MEDIUM` / `LOW` confidence.
   - Jump discontinuities $>40$ m/s are flagged as `INVALID_SPEED`.
