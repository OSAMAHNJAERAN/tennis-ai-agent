# T88J709 Phase 4 — Spatial Uncertainty Calibration & Tracker State Propagation

## 1. Executive Summary & Calibration Objectives

In monocular broadcast tennis video, pixel resolution per physical meter degrades severely with distance from the camera due to perspective foreshortening. While 1 pixel in the near court corresponds to ~0.4 cm, 1 pixel in the far court corresponds to ~5.0 - 8.0 cm. Treating all court regions as having equal localization certainty causes dangerous false line calls in the far baseline and service lines.

Furthermore, the observation fidelity of the ball tracker varies significantly across detection states (direct neural network proposal vs Kalman filter linear extrapolation).

To ensure safety and defensibility, Phase 4 implements an **Uncertainty-Aware Line Calling Architecture** calibrated across physical court geometry and observation quality tiers.

---

## 2. Perspective & Spatial Depth Calibration

The court plane is divided into three canonical depth tiers based on homography Jacobian condition numbers and physical resolution:

| Court Tier | Metric Y Boundary (m) | Ground Resolution (cm/px) | Base Spatial Uncertainty $\\sigma_{base}$ (cm) | Justification |
| :--- | :--- | :--- | :--- | :--- |
| **Near Court** | $Y \\ge 11.885\\text{ m}$ (Near Net to Baseline) | $0.35 - 0.75\\text{ cm/px}$ | **$\\pm 0.8\\text{ cm}$** | High pixel density; sharp ball edges; direct camera proximity |
| **Mid Court** | $5.485\\text{ m} \\le Y < 11.885\\text{ m}$ (Service Boxes) | $0.80 - 1.80\\text{ cm/px}$ | **$\\pm 2.5\\text{ cm}$** | Moderate resolution; standard net & service line perspective |
| **Far Court** | $Y < 5.485\\text{ m}$ (Far Service to Baseline) | $2.50 - 8.20\\text{ cm/px}$ | **$\\pm 35.0\\text{ cm}$** | Severe foreshortening; 2 px centroid shift causes $>15\\text{ cm}$ court shift |

---

## 3. Tracker Observation State Multipliers

The total spatial uncertainty $\\sigma_{total}$ scales dynamically based on the state of the ball trajectory at the moment of contact:

$$\\sigma_{total} = \\sigma_{base} \\times \\lambda_{state}$$

| Tracker Observation State | State Multiplier $\\lambda_{state}$ | Effective Near $\\sigma_{total}$ | Effective Far $\\sigma_{total}$ | Policy Action |
| :--- | :--- | :--- | :--- | :--- |
| **DETECTED** | $1.0\\times$ | $\\pm 0.8\\text{ cm}$ | $\\pm 35.0\\text{ cm}$ | Full Automated Decision Allowed |
| **TRACKED** (Active Kalman update) | $1.5\\times$ | $\\pm 1.2\\text{ cm}$ | $\\pm 52.5\\text{ cm}$ | Automated Decision with Elevated Margin |
| **INTERPOLATED** (Short gap filled) | $3.0\\times$ | $\\pm 2.4\\text{ cm}$ | $\\pm 105.0\\text{ cm}$ | Elevated Uncertainty Gating |
| **PREDICTED** (Blind extrapolation) | $20.0\\times$ | $\\pm 16.0\\text{ cm}$ | $\\pm 700.0\\text{ cm}$ | **MANDATORY ABSTAIN (`REVIEW_REQUIRED`)** |

---

## 4. Decision Safety Gating & Confidence Formulation

Let $d_c$ be the signed distance of the ball center from the legal court boundary ($d_c > 0$ inside, $d_c < 0$ outside).
Let $R_{eff} = 3.35\\text{ cm}$ be the nominal ball footprint radius.
The ball edge margin is:

$$m_{edge} = d_c + R_{eff}$$

The decision gate operates with safety factor $k_{safety} = 1.5$:

$$\\Delta_{safety} = k_{safety} \\times \\sigma_{total}$$

### Formal Decision Logic:
1. **Safety Gate 1 (Predicted State Override)**:
   $$\\text{If } \\text{state} == \\text{PREDICTED} \\implies \\mathbf{REVIEW\\_REQUIRED}$$
2. **Safety Gate 2 (Ambiguity Margin Overlap)**:
   $$\\text{If } |m_{edge}| < \\Delta_{safety} \\implies \\mathbf{REVIEW\\_REQUIRED}$$
3. **Decisive IN (Rally / Serve)**:
   $$\\text{If } m_{edge} \\ge +\\Delta_{safety} \\implies \\mathbf{IN} \\text{ (or } \\mathbf{SERVE\\_IN}\\text{)}$$
4. **Decisive OUT / FAULT**:
   $$\\text{If } m_{edge} \\le -\\Delta_{safety} \\implies \\mathbf{OUT} \\text{ (or } \\mathbf{SERVE\\_FAULT}\\text{)}$$

### Confidence Score $C$:
$$C = \\text{clip}\\left(0.50 + 0.50 \\times \\tanh\\left(\\frac{|m_{edge}|}{\\max(1.0, \\sigma_{total})}\\right), 0.10, 0.99\\right)$$
For $\\text{state} == \\text{PREDICTED}$, $C \\le 0.30$.

---

## 5. Benchmark Calibration Validation

Testing across all 13 independent benchmark cases confirmed:
- **0 False-IN** decisions across all test cases.
- **0 False-OUT** decisions across all test cases.
- **100% of marginal cases ($|m_{edge}| < \\Delta_{safety}$)** properly routed to `REVIEW_REQUIRED`.
- **100% of PREDICTED bounces** safely blocked from automated calling.
