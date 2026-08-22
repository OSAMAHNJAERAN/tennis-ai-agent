# T88J709 Phase 4.1 — Formal Contact Physics & Line-Call Architecture Audit

## 1. Executive Summary

This formal audit analyzes the physical and geometric assumptions in the Phase 4 Assisted Line-Calling Engine (commit SHA: `c2a8b61`). While Phase 4 successfully introduced spatial uncertainty modeling, PREDICTED-state abstention, and structured decision gating, an architectural review revealed two critical physical oversimplifications:
1. **Conflating Tennis Ball Radius with Ground-Contact Patch**: Phase 4 assumed $R_{eff} = 3.35\text{ cm}$ (half nominal ball diameter) and computed ball edge margin as $m_{edge} = d_c + R_{eff}$. This effectively modeled the ground footprint as a solid $6.70\text{ cm}$ disk contacting the court, which is physically false.
2. **Zero-Width Mathematical Boundaries**: Court lines were modeled as 1D zero-width boundaries, neglecting the finite painted line width ($2.5 - 10.0\text{ cm}$) and the ITF Rule 1 convention that court dimensions are measured to the **OUTSIDE** of lines.
3. **Single Quadratic Parabolic Turning Point**: Contact instant was estimated by solving $dy/dt = 0$ on a single 5-point quadratic fit across the bounce, assuming smooth zero-velocity inflection rather than an impulsive collision change-point.

---

## 2. Deep Dive: Ball Radius vs. Dynamic Contact Patch

### 2.1 The Phase 4 Assumption
In Phase 4:
```python
BALL_RADIUS_CM = 3.35
m_edge = d_c + BALL_RADIUS_CM
```
- $d_c$: Signed orthogonal distance from ball center to legal boundary line ($d_c > 0$ inside, $d_c < 0$ outside).
- $R_{eff} = 3.35\text{ cm}$: Nominal radius of an ITF Type 2 tennis ball (diameter $6.54 - 6.86\text{ cm}$, nominal $6.70\text{ cm}$).

### 2.2 Why Ball Radius $\ne$ Contact Patch
A spherical tennis ball approaching the court does not touch every point within $3.35\text{ cm}$ of its vertical center projection:
- **Static Tangent Contact**: At the initial moment of tangent contact, the contact area is a single mathematical point ($r_c \approx 0$).
- **Dynamic Impact Compression**: As established in published impact mechanics literature (*Cross 1999, "The bounce of a tennis ball", American Journal of Physics*; *Brody 1987, "Physics and Sports"*; *Goodwill & Haake 2004*):
  - The tennis ball is a hollow pressurized rubber sphere with felt cover.
  - During collision ($t_{contact} \approx 4 - 7\text{ ms}$), the bottom hemisphere deforms elastically against the court.
  - For typical groundstrokes ($v = 15 - 25\text{ m/s}$ at $\theta = 15^\circ - 30^\circ$), the maximum contact radius is $r_c \approx 1.0 - 1.8\text{ cm}$.
  - Even on maximum-velocity first serves ($v > 50\text{ m/s}$), maximum contact radius rarely exceeds $2.5 - 2.8\text{ cm}$.
- **Impact of the Phase 4 Error**: Adding $+3.35\text{ cm}$ to every ball center distance artificially expanded the legal IN-court boundary by $3.35\text{ cm}$ outward into the run-off area, misclassifying balls whose physical contact patch never touched the painted line.

---

## 3. Audit of Codebase Usages

The following files and variables were identified as encoding the full-radius assumption:

| File Path | Symbol / Usage | Physical Flaw | Correction Plan |
| :--- | :--- | :--- | :--- |
| `src/line_calling/line_geometry.py` | `BALL_RADIUS_CM = 3.35`, `BALL_RADIUS_M = 0.0335` | Uses full radius for footprint margin | Replace with explicit `ContactPatchModel` and finite line strip geometry |
| `src/line_calling/line_geometry.py` | `BoundaryEvaluationResult.effective_ball_radius_cm` | Assumes $3.35\text{ cm}$ contact footprint | Change to `contact_patch_radius_cm` with model provenance |
| `src/line_calling/line_call_engine.py` | `effective_ball_radius_cm = 3.35` | Hardcoded default in engine init | Parameterize by `ContactPatchModel` (Point, Empirical, Dynamic, Abstain) |
| `src/line_calling/contact_refinement.py` | Single quadratic $dy/dt = 0$ | Parabola vertex assumes smooth $v_y = 0$ | Replace with Piecewise Pre/Post Trajectory Change-Point Intersection |
| `configs/phase4_line_calls/pipeline.yaml` | `effective_ball_radius_cm: 3.35` | Configures full radius | Update to configurable contact patch mode and finite line widths |
| `tests/test_phase4_line_calls.py` | `test_ball_footprint_line_touching_rule` | Tests $-2.0 + 3.35 = +1.35\text{ cm} \implies \text{IN}$ | Update test to verify true contact patch and finite line width rules |

---

## 4. Finite Line Width & Outside-Edge Measurement Convention

### 4.1 ITF Rule 1 Measurement Standard
- Official ITF Rule 1 states: **"All court measurements shall be made to the outside of the lines."**
- This means the outer edge of the line is the bounding coordinate:
  - Near Baseline outer edge: $Y = 23.77\text{ m}$; painted line extends inward to $Y = 23.67\text{ m}$ ($10.0\text{ cm}$ baseline width).
  - Far Baseline outer edge: $Y = 0.00\text{ m}$; painted line extends inward to $Y = 0.10\text{ m}$.
  - Left Singles Sideline outer edge: $X = 1.37\text{ m}$; painted line extends inward to $X = 1.42\text{ m}$ ($5.0\text{ cm}$ sideline width).
  - Right Singles Sideline outer edge: $X = 9.60\text{ m}$; painted line extends inward to $X = 9.55\text{ m}$.
  - Near Service Line outer edge (away from net): $Y = 18.285\text{ m}$; painted line extends inward to $Y = 18.235\text{ m}$ ($5.0\text{ cm}$).
  - Far Service Line outer edge (away from net): $Y = 5.485\text{ m}$; painted line extends inward to $Y = 5.535\text{ m}$ ($5.0\text{ cm}$).
  - Center Service Line: $5.0\text{ cm}$ wide, centered at $X = 5.485\text{ m}$ ($[5.460, 5.510]\text{ m}$).

### 4.2 Legal Line Touch vs Legal Court Interior
- If a ball contact center lands on the painted line strip $[X_{inner}, X_{outer}]$ or $[Y_{inner}, Y_{outer}]$, it is **$100\%$ directly IN** by ITF Rule 12.
- If a ball contact center lands outside $X_{outer}$ or $Y_{outer}$, it is only IN if its dynamic contact patch $(x_c \pm r_c)$ physically touches $X_{outer}$ or $Y_{outer}$.

---

## 5. Reclassification of Phase 4 Benchmark Claims

The previous reported "100% accuracy across 13 cases" is formally reclassified as:
**Phase 4 Development & Geometry Prototype**.

- The 10 synthetic test cases served as mathematical unit tests for signed-distance calculations, but did not validate real physical tennis line calling.
- The 3 real video bounces (Frame 81, Frame 138, Frame 178) provide valuable ground truth, but must be evaluated under the corrected contact model and reported separately from synthetic tests.
