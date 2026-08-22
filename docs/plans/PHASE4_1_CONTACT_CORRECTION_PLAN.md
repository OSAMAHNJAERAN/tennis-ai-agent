# T88J709 Phase 4.1 — Implementation Plan: Physics & Validation Correction

## 1. Goal & Rationale
Phase 4.1 systematically corrects the physical contact footprint, court line width geometry, impact change-point timing, and server-relative orientation in the Assisted Tennis Line-Calling Engine, establishing a defensible foundation for Phase 5 automated scoring.

---

## 2. Core Architectural Upgrades

### Upgrade 1: Explicit Finite Line Width & Outside-Edge Representation
- Represent each of the 7 court lines as a 2D geometric strip $[X_{min}, X_{max}] \times [Y_{min}, Y_{max}]$.
- Respect the ITF Rule 1 convention: measurements are to the **OUTSIDE** of lines.
- Store `line_outer_edge`, `line_center`, `line_inner_edge`, and `line_width_cm` with configurable/estimated line width uncertainty.

### Upgrade 2: Physics-Defensible Contact Patch Modeling
- Replace hardcoded $R_{eff} = 3.35\text{ cm}$ with parameterized `ContactPatchModel`:
  - `POINT_CONTACT`: Contact center ($r_c = 0.0\text{ cm}$) with spatial uncertainty gating.
  - `EMPIRICAL_PATCH`: Documented $r_c = 1.25\text{ cm}$ ($0.0125\text{ m}$) from tennis impact literature.
  - `DYNAMIC_IMPACT`: Velocity-aware contact patch $r_c(v_\perp) = r_0 + k \sqrt{v_\perp}$ ($0.8 - 2.5\text{ cm}$).
  - `ABSTENTION_GEOMETRIC`: Safe abstention if contact center is outside painted line outer edge but margin $< 2.5\text{ cm}$.

### Upgrade 3: Piecewise Impulsive Impact Change-Point Refinement
- Fit pre-impact trajectory ($t-3..t-1$) and post-impact trajectory ($t+1..t+3$) independently.
- Solve analytical line-line / arc-arc intersection to extract the impulsive velocity change-point $(t^*, x^*, y^*)$.
- Compare against single quadratic and raw discrete frame in experiments.

### Upgrade 4: Server-Relative Deuce/Ad Service Box Switching
- Map serving player identity, court end (Near vs Far), point index (even = Deuce, odd = Ad), and cross-court target service box.
- Audit Frame 81 serve sequence.

### Upgrade 5: Recalibrated Local Spatial Uncertainty & Benchmark v2
- Compute local ground resolution ($	ext{cm/px}$) along each line via homography Jacobian.
- Establish `data/benchmarks/line_calls_independent_v2/` strictly separating real video cases from synthetic geometry tests.
