# T88J709 Phase 4.1 — Contact Patch Modeling & Provenance Experiments

## 1. Executive Summary & Physics Background

In Phase 4, the ball footprint margin was calculated as $m_{edge} = d_c + R_{eff}$ with $R_{eff} = 3.35\text{ cm}$. This experiment formalizes the physical contact patch models, documents their scientific provenance, and evaluates their impact on line-call accuracy.

---

## 2. Scientific Provenance of Tennis Ball Deformation

According to published experimental research in tennis impact dynamics (*Brody 1987; Cross 1999; Goodwill & Haake 2004; Miller 2006*):
1. **Static / Low-Speed Contact**: Tangent contact radius $r_c \approx 0.0 - 0.5\text{ cm}$.
2. **Standard Groundstroke Impact ($15 - 25\text{ m/s}$, $\theta = 20^\circ$)**:
   - Impact duration: $t_c \approx 5.0\text{ ms}$.
   - Vertical compression: $\delta_y \approx 0.5 - 1.2\text{ cm}$.
   - Geometric contact circle radius: $r_c = \sqrt{R_{ball}^2 - (R_{ball} - \delta_y)^2} \approx 1.25 - 1.80\text{ cm}$.
3. **High-Speed Serve Impact ($>45\text{ m/s}$)**:
   - Dynamic contact patch radius reaches up to $2.2 - 2.6\text{ cm}$.
4. **Physical Conclusion**: At no point during normal play does the contact patch equal the full ball radius ($3.35\text{ cm}$).

---

## 3. Comparative Evaluation of Contact Patch Models

| Model | Formulation | Provenance | Edge Margin on $d_c = -2.0\text{ cm}$ (Outside) | Decision ($\\sigma = 0.8\text{ cm}$) |
| :--- | :--- | :--- | :--- | :--- |
| **Model A: Point Contact** | $r_c = 0.0\text{ cm}$ | Pure center point approximation | $m_{edge} = -2.00\text{ cm}$ | `OUT` |
| **Model B: Nominal Empirical Patch** | $r_c = 1.25\text{ cm}$ | Measured rally groundstroke impact (Cross 1999) | $m_{edge} = -0.75\text{ cm}$ | `REVIEW_REQUIRED` (close call) |
| **Model C: Dynamic Velocity Model** | $r_c(v_\perp) = 0.8 + 0.15 \sqrt{v_\perp}$ | Velocity-dependent compression scaling | $m_{edge} = -0.65\text{ cm}$ | `REVIEW_REQUIRED` |
| **Model D: Geometric Abstention** | Point center, abstain if $d_c \in [-2.5, 0]$ | Zero physical assumption; safety priority | $d_c = -2.00\text{ cm}$ | `REVIEW_REQUIRED` |
| **Phase 4 Baseline (Flawed)** | $R_{eff} = 3.35\text{ cm}$ | Conflated radius with patch | $m_{edge} = +1.35\text{ cm}$ | **Confident `IN` (FALSE POSITIVE)** |

---

## 4. Engineering Selection & Fallback Policy

For Phase 4.1:
1. **Primary Model**: **Model B (Nominal Empirical Contact Patch $r_c = 1.25\text{ cm}$)** with explicit provenance documentation.
2. **Safe Fallback Policy**: Whenever $|m_{edge}| < 1.5 \sigma_{total}$, the engine must issue **`REVIEW_REQUIRED`**.
3. **Zero Assumption Override**: The engine does NOT automatically expand the ball footprint by $3.35\text{ cm}$ to force an `IN` call.
