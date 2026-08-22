# Line Call Annotation Protocol v2

## 1. Frame Selection
1. Identify bounce frame $ via vertical trajectory reversal and court proximity.
2. Mark pre-impact (-3..t-1$) and post-impact (+1..t+3$) frames for piecewise change-point fitting.

## 2. Geometry & Contact Standards
1. Court boundaries are measured to the OUTSIDE of lines (ITF Rule 1).
2. Sidelines and service lines are .0\text{ cm}$ wide; baselines are .0\text{ cm}$ wide.
3. Nominal dynamic contact patch radius is  = 1.25\text{ cm}$ (Cross 1999).
4. If contact margin $|m_{edge}| < 1.5 \sigma_{total}$, decision must be \REVIEW_REQUIRED\.
