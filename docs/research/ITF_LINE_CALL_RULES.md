# ITF Rules of Tennis: Line Calling, Court Dimensions & Ball Geometry Reference

## 1. Authoritative Citations & Overview
This reference compiles the official rules and physical specifications from the **International Tennis Federation (ITF) Rules of Tennis (2024)** governing court boundaries, ball dimensions, line-touching rules, and service box definitions.

**Primary Reference:**
- *ITF Rules of Tennis* (ITF Ltd., Approved by General Assembly)
- Rule 1: The Court
- Rule 3: The Ball (Appendix I: Regulations for Rule 3)
- Rule 12: Ball Touches a Line
- Rule 17: The Service

---

## 2. Court Dimensions & Boundary Specifications (ITF Rule 1)

```
(0,0) [Doubles TL]                                                  (10.97,0) [Doubles TR]
  +-----------------------------------------------------------------------+
  | Doubles Alley (1.37m)                                                 |
  +-------+-------------------------------------------------------+-------+
  | (1.37,0) [Singles TL]                                 (9.60,0) [Singles TR]
  |       |                     Backcourt (5.485m)                |       |
  |       +---------------------------+---------------------------+       |
  |       | (1.37, 5.485)             | (5.485, 5.485)            | (9.60, 5.485)
  |       | Left Service Box          | Right Service Box         |       |
  |       | (6.40m x 4.115m)          | (6.40m x 4.115m)          |       |
  |       |                           |                           |       |
  +-------+===========================#===========================+-------+ Net (Y = 11.885m)
  |       |                           |                           |       |
  |       | Left Service Box          | Right Service Box         |       |
  |       | (6.40m x 4.115m)          | (6.40m x 4.115m)          |       |
  |       +---------------------------+---------------------------+       |
  |       | (1.37, 18.285)            | (5.485, 18.285)           | (9.60, 18.285)
  |       |                     Backcourt (5.485m)                |       |
  +-------+-------------------------------------------------------+-------+
  | (1.37, 23.77) [Singles BL]                             (9.60, 23.77) [Singles BR]
  | Doubles Alley (1.37m)                                                 |
  +-----------------------------------------------------------------------+
(0, 23.77) [Doubles BL]                                             (10.97, 23.77) [Doubles BR]
```

### Standard Dimensions (Metric):
- **Court Length:** $23.77\text{ m}$ ($78\text{ ft}$)
- **Singles Court Width:** $8.23\text{ m}$ ($27\text{ ft}$)
- **Doubles Court Width:** $10.97\text{ m}$ ($36\text{ ft}$)
- **Doubles Alley Width:** $1.37\text{ m}$ ($4.5\text{ ft}$)
- **Net Position:** $Y = 11.885\text{ m}$ (center of court length)
- **Net Height:** $0.914\text{ m}$ ($3.0\text{ ft}$) at center strap; $1.067\text{ m}$ ($3.5\text{ ft}$) at net posts
- **Service Line Distance from Net:** $6.40\text{ m}$ ($21\text{ ft}$) on each side
  - Top Service Line: $Y = 11.885 - 6.40 = 5.485\text{ m}$
  - Bottom Service Line: $Y = 11.885 + 6.40 = 18.285\text{ m}$
- **Service Box Width:** $4.115\text{ m}$ ($13.5\text{ ft}$) per half
- **Center Service Line:** $X = 5.485\text{ m}$ (divides service boxes)

### Line Widths & Inclusions:
- **Center Service Line & Center Mark:** $5\text{ cm}$ ($2\text{ in}$) wide
- **Service Lines & Sidelines:** Between $2.5\text{ cm}$ and $5\text{ cm}$ ($1\text{ to }2\text{ in}$) wide
- **Baselines:** Up to $10\text{ cm}$ ($4\text{ in}$) wide
- **Rule of Inclusions:** All court measurements are taken to the **outside edges** of the boundary lines. Therefore, the physical lines themselves are legally part of the in-court area bounded by those lines.

---

## 3. Ball Dimensions & Physical Characteristics (ITF Rule 3 / Appendix I)
- **Type 2 (Standard Medium Speed Ball - Most common in professional tournaments):**
  - **Diameter:** $6.54\text{ cm}$ to $6.86\text{ cm}$ ($2.57\text{ to }2.70\text{ in}$)
  - **Nominal Diameter:** $\mathbf{6.70\text{ cm}}$ ($0.067\text{ m}$)
  - **Nominal Ball Radius ($R_{ball}$):** $\mathbf{3.35\text{ cm}}$ ($0.0335\text{ m}$)
  - **Mass:** $56.0\text{ g}$ to $59.4\text{ g}$
  - **Rebound Height:** $135\text{ cm}$ to $147\text{ cm}$ when dropped from $254\text{ cm}$ onto concrete.

---

## 4. Fundamental Line-Calling Rules

### 4.1 ITF Rule 12: Ball Touches a Line
> *"If a ball touches a line, it is regarded as touching the court bounded by that line."*

**Operational Criterion:**
If any portion of the tennis ball's physical footprint or ground contact patch intersects or overlaps any part of the boundary line (or its outer edge), the shot is legally **IN**.

### 4.2 Singles Rally Boundary Polygon
During standard singles rally play (all shots after the serve return):
- **Left Boundary:** Singles Left Sideline ($X = 1.37\text{ m}$)
- **Right Boundary:** Singles Right Sideline ($X = 9.60\text{ m}$)
- **Top Boundary:** Far Baseline ($Y = 0.00\text{ m}$)
- **Bottom Boundary:** Near Baseline ($Y = 23.77\text{ m}$)

### 4.3 Service Box Legal Polygons (ITF Rule 17)
A legal serve must cross the net and land within the diagonally opposite service court:
- **Server Serving from Far-Right (Top Deuce Court):**
  - Target: Near Deuce Court (Bottom-Left from camera view)
  - Boundaries: $X \in [1.37, 5.485\text{ m}]$, $Y \in [11.885, 18.285\text{ m}]$
- **Server Serving from Far-Left (Top Ad Court):**
  - Target: Near Ad Court (Bottom-Right from camera view)
  - Boundaries: $X \in [5.485, 9.60\text{ m}]$, $Y \in [11.885, 18.285\text{ m}]$
- **Server Serving from Near-Right (Bottom Deuce Court):**
  - Target: Far Deuce Court (Top-Left from camera view)
  - Boundaries: $X \in [1.37, 5.485\text{ m}]$, $Y \in [5.485, 11.885\text{ m}]$
- **Server Serving from Near-Left (Bottom Ad Court):**
  - Target: Far Ad Court (Top-Right from camera view)
  - Boundaries: $X \in [5.485, 9.60\text{ m}]$, $Y \in [5.485, 11.885\text{ m}]$

---

## 5. Contact Footprint Approximation vs Point Geometry
A dimensionless point test ($P_{center} \in \text{Polygon}$) fails to account for real-world tennis physics:
1. **Center Signed Distance ($d_c$):** Perpendicular distance from ball center to legal boundary line ($+d_c$ for inside, $-d_c$ for outside).
2. **Effective Ball Radius ($R_{eff}$):** Nominal ball radius $R_{ball} = 3.35\text{ cm}$ (or effective impact contact radius).
3. **Ball Edge Margin ($m_{edge}$):**
   $$m_{edge} = d_c + R_{eff}$$
4. **Physical Criterion:**
   - If $m_{edge} \ge 0$: Ball touches or is inside the line $\implies$ **IN**
   - If $m_{edge} < 0$: Entire ball footprint is outside the line $\implies$ **OUT**
