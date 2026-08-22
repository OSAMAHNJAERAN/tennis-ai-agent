# T88J709 — Shot Direction & Court Zoning Geometric Definitions

## 1. Canonical Coordinate Reference Frame

- **Court Origin $(0, 0)$**: Far-left doubles sideline corner.
- **Dimensions**: Width $W = 10.97\text{ m}$ (Singles $8.23\text{ m}$ with $1.37\text{ m}$ alleys), Length $L = 23.77\text{ m}$.
- **Centerline**: $X_{\text{center}} = 5.485\text{ m}$.
- **Net Line**: $Y_{\text{net}} = 11.885\text{ m}$.
- **Court Ends**:
  - **Far End (Player 2)**: $0 \le Y < 11.885\text{ m}$, Court side sign $S = -1$.
  - **Near End (Player 1)**: $11.885 < Y \le 23.77\text{ m}$, Court side sign $S = +1$.

---

## 2. Shot Direction Vector Mathematics

Given stroke initiation $(X_0, Y_0)$ and landing contact $(X_1, Y_1)$:
$$\Delta X = X_1 - X_0, \quad \Delta Y = Y_1 - Y_0$$
$$\theta = \arctan2(\Delta X, |\Delta Y|)$$

### 2.1 Direction Taxonomy
1. **`CROSS_COURT`**:
   - The ball travels across the court centerline from one lateral flank to the opposite flank:
     $$\text{sign}(X_0 - X_{\text{center}}) \cdot \text{sign}(X_1 - X_{\text{center}}) < 0 \quad \text{and} \quad |\Delta X| \ge 1.50\text{ m}$$
   - Or when starting near the center ($|X_0 - X_{\text{center}}| < 1.0\text{ m}$), the trajectory deflects laterally into the deep outer corner ($|\theta| \ge 15.0^\circ$).

2. **`DOWN_THE_LINE`**:
   - The ball travels along the same lateral side of the court:
     $$\text{sign}(X_0 - X_{\text{center}}) \cdot \text{sign}(X_1 - X_{\text{center}}) > 0 \quad \text{and} \quad |\Delta X| < 1.50\text{ m}$$
   - The trajectory remains parallel to the sidelines ($|\theta| < 12.0^\circ$).

3. **`MIDDLE`**:
   - The ball lands within the central corridor of the court ($3.65\text{ m} \le X_1 \le 7.32\text{ m}$) with low lateral angular deflection ($|\theta| \le 8.0^\circ$).

4. **`UNKNOWN`**:
   - Either $(X_0, Y_0)$ or $(X_1, Y_1)$ is unobserved, or the trajectory does not cross the net.

---

## 3. Canonical 3x3 Court Zoning & Service Box Geometry

### 3.1 3x3 Grid Definitions (Player-Relative to Receiving End)
- **Lateral Zones** ($X$ coordinate in meters):
  - `LEFT`: $X < 3.65\text{ m}$
  - `CENTER`: $3.65\text{ m} \le X \le 7.32\text{ m}$
  - `RIGHT`: $X > 7.32\text{ m}$
- **Depth Zones** (Distance from Net $d_{\text{net}} = |Y - Y_{\text{net}}|$):
  - `SHORT`: $d_{\text{net}} < 4.50\text{ m}$ (Inside service box / near net)
  - `MID`: $4.50\text{ m} \le d_{\text{net}} \le 9.00\text{ m}$ (Mid-court / service line region)
  - `DEEP`: $d_{\text{net}} > 9.00\text{ m}$ (Baseline region)

### 3.2 Service Placement Categories (`WIDE`, `BODY`, `T`)
For a legal serve landing inside the designated target service box ($6.40\text{ m}$ depth from net):
- **`T` (Center Service Line Target)**:
  - Lateral distance to center service line $d_{\text{center}} \le 0.85\text{ m}$.
- **`WIDE` (Sideline Target)**:
  - Lateral distance to singles sideline $d_{\text{sideline}} \le 0.85\text{ m}$ (or $d_{\text{center}} \ge 2.50\text{ m}$).
- **`BODY` (Central Receiver Target)**:
  - Intermediate landing location ($0.85\text{ m} < d_{\text{center}} < 2.50\text{ m}$).
