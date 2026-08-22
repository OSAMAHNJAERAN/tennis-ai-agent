# T88J709 — Tennis Shot Classification Feature Engineering

## 1. Mathematical Formulation & Normalization

To ensure invariant classification across near/far court ends and camera perspective distortions, all visual and kinematic features extracted around a hit event are normalized into **Player-Centric Normalized Space**.

Let the hit event occur at frame $t_{\text{hit}}$. We consider the temporal window $W = [t_{\text{hit}} - K, \dots, t_{\text{hit}} + K]$ (default $K=4$).

### 1.1 Court-Side Inversion Factor
Let $S \in \{+1, -1\}$ denote the court-side multiplier:
- **Near Court Player** (facing up/far end): $S = +1$
- **Far Court Player** (facing down/camera): $S = -1$

### 1.2 Handedness Inversion Factor
Let $H \in \{+1, -1\}$ denote player handedness:
- **Right-Handed Player**: $H = +1$ (Forehand on right, Backhand on left)
- **Left-Handed Player**: $H = -1$ (Forehand on left, Backhand on right)

---

## 2. Feature Definitions

### Feature 1: Lateral Ball-to-Player Offset ($\Delta X_{\text{rel}}$)
$$\Delta X_{\text{rel}} = S \cdot H \cdot \frac{X_{\text{ball}}(t_{\text{hit}}) - X_{\text{player\_center}}(t_{\text{hit}})}{W_{\text{player\_bbox}}}$$
- Positive values indicate the ball is on the player's dominant (forehand) side.
- Negative values indicate the non-dominant (backhand) side.

### Feature 2: Dominant Wrist Extension Angle ($\theta_{\text{wrist}}$)
Using YOLO11 keypoints for Right Shoulder ($K_6$), Left Shoulder ($K_5$), Right Wrist ($K_{10}$), Left Wrist ($K_9$):
- For a right-handed player ($H=+1$), we measure the vector from Neck/Torso Center to Dominant Wrist:
  $$\vec{v}_{\text{arm}} = \mathbf{P}_{\text{dominant\_wrist}} - \mathbf{P}_{\text{dominant\_shoulder}}$$
- Lateral arm displacement normalized by shoulder width $D_{\text{shoulder}} = \|\mathbf{P}_{R\_shoulder} - \mathbf{P}_{L\_shoulder}\|_2$:
  $$\Delta X_{\text{arm}} = S \cdot H \cdot \frac{X_{\text{dominant\_wrist}} - X_{\text{dominant\_shoulder}}}{D_{\text{shoulder}}}$$

### Feature 3: Shoulder Axis Rotation ($\phi_{\text{torso}}$)
$$\phi_{\text{torso}} = \arctan2(Y_{R\_shoulder} - Y_{L\_shoulder}, X_{R\_shoulder} - X_{L\_shoulder})$$
Measures rotational coil/uncoil during stroke execution.

### Feature 4: Trajectory Deflection Angle ($\Delta \theta_{\text{trajectory}}$)
Deflection between incoming trajectory vector $\vec{v}_{\text{in}} = \mathbf{P}_{\text{ball}}(t_{\text{hit}}) - \mathbf{P}_{\text{ball}}(t_{\text{hit}} - 4)$ and outgoing trajectory vector $\vec{v}_{\text{out}} = \mathbf{P}_{\text{ball}}(t_{\text{hit}} + 4) - \mathbf{P}_{\text{ball}}(t_{\text{hit}})$.

---

## 3. Decision Boundary & Safe Abstention Gating

```
If EventType == SERVE_CONTACT:
    Return (SERVE, Confidence = 0.95)

If Pose Keypoint Confidence < 0.35 OR Critical Joints Missing:
    Fallback to Geometry Offset Delta X_rel

If |Combined Feature Score| < Threshold_Ambiguity:
    Return (UNKNOWN, Confidence = 0.50)  # Safe Abstention!

If Combined Score > 0:
    Return (FOREHAND, Confidence = Sigmoid(Score))
Else:
    Return (BACKHAND, Confidence = Sigmoid(-Score))
```
