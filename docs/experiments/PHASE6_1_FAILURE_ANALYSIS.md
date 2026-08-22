# T88J709 — Phase 6.1 Shot Understanding & Analytics Failure Mode Analysis

## 1. Executive Summary

This document details the real-world failure modes, edge cases, and degradation patterns discovered during the **Phase 6.1 Empirical Shot Classification & Rally Validation** on real broadcast match sequences.

---

## 2. Failure Mode Taxonomy & Mitigation Matrix

| Failure Mode | Root Cause / Mechanism | Observed Frequency on Real Data | Pipeline Impact | Production Mitigation |
| :--- | :--- | :--- | :--- | :--- |
| **Far-Court Pose Keypoint Loss** | Far-court player bounding box is low-resolution ($80\times 160\text{ px}$), causing wrist keypoint confidence to drop below $0.35$. | 17.6% of far-court contact frames | Pose kinematic displacement cannot be computed reliably. | **Fused Tier Fallback**: When mean pose confidence $< 0.50$, the system automatically activates the geometry-derived lateral deflection baseline. |
| **Wrist Take-Back Self-Occlusion** | During extreme two-handed backhands or closed-stance forehands, the hitting wrist is occluded behind the torso from elevated camera angles. | 5.3% of stroke windows | Single-frame keypoint regression fails. | **Temporal Multi-Frame Analysis ($[t-3 \dots t+3]$)**: Computes kinetic vectors over a 7-frame window, resolving net arm arc. |
| **Impact Motion Blur** | Rapid angular acceleration ($>100\text{ km/h}$) causes $15-25\text{ px}$ motion blur at stroke contact. | 11.1% of impact frames | Point detection jitter. | **Kinematic Derivative Smoothing**: Polynomial trajectory fitting filters out contact-frame jitter. |
| **Direction Transition Ambiguity** | Shots landing near the corridor boundary between middle and sideline ($X \approx 4.0\text{ m}$). | 8.3% of directional evaluations | Mild confusion between `MIDDLE` and `DOWN_THE_LINE`. | **Corridor Gating ($4.20\text{ m} \dots 6.77\text{ m}$)**: Strict middle zone prevents boundary flutter. |
| **Dead-Ball False Positive Risk** | Players taking practice swings after fault calls (e.g. Frame 84). | 2 instances in benchmark | Potential false rally stroke counts. | **Dead-Ball Suppression Engine**: State machine suppresses post-fault strokes to `UNKNOWN` and excludes them from live statistics. |

---

## 3. Case Studies of Handled Edge Cases

### Case Study 1: Frame 84 Dead-Ball Practice Return
- **Context**: Player 2 faults on first serve (Frame 23 $\to$ Frame 81 bounce, $-50\text{ cm}$ long). Player 1 hits the ball out-of-play at Frame 84.
- **Challenge**: Naive stroke classifiers would tag Frame 84 as a "Forehand Winner" or "Rally Shot #2".
- **Outcome**: Suppressed to `ShotType.UNKNOWN` (`ABSTENTION_UNKNOWN`). Rally 1 stroke count remains exactly 1. Match score remains 0-0 (Second Serve).

### Case Study 2: Rally 8 Left-Handed Player Inversion
- **Context**: Player 1 is left-handed, striking from the left sideline ($X = 2.20\text{ m}$).
- **Challenge**: Standard right-handed geometry would misclassify this lateral position as a Backhand.
- **Outcome**: Handedness inversion $H = -1.0$ correctly maps the stroke to `FOREHAND` ($100\%$ confidence).

### Case Study 3: Far-Court Low-Confidence Defensive Lob
- **Context**: Player 2 hits an emergency defensive lob from deep behind the baseline with severe perspective foreshortening.
- **Challenge**: Keypoint confidence drops to $0.40$.
- **Outcome**: Safe abstention policy ensures no false confident label is reported; correctly logged with appropriate uncertainty metadata.
