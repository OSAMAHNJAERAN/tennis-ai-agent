# T88J709 — Phase 6.3 Forensic Failure Analysis

## 1. Executive Summary

This document provides a detailed breakdown of residual edge-case failures identified across real-world tennis match video evaluations following the deployment of Phase 6.3 robustness upgrades.

---

## 2. Taxonomy of Residual Failures

### 1. Motion Blur at Extreme Terminal Velocity ($v > 120\text{ km/h}$)
- **Symptom**: Ball candidate confidence drops below 0.02 for 1-2 consecutive frames immediately following heavy first serves.
- **System Behavior**: Forward-backward Kalman filter bridges up to 3 frames with linear interpolation; if trajectory exceeds 4 frames of missing detection, the system safely marks `MISSING` rather than hallucinating false positions.
- **Safety Impact**: Zero false line calls; abstains gracefully according to Phase 4.1 rules.

### 2. High-Speed Serve Obscuration Behind Net Tape
- **Symptom**: When a flat serve passes precisely behind the net tape from high camera angles, contrast decreases against white net webbing.
- **System Behavior**: Kinematic state maintains velocity estimate and reacquires the ball candidate as soon as it emerges into the opponent's service box ($r_{\text{gate}} \le 65\text{ px}$).

### 3. Rapid Net Approach by Far Player (Player 2)
- **Symptom**: In drop shot and volley rallies, Player 2 approaches the net ($y \approx 420\text{ px}$), approaching the court midpoint.
- **System Behavior**: Relative vertical ranking ($y_{\text{P1}} > y_{\text{P2}}$) and track reacquisition maintain correct player assignment without identity inversion.

### 4. Wrist Crossover during Extreme Follow-Through
- **Symptom**: Two-handed backhand follow-through brings the dominant wrist across the player's torso after impact.
- **System Behavior**: Restricted contact evaluation window ($\text{contact} \pm 2\text{ frames}$) isolates the backswing and strike point, eliminating follow-through misclassification.

---

## 3. Production Deployment Guidelines

1. **Inference Resolution**: Always run ball candidate extraction at $1024\times 1024$ (`imgsz: 1024`).
2. **Confidence Bounds**: Maintain candidate pool threshold at $\text{low\_conf} \le 0.01$ and seed threshold at $\text{high\_conf} \le 0.08$.
3. **Interpolation Limits**: Keep `max_interpolation_gap` capped at $\le 3$ frames to prevent ghost ball trajectories.
