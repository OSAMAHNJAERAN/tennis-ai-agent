# T88J709 Phase 4.1 — Failure Modes, Contact Limits & Safety Policy Analysis

## 1. Executive Summary

Automated tennis line calling from single broadcast cameras is constrained by camera resolution, discrete frame sampling, and ball deformation physics. This document formalizes the updated failure modes and explains the safety abstention policies enforced in Phase 4.1.

---

## 2. Analysis of Primary Failure Modes

### 2.1 Failure Mode 1: Close Calls in Low-Resolution Far Court
- **Physical Reality**: At the far baseline ($Y = 0.0\text{ m}$), ground resolution along the optical vertical axis is $6.90\text{ cm/px}$. A tiny sub-pixel localization error of 1 px in image space maps to nearly $7\text{ cm}$ on the court.
- **Safety Policy**: The spatial uncertainty for far-court bounces is set to $\pm 35.0\text{ cm}$. Any far-court bounce landing within $52.5\text{ cm}$ ($1.5 \times 35.0\text{ cm}$) of the far baseline or service line automatically triggers **`REVIEW_REQUIRED`**.

### 2.2 Failure Mode 2: Unobservable Ball Deformation on Monocular 30 FPS Video
- **Physical Reality**: High-speed ball deformation ($1.0 - 2.0\text{ cm}$) occurs over $4 - 7\text{ ms}$, which is completely invisible in standard $33.3\text{ ms}$ video frames.
- **Safety Policy**: If the ball center lands outside the painted line outer edge by less than the maximum possible dynamic deformation radius ($d_c \in [-2.5\text{ cm}, 0.0\text{ cm}]$) and cannot be verified beyond uncertainty, the system abstains via **`REVIEW_REQUIRED`**.

### 2.3 Failure Mode 3: Trajectory Gaps and Tracker Interpolation
- **Physical Reality**: When the ball is occluded by players or the net, tracker states degrade to `INTERPOLATED` or `PREDICTED`.
- **Safety Policy**:
  - `PREDICTED`: 100% Mandatory Abstention (`REVIEW_REQUIRED`, confidence capped at 0.25).
  - `INTERPOLATED`: Uncertainty multiplier $\lambda = 3.0\times$ ensures close calls are safely abstained.
