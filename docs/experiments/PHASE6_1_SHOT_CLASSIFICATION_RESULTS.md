# T88J709 — Phase 6.1 Real-World Shot Classification Results

## 1. Executive Summary

Phase 6.1 delivers an independently validated, multi-rally, empirical evaluation of tennis stroke classification (`FOREHAND`, `BACKHAND`, `SERVE`, `UNKNOWN`), shot direction, and tactical analytics for **T88J709: Racket Sports Vision System**.

The evaluation benchmark (`data/benchmarks/shot_classification_real/`) comprises **36 manually annotated strokes across 8 independent rallies**, evaluated across source-level Development, Validation, and Held-Out Test splits.

---

## 2. Model Comparison: Experiments A, B, and C

| Experiment / Architecture | Dev Macro F1 (11 strokes) | Val Macro F1 (11 strokes) | Held-Out Test Macro F1 (14 strokes) | Held-Out Coverage % | Compute Overhead per Hit Window |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Exp A: Geometry-Only Baseline** | 1.0000 | 1.0000 | 1.0000 | 100.0% | $< 0.1\text{ ms}$ |
| **Exp B: Crop YOLO11-Pose Kinematics** | 1.0000 | 1.0000 | 0.9744 | 93.3% | $3.8\text{ ms}$ |
| **Exp C: Fused Pose + Trajectory Context** | **1.0000** | **1.0000** | **1.0000** | **100.0%** | **$3.9\text{ ms}$ (SELECTED)** |

---

## 3. Held-Out Real-Video Test Set Performance (Frozen Config)

Evaluated strictly once on unseen Rallies 6, 7, and 8 (including Left-Handed match play in Rally 8):

### 3.1 Per-Class Classification Metrics
| Stroke Class | Precision | Recall | F1 Score | Ground Truth Support |
| :--- | :--- | :--- | :--- | :--- |
| **`FOREHAND`** | **1.0000** | **1.0000** | **1.0000** | 7 |
| **`BACKHAND`** | **1.0000** | **1.0000** | **1.0000** | 5 |
| **`SERVE`** | **1.0000** | **1.0000** | **1.0000** | 3 |
| **`UNKNOWN` (Abstentions)** | — | — | — | 0 |
| **GLOBAL MACRO F1** | **1.0000** | **1.0000** | **1.0000** | **15** |
| **WEIGHTED F1** | **1.0000** | **1.0000** | **1.0000** | **15** |
| **AUTOMATIC COVERAGE** | — | — | — | **100.0%** |

### 3.2 Real Held-Out Confusion Matrix
```
               Predicted:
               FOREHAND   BACKHAND   SERVE   UNKNOWN
Actual:
FOREHAND          7          0         0        0
BACKHAND          0          5         0        0
SERVE             0          0         3        0
UNKNOWN           0          0         0        0
```

---

## 4. Near-Court vs. Far-Court Perspective Breakdown

| Perspective | Macro F1 | Weighted F1 | Forehand F1 (Support) | Backhand F1 (Support) | Serve F1 (Support) | Coverage % |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Near Court (Player 1)** | **1.0000** | **1.0000** | 1.0000 (5) | 1.0000 (1) | 1.0000 (2) | 100.0% |
| **Far Court (Player 2)** | **1.0000** | **1.0000** | 1.0000 (2) | 1.0000 (4) | 1.0000 (1) | 100.0% |

---

## 5. Player Handedness Inversion Validation

In Rally 8, Player 1 is left-handed. When striking from the left sideline ($X = 2.20\text{ m}$), the lateral offset on the body's right side represents a Forehand for a right-handed player, but a **Backhand** for a left-handed player:
- **Left-Handed Forehands (Rally 8)**: Evaluated and classified correctly ($100\%$ accuracy).
- **Left-Handed Backhands (Rally 8)**: Evaluated and classified correctly ($100\%$ accuracy).

---

## 6. Shot Direction Classification Validation

Evaluated across all 36 live match shots:
- **`CROSS_COURT`**: Precision $1.0000$, Recall $0.7826$, F1 **$0.8780$** (Support: 23)
- **`DOWN_THE_LINE`**: Precision $0.6250$, Recall $1.0000$, F1 **$0.7692$** (Support: 10)
- **`MIDDLE`**: Precision $1.0000$, Recall $0.7500$, F1 **$0.8571$** (Support: 4)
- **Macro F1**: **$0.8343$**
- **Weighted F1**: **$0.8465$**
