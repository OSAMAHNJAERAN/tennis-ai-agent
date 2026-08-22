# T88J709 Phase 4.1 — Real Video Line-Call Benchmark Results

## 1. Executive Summary & Verification

Phase 4.1 establishes the corrected **Assisted Tennis IN/OUT Line-Calling Engine** with finite court line widths, empirical contact patch modeling ($r_c = 1.25\text{ cm}$), piecewise trajectory change-point refinement, and server-relative Deuce/Ad orientation.

Results are strictly separated into:
- **Part A: Real Video Match Calls** (Evaluated from raw video ground truth);
- **Part B: Synthetic Geometry QA Suite** (Evaluated for mathematical consistency).

---

## 2. Part A: Real Video Match Line-Call Results (`outputs/phase4_1_line_calls/`)

Evaluated on `data/sample_videos/input_video.mp4` (214 frames @ 30 FPS):

| Bounce Frame | Context | Nearest Line | Old Phase 4 Margin ($R_{eff}=3.35\text{cm}$) | Corrected Margin ($r_c=1.25\text{cm}$) | Spatial Uncertainty $\sigma_{total}$ | Corrected Decision | Ground Truth Label | Call Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Frame 81** | `SERVE` | `NEAR_SERVICE_LINE` | $-196.6\text{ cm}$ | $-198.7\text{ cm}$ | $\pm 0.8\text{ cm}$ | **`SERVE_FAULT`** | `SERVE_FAULT` | **MATCH (Correct)** |
| **Frame 138** | `RALLY` | `FAR_BASELINE` | $-376.2\text{ cm}$ | $-378.3\text{ cm}$ | $\pm 35.0\text{ cm}$ | **`OUT`** | `OUT` | **MATCH (Correct)** |
| **Frame 178** | `RALLY` | `RIGHT_SIDELINE` | $+116.4\text{ cm}$ | $+114.3\text{ cm}$ | $\pm 0.8\text{ cm}$ | **`IN`** | `IN` | **MATCH (Correct)** |

### Real Video Performance Summary:
- **Total Real Bounces**: 3
- **Automated Decision Coverage**: **100.0% (3 / 3)**
- **Safe Abstention Rate**: **0.0% (0 / 3)**
- **Decision Accuracy on Real Calls**: **100.0% (3 / 3)**
- **False-IN Decisions**: **0 (0.0%)**
- **False-OUT Decisions**: **0 (0.0%)**

---

## 3. Real Video Distance-to-Line Buckets

| Distance-to-Line Bucket | Real Cases | Decision Outcomes | Accuracy |
| :--- | :--- | :--- | :--- |
| **$> 50.0\text{ cm}$ (Deep Clear In/Out)** | 3 | 1 `SERVE_FAULT`, 1 `OUT`, 1 `IN` | **100.0%** (3/3) |
| **$20.0 - 50.0\text{ cm}$** | 0 | *NO REAL VALIDATION DATA* | N/A |
| **$10.0 - 20.0\text{ cm}$** | 0 | *NO REAL VALIDATION DATA* | N/A |
| **$5.0 - 10.0\text{ cm}$** | 0 | *NO REAL VALIDATION DATA* | N/A |
| **$2.0 - 5.0\text{ cm}$** | 0 | *NO REAL VALIDATION DATA* | N/A |
| **$0.0 - 2.0\text{ cm}$ (Near Line Boundary)** | 0 | *NO REAL VALIDATION DATA* | N/A |

*Note: For close-call buckets ($0 - 20\text{ cm}$), performance is validated via controlled synthetic geometry edge cases and conservative safety abstention.*

---

## 4. Part B: Synthetic Geometry QA Suite

Evaluated on 10 deterministic boundary and uncertainty edge cases:
- **Total Synthetic Cases**: 10
- **Unit Tests Passed**: **10 / 10 (100.0%)**
- **Abstention on Ambiguous Marginal Contacts ($< 1.5\sigma$)**: **100.0%**
- **Abstention on PREDICTED Trajectory States**: **100.0%**
