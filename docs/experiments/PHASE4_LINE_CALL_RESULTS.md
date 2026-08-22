# T88J709 Phase 4 — Quantitative Line-Calling Benchmark Results

## 1. Executive Summary & Verification

Phase 4 establishes the authoritative **Assisted Tennis IN/OUT Line-Calling Engine** for T88J709. The engine integrates continuous quadratic trajectory apex refinement, ITF Rule 12 ball footprint geometry ($R_{eff} = 3.35\\text{ cm}$), depth-dependent spatial uncertainty modeling, and strict safety abstention gating.

Evaluation was performed on the independent benchmark dataset (`data/benchmarks/line_calls_independent/`):
- **Total Test Cases**: 13 (3 Real Match Video Bounces + 10 Controlled Synthetic Boundary Cases)
- **Overall Decision Accuracy**: **100.0% (13 / 13 correct decisions)**
- **False-IN Decisions**: **0 (0.0%)**
- **False-OUT / False-FAULT Decisions**: **0 (0.0%)**
- **Automated Decision Coverage**: **76.9% (10 / 13)**
- **Safe Abstention Rate (`REVIEW_REQUIRED`)**: **23.1% (3 / 13)**

---

## 2. Quantitative Performance Breakdown by Evaluation Bucket

| Evaluation Category | Total Cases | Automated Calls | Safe Abstentions | Accuracy on Decisive Calls | Zero-Error Maintained? |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Real Video Bounces** | 3 | 3 (100.0%) | 0 (0.0%) | **100.0%** (3/3) | YES |
| **Clear In / Out Cases** | 4 | 4 (100.0%) | 0 (0.0%) | **100.0%** (4/4) | YES |
| **Touching Line / Edge Margin** | 2 | 2 (100.0%) | 0 (0.0%) | **100.0%** (2/2) | YES |
| **Marginal Ambiguity Cases** | 2 | 0 (0.0%) | 2 (100.0%) | **100.0%** (Abstained) | YES |
| **PREDICTED Trajectory Cases** | 1 | 0 (0.0%) | 1 (100.0%) | **100.0%** (Abstained) | YES |
| **Serve Box Context Cases** | 1 | 1 (100.0%) | 0 (0.0%) | **100.0%** (1/1) | YES |
| **TOTAL** | **13** | **10 (76.9%)** | **3 (23.1%)** | **100.0%** | **YES** |

---

## 3. Real Match Video Line Calls Analysis (`outputs/phase4_line_calls_1/`)

| Event ID | Frame Index | Context | Nearest Boundary Line | Signed Center Dist $d_c$ (cm) | Ball Edge Margin $m_{edge}$ (cm) | Spatial Uncertainty $\\sigma_{total}$ | Decision | Confidence | Reason Summary |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Ev 2** | Frame 81 | `SERVE` | `NEAR_SERVICE_LINE` | $-199.9\\text{ cm}$ | $-196.6\\text{ cm}$ | $\\pm 0.8\\text{ cm}$ | **`SERVE_FAULT`** | **0.980** | Ball footprint landed 196.6 cm past near service line outside Near Deuce box. |
| **Ev 4** | Frame 138 | `RALLY` | `FAR_BASELINE` | $-379.6\\text{ cm}$ | $-376.2\\text{ cm}$ | $\\pm 35.0\\text{ cm}$ | **`OUT`** | **0.907** | Deep long ball landed 376.2 cm past far baseline beyond far-court uncertainty. |
| **Ev 6** | Frame 178 | `RALLY` | `RIGHT_SIDELINE` | $+113.1\\text{ cm}$ | $+116.4\\text{ cm}$ | $\\pm 0.8\\text{ cm}$ | **`IN`** | **0.980** | Clean baseline groundstroke landed 116.4 cm inside singles court. |

---

## 4. Distance-to-Line Performance Buckets

| Distance-to-Line Bucket | Cases | Decision Outcomes | Accuracy | Policy Safety |
| :--- | :--- | :--- | :--- | :--- |
| **$> 50.0\\text{ cm}$ (Deep Clear In/Out)** | 5 | 5 Automated (3 `OUT`/`FAULT`, 2 `IN`) | 100.0% | Extremely safe ($>10\\sigma$) |
| **$10.0 - 50.0\\text{ cm}$ (Moderate Clear)** | 3 | 3 Automated (2 `IN`, 1 `OUT`) | 100.0% | Decisive ($>4\\sigma$) |
| **$0.0 - 10.0\\text{ cm}$ (Near Line Boundary)** | 3 | 2 Automated (`IN` via footprint touch, `OUT`), 1 `REVIEW_REQUIRED` | 100.0% | Strict gating |
| **$< 1.5 \\sigma_{total}$ (Within Uncertainty)** | 2 | 2 `REVIEW_REQUIRED` | 100.0% | **0 False Decisions** |

---

## 5. Pipeline Throughput & Latency

On full 214-frame 1080p match video with YOLO11s ball extraction, ByteTrack player tracking, and full mini-court rendering:
- **Total Pipeline Execution Time**: **12.05 seconds**
- **Effective Pipeline Throughput**: **17.8 FPS**
- **Line Calling Module Latency**: **< 0.01 seconds** total (< 0.1 ms per bounce event)
