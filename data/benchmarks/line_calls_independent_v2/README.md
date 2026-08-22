# Independent Tennis Line Call Benchmark v2

## 1. Overview
This benchmark evaluates the T88J709 Assisted Tennis IN/OUT Line-Calling Engine under ITF 2026 court geometry, finite line widths, empirical contact patch deformation, and depth-dependent spatial uncertainty.

## 2. Strict Metric Separation Policy
- **Real Video Ground Truth**: Evaluates model performance on real broadcast video footage (eal_video_ground_truth.json\).
- **Synthetic Geometry QA Suite**: Evaluates geometric edge cases, signed distance boundary limits, and safety abstentions (\synthetic_geometry_cases.json\).
- **Rule**: Metrics from real video and synthetic unit tests MUST NEVER be aggregated into a single headline accuracy number.

## 3. Categories & Definitions
- \CLEAR_IN\: Ball lands cleanly inside the court boundaries ({edge} \ge +1.5\sigma$).
- \PAINTED_LINE_TOUCH\: Ball center lands directly on the finite painted line strip $[X_{inner}, X_{outer}]$ (ITF Rule 12 $\implies$ IN).
- \CLEAR_OUT\: Ball lands clearly outside outer legal boundary ({edge} \le -1.5\sigma$).
- \AMBIGUOUS_CLOSE_CALL\: Ball contact margin is within the spatial uncertainty envelope ($|m_{edge}| < 1.5\sigma \implies$ \REVIEW_REQUIRED\).
- \PREDICTED_STATE_ABSTAIN\: Ball state was predicted without direct optical detection $\implies$ mandatory \REVIEW_REQUIRED\.
