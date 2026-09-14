# Fixed court-calibration support comparison

Reject a global reduction of the calibration inlier ratio from 0.75 to 5/7. The complete frozen comparison recovers two useful Madrid initializations but also accepts a clearly misplaced court over the spectators in an amateur Wimbledon video. All three newly accepted fits were visually inspected. The runtime calibration remains unchanged.

The experiment replays 183 saved predictions: 125 train-partition images, 32 selection images, eight external-source images and eighteen camera/model controls. These are reused development cases, including repeated sources and models. Publisher labels are unverified, some have known errors, and the ResNet detector's training independence is unknown. This is not qualification evidence.

| Group | Cases | Accepted at 0.75 | Accepted at 5/7 |
| --- | ---: | ---: | ---: |
| Train partition | 125 | 123 | 125 |
| Selection | 32 | 31 | 31 |
| External source | 8 | 7 | 7 |
| Camera/model controls | 18 | 3 | 4 |
| Total | 183 | 164 | 167 |

The native RANSAC threshold remains three pixels and the minimum inlier count remains six. Each new fit has ten inliers among fourteen correspondences. Every previously accepted calibration, including its matrix and projected points, is exactly unchanged. The runner also reproduces saved default validity decisions and inlier counts for all cases.

Across the 165 labeled images, there are 2,304 eligible landmarks after the unchanged outside-image exclusions. These scores concern calibrated projection before line refinement:

| Ratio | TP / FP / FN | Precision | Recall | Available predictions |
| --- | --- | ---: | ---: | ---: |
| 0.75 | 1453 / 796 / 851 | 64.6065% | 63.0642% | 2249 |
| 5/7 | 1471 / 806 / 833 | 64.6025% | 63.8455% | 2277 |

Thus the relaxed calibration adds eighteen correctly located points and ten wrongly located points. It increases labeled coverage without improving precision. The unlabeled amateur false acceptance is separate from these counts; it must not disappear behind the labeled aggregate.

Only the three newly accepted fits undergo the frozen no-white proposal and boundary-preserving refinement. No thresholds are changed after seeing the results.

| Newly accepted case | Projected matches | Refined matches | Mean line support before / after | Refinement decision |
| --- | ---: | ---: | --- | --- |
| `7LLSADF9t48_650` | 8 / 14 | 14 / 14 | 0.5033 / 0.8756 | Accept correction |
| `7LLSADF9t48_2700` | 10 / 14 | 14 / 14 | 0.6789 / 0.9167 | Accept correction |
| UVY baseline `tennis_V02` | Unlabeled | Unlabeled | 0 / 0 | Insufficient compatible lines |

Both Madrid corrections visibly improve the near baseline and sidelines. All 28 labels agree within the unchanged inclusive seven-native-pixel tolerance after refinement; mean error is 2.3252 pixels and median error 2.1146 pixels. This is the result on two related images selected by the changed calibration decision, not a 100% accuracy estimate for the system.

The Wimbledon projection lies entirely over the spectators above the actual court. Despite that, its ten inlier landmarks have only 0.9127 pixels mean fit residual, similar to 0.8028 and 0.8892 pixels for the Madrid cases. Internally consistent predictions can still describe the wrong place. Refinement rejects the correction but returns the initialization unchanged, so rejection by the existing refinement does not invalidate this bad calibration.

A possible follow-up is to permit recovery of a rejected calibration only when image-supported refinement succeeds, withholding the alternate initialization otherwise. That rule is suggested by these observed outcomes and would be a new development hypothesis. This report does not silently apply it or claim independent validation of it. Existing valid-but-wrong calibrations also remain an unresolved issue.

Validation includes source/code/protocol hashes, unchanged default replay, independent scoring of every available label, verification of every saved projected matrix and the two accepted refinement transformations, and source-context visual inspection of all three new fits. Maximum independent projection difference is 4.55e-13 native pixels; maximum refinement-transform difference is 2.85e-14. No model inference, checkpoint loading, new training or runtime mutation occurs. The unchanged algorithms' earlier focused tests are historical evidence; this experiment's checks are the complete replay and independent review.

Protocol: `docs/experiments/COURT_CALIBRATION_SUPPORT_PROTOCOL.md`. Runner: `scripts/evaluate/benchmark_court_calibration_support.py`. Reviewer: `scripts/evaluate/review_court_calibration_support.py`. Report, separate review and all three visual panels: `outputs/vision_upgrade_audit/court_calibration_support01`.

Report SHA-256: `2b1e249d50bfa6cc133f2db2a6e89b8f86a8f6a671f29ff4d05e07ac168ffa9b`.
