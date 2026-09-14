# Court projective refinement results

The frozen local refinement corrects the visibly displaced court in all three sampled match143 frames while retaining a single projective court geometry. Six corrections are accepted across the two pipeline clips; all accepted before/after overlays were visually inspected. This is a measured image-evidence improvement, not independent metric court accuracy or camera-general detection.

The original protocol remains unchanged in `COURT_PROJECTIVE_LINE_REFINEMENT.md`, SHA-256 `a5581f021033b8eb4f9bb9131f88afcc601d448d94500a47c26448a541d6588c`. All15 refinement and existing calibration tests pass. Eight specifically test refinement on known synthetic geometry and unsupported inputs.

| Clip / native frame | Original line support | Refined line support | Maximum displacement at width960 |
| --- | ---: | ---: | ---: |
| match143 / 0 | 0.4178 | 0.9211 | 13.79 px |
| match143 / 125 | 0.4267 | 0.9378 | 13.57 px |
| match143 / 249 | 0.3911 | 0.9111 | 12.91 px |
| match148 / 0 | 0.8100 | 0.8878 | 4.49 px |
| match148 / 300 | 0.8189 | 0.8778 | 4.70 px |
| match148 / 599 | 0.7144 | 0.8878 | 5.13 px |

Line support is the mean fraction of visible samples close to selected bright-line segments, not a probability or an independently annotated accuracy score. The same segments supply fitting and scoring. Visual review confirms correction of the far baseline and left-side lines in match143, with smaller paint-alignment changes on match148.

All24 planned cases complete. Eighteen earlier controls produce no accepted correction: fifteen fail the existing native-pixel geometric initialization gate, and three lack six compatible image lines. The latter three are visibly misplaced UVY baseline/geoaug V01 and CalTennis baseline clip1 initializations; their unchanged before/after overlays were inspected. Critically, all six broadcast heatmap controls fail before refinement. This means their unchanged status does not establish that refinement preserves good heatmap estimates. Do not describe this as eighteen successful negative tests or broad court qualification.

Detailed evidence, source/code/video hashes, selected line correspondences, solver transforms and accepted/rejected outputs are in `outputs/vision_upgrade_audit/court_line_refinement01/report.json`. `review.json` records the independent integrity checks and nine inspected overlays. Fifteen rejected initializations have no new overlay because refinement was never attempted.

## Continuous first-frame correction replay

A second fixed experiment takes only each clip's accepted frame0 correction and propagates it through the original saved camera transforms. No subsequent frame is refitted or smoothed. All850 native frames are rendered into paired videos, with original geometry on the left and corrected geometry on the right. The videos fully decode at250 frames/25FPS and600 frames/60FPS. Every saved transform is independently checked against the composition of the original camera motion and corrected first-frame anchor.

Image support is sampled at frame0, every whole native second, and the final frame: eleven samples per clip. All22 improve. Match143 support changes from0.3911–0.4422 to0.8511–0.9422; match148 from0.6567–0.8356 to0.8011–0.9011. Minimum sample improvements are0.4211 and0.0511 respectively. First/middle/last rendered snapshots from both videos were visually inspected; no obvious drift is visible at those inspected points. This does not establish framewise physical accuracy or eliminate camera-registration failure modes.

Artifacts are in `outputs/vision_upgrade_audit/court_refined_anchor_replay01/`:

- `match143_000.mp4`: SHA-256 `4ac2fc36b77e52bbca0a6f54aca182ac9e0482c2fb03d9a16662feb35509a61f`.
- `match148_000.mp4`: SHA-256 `b0e1036c174075ce3a7a5ebf259b4063ed6b48e7bc98d6b94a055a27b923427a`.
- `report.json`: all850 matrices and projected points,22 support comparisons and source provenance.
- `review.json`: transform-composition verification, decoded-video and snapshot hashes, visual-inspection scope.

No runtime court decision or production default changed. The next requirement is broader well-initialized court controls and independently labeled court geometry before using these corrections for player distances, ground projections or bounce analysis. The local method cannot recover the grossly misplaced spectator and ground-level initializations shown here, and it cannot turn an airborne-ball ground-plane projection into a physical3D measurement.


A follow-up on all eight existing external-source publisher images completes with23 gained landmark matches and no losses relative to projected geometry, using unchanged refinement settings. All eight panels were inspected; known label errors remain. See `COURT_REFINEMENT_RAW_LABEL_RESULTS.md` for coverage-adjusted interpretation, raw counts, source-independence limits and18 passing targeted tests.
