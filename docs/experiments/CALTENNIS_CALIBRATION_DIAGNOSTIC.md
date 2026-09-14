# CalTennis calibration and timing diagnostic

## Source and selection

The [publisher dataset card](https://huggingface.co/datasets/demalenk/caltennis) describes synchronized multi-view tennis practice and match recordings for label-free monocular-to-3D evaluation. It provides camera intrinsics/extrinsics and timestamp arrays under CC BY-NC 4.0. Research evaluation here does not establish permission for commercial incorporation. Multi-view consistency is not independently measured 3D pose or ball-speed ground truth.

Pinned revision: `6c1c10b2e6c16d46ab1a83614f439d2fb91e9d7d`. Saved documentation and file tree: `artifacts/validation/vision_upgrade/caltennis_source_audit/`. README SHA-256 is `540e2de40207558028b07d03a0c5026187ea6e29b82cc919703d24329dcdac22`; reviewed tree SHA-256 is `63be664549e351af452a2e954a55c6adf46597a0e54f7fc6e080306929e07da1`. The mini index has38 rows. The retrieved recursive tree contains409 entries and94 videos; no claim is made that this inventory is a verified complete publication inventory.

`scripts/data/acquire_caltennis_diagnostic.py` selects the three smallest listed video files before any model inference and acquires their corresponding calibration/time files within a fixed100MB limit. Selection favors compact diagnostics, not representative statistical sampling. Two clips belong to the same September18 session, and one to October2; all use court5. These are not three independent matches. Player skill level is not established from the card or appearance.

Acquisition completed at `data/external/caltennis_diagnostic/manifest.json`. Each media/time file matches its publisher LFS SHA-256, and each calibration matches its publisher Git blob hash. The manifest additionally records local SHA-256 for every saved file. Three videos total94,095,006 bytes; nine selected video/time/calibration files total94,177,261 bytes, excluding saved source documentation. No archive extraction or dataset-provided code execution occurs. Source metadata's workstation paths are provenance strings, never local file access instructions.

## Full decode and timing checks

All9,404 video frames decode. Initial audit: `outputs/vision_upgrade_audit/caltennis_initial_audit/report.json`. Every presentation timestamp was read with ffprobe; comparison: `timing.json` in the same directory. Source NPY arrays are loaded with `allow_pickle=False` and contain finite numeric time values, one per decoded frame.

| Clip / view filename | Decoded frames | Measured FPS | Video size | Calibration size |
| --- | --- | --- | --- | --- |
| September18 NW01 |2491 |59.95 |1920x1088 |1920x1080 |
| October2 N1 |1628 |29.99 |1920x1088 |1920x1080 |
| September18 NE01 |5285 |59.94 |1920x1088 |1920x1080 |

Each stream has strictly increasing presentation timestamps and constant frame spacing to ffprobe's printed precision. The publisher time arrays are also strictly increasing but have variable deltas. Interpreting their values as milliseconds gives maximum relative-time discrepancies from video PTS of9.569ms,8.084ms and2.103ms respectively. This is a checked unit hypothesis, not documentation of source units or a proof of synchronization between views. Do not replace measured video rates with the card's nominal60Hz or assume the two September clips share the same zero timestamp.

Calibration rotations are proper orthogonal matrices (determinant approximately1), camera/world rotations invert within3.4e-16, and translations satisfy `t_w2c = -R_w2c C` within7.2e-15. These algebraic checks establish internal consistency only. Publisher `reprojection_error` fields are3.709,5.984 and2.644; original annotation points, error definition and independent calibration accuracy are not established. The card does not document the court coordinate origin/axis convention.

## Image geometry and visual scope

First/middle/last previews were generated for each video. All three first frames were visually inspected. They show outdoor hard courts from near-ground cameras, with partial court visibility in the oblique views and strong sunlight/shadows in the October clip. This adds useful diagnostic conditions beyond the broadcast and Wimbledon fan footage already studied. It does not by itself establish amateur-match coverage.

Both OpenCV and ffprobe report1920x1088 video frames, whereas all three supplied calibration files specify1920x1080. First-frame bottom-row checks show nonzero image content in all eight extra rows, so simple black padding is not demonstrated. Resizing, cropping or edge extension during publication could have changed the relation to calibration pixels. No transform is silently chosen. This discrepancy must be resolved before using projected points as an independent court benchmark or interpreting world coordinates as validated speed/distance evidence.

No model accuracy, ball labels, player-role identities, 3D pose accuracy or speed error is claimed from these acquisitions. Next steps are to establish the publisher coordinate convention and video-to-calibration image transform, inspect projected court lines, and then compare court candidates on the verified representation. Production defaults and event/speed authority remain unchanged.


## Projection and checkpoint probes completed

`scripts/evaluate/audit_caltennis_projection.py` applies the explicit hypothesis that the publisher uses a doubles-corner origin, X along court length, Y along court width and Z upward. It projects with supplied K/R/t and independently cross-checks the matrix operation against OpenCV projectPoints within1e-8 pixels. The report preserves both unchanged calibration-pixel coordinates and a 1088/1080 vertical-resize hypothesis. All three first-frame projection panels were visually inspected and broadly follow the visible court; this supports further investigation but does not independently prove the exact coordinate convention or subpixel image transform. Artifact: `outputs/vision_upgrade_audit/caltennis_projection_hypotheses/report.json`.

Primary-source research found a plausible cause for the height change: the paper's cited deface anonymizer delegates video encoding to ImageIO, whose imageio-ffmpeg writer can round1080 upward to1088 by scaling to a16-pixel macroblock multiple. Exact publisher versions/arguments and original images remain unavailable. The court origin/axis labels and whether timestamp synchronization offsets were applied also remain undocumented. See `docs/research/CALTENNIS_CAMERA_CONVENTIONS.md` for pinned source lines, paper citations and limitations. No publication transform is silently assumed.

`scripts/evaluate/benchmark_caltennis_court_probe.py` completed both frozen ResNet court checkpoints on each clip's first/middle/last frame (nine native1920x1088 frames, eighteen predictions). Dataset/checkpoint/code hashes and all predictions are saved in `outputs/vision_upgrade_audit/caltennis_court_probe/report.json`. Existing geometric calibration accepts all three baseline predictions on clip1 and rejects the other six; the geoaug checkpoint fails all nine. Passing fit counts are internal consistency diagnostics, not court accuracy.

The paired first-frame previews for all three clips were visually inspected. Both checkpoints produce misplaced courts: clip1's passing baseline puts the far baseline above the net and imposes a much narrower end-on template; clip2 projects into sky/fences; clip3 similarly misplaces the template. The six middle/last previews are saved but have not yet received visual review. There is no legitimate accuracy score against the still-unverified supplied calibration coordinates. Neither model is promoted from this probe, and no fit threshold is relaxed. Next candidate work evaluates a heatmap architecture with spatial localization rather than another threshold on these regressed points.
