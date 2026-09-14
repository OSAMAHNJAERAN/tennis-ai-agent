# Ball training expansion: sixty clips verified

Completed 2026-09-14. Added exactly the twenty predeclared publisher-training entries 40:60 at RacketVision revision `85157ca21faa2abca96d837dd2b963738029bcc8`. The new acquisition contains 1,000 explicit labels in 6,751 fully decoded frames, totaling 198.75 seconds of footage and 94,495,649 downloaded bytes including split metadata/CSVs. No model inference, training, checkpoint change or test-set acquisition occurred.

The expanded corpus now contains sixty publisher clip IDs and 2,999 labels. All forty existing clip assignments remain unchanged under the every-fifth-distinct-match selection rule. There are no duplicate clip IDs or identical video hashes across the sixty files, and no publisher validation match-ID overlap. This does not rule out shared broadcasts, repeated players or near-duplicate footage.

| Partition | Clips | Explicit labels | Visible | Absent |
| --- | ---: | ---: | ---: | ---: |
| Training | 48 | 2,399 | 2,220 | 179 |
| Internal selection | 12 | 600 | 569 | 31 |
| Combined | 60 | 2,999 | 2,789 | 210 |
| New twenty only, across both partitions | 20 | 1,000 | 945 | 55 |

All new videos are 1920x1080: fourteen at 25 FPS, five at 60 FPS and one at approximately 29.97 FPS. Every one of the 2,999 labels has at least one real three-frame context under the deployed spacing rule `max(1, floor(fps/30 + .5))`. This establishes availability for future training preparation; the older trainer still uses adjacent-frame context, and has not been silently modified or rerun.

## Provenance and validation

The existing downloader enforces the fixed revision, membership in publisher training, a 200 MB video budget and publisher SHA-256 values for every video. It checks and saves the pinned source card, which declares MIT. The audit rechecks all files against the three local manifests and compares their original publisher split lists. No validation or test labels become training samples.

The new audit validates each explicit frame index, visibility state and visible coordinate at native dimensions. It rejects duplicates and malformed labels. Absent target coordinates remain `None`, and unannotated frames remain unannotated. A second implementation independently rechecks every CSV and reproduces the partition counts, source hashes, context availability and fixed visual selection. Full decode of the twenty new clips is checked against reported frame counts and dimensions; the original forty are hash/label checked without repeating their earlier full processing.

Eight new tests pass in 0.12 seconds in the available standard Python environment. They cover sparse target semantics and coordinate scaling, invalid/nonfinite/out-of-range labels, publisher leakage and duplicate clips, grouped rally assignments, preservation of existing selection roles and deterministic visual sampling. These tests do not measure model accuracy.

## Visual inspection

All twenty review boards were inspected, containing thirty-five preselected examples: the first visible label in every clip and the first absent label in each of fifteen clips. The selection is fixed by CSV frame order before viewing, independent of model errors. Source frames and adjacent unmarked crops are preserved. Visual observations are in the separate `review.json`; the original generated `report.json` remains unchanged, including its pre-review state flag.

Nineteen visible examples show a plausible moving ball or motion streak near the target. `match209_000:9` lies in far-player hand/racket contact clutter and cannot be isolated confidently in the displayed crops. The absence examples show full-frame temporal context, but their reduced display scale cannot conclusively certify the absence of a tiny/occluded ball. No label was rewritten, dropped, or converted into inferred ground truth. A fixed sample cannot certify all 1,000 new labels, and it does not replace the earlier training-negative quality findings.

Assistant scene tags identify five clay clips, one grass clip, four indoor-hard clips, and ten other hard-court clips. Two low-baseline examples are present (`match209_000`, `match217_000`); most views are elevated broadcast cameras. `match213_000` and `match22_000` contain doubles. These are useful ball-only contexts, not evidence for supported doubles player/role tracking. These visual categories are approximate review tags, not independently annotated dataset metadata or proof of amateur-camera coverage.

## Training implication

The data expansion adds actual tennis examples and preserves a twelve-match internal selection partition. It enables a larger bounded transfer-learning experiment without training on the 900 repeatedly examined development labels. The next training protocol should explicitly choose the treatment of ambiguous absent/contact supervision, use detection metrics rather than focal loss alone for model selection, and account for the existing training/deployment temporal-spacing mismatch at 60 FPS. Changes to temporal sampling, data quantity and loss must be distinguished in the experiment record; their combined effect must not be called a single-variable ablation.

No new model is claimed improved from dataset acquisition alone. The original threshold and all runtime defaults remain unchanged. Independent ball/player/court/event and physical-speed qualification remains incomplete, so production readiness remains **NO**.

## Artifacts

- New data: `data/external/racketvision_training_third20/` (ignored video binaries, pinned source card and manifest).
- Corpus registry: `data/data_registry.yaml`, entry `racketvision_ball_training60`.
- Audit and twenty boards: `outputs/vision_upgrade_audit/ball_training_expansion60/`.
- Independent review: `review.json` in that audit directory, with 35 explicit inspection notes and twenty scene tags.
- Protocol: `docs/experiments/BALL_TRAINING_EXPANSION60_PROTOCOL.md`.
- Tools: `scripts/data/audit_ball_training_expansion.py` and `scripts/data/review_ball_training_expansion.py`.

New dataset manifest SHA-256: `8394fc2d5d97ddd139035377f12b022e7159fe44f5fe5c8113520baae12a6d6a`.

Preserved audit SHA-256: `cad5b629feed6397c7a5d2e3258f688950f32c79968c8fc7838347729091b814`.

Reproduction uses the established Python 3.13 environment for OpenCV decoding. The downloader/audit/reviewer protect existing directories or completed outputs; do not overwrite them to rerun the same preparation. The first metadata-only preparation tests also run without GPU libraries.
