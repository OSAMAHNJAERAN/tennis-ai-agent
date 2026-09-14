# WASB head-only adaptation results

Completed 2026-09-14. The frozen one-epoch candidate passes the internal gate by only **0.023724 percentage points of F1** on 600 reserved publisher-training labels. This authorizes the predeclared external development comparison, not a runtime replacement or a reliability claim.

| Model and selected threshold | TP | FP | FN | TN | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original, 0.35 | 527 | 34 | 42 | 18 | 93.9394% | 92.6186% | 93.2743% |
| Head epoch 01, 0.35 | 529 | 36 | 40 | 16 | 93.6283% | 92.9701% | 93.2981% |

Both thresholds were independently selected from the same predeclared five values using pooled F1 on twelve reserved training clips (569 visible labels, 31 absent). Precision falls 0.3111 percentage points, recall rises 0.3515. Absence false detections rise from 13 to 15. Wrong visible localizations remain 21, and visible abstentions fall from 21 to 19. This is internal model-selection evidence, not an estimate on independent footage.

At the selected thresholds, two clips improve and two regress. `match199_000` gains one correctly located ball (45 to 46); `match225_000` gains one (34 to 35) by correcting a wrong location. `match176_000` adds two false detections: one on an absent label and one replacing a visible abstention. `match207_000` adds one absent false detection. Other eight clip counts are unchanged. The four correctness transitions therefore omit a fifth error-type transition, `match176_000:44`, from visible abstention to wrong location; it is included in the metrics and this interpretation.

## Training and verification

The original pretrained HRNet is retained, with only `final_layers.0.weight` and `final_layers.0.bias` updated. Independent tensor comparison confirms every non-head parameter and buffer is exactly unchanged. Batch normalization remains frozen. A three-update smoke has finite positive losses and nonzero gradients; model, optimizer, scaler and seeds are reset before the real epoch. One pass consumes all 2,399 training samples from 48 clips, drawing 1,378 visible and 1,021 absent spatial views. Seed 7, AdamW 1e-5, weight decay 1e-4, AMP and gradient clipping 1 follow the frozen protocol.

Preparation takes 226.62 seconds and the training epoch 200.43 seconds. Mean training loss is 0.00008167936. Original selection reuses eight verified clip results and newly infers four in 134.78 seconds. All twelve adapted selection clips take 438.93 seconds. Historical cached original time is 310.37 seconds; these mixed cached/new sparse timings do not establish a model speed gain. Peak allocated GPU memory is 275,635,200 bytes (262.87 MiB), not total process/device memory. Runtime: Python 3.13.5, PyTorch 2.13.0+cu126, CUDA 12.6, RTX 4050 Laptop GPU.

Independent verification checks source/video/label hashes, train-selection separation, all 600 targets and real temporal windows, unchanged inference code, cached original rows, selected candidate identity, all ten pooled scores, all selected per-clip scores, and both threshold choices. Twenty-one focused training/data/inference tests pass. Frozen checkpoint SHA256 is `5e85ba24172e80e72a0c51607ce6d46b410c5b66955506535cf6281c051c96f1`.

## Visual review

All four preselected correctness changes are inspected in the two saved comparison boards. `match176_000:293` selects static bright crowd/seat detail near the image edge; this supports a clutter false detection. `match207_000:93` selects a pale streak beside a court line; the three-frame crop does not establish that it is a ball, so the publisher absence error is retained without certifying full-frame absence. `match199_000:207` agrees closely with a label beside the far player's leg, but body/racket/background clutter makes independent ball visibility ambiguous at this resolution. `match225_000:126` shows a small moving light feature at the net aligned with the label, replacing an original prediction in the near foreground. No source labels are changed.

## Decision and evidence

The candidate advances only to the frozen eighteen-clip raw development comparison described in `WASB_HEAD_EXTERNAL_PROTOCOL.md`. Neither model exceeds 95% precision and recall internally. The gain is extremely small, frame outcomes are dependent, and no significance claim is made. Training uses adjacent temporal context while deployment selection uses FPS-adjusted spacing; source-broadcast independence, publisher absence quality and pretrained exposure remain unknown. Court/player generalization, true ball speed and independent end-to-end qualification remain unresolved. Overall production readiness remains **NO**.

Evidence files:

- `artifacts/training/vision_upgrade/wasb_head_pilot04/manifest.json` (SHA256 `950a88e8c65c2070a38a6e84cfe821df1bae30b5dc865b093a0179cdc2ac19dd`).
- `outputs/vision_upgrade_audit/wasb_head_pilot04/review.json` contains exact independent scores and paired correctness changes.
- `outputs/vision_upgrade_audit/wasb_head_pilot04/visual_review.json` records selected cases, image hashes and inspection notes.
- `outputs/vision_upgrade_audit/wasb_head_pilot04/comparison_00.jpg` and `comparison_01.jpg` preserve the inspected frames.
- `WASB_HEAD_ADAPTATION_PROTOCOL.md` is the unchanged pre-training protocol.
