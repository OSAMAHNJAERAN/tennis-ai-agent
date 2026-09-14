# Pretrained context verifier pilot05

Protocol fixed before feature extraction or training. Test a frozen ImageNet ResNet50 encoder with wider context because the small CNN verifier confuses balls with body/racket details. Use cached TorchVision IMAGENET1K_V2 weights SHA256 `11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca` (102,540,417 bytes), loaded with weights_only=True. Official preprocessing: https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html. ImageNet accuracy is not tennis accuracy.

Keep existing .05 WASB proposals, sample IDs and labels from training20 and expansion12. Extract64x64 RGB patches at reference960x540 around each actual proposal, current and exact cached past frame, with reflected borders. Apply official resize232/center224/ImageNet normalization. Freeze the encoder in evaluation mode, remove its classification layer, and concatenate current2048, past2048 and absolute difference2048 features. Store6144 features as float16, retaining original confidence/reciprocal rank separately. No labels enter feature extraction.

Train LayerNorm6144, Linear6146-to128, ReLU, dropout0.2, Linear128-to1. Eight epochs, seed7, batch64, AdamW .001/weight decay .0001, balanced replacement sampling4,719 draws per epoch. Use all original16 training-clip labels, including the nine previously reviewed uncertain negatives. Select strict frame F1 then precision then recall on the original four internal-selection clips over thresholds .3/.5/.7/.9 and all eight epochs. Freeze selection before external evaluation. Do not relabel external errors or tune thresholds on validation. Pretrained features replace image augmentation; encoder, context and head changes are bundled, not a single-factor ablation.

Verify complete source/feature hashes. Enforce deterministic algorithms and CUBLAS workspace configuration and record runtime. Compare unchanged proposal baseline and prior matched CNN control. No automatic integration; broad camera coverage, independent labels, court/player/racket accuracy and physical analytics remain unresolved.


## Training completed; external evaluation pending

All 5,379 original training-cache candidates were encoded in 867.17 seconds, excluding encoder loading. The feature file is approximately 66 MB. Nineteen targeted tests passed in 18.77 seconds, covering crop centers, resolution scaling, reflection, invalid observations, score shape, feature/source hashes, clip completeness and unchanged label mappings.

All eight head-training epochs completed, selecting epoch 8 at threshold 0.3. Internal counts are 150 TP, 20 FP, 32 FN and 12 TN: precision 88.24%, recall 82.42%, F1 85.23%. The head has 799,233 parameters; training plus internal selection took 4.19 seconds, excluding startup and cache loading. This internal result is worse than the matched small-CNN control.

The selected head was frozen before starting external feature extraction: SHA256 `53bbb842ac0a7f3b076043215faaf854eb501b420c3fe0b7206c5e8acf03bf0b`. External evaluation remains pending until the complete twelve-clip feature cache is verified. No external accuracy result is inferred from training loss or internal selection.


## Completed external result and rejection

The external cache contains all 4,459 original proposals; extraction took 257.79 seconds excluding encoder loading. Frozen epoch 8 / threshold 0.3 gives **450 TP, 83 FP, 93 FN, 35 TN**, precision **84.43%**, recall **82.87%**, F1 **83.64%**, on all 600 unchanged labels. The matched small-CNN control gives 471/28/72/38 (94.39% precision, 86.74% recall). Paired results recover 32 correct visible frames but lose 53; correct absences gain 14 and lose 17. The pretrained context candidate is rejected for production.

There are 61 wrong visible localizations, compared with nine for the CNN control. Forty-eight still have a correct candidate available. In 24 errors, the publisher ball target lies within the incorrectly selected 64-pixel context. This geometry does not prove a single cause, but two inspected examples show the off-center ball inside a highly scored wrong crop. Other examples show high scores for player/racket context while a correct blurred-ball proposal scores lower. Four deterministic review cases are saved in `outputs/vision_upgrade_audit/verifier_context_pilot05/rank_failure_examples.jpg` with selection metadata. These are diagnostics, not independent labels.

External report: `artifacts/validation/vision_upgrade/expansion12_candidate_verifier_pilot05_context.json`. Paired comparison and all 61 wrong-location diagnostics: `outputs/vision_upgrade_audit/verifier_context_pilot05/comparison.json`. No default changes or label edits. The next research hypothesis separates candidate rejection from localization ranking rather than treating a global context score as precise center evidence.
